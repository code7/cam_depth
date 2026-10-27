# -*- coding: utf-8 -*-
"""生产界面（前端）：订阅相机服务画面/状态、Modbus 状态，配置走配置服务。

本进程不导入相机 SDK，也不碰串口和配置文件。
"""

import argparse
import base64
import io
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cam_config as cfgmod
from common import topics
from common.mqtt_node import MqttNode

LABELS = ("A", "B")
DISPLAY_INTERVAL_MS = 80
REFRESH_CHOICES = ("30", "50", "80", "120", "200", "300")

C_OK = "#0a7a2f"
C_ERR = "#c00000"
C_WARN = "#b06000"
C_IDLE = "#888888"
C_TX = "#1a4fa0"

UI_FONT = ("Microsoft YaHei UI", 10)
MONO_FONT = ("Consolas", 10)

KIND_COLOR = {"ok": C_OK, "warn": C_WARN, "err": C_ERR, "idle": C_IDLE}


def stamp():
    return time.strftime("%H:%M:%S")


def _put_latest(q, obj):
    try:
        q.put_nowait(obj)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(obj)
        except queue.Full:
            pass


class CameraPanel:
    def __init__(self, app, parent, index):
        self.app = app
        self.index = index
        self.panel = LABELS[index]
        self.name = f"{self.panel} 面相机"

        self.serial = None
        self.status = {}
        self.meta = None
        self.preview = None
        self.src_w = 0
        self.src_h = 0
        self.depth_w = 0
        self.depth_h = 0
        self.scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._photo = None
        self._img_item = None
        self._roi_item = None
        self._last_frame_wall = None

        self._build_ui(parent)

    def _build_ui(self, parent):
        self.frame = ttk.LabelFrame(parent, text=self.name)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        head = ttk.Frame(self.frame)
        head.grid(row=0, column=0, sticky="we", padx=4, pady=(4, 2))
        ttk.Label(head, text=self.name, font=(UI_FONT[0], 11, "bold")).pack(side="left")
        self.lbl_state = tk.Label(head, text="等待服务", fg=C_IDLE, font=UI_FONT)
        self.lbl_state.pack(side="right")
        self.lbl_serial = tk.Label(head, text="", fg="#555", font=MONO_FONT)
        self.lbl_serial.pack(side="right", padx=(0, 12))
        self.btn_retry = ttk.Button(head, text="重试连接", command=self.retry_open,
                                    state="disabled")
        self.btn_retry.pack(side="right", padx=(0, 10))

        self.canvas = tk.Canvas(self.frame, width=600, height=450, bg="black",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel(e, -120))
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_zoom())

        self.lbl_coord = tk.Label(self.frame, text="坐标: (---, ---)   深度: ---",
                                  anchor="w", font=MONO_FONT, fg="#555")
        self.lbl_coord.grid(row=2, column=0, sticky="we", padx=4)
        self.lbl_status = tk.Label(self.frame, text="等待相机服务…", anchor="w",
                                   justify="left", wraplength=600,
                                   font=UI_FONT, fg="#555")
        self.lbl_status.grid(row=3, column=0, sticky="we", padx=4, pady=(0, 4))

    # -------------------------------------------------- 数据
    def apply_frame(self, obj):
        raw = obj.get("jpeg")
        if not raw:
            return
        try:
            data = base64.b64decode(raw)
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return
        self.preview = np.asarray(image)
        self.meta = obj
        self.src_h, self.src_w = self.preview.shape[:2]
        self.depth_w = int(obj.get("depth_w") or self.src_w)
        self.depth_h = int(obj.get("depth_h") or self.src_h)
        self.serial = obj.get("serial")
        self._last_frame_wall = time.monotonic()
        self._redraw()

    def set_status(self, st):
        self.status = st or {}
        self.serial = self.status.get("serial")
        state = self.status.get("state") or "等待服务"
        kind = self.status.get("state_kind") or "idle"
        self.lbl_state.config(text=state, fg=KIND_COLOR.get(kind, C_IDLE))
        self.lbl_serial.config(text=f"ID {self.serial}" if self.serial else "")
        online = self.status.get("online", True)
        self.btn_retry.config(
            state="normal" if online and self.serial
            and not self.status.get("running") else "disabled")
        self._refresh_status_text()
        self._update_roi_overlay()

    def _refresh_status_text(self):
        st = self.status
        if not st:
            return
        parts = []
        fps = st.get("fps")
        if fps:
            parts.append(f"帧间隔 {st.get('arrival_ms', 0):.0f} ms ≈ {fps:.1f} fps")
            parts.append(f"收 {st.get('frames_received', 0)} 帧")
            if st.get("frames_dropped"):
                parts.append(f"丢 {st.get('frames_dropped')} 帧")
        else:
            parts.append("等待首帧")
        plane = st.get("plane")
        if plane:
            center = st.get("plane_center_mm")
            extra = f"，中心距离 {center:.2f} mm" if center is not None else ""
            parts.append(
                f"基准面 z={plane[0]:.5f}x+{plane[1]:.5f}y+{plane[2]:.2f}{extra}")
        elif st.get("calibrated") is False:
            parts.append("未标定基准面")
        roi = st.get("roi")
        if roi:
            parts.append(f"范围 {roi[0]},{roi[1]} → {roi[2]},{roi[3]}"
                         f"（{'自动' if st.get('roi_mode') == 'auto' else '手动'}）")
        else:
            parts.append("范围未设置")
        if st.get("reason") and not st.get("ready"):
            parts.append(st["reason"])
        if st.get("last_error"):
            parts.append(st["last_error"])
        self.lbl_status.config(text="　".join(parts),
                               fg=C_WARN if not st.get("ready") else "#555")

    # -------------------------------------------------- 绘制
    def _redraw(self):
        if self.preview is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 600, 450
        iw, ih = self.src_w, self.src_h
        if not iw or not ih:
            return
        fit = min(cw / iw, ch / ih)
        scale = fit * self.zoom
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        self.scale = scale
        self.offset_x, self.offset_y = self._clamp_offset(new_w, new_h, cw, ch)
        interp = cv2.INTER_AREA if new_w < iw else cv2.INTER_LINEAR
        resized = cv2.resize(self.preview, (new_w, new_h), interpolation=interp)
        image = Image.fromarray(resized)
        if self._img_item is None:
            self._photo = ImageTk.PhotoImage(image)
            self._img_item = self.canvas.create_image(
                self.offset_x, self.offset_y, anchor="nw", image=self._photo, tags="img")
        elif self._photo.width() != new_w or self._photo.height() != new_h:
            self._photo = ImageTk.PhotoImage(image)
            self.canvas.coords(self._img_item, self.offset_x, self.offset_y)
            self.canvas.itemconfigure(self._img_item, image=self._photo)
        else:
            self._photo.paste(image)
            self.canvas.coords(self._img_item, self.offset_x, self.offset_y)
        self._update_roi_overlay()

    def _clamp_offset(self, new_w, new_h, cw, ch):
        if new_w <= cw:
            ox = (cw - new_w) // 2
        else:
            ox = min(max(self.offset_x, cw - new_w), 0)
        if new_h <= ch:
            oy = (ch - new_h) // 2
        else:
            oy = min(max(self.offset_y, ch - new_h), 0)
        return ox, oy

    def _roi_in_preview(self):
        roi = (self.status or {}).get("roi") or (self.meta or {}).get("roi")
        if not roi or not self.depth_w or not self.depth_h:
            return None
        sx = self.src_w / float(self.depth_w)
        sy = self.src_h / float(self.depth_h)
        return (roi[0] * sx, roi[1] * sy, roi[2] * sx, roi[3] * sy)

    def _update_roi_overlay(self):
        roi = self._roi_in_preview()
        if not roi:
            if self._roi_item is not None:
                self.canvas.delete(self._roi_item)
                self._roi_item = None
            return
        x0, y0, x1, y1 = roi
        cx0 = self.offset_x + x0 * self.scale
        cy0 = self.offset_y + y0 * self.scale
        cx1 = self.offset_x + x1 * self.scale
        cy1 = self.offset_y + y1 * self.scale
        color = "#ffb000" if self.index == 0 else "#00c8ff"
        if self._roi_item is None:
            self._roi_item = self.canvas.create_rectangle(
                cx0, cy0, cx1, cy1, outline=color, width=2, tags="roi")
        else:
            self.canvas.coords(self._roi_item, cx0, cy0, cx1, cy1)
            self.canvas.itemconfigure(self._roi_item, outline=color)

    # -------------------------------------------------- 交互
    def on_mouse_wheel(self, event, delta=None):
        if self.preview is None:
            return
        if delta is None:
            delta = event.delta
        factor = 1.1 if delta > 0 else 1 / 1.1
        new_zoom = min(max(self.zoom * factor, 0.05), 20.0)
        if new_zoom == self.zoom:
            return
        img_x = (event.x - self.offset_x) / self.scale
        img_y = (event.y - self.offset_y) / self.scale
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 600, 450
        self.zoom = new_zoom
        scale = min(cw / self.src_w, ch / self.src_h) * self.zoom
        self.scale = scale
        self.offset_x = event.x - img_x * scale
        self.offset_y = event.y - img_y * scale
        self._redraw()

    def reset_zoom(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._redraw()

    def on_mouse_move(self, event):
        if self.preview is None or not self.src_w:
            self.lbl_coord.config(text="坐标: (---, ---)   深度: ---")
            return
        px = (event.x - self.offset_x) / self.scale
        py = (event.y - self.offset_y) / self.scale
        x = int(px * self.depth_w / self.src_w)
        y = int(py * self.depth_h / self.src_h)
        if 0 <= x < self.depth_w and 0 <= y < self.depth_h:
            self.lbl_coord.config(text=f"坐标: ({x}, {y})   深度: 查询中…")
            self.app.request_probe(self.index, x, y)
        else:
            self.lbl_coord.config(text="坐标: (---, ---)   深度: ---")

    def set_probe(self, probe):
        if not probe:
            return
        text = (f"坐标: ({probe['x']}, {probe['y']})   "
                f"深度: {probe['depth']:.1f} mm")
        if probe.get("height") is not None:
            text += f"   高度: {probe['height']:.1f} mm"
        self.lbl_coord.config(text=text)

    def retry_open(self):
        def done(resp, err):
            if err or not resp.get("ok"):
                self.app.log(f"{self.name} 重试打开失败：{err or resp.get('error')}",
                             "err")
        self.app.call_camera("open", {"panel": self.panel}, done)


class App:
    def __init__(self, root, mqtt_host="127.0.0.1", mqtt_port=1883, smoke=0):
        self.root = root
        self.closed = False
        self.smoke = smoke
        self.root.title("双深度相机 A/B 面测量（前后端分离 / MQTT）")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1320, sw - 60), min(900, sh - 80)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        root.minsize(1000, 650)

        self.node = MqttNode("ui", mqtt_host, mqtt_port)
        self.display_interval_ms = DISPLAY_INTERVAL_MS
        self.inventory = {}
        self.config_ready = {}
        self.modbus_status = {}
        self.modbus_regs = {}
        self.config_window = None
        self.log_lines = []

        self.frame_queues = [queue.Queue(maxsize=1) for _ in range(2)]
        self.status_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.modbus_status_queue = queue.Queue(maxsize=1)
        self.modbus_regs_queue = queue.Queue(maxsize=1)
        self.action_results = queue.Queue()
        self._probe_pending = set()
        self._probe_last = {}
        self._config_retry_at = 0.0

        self._subscribe()
        self.node.start(wait=1.5)

        self.show_height = tk.BooleanVar(value=False)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(120, self._initial_sync)
        self.root.after(200, self._tick)
        if smoke:
            self.root.after(int(smoke * 1000), self.on_close)

    # -------------------------------------------------- MQTT
    def _subscribe(self):
        self.node.subscribe(topics.CAMERA_INVENTORY,
                            lambda t, o: setattr(self, "inventory", o or {}))
        self.node.subscribe(f"{topics.NS}/camera/+/status",
                            lambda t, o: self._on_camera_status(t, o))
        self.node.subscribe(f"{topics.NS}/camera/+/frame",
                            lambda t, o: self._on_camera_frame(t, o))
        self.node.subscribe(topics.CONFIG_READY, self._on_config_ready)
        self.node.subscribe(topics.MODBUS_STATUS,
                            lambda t, o: _put_latest(self.modbus_status_queue, o or {}))
        self.node.subscribe(topics.MODBUS_REGISTERS,
                            lambda t, o: _put_latest(self.modbus_regs_queue, o or {}))
        self.node.subscribe(f"{topics.NS}/log/+",
                            lambda t, o: self._on_log(t, o))

    def _on_camera_status(self, topic, obj):
        panel = topic.rsplit("/", 2)[-2]
        if panel in LABELS:
            self.status_queue.put((LABELS.index(panel), obj))

    def _on_camera_frame(self, topic, obj):
        panel = topic.rsplit("/", 2)[-2]
        if panel in LABELS:
            _put_latest(self.frame_queues[LABELS.index(panel)], obj)

    def _on_config_ready(self, topic, obj):
        if not isinstance(obj, dict):
            return
        self.config_ready.update(obj)

    def _on_log(self, topic, obj):
        if not isinstance(obj, dict):
            return
        line = f"{stamp()}  [{obj.get('service', '?')}] {obj.get('msg', '')}"
        self.log_queue.put(line)

    # -------------------------------------------------- UI
    def _build_ui(self):
        bar = ttk.Frame(self.root)
        bar.grid(row=0, column=0, sticky="we", padx=6, pady=(6, 2))
        ttk.Button(bar, text="重新扫描", command=self.rescan).pack(side="left")
        ttk.Button(bar, text="配置", command=self.open_config).pack(side="left", padx=6)
        self.btn_modbus = ttk.Button(bar, text="连接485", command=self.toggle_modbus)
        self.btn_modbus.pack(side="left", padx=(18, 2))
        ttk.Button(bar, text="寄存器监视", command=self.show_registers).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="高度图", variable=self.show_height,
                        command=self._on_show_height).pack(side="left", padx=(18, 2))

        self.lbl_status = tk.Label(self.root, text="等待相机服务…", anchor="w",
                                   font=UI_FONT, fg="#555")
        self.lbl_status.grid(row=1, column=0, sticky="we", padx=6)
        self.lbl_health = tk.Label(self.root, text="", anchor="w", font=UI_FONT, fg="#555")
        self.lbl_health.grid(row=2, column=0, sticky="we", padx=6)
        self.lbl_modbus = tk.Label(self.root, text="Modbus：未知", anchor="w",
                                   font=UI_FONT, fg=C_IDLE)
        self.lbl_modbus.grid(row=3, column=0, sticky="we", padx=6)

        container = ttk.Frame(self.root)
        container.grid(row=4, column=0, sticky="nsew", padx=6, pady=6)
        container.rowconfigure(0, weight=1)
        for col in range(2):
            container.columnconfigure(col, weight=1)
        self.panels = []
        for i in range(2):
            panel = CameraPanel(self, container, i)
            panel.frame.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            self.panels.append(panel)

        logf = ttk.LabelFrame(self.root, text="运行日志", padding=(6, 4))
        logf.grid(row=5, column=0, sticky="nsew", padx=6, pady=(0, 6))
        holder = ttk.Frame(logf)
        holder.pack(fill="both", expand=True)
        self.txt_log = tk.Text(holder, height=7, wrap="none", font=MONO_FONT,
                               state="disabled", bg="#fbfbfb")
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt_log.pack(side="left", fill="both", expand=True)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(4, weight=1)
        self.root.rowconfigure(5, weight=0)

    def log(self, msg, level="info"):
        line = f"{stamp()}  {msg}"
        if level == "err":
            print(line, file=sys.stderr, flush=True)
        else:
            print(line, flush=True)
        self.log_queue.put(line)

    def _append_log(self, line):
        self.log_lines.append(line)
        if len(self.log_lines) > 800:
            del self.log_lines[:-800]
        try:
            self.txt_log.configure(state="normal")
            self.txt_log.insert("end", line + "\n")
            n = int(self.txt_log.index("end-1c").split(".")[0])
            if n > 2000:
                self.txt_log.delete("1.0", f"{n - 2000}.0")
            self.txt_log.configure(state="disabled")
            self.txt_log.see("end")
        except Exception:
            pass

    # -------------------------------------------------- 异步
    def run_async(self, fn, on_done):
        def worker():
            try:
                result = fn()
            except Exception as e:
                self.action_results.put((on_done, None, e))
            else:
                self.action_results.put((on_done, result, None))
        threading.Thread(target=worker, daemon=True).start()

    def call_camera(self, action, data, on_done, timeout=5.0):
        self.run_async(
            lambda: self.node.request(topics.CAMERA_REQ, topics.CAMERA_RES,
                                      action, data, timeout), on_done)

    def call_config(self, action, data, on_done, timeout=8.0):
        self.run_async(
            lambda: self.node.request(topics.CONFIG_REQ, topics.CONFIG_RES,
                                      action, data, timeout), on_done)

    def _initial_sync(self):
        if self.closed:
            return
        self.call_camera("state", {}, self._on_initial_camera, timeout=3.0)
        self.call_config("get_all", {}, self._on_initial_config, timeout=3.0)
        self.call_camera("set_stream", {"enabled": True},
                         lambda resp, err: None, timeout=2.0)
        self.call_camera("set_show_height", {"enabled": bool(self.show_height.get())},
                         lambda resp, err: None, timeout=2.0)

    def _on_initial_camera(self, resp, err):
        if err or not resp.get("ok"):
            self.log(f"相机服务连接失败：{err or resp.get('error')}", "err")
            return
        data = resp.get("data") or {}
        self.inventory = data.get("inventory") or self.inventory
        for panel in self.panels:
            st = (data.get("panels") or {}).get(panel.panel)
            if st:
                panel.set_status(st)

    def _on_initial_config(self, resp, err):
        if err or not resp.get("ok"):
            self.config_ready["online"] = False
            self.log(f"配置服务连接失败：{err or resp.get('error')}", "warn")
            return
        data = resp.get("data") or {}
        data["online"] = True
        self.config_ready = data
        app = data.get("app") or {}
        try:
            self.display_interval_ms = max(10, int(app.get(
                "display_interval_ms", DISPLAY_INTERVAL_MS)))
        except (TypeError, ValueError):
            self.display_interval_ms = DISPLAY_INTERVAL_MS

    # -------------------------------------------------- 主节拍
    def _tick(self):
        if self.closed:
            return
        for i, panel in enumerate(self.panels):
            latest = None
            while True:
                try:
                    latest = self.frame_queues[i].get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                panel.apply_frame(latest)
        self._drain_status()
        self._drain_modbus()
        self._drain_actions()
        self._drain_logs()
        self._update_health()
        self._maybe_retry_config()
        self.root.after(max(10, self.display_interval_ms), self._tick)

    def _drain_status(self):
        latest = {}
        while True:
            try:
                index, obj = self.status_queue.get_nowait()
            except queue.Empty:
                break
            latest[index] = obj
        for index, obj in latest.items():
            self.panels[index].set_status(obj)

    def _drain_modbus(self):
        try:
            self.modbus_status = self.modbus_status_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.modbus_regs = self.modbus_regs_queue.get_nowait()
        except queue.Empty:
            pass
        self._refresh_modbus_label()

    def _drain_actions(self):
        while True:
            try:
                callback, result, err = self.action_results.get_nowait()
            except queue.Empty:
                return
            if self.closed:
                continue
            try:
                callback(result, err)
            except Exception as e:
                self.log(f"界面回调异常：{e}", "err")

    def _drain_logs(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                return
            self._append_log(line)

    def _maybe_retry_config(self):
        if self.config_ready.get("online"):
            return
        now = time.monotonic()
        if now < self._config_retry_at:
            return
        self._config_retry_at = now + 5.0
        self.call_config("get_all", {}, self._on_initial_config, timeout=2.0)

    def _update_health(self):
        inv = self.inventory or {}
        cameras = inv.get("cameras") or []
        if inv.get("error"):
            self.lbl_status.config(text=f"相机服务：{inv['error']}", fg=C_ERR)
        elif cameras:
            names = "、".join(c.get("serial", "?") for c in cameras)
            extra = "（模拟）" if inv.get("simulate") else ""
            self.lbl_status.config(text=f"发现 {len(cameras)} 台相机{extra}：{names}",
                                   fg=C_OK)
        else:
            self.lbl_status.config(text="等待相机服务扫描相机…", fg=C_WARN)

        parts = []
        ready_all = True
        for i, panel in enumerate(self.panels):
            st = panel.status or {}
            if not st.get("serial"):
                state = "未分配"
                ready_all = False
            else:
                state = st.get("state") or "未知"
                if not st.get("ready"):
                    ready_all = False
            parts.append(f"{LABELS[i]}：{state}")
        self.lbl_health.config(text="　|　".join(parts),
                               fg=C_OK if ready_all else C_WARN)

    def _refresh_modbus_label(self):
        st = self.modbus_status or {}
        if not st:
            self.lbl_modbus.config(text="Modbus：等待服务…", fg=C_WARN)
            return
        if not st.get("online"):
            self.lbl_modbus.config(text="Modbus：服务离线", fg=C_ERR)
            return
        if not st.get("running"):
            err = st.get("last_error") or "未连接"
            self.lbl_modbus.config(text=f"Modbus：未连接（{err}）", fg=C_ERR)
            return
        text = (f"Modbus：已连接 {st.get('port')} @{st.get('baudrate')} "
                f"站号{st.get('slave_addr')} 基址{st.get('addr_base')}　"
                f"状态 {st.get('status_name')}")
        if st.get("last_result"):
            a, b = st["last_result"]
            text += f"　A {a:.1f} mm，B {b:.1f} mm"
        if st.get("last_error"):
            text += f"　故障：{st['last_error']}"
        self.lbl_modbus.config(text=text, fg=C_OK if not st.get("last_error") else C_WARN)

    # -------------------------------------------------- 操作
    def request_probe(self, index, x, y):
        now = time.monotonic()
        if index in self._probe_pending or now - self._probe_last.get(index, 0) < 0.06:
            return
        self._probe_pending.add(index)
        self._probe_last[index] = now

        def done(resp, err):
            self._probe_pending.discard(index)
            if err or not resp.get("ok"):
                return
            probe = (resp.get("data") or {}).get("probe")
            if probe and not self.closed:
                self.panels[index].set_probe(probe)

        self.call_camera("probe", {"panel": LABELS[index], "x": x, "y": y},
                         done, timeout=1.5)

    def rescan(self):
        self.log("请求重新扫描相机")

        def done(resp, err):
            if err or not resp.get("ok"):
                self.log(f"重新扫描失败：{err or resp.get('error')}", "err")
        self.call_camera("rescan", {}, done, timeout=3.0)

    def toggle_modbus(self):
        running = bool((self.modbus_status or {}).get("running"))
        action = "disconnect" if running else "connect"
        self.log(f"请求 Modbus {'断开' if running else '连接'}")
        self.run_async(
            lambda: self.node.request(topics.MODBUS_REQ, topics.MODBUS_RES,
                                      action, {}, 8.0),
            lambda resp, err: self.log(
                f"Modbus 操作失败：{err or resp.get('error')}", "err")
            if err or not resp.get("ok") else None)

    def _on_show_height(self):
        enabled = bool(self.show_height.get())

        def done(resp, err):
            if err or not resp.get("ok"):
                self.log(f"切换高度图失败：{err or resp.get('error')}", "err")
        self.call_camera("set_show_height", {"enabled": enabled}, done, timeout=2.0)

    def show_registers(self):
        st = self.modbus_regs or self.modbus_status
        if not st:
            self.run_async(
                lambda: self.node.request(topics.MODBUS_REQ, topics.MODBUS_RES,
                                          "state", {}, 3.0),
                self._show_registers_report)
            return
        self._show_registers_report({"ok": True, "data": st}, None)

    def _show_registers_report(self, resp, err):
        if err or not resp.get("ok"):
            messagebox.showerror("寄存器监视", f"读取失败：{err or resp.get('error')}")
            return
        st = resp.get("data") or {}
        regs = st.get("registers") or []
        names = st.get("names") or {}
        readonly = set(st.get("readonly") or [])
        base = int(st.get("addr_base") or 0)
        scale = float(st.get("dist_scale") or 10.0)
        signed = bool(st.get("dist_signed"))
        abs_max = float(st.get("dist_abs_max") or 6000.0)
        lines = ["寄存器一览（文档编号 / 线上地址 / 名称 / 值）", ""]
        for i, val in enumerate(regs):
            rw = "R" if i in readonly else "W"
            extra = ""
            if i == 0:
                extra = f"   ({st.get('status_name', '?')})"
            elif i in (2, 3):
                raw = int(val)
                mm = (raw - 65536 if signed and raw >= 32768 else raw) / scale
                extra = f"   ({mm:+.1f} mm)"
            lines.append(f"  {40001 + i:<6} wire={i + base:<3} {rw}  "
                         f"{names.get(str(i), '?'):<20} {val:<7}{extra}")
        lines += ["", f"距离编码：0.1mm/LSB，带符号量程 "
                      f"{st.get('dist_min', 0):.1f}~{st.get('dist_max', 0):.1f} mm"
                      f"（近摄像头为负、远离为正）",
                  f"有效范围：|距离| ≤ {abs_max / 1000.0:.0f} m，超过按无效数据处理",
                  f"连接状态：{'已连接' if st.get('running') else '未连接'}"]
        if st.get("last_result"):
            a, b = st["last_result"]
            lines.append(f"上次相对校准面位置：A面 {a:.1f} mm，B面 {b:.1f} mm"
                         f"（周期 {st.get('cycle_ms') or 0:.0f} ms）")
        if st.get("last_error"):
            lines.append(f"上次故障：{st['last_error']}")
        stats = st.get("stats") or {}
        if stats:
            lines += ["", f"通讯统计：收 {stats.get('rx', 0)} 帧，发 {stats.get('tx', 0)} 帧，"
                          f"CRC 错 {stats.get('crc_err', 0)}，站号不符 {stats.get('addr_skip', 0)}，"
                          f"异常响应 {stats.get('exc', 0)}"]
        lines += ["", "PLC 写入记录（最近 20 条）："]
        for item in (st.get("write_log") or [])[-20:]:
            try:
                idx, val, ts = item
            except (TypeError, ValueError):
                continue
            lines.append(f"  {ts}  40001+{idx} ({names.get(str(idx), '?')}) = {val}")
        lines += ["", "服务日志（最近 40 条）："]
        lines += ["  " + x for x in self.log_lines[-40:]]
        self.show_report("寄存器 / 通讯监视", "\n".join(lines))

    def show_report(self, title, text):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("980x620")
        holder = ttk.Frame(win)
        holder.pack(fill="both", expand=True, padx=6, pady=6)
        txt = tk.Text(holder, wrap="none", font=MONO_FONT)
        sb = ttk.Scrollbar(holder, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=(0, 8))
        return win

    # -------------------------------------------------- 配置
    def inventory_serials(self):
        serials = []
        for cam in (self.inventory or {}).get("cameras") or []:
            serial = str(cam.get("serial") or "")
            if serial:
                serials.append(serial)
        for panel in self.panels:
            if panel.serial:
                serials.append(str(panel.serial))
        return sorted(set(serials))

    def panel_serial(self, index):
        panels = (self.inventory or {}).get("panels") or {}
        serial = panels.get(LABELS[index])
        if serial:
            return str(serial)
        status = self.panels[index].status or {}
        return str(status.get("serial") or "")

    def open_config(self):
        if self.config_window is not None:
            try:
                self.config_window.win.lift()
            except Exception:
                pass
            return
        self.config_window = ConfigWindow(self)

    # -------------------------------------------------- 关闭
    def on_close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.node.publish_json(topics.CAMERA_REQ,
                                   {"id": "close", "action": "set_stream",
                                    "enabled": False}, qos=0)
        except Exception:
            pass
        try:
            self.node.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


class ConfigWindow:
    def __init__(self, app):
        self.app = app
        self.root = app.root
        self.win = tk.Toplevel(app.root)
        self.win.title("配置")
        sw, sh = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
        w, h = min(1120, sw - 60), min(760, sh - 80)
        self.win.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.win.minsize(980, 680)
        self.serials = []
        self.drafts = {}
        self.sections = [{}, {}]
        self.configs = {}
        self.app_cfg = {}
        rep = (app.config_ready or {}).get("report") or {}
        self.persist_serials = set(rep.get("unlocked") or []) | set(
            rep.get("missing") or [])
        self._closing = False
        self._build()
        self._init_sections()
        self._refresh_previews()
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.app.call_config("get_all", {}, self._on_snapshot, timeout=4.0)

    def _build(self):
        holder = ttk.Frame(self.win)
        holder.pack(fill="both", expand=True, padx=8, pady=8)
        nb = ttk.Notebook(holder)
        nb.pack(fill="both", expand=True)
        cam_tab = ttk.Frame(nb)
        app_tab = ttk.Frame(nb)
        nb.add(cam_tab, text="A/B 相机与计算范围")
        nb.add(app_tab, text="485 / 运行")

        for col in range(2):
            cam_tab.columnconfigure(col, weight=1)
        cam_tab.rowconfigure(0, weight=1)
        for i in range(2):
            box = ttk.LabelFrame(cam_tab, text=f"{LABELS[i]} 面窗口", padding=6)
            box.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            self._build_camera_section(i, box)
        self._build_app_tab(app_tab)

        bottom = ttk.Frame(self.win)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bottom, text="取消", command=self.close).pack(side="right")
        ttk.Button(bottom, text="保存并应用", command=self.save).pack(side="right", padx=6)

    def _build_camera_section(self, index, box):
        s = self.sections[index]
        s["serial"] = ""
        s["fit"] = None
        s["drag"] = None
        s["rect"] = None
        s["photo"] = None

        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="窗口绑定:").pack(side="left")
        combo = ttk.Combobox(row, state="readonly", width=20, values=[""] + self.serials)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e, i=index: self._on_serial_change(i))
        s["combo"] = combo

        s["var_auto_cal"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="启动时自动校准基准面（使用下面的计算范围）",
                        variable=s["var_auto_cal"]).pack(anchor="w", pady=(6, 2))

        filt = ttk.Frame(box)
        filt.pack(fill="x", pady=2)
        ttk.Label(filt, text="显示距离 min/max (mm):").pack(side="left")
        s["var_min"] = tk.StringVar()
        s["var_max"] = tk.StringVar()
        ttk.Entry(filt, textvariable=s["var_min"], width=7).pack(side="left", padx=2)
        ttk.Entry(filt, textvariable=s["var_max"], width=7).pack(side="left", padx=(0, 10))
        ttk.Label(filt, text="最大单坑(mm²):").pack(side="left")
        s["var_amax"] = tk.StringVar()
        ttk.Entry(filt, textvariable=s["var_amax"], width=8).pack(side="left", padx=2)

        roi_row = ttk.Frame(box)
        roi_row.pack(fill="x", pady=(6, 2))
        s["var_roi_mode"] = tk.StringVar(value="auto")
        ttk.Radiobutton(roi_row, text="自动计算范围", value="auto",
                        variable=s["var_roi_mode"],
                        command=lambda i=index: self._mode_changed(i)).pack(side="left")
        ttk.Radiobutton(roi_row, text="手动框选", value="manual",
                        variable=s["var_roi_mode"],
                        command=lambda i=index: self._mode_changed(i)).pack(side="left", padx=8)
        ttk.Button(roi_row, text="自动计算",
                   command=lambda i=index: self._auto_roi(i)).pack(side="left", padx=(12, 2))
        ttk.Button(roi_row, text="清除",
                   command=lambda i=index: self._clear_roi(i)).pack(side="left", padx=2)

        s["lbl_roi"] = tk.Label(box, text="范围：未设置", anchor="w", fg="#555",
                                font=MONO_FONT)
        s["lbl_roi"].pack(fill="x")
        s["lbl_hint"] = tk.Label(box, text="在下方预览图上拖拽即可手动框选",
                                 anchor="w", fg="#777", font=UI_FONT)
        s["lbl_hint"].pack(fill="x")

        canvas = tk.Canvas(box, width=430, height=300, bg="black",
                           highlightthickness=1, highlightbackground="#999")
        canvas.pack(fill="both", expand=True, pady=4)
        canvas.bind("<ButtonPress-1>", lambda e, i=index: self._drag_start(i, e))
        canvas.bind("<B1-Motion>", lambda e, i=index: self._drag_move(i, e))
        canvas.bind("<ButtonRelease-1>", lambda e, i=index: self._drag_end(i, e))
        s["canvas"] = canvas

        cal_row = ttk.Frame(box)
        cal_row.pack(fill="x")
        ttk.Button(cal_row, text="立即校准平面",
                   command=lambda i=index: self._calibrate_now(i)).pack(side="left")
        s["lbl_cal"] = tk.Label(cal_row, text="", anchor="w", fg="#555", font=UI_FONT)
        s["lbl_cal"].pack(side="left", padx=8)

    def _build_app_tab(self, parent):
        box = ttk.LabelFrame(parent, text="RS-485 / Modbus 从站", padding=8)
        box.pack(fill="x", padx=4, pady=6)
        mcfg = self.app.config_ready.get("app") or {}
        mcfg = mcfg.get("modbus") or cfgmod.DEFAULT_MODBUS
        self.var_auto_modbus = tk.BooleanVar(
            value=bool((self.app.config_ready.get("app") or {}).get(
                "auto_start_modbus", True)))
        ttk.Checkbutton(box, text="开机自动连接 485，等待 PLC 触发",
                        variable=self.var_auto_modbus).grid(
                            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        self.var_port = tk.StringVar(value=str(mcfg.get("port", "")))
        self.var_baud = tk.StringVar(value=str(mcfg.get("baudrate", 9600)))
        self.var_slave = tk.StringVar(value=str(mcfg.get("slave_addr", 1)))
        self.var_base = tk.StringVar(value=str(mcfg.get("addr_base", 0)))
        self.var_timeout = tk.StringVar(value=str(mcfg.get("timeout_s", 3.0)))
        self.var_delay = tk.StringVar(value=str(mcfg.get("measure_delay_s", 2.0)))

        fields = (("串口:", self.var_port, 18), ("波特率:", self.var_baud, 8),
                  ("站号:", self.var_slave, 5), ("地址基址(0/1):", self.var_base, 4),
                  ("测量超时(s):", self.var_timeout, 6),
                  ("触发后延迟(s):", self.var_delay, 6))
        for i, (label, var, width) in enumerate(fields):
            ttk.Label(box, text=label).grid(row=1 + i // 2, column=(i % 2) * 2,
                                            sticky="w", pady=2)
            ttk.Entry(box, textvariable=var, width=width).grid(
                row=1 + i // 2, column=(i % 2) * 2 + 1, sticky="w",
                padx=(2, 16), pady=2)

        run = ttk.LabelFrame(parent, text="显示", padding=8)
        run.pack(fill="x", padx=4, pady=6)
        ttk.Label(run, text="刷新间隔(ms):").pack(side="left")
        ms = int((self.app.config_ready.get("app") or {}).get(
            "display_interval_ms", DISPLAY_INTERVAL_MS))
        self.var_refresh = tk.StringVar(value=str(ms) if str(ms) in REFRESH_CHOICES
                                        else str(DISPLAY_INTERVAL_MS))
        ttk.Combobox(run, textvariable=self.var_refresh, state="readonly", width=6,
                     values=REFRESH_CHOICES).pack(side="left", padx=4)

    # -------------------------------------------------- 初始化
    def _init_sections(self):
        inv = self.app.inventory_serials()
        if self.configs:
            inv = [s for s in inv
                   if s in self.configs or s in self.persist_serials]
        self.serials = sorted(set(inv) | set(self.configs))
        for i in range(2):
            s = self.sections[i]
            s["combo"].config(values=[""] + self.serials)
        bound = {}
        for serial, cfg in self.configs.items():
            panel = cfg.get("panel")
            if panel in (0, 1) and panel not in bound:
                bound[panel] = serial
        used = set()
        for i in range(2):
            s = self.sections[i]
            sel = bound.get(i) or self.app.panel_serial(i)
            if sel not in self.serials or sel in used:
                sel = ""
                for serial in self.serials:
                    if serial not in used:
                        sel = serial
                        break
            s["serial"] = sel
            if sel:
                used.add(sel)
            s["combo"].set(sel)
            self._load_section(i)

    def _on_snapshot(self, resp, err):
        if self._closing or self.app.closed:
            return
        if err or not resp.get("ok"):
            self.app.log(f"读取配置失败，界面使用默认值：{err or resp.get('error')}", "warn")
            return
        data = resp.get("data") or {}
        self.configs = data.get("cameras") or {}
        self.app_cfg = data.get("app") or {}
        rep = data.get("report") or {}
        self.persist_serials = set(rep.get("unlocked") or []) | set(
            rep.get("missing") or []) | set(self.configs)
        app = self.app_cfg
        mcfg = dict(app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        try:
            self.var_auto_modbus.set(bool(app.get("auto_start_modbus", True)))
            self.var_port.set(str(mcfg.get("port", "")))
            self.var_baud.set(str(mcfg.get("baudrate", 9600)))
            self.var_slave.set(str(mcfg.get("slave_addr", 1)))
            self.var_base.set(str(mcfg.get("addr_base", 0)))
            self.var_timeout.set(str(mcfg.get("timeout_s", 3.0)))
            self.var_delay.set(str(mcfg.get("measure_delay_s", 2.0)))
            ms = int(app.get("display_interval_ms", DISPLAY_INTERVAL_MS))
            self.var_refresh.set(str(ms) if str(ms) in REFRESH_CHOICES
                                 else str(DISPLAY_INTERVAL_MS))
        except Exception:
            pass
        self.drafts = {}
        self._init_sections()

    def _draft(self, serial):
        if serial not in self.drafts:
            cfg = self.configs.get(serial) or cfgmod.default_camera_config(serial)
            cal = cfg.get("calibration") or {}
            cut = cfg.get("cut") or {}
            plane = None
            if cfgmod.is_calibrated(cfg):
                plane = tuple(float(v) for v in cal["coefficients"])
            self.drafts[serial] = {
                "auto_calibrate": bool(cfg.get("auto_calibrate", True)),
                "min_mm": float((cfg.get("filter") or {}).get(
                    "min_mm", cfgmod.DEFAULT_FILTER["min_mm"])),
                "max_mm": float((cfg.get("filter") or {}).get(
                    "max_mm", cfgmod.DEFAULT_FILTER["max_mm"])),
                "a_max_mm2": float(cut.get("a_max_mm2", cfgmod.DEFAULT_CUT["a_max_mm2"])),
                "roi_mode": str(cut.get("roi_mode") or cfgmod.DEFAULT_CUT["roi_mode"]),
                "roi": list(cut.get("roi") or cal.get("roi"))
                if (cut.get("roi") or cal.get("roi")) else None,
                "plane": plane,
                "plane_center_mm": cal.get("plane_center_mm"),
            }
        return self.drafts[serial]

    def _load_section(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            s["var_auto_cal"].set(True)
            s["var_min"].set(str(int(cfgmod.DEFAULT_FILTER["min_mm"])))
            s["var_max"].set(str(int(cfgmod.DEFAULT_FILTER["max_mm"])))
            s["var_amax"].set(str(int(cfgmod.DEFAULT_CUT["a_max_mm2"])))
            s["var_roi_mode"].set("auto")
            s["lbl_roi"].config(text="范围：未绑定相机")
            s["lbl_cal"].config(text="")
            self._update_preview(index)
            return
        d = self._draft(serial)
        s["var_auto_cal"].set(d["auto_calibrate"])
        s["var_min"].set(f"{d['min_mm']:.0f}")
        s["var_max"].set(f"{d['max_mm']:.0f}")
        s["var_amax"].set(f"{d['a_max_mm2']:.0f}")
        s["var_roi_mode"].set(d["roi_mode"])
        self._update_roi_label(index)
        if d.get("plane") is not None:
            p = d["plane"]
            s["lbl_cal"].config(text=f"已标定：z={p[0]:.5f}x+{p[1]:.5f}y+{p[2]:.2f}",
                                fg=C_OK)
        else:
            s["lbl_cal"].config(text="未标定", fg=C_WARN)
        self._update_preview(index)

    def _on_serial_change(self, index):
        s = self.sections[index]
        if s.get("serial"):
            self._capture_section(index)
        s["serial"] = s["combo"].get()
        self._load_section(index)

    def _capture_section(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            return True
        d = self._draft(serial)
        try:
            d["auto_calibrate"] = bool(s["var_auto_cal"].get())
            d["min_mm"] = float(s["var_min"].get())
            d["max_mm"] = float(s["var_max"].get())
            d["a_max_mm2"] = max(1.0, float(s["var_amax"].get()))
        except ValueError:
            return False
        if d["max_mm"] <= d["min_mm"]:
            return False
        d["roi_mode"] = s["var_roi_mode"].get()
        return True

    def _update_roi_label(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            s["lbl_roi"].config(text="范围：未绑定相机")
            return
        d = self._draft(serial)
        roi = d.get("roi")
        if roi:
            s["lbl_roi"].config(
                text=f"范围：x {roi[0]}~{roi[2]}，y {roi[1]}~{roi[3]}"
                     f"（{'自动' if d['roi_mode'] == 'auto' else '手动'}）")
        else:
            s["lbl_roi"].config(text="范围：未设置")

    def _mode_changed(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            return
        self._draft(serial)["roi_mode"] = s["var_roi_mode"].get()
        self._update_roi_label(index)
        self._update_preview(index)

    def _clear_roi(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            return
        self._draft(serial)["roi"] = None
        self._update_roi_label(index)
        self._update_preview(index)

    def _auto_roi(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            return
        panel = self.app.panels[index]
        if panel.serial != serial or panel.preview is None:
            messagebox.showwarning("提示", "该窗口还没有这台相机的画面，请等相机打开后再试",
                                   parent=self.win)
            return

        def done(resp, err):
            if self._closing:
                return
            if err or not resp.get("ok"):
                messagebox.showwarning("提示", f"自动计算范围失败：{err or resp.get('error')}",
                                       parent=self.win)
                return
            roi = (resp.get("data") or {}).get("roi")
            if not roi:
                return
            d = self._draft(serial)
            d["roi"] = list(roi)
            d["roi_mode"] = "auto"
            s["var_roi_mode"].set("auto")
            self._update_roi_label(index)
            self._update_preview(index)

        self.app.call_camera("auto_roi", {"panel": LABELS[index]}, done, timeout=8.0)

    def _calibrate_now(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            return
        panel = self.app.panels[index]
        if panel.serial != serial or panel.preview is None:
            messagebox.showwarning("提示", "该窗口还没有这台相机的画面，请等相机打开后再试",
                                   parent=self.win)
            return
        self._capture_section(index)
        d = self._draft(serial)
        s["lbl_cal"].config(text="正在拟合…", fg=C_WARN)

        def done(resp, err):
            if self._closing:
                return
            if err or not resp.get("ok"):
                s["lbl_cal"].config(text=f"校准失败：{err or resp.get('error')}", fg=C_ERR)
                return
            data = resp.get("data") or {}
            fit = data.get("plane")
            if not fit:
                s["lbl_cal"].config(text="校准失败：没有结果", fg=C_ERR)
                return
            d["plane"] = tuple(float(v) for v in fit)
            if data.get("plane_center_mm") is not None:
                d["plane_center_mm"] = float(data["plane_center_mm"])
            if s.get("serial") == serial:
                s["lbl_cal"].config(
                    text=f"已标定：z={fit[0]:.5f}x+{fit[1]:.5f}y+{fit[2]:.2f}", fg=C_OK)

        self.app.call_camera("fit_plane",
                             {"panel": LABELS[index], "roi": d.get("roi")},
                             done, timeout=12.0)

    # -------------------------------------------------- 预览 / 框选
    def _update_preview(self, index):
        s = self.sections[index]
        canvas = s["canvas"]
        canvas.delete("all")
        s["photo"] = None
        s["fit"] = None
        serial = s.get("serial") or ""
        if not serial:
            canvas.create_text(10, 10, anchor="nw", fill="#aaa", text="未绑定相机")
            return
        panel = self.app.panels[index]
        if panel.serial != serial or panel.preview is None:
            canvas.create_text(10, 10, anchor="nw", fill="#aaa",
                               text="等待该相机的画面（保存后生效）")
            return
        cw = max(20, canvas.winfo_width())
        ch = max(20, canvas.winfo_height())
        iw, ih = panel.src_w, panel.src_h
        if not iw or not ih:
            return
        scale = min(cw / iw, ch / ih)
        ow, oh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox, oy = (cw - ow) // 2, (ch - oh) // 2
        resized = cv2.resize(panel.preview, (ow, oh),
                             interpolation=cv2.INTER_AREA if ow < iw else cv2.INTER_LINEAR)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        s["photo"] = photo
        s["fit"] = (scale, ox, oy, iw, ih, panel.depth_w, panel.depth_h)
        canvas.create_image(ox, oy, anchor="nw", image=photo)
        d = self._draft(serial)
        roi = d.get("roi")
        if roi and panel.depth_w:
            sx = iw / float(panel.depth_w)
            sy = ih / float(panel.depth_h)
            color = "#ffb000" if index == 0 else "#00c8ff"
            canvas.create_rectangle(ox + roi[0] * sx * scale, oy + roi[1] * sy * scale,
                                    ox + roi[2] * sx * scale, oy + roi[3] * sy * scale,
                                    outline=color, width=2)

    def _refresh_previews(self):
        if self._closing:
            return
        if any(self.sections[i].get("drag") is not None for i in range(2)):
            self.win.after(500, self._refresh_previews)
            return
        for i in range(2):
            self._update_preview(i)
        self.win.after(500, self._refresh_previews)

    def _canvas_to_depth(self, index, cx, cy):
        fit = self.sections[index].get("fit")
        if not fit:
            return None
        scale, ox, oy, iw, ih, depth_w, depth_h = fit
        x = int((cx - ox) / scale * depth_w / iw)
        y = int((cy - oy) / scale * depth_h / ih)
        x = max(0, min(depth_w - 1, x))
        y = max(0, min(depth_h - 1, y))
        return x, y

    def _drag_start(self, index, event):
        s = self.sections[index]
        if not (s.get("serial") or "") or s.get("fit") is None:
            return
        s["drag"] = (event.x, event.y)
        s["rect"] = s["canvas"].create_rectangle(event.x, event.y, event.x, event.y,
                                                 outline="red", width=2)

    def _drag_move(self, index, event):
        s = self.sections[index]
        if s.get("drag") is None or s.get("rect") is None:
            return
        x0, y0 = s["drag"]
        s["canvas"].coords(s["rect"], x0, y0, event.x, event.y)

    def _drag_end(self, index, event):
        s = self.sections[index]
        if s.get("drag") is None:
            return
        x0c, y0c = s["drag"]
        s["drag"] = None
        if s.get("rect") is not None:
            s["canvas"].delete(s["rect"])
            s["rect"] = None
        if abs(event.x - x0c) < 8 or abs(event.y - y0c) < 8:
            return
        a = self._canvas_to_depth(index, min(x0c, event.x), min(y0c, event.y))
        b = self._canvas_to_depth(index, max(x0c, event.x), max(y0c, event.y))
        if not a or not b:
            return
        serial = s.get("serial") or ""
        panel = self.app.panels[index]
        shape = (panel.depth_h or 10 ** 6, panel.depth_w or 10 ** 6)
        from common.measure import clamp_roi
        roi = clamp_roi((a[0], a[1], b[0], b[1]), shape)
        if roi is None:
            messagebox.showwarning("提示", "框选区域太小", parent=self.win)
            return
        d = self._draft(serial)
        d["roi"] = list(roi)
        d["roi_mode"] = "manual"
        s["var_roi_mode"].set("manual")
        self._update_roi_label(index)
        self._update_preview(index)

    # -------------------------------------------------- 保存
    def save(self):
        if (self.app.inventory or {}).get("simulate"):
            messagebox.showwarning("模拟模式", "当前相机服务是模拟模式，不会写入配置文件。",
                                   parent=self.win)
            return
        for i in range(2):
            if not self._capture_section(i):
                messagebox.showerror(
                    "参数错误", f"{LABELS[i]} 距离范围无效，请检查最小值、最大值和面积阈值。",
                    parent=self.win)
                return
        serials = [self.sections[i]["serial"] for i in range(2)]
        filled = [s for s in serials if s]
        if len(filled) != len(set(filled)):
            messagebox.showerror("配置错误", "A/B 两个窗口不能绑定同一台相机",
                                 parent=self.win)
            return
        try:
            baud = int(float(self.var_baud.get()))
            slave = int(float(self.var_slave.get()))
            base = int(float(self.var_base.get()))
            timeout = float(self.var_timeout.get())
            delay = float(self.var_delay.get())
            refresh = int(float(self.var_refresh.get()))
        except ValueError:
            messagebox.showerror("配置错误", "485 参数、延迟或刷新间隔必须是数字",
                                 parent=self.win)
            return
        if base not in (0, 1):
            messagebox.showerror("配置错误", "地址基址只能是 0 或 1", parent=self.win)
            return
        if baud <= 0 or not 1 <= slave <= 247 or timeout <= 0 or delay < 0:
            messagebox.showerror(
                "配置错误", "波特率/超时时间必须大于 0，触发延迟不能小于 0，"
                          "站号必须在 1~247 之间。", parent=self.win)
            return

        all_serials = [s for s in self.app.inventory_serials()
                       if s in self.persist_serials]
        for serial in all_serials:
            if serial not in self.serials:
                self.serials.append(serial)

        patches = {}
        for serial in all_serials:
            patches[serial] = {"panel": None}
        for i, serial in enumerate(serials):
            if not serial:
                continue
            d = self._draft(serial)
            if d["roi_mode"] == "manual" and not d["roi"]:
                messagebox.showerror(
                    "参数错误", f"{LABELS[i]} 面为手动模式，请先在预览图上框选计算范围。",
                    parent=self.win)
                return
            patch = {
                "panel": i,
                "auto_calibrate": bool(d["auto_calibrate"]),
                "filter": {"min_mm": float(d["min_mm"]),
                           "max_mm": float(d["max_mm"])},
                "cut": {
                    "a_max_mm2": float(d["a_max_mm2"]),
                    "roi_mode": d["roi_mode"],
                    "roi": list(d["roi"]) if d["roi"] else None,
                },
            }
            if d.get("plane") is not None:
                a, b, c = d["plane"]
                cal = {
                    "type": "plane_xy",
                    "coefficients": [float(a), float(b), float(c)],
                    "unit": "mm",
                    "roi": list(d["roi"]) if d["roi"] else None,
                    "inlier_ratio": None,
                    "calibrated_at": cfgmod.now_str(),
                }
                if d.get("plane_center_mm") is not None:
                    cal["plane_center_mm"] = float(d["plane_center_mm"])
                patch["calibration"] = cal
            patches[serial] = patch

        app_patch = {
            "display_interval_ms": max(10, refresh),
            "auto_start_modbus": bool(self.var_auto_modbus.get()),
            "modbus": {
                "port": self.var_port.get().strip(),
                "baudrate": baud,
                "slave_addr": slave,
                "addr_base": base,
                "timeout_s": timeout,
                "measure_delay_s": delay,
            },
        }

        def work():
            for serial, patch in patches.items():
                self.app.node.request_ok(
                    topics.CONFIG_REQ, topics.CONFIG_RES, "set_camera",
                    {"serial": serial, "patch": patch, "save": False,
                     "source": "ui", "notify": False}, timeout=10.0)
            self.app.node.request_ok(
                topics.CONFIG_REQ, topics.CONFIG_RES, "set_app",
                {"patch": app_patch, "save": False, "source": "ui",
                 "notify": False}, timeout=10.0)
            self.app.node.request_ok(
                topics.CONFIG_REQ, topics.CONFIG_RES, "save",
                {"source": "ui"}, timeout=15.0)
            try:
                self.app.node.request_ok(
                    topics.CAMERA_REQ, topics.CAMERA_RES, "reload", {}, timeout=15.0)
            except Exception as e:
                return {"warning": f"配置已保存，但相机服务未应用：{e}"}
            return {}

        def done(result, err):
            if self._closing:
                return
            if err:
                messagebox.showerror("保存失败", f"配置写入失败：{err}", parent=self.win)
                return
            self.app.display_interval_ms = max(10, refresh)
            warning = (result or {}).get("warning")
            if warning:
                self.app.log(warning, "warn")
            self.app.log("配置已保存并应用", "ok")
            self.close()

        self.app.run_async(work, done)

    def close(self):
        self._closing = True
        try:
            self.win.destroy()
        except Exception:
            pass
        if self.app.config_window is self:
            self.app.config_window = None


def main():
    parser = argparse.ArgumentParser(description="深度相机测量前端")
    parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--mqtt-port", type=int,
                        default=int(os.environ.get("MQTT_PORT", "1883")))
    parser.add_argument("--smoke", type=float, default=0.0,
                        help="启动 N 秒后自动关闭（冒烟测试用）")
    args = parser.parse_args()
    root = tk.Tk()
    app = App(root, args.mqtt_host, args.mqtt_port, smoke=args.smoke)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()
