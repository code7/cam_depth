# -*- coding: utf-8 -*-
"""生产界面：双深度相机 A/B 面自动测量。

启动后自动扫描相机、按序列号解锁配置、绑定左右窗口、打开并开始采集，
随后等待 PLC 通过 RS-485 触发测量。配置集中在一个独立窗口里：
A/B 窗口绑定、启动自动校准、计算范围（自动/手动）。
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import vcamera as vcam
except ImportError:
    vcam = None

import cam_config as cfgmod
import cut_plane
import modbus_rtu as mb
from modbus_service import ModbusService


CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, cfgmod.DEFAULT_FILE_NAME)

NUM_CAMERAS = 2
LABELS = ("A", "B")
DISPLAY_INTERVAL_MS = 80
REFRESH_CHOICES = ("30", "50", "80", "120", "200", "300")
STALL_WARN_S = 3.0
STALL_RECOVER_S = 8.0
RECOVER_MAX_ATTEMPTS = 6
SCAN_RETRY_S = 5.0
MODBUS_RETRY_S = 15.0

C_OK = "#0a7a2f"
C_ERR = "#c00000"
C_WARN = "#b06000"
C_IDLE = "#888888"
C_TX = "#1a4fa0"

UI_FONT = ("Microsoft YaHei UI", 10)
MONO_FONT = ("Consolas", 10)
BIG_FONT = ("Consolas", 24, "bold")


def value_to_python(v):
    try:
        if v.IsBool():
            return bool(v.as_bool())
        if v.IsString():
            return str(v.as_string())
        if v.IsFloat() or v.IsDouble():
            return float(v.as_float())
        if v.IsArray() or v.IsDictionary():
            return None
        return int(v.as_number())
    except Exception:
        return None


def fit_plane_ransac(depth_data, roi, step=2, max_iter=300,
                     inlier_thresh=4.0, min_inliers=100):
    x0, y0, x1, y1 = roi
    sub = depth_data[y0:y1:step, x0:x1:step]
    h, w = sub.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xs = (xs * step + x0).astype(np.float64).ravel()
    ys = (ys * step + y0).astype(np.float64).ravel()
    zs = sub.ravel().astype(np.float64)

    valid = np.isfinite(zs) & (zs > 0)
    pts = np.column_stack([xs[valid], ys[valid], zs[valid]])
    n = len(pts)
    if n < 3:
        return None

    rng = np.random.default_rng()
    best_cnt = 0
    best_inliers = None
    for _ in range(max_iter):
        idx = rng.choice(n, 3, replace=False)
        A = np.column_stack([pts[idx, 0], pts[idx, 1], np.ones(3)])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        abc = np.linalg.solve(A, pts[idx, 2])
        residual = np.abs(pts[:, 2] - (abc[0] * pts[:, 0] + abc[1] * pts[:, 1] + abc[2]))
        inliers = residual < inlier_thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers
            if cnt > n * 0.95:
                break

    if best_inliers is None or best_cnt < min_inliers:
        return None

    inliers = best_inliers
    for _ in range(3):
        ip = pts[inliers]
        if len(ip) < 3:
            break
        A = np.column_stack([ip[:, 0], ip[:, 1], np.ones(len(ip))])
        abc, _, _, _ = np.linalg.lstsq(A, ip[:, 2], rcond=None)
        residual = np.abs(pts[:, 2] - (abc[0] * pts[:, 0] + abc[1] * pts[:, 1] + abc[2]))
        inliers = residual < inlier_thresh

    a, b, c = abc
    inlier_ratio = int(inliers.sum()) / n
    return a, b, c, inlier_ratio


def clamp_roi(roi, shape):
    if not roi:
        return None
    h, w = shape[:2]
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def valid_bbox(valid):
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_from_mask(mask, min_pixels=200):
    if mask is None or int(mask.sum()) < min_pixels:
        return None
    return valid_bbox(mask)


def auto_roi_bbox(depth, plane=None):
    valid = cut_plane.valid_mask(depth)
    if not valid.any():
        return None
    if plane is None:
        box = valid_bbox(valid)
        fit = fit_plane_ransac(depth, box, step=4, max_iter=240, min_inliers=150)
        if fit is not None:
            plane = (fit[0], fit[1], fit[2])
        else:
            plane = (0.0, 0.0, float(np.median(depth[valid])))
    plane_depth = cut_plane.plane_from_coeffs(plane, depth.shape)
    h = cut_plane.build_height(depth, plane_depth)
    mask = cut_plane.auto_block_roi(h, valid)
    return bbox_from_mask(mask) or valid_bbox(valid)


def restrict_valid(valid, roi):
    if not roi:
        return valid
    x0, y0, x1, y1 = roi
    valid[:y0] = False
    valid[y1:] = False
    valid[:, :x0] = False
    valid[:, x1:] = False
    return valid


def solve_panel(p, default_a_max=500.0):
    depth = p["depth"]
    plane = p.get("plane")
    if plane is None:
        raise RuntimeError("未标定基准面")
    valid = cut_plane.valid_mask(depth)
    if not valid.any():
        raise RuntimeError("没有有效深度数据")
    restrict_valid(valid, clamp_roi(p.get("roi"), depth.shape))
    if not valid.any():
        raise RuntimeError("计算范围内没有有效深度数据")
    plane_depth = cut_plane.plane_from_coeffs(plane, depth.shape)
    h = cut_plane.build_height(depth, plane_depth)
    fx, fy = p.get("fx"), p.get("fy")
    mm2 = cut_plane.mm2_per_pixel(depth, fx, fy) if fx and fy else 1.0
    a_max = float(p.get("a_max_mm2") or default_a_max)
    result = cut_plane.solve_cut_plane(h, valid, mm2, a_max)
    H, W = depth.shape[:2]
    center = plane[0] * (W / 2.0) + plane[1] * (H / 2.0) + plane[2]
    return center + result["delta_mm"]


def compute_measure(a, b, a_max_mm2):
    return solve_panel(a, a_max_mm2), solve_panel(b, a_max_mm2)


def calibrate_pair(a, b):
    out = {}
    for key, p in (("A", a), ("B", b)):
        depth = p["depth"]
        roi = clamp_roi(p.get("roi"), depth.shape) or auto_roi_bbox(depth, p.get("plane"))
        if roi is None:
            raise RuntimeError(f"{key}面无法确定有效区域")
        fit = fit_plane_ransac(depth, roi, step=2, max_iter=400)
        if fit is None:
            raise RuntimeError(f"{key}面基准面拟合失败")
        out[key] = (fit[0], fit[1], fit[2])
    return out


def stamp():
    return time.strftime("%H:%M:%S")


class CameraView:
    def __init__(self, app, parent, index):
        self.app = app
        self.root = app.root
        self.index = index
        self.name = f"{LABELS[index]} 面相机"

        self.cam = None
        self.serial = None
        self.camera_cfg = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.is_running = False
        self.opening = False
        self._want_close = False
        self._open_gen = 0

        self._photo = None
        self._img_item = None
        self._roi_item = None
        self._preview = None
        self.src_w = 0
        self.src_h = 0
        self.depth_data = None
        self.depth_w = 0
        self.depth_h = 0
        self.depth_scale = cfgmod.DEFAULT_DEPTH_SCALE_MM
        self.scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.calibrated = False
        self.plane_a = 0.0
        self.plane_b = 0.0
        self.plane_c = 0.0
        self.roi = None
        self.roi_mode = "auto"
        self.auto_calibrate = True
        self.min_mm = cfgmod.DEFAULT_FILTER["min_mm"]
        self.max_mm = cfgmod.DEFAULT_FILTER["max_mm"]
        self.a_max_mm2 = cfgmod.DEFAULT_CUT["a_max_mm2"]

        self.fx = None
        self.fy = None
        self.frame_seq = 0
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_arrival = None
        self.arrival_ms = None
        self.arrival_history = []
        self.last_render_wall = None
        self.last_frame_wall = None
        self.stall_notified = False
        self.recovering = False
        self.recover_count = 0
        self._recover_result = None
        self._recover_gen = 0

        self._prepared = False
        self._prepare_pending = False
        self._prepare_attempts = 0
        self._prepare_retry_at = 0.0
        self._prepare_gen = 0

        self._build_ui(parent)

    def _build_ui(self, parent):
        self.frame = ttk.LabelFrame(parent, text=self.name)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        head = ttk.Frame(self.frame)
        head.grid(row=0, column=0, sticky="we", padx=4, pady=(4, 2))
        ttk.Label(head, text=self.name, font=(UI_FONT[0], 11, "bold")).pack(side="left")
        self.lbl_state = tk.Label(head, text="等待扫描", fg=C_IDLE, font=UI_FONT)
        self.lbl_state.pack(side="right")
        self.lbl_serial = tk.Label(head, text="", fg="#555", font=MONO_FONT)
        self.lbl_serial.pack(side="right", padx=(0, 12))
        self.btn_retry = ttk.Button(head, text="重试连接", command=self.retry_open, state="disabled")
        self.btn_retry.pack(side="right", padx=(0, 10))

        self.canvas = tk.Canvas(self.frame, width=600, height=450, bg="black",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        self.canvas.bind("<Configure>", lambda e: self._redraw_image())
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel(e, -120))
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_zoom())

        self.lbl_coord = tk.Label(self.frame, text="坐标: (---, ---)   深度: ---",
                                  anchor="w", font=MONO_FONT, fg="#555")
        self.lbl_coord.grid(row=2, column=0, sticky="we", padx=4)
        self.lbl_status = tk.Label(self.frame, text="等待自动扫描", anchor="w",
                                   justify="left", wraplength=600,
                                   font=UI_FONT, fg="#555")
        self.lbl_status.grid(row=3, column=0, sticky="we", padx=4, pady=(0, 4))

    def _set_state(self, text, color=C_IDLE):
        self.lbl_state.config(text=text, fg=color)

    def bind_config(self, serial, cfg):
        self.serial = serial
        self.camera_cfg = cfg
        self.depth_scale = float((cfg.get("depth") or {}).get(
            "scale_mm", cfgmod.DEFAULT_DEPTH_SCALE_MM))
        flt = cfg.get("filter") or {}
        self.min_mm = float(flt.get("min_mm", cfgmod.DEFAULT_FILTER["min_mm"]))
        self.max_mm = float(flt.get("max_mm", cfgmod.DEFAULT_FILTER["max_mm"]))
        cut = cfg.get("cut") or {}
        self.a_max_mm2 = float(cut.get("a_max_mm2", cfgmod.DEFAULT_CUT["a_max_mm2"]))
        self.roi_mode = str(cut.get("roi_mode") or cfgmod.DEFAULT_CUT["roi_mode"])
        self.auto_calibrate = bool(cfg.get("auto_calibrate", True))
        cal = cfg.get("calibration") or {}
        raw_roi = cut.get("roi") or cal.get("roi")
        if self.roi_mode == "manual" and not raw_roi:
            self.roi_mode = "auto"
        if self.depth_h:
            self.roi = clamp_roi(raw_roi, (self.depth_h, self.depth_w))
        else:
            self.roi = tuple(int(v) for v in raw_roi) if raw_roi else None
        if cfgmod.is_calibrated(cfg):
            a, b, c = (float(v) for v in cal["coefficients"])
            self.plane_a, self.plane_b, self.plane_c = a, b, c
            self.calibrated = True
        else:
            self.calibrated = False
        self.lbl_serial.config(text=f"ID {serial}")
        self._prepare_gen += 1
        self._prepared = False
        self._prepare_pending = False
        self._prepare_attempts = 0
        self._prepare_retry_at = 0.0
        self._refresh_status()
        self._update_roi_overlay()

    def set_idle(self, message="未发现相机"):
        self._set_state("未连接", C_IDLE)
        self.lbl_serial.config(text="")
        self.lbl_status.config(text=message, fg=C_WARN)
        self.btn_retry.config(state="disabled")
        self.canvas.delete("all")
        self._img_item = None
        self._roi_item = None
        self._photo = None
        self._preview = None
        self.depth_data = None
        self.calibrated = False
        self.roi = None
        self.serial = None

    def retry_open(self):
        if self.serial:
            self.app.open_panel(self.index)

    def open_async(self, info):
        if self.cam is not None or self.opening:
            return
        if str(info.serial_number) != str(self.serial):
            return
        if vcam is None:
            self._set_state("SDK 缺失", C_ERR)
            self.lbl_status.config(text="未找到 vcamera 模块，无法打开相机", fg=C_ERR)
            return
        self._want_close = False
        self.opening = True
        self._open_gen += 1
        gen = self._open_gen
        self.btn_retry.config(state="disabled")
        self._set_state("正在连接", C_WARN)
        self.lbl_status.config(text=f"正在打开相机 {self.serial}…", fg=C_WARN)
        cfg = self.camera_cfg or {}
        threading.Thread(target=self._open_worker,
                         args=(gen, info, cfg), daemon=True).start()

    def _open_worker(self, gen, info, cfg):
        cam = None
        err = None
        warnings = []
        features = {}
        image_mode = None
        try:
            cam = vcam.CameraFactory.GetCameraBySerialNumber(info.serial_number)
            status = cam.Connect()
            if not status:
                raise RuntimeError(status.message())
            acq_mode, status = cam.GetFeature("AcquisitionMode")
            if not status:
                raise RuntimeError(f"获取 AcquisitionMode 失败：{status.message()}")
            status = acq_mode.SetValue(2)
            if not status:
                raise RuntimeError(f"设置连续采集失败：{status.message()}")
            status = cam.SetSensorEnabled(vcam.SensorType.Depth, True)
            if not status:
                raise RuntimeError("使能深度传感器失败")
            warnings = self._apply_acquisition_config(cam, cfg)
            features = self._read_acquisition_features(cam)
            image_mode = self._read_image_mode(cam)
            cam.RegisterFrameSetCallback(self.frame_callback)
            if not cam.StartCapture():
                raise RuntimeError("启动采集失败")
        except Exception as e:
            err = str(e)
            if cam is not None:
                try:
                    cam.StopCapture()
                except Exception:
                    pass
                try:
                    cam.Disconnect()
                except Exception:
                    pass
                cam = None
        self.app.ui_after(
            lambda: self._open_done(gen, cam, err, warnings, features, image_mode))

    def _open_done(self, gen, cam, err, warnings, features, image_mode):
        if gen != self._open_gen:
            if cam is not None:
                try:
                    cam.StopCapture()
                    cam.Disconnect()
                except Exception:
                    pass
            return
        self.opening = False
        if self.app.closed:
            if cam is not None:
                try:
                    cam.Disconnect()
                except Exception:
                    pass
            return
        if self._want_close:
            if cam is not None:
                try:
                    cam.StopCapture()
                    cam.Disconnect()
                except Exception:
                    pass
            self._want_close = False
            return
        if err or cam is None:
            self._set_state("打开失败", C_ERR)
            self.lbl_status.config(text=f"打开失败：{err}", fg=C_ERR)
            self.btn_retry.config(state="normal")
            self.app.log(f"{self.name} 打开失败：{err}", "err")
            self.app.on_panel_open_done(self)
            return

        self.cam = cam
        self._clear_queue()
        self.is_running = True
        self._reset_frame_stats()
        self.last_render_wall = time.monotonic()
        self.last_frame_wall = self.last_render_wall

        cfg = self.app.store.camera(self.serial)
        cfg["panel"] = self.index
        cfg["model"] = str(getattr(self.app.info_of(self.serial), "model", "") or "")
        cfg["firmware"] = str(getattr(self.app.info_of(self.serial), "firmware_version", "") or "")
        self.app.store.apply_acquisition_from_camera(self.serial, features, image_mode, warnings)
        self.bind_config(self.serial, cfg)
        self.app.save_config()

        self._set_state("采集中", C_OK)
        self.lbl_status.config(text="已打开并开始采集，等待首帧后自动准备测量范围…", fg=C_WARN)
        self.btn_retry.config(state="normal")
        self.app.log(f"{self.name} 已打开（{self.serial}），采集已启动", "ok")
        if warnings:
            self.app.log(f"{self.name} 采集参数告警：" + "；".join(warnings), "warn")
        self.app.on_panel_open_done(self)

    def _apply_acquisition_config(self, cam, cfg):
        warnings = []
        acq = (cfg or {}).get("acquisition") or {}
        feats = acq.get("features") or {}
        for name, item in feats.items():
            try:
                feat, st = cam.GetFeature(name)
                if not st:
                    warnings.append(f"{name} 不支持")
                    continue
                acc, _ = feat.GetAccessMode()
                if str(acc).split(".")[-1] not in ("Writable", "ReadWritable"):
                    warnings.append(f"{name} 只读")
                    continue
                st = feat.SetValue(item.get("value"))
                if not st:
                    warnings.append(f"{name} 写入失败：{st.message()}")
            except Exception as e:
                warnings.append(f"{name} 写入异常：{e}")
        mode = acq.get("image_mode")
        if mode:
            try:
                modes, st = cam.GetImageModes(vcam.SensorType.Depth)
                if st and modes:
                    match = None
                    for m in modes:
                        if (int(m.width) == int(mode.get("width", -1))
                                and int(m.height) == int(mode.get("height", -1))):
                            match = m
                            break
                    if match is not None:
                        st = cam.SetImageMode(vcam.SensorType.Depth, match)
                        if not st:
                            warnings.append(f"图像模式写入失败：{st.message()}")
            except Exception as e:
                warnings.append(f"图像模式写入异常：{e}")
        return warnings

    def _read_acquisition_features(self, cam):
        out = {}
        for name in cfgmod.ACQ_FEATURES:
            try:
                feat, st = cam.GetFeature(name)
                if not st:
                    continue
                val, vs = feat.GetValue()
                if not vs:
                    continue
                py = value_to_python(val)
                if py is None:
                    continue
                out[name] = {"type": str(feat.GetType()).split(".")[-1], "value": py}
            except Exception:
                continue
        return out

    def _read_image_mode(self, cam):
        try:
            mode, st = cam.GetCurrentImageMode(vcam.SensorType.Depth)
            if not st or mode is None:
                return None
            return {"width": int(mode.width), "height": int(mode.height),
                    "pixel_format": str(getattr(mode, "pixel_format", "")).split(".")[-1]}
        except Exception:
            return None

    def close_camera(self):
        self._want_close = True
        self._open_gen += 1
        self.opening = False
        self._recover_gen += 1
        self.recovering = False
        self._recover_result = None
        self.is_running = False
        cam, self.cam = self.cam, None
        if cam is not None:
            try:
                cam.StopCapture()
            except Exception:
                pass
            try:
                cam.Disconnect()
            except Exception:
                pass
        self._clear_queue()
        self.depth_data = None
        self.depth_w = self.depth_h = 0
        self._preview = None
        self.src_w = self.src_h = 0
        self._photo = None
        self._img_item = None
        self._roi_item = None
        self.calibrated = False
        self.roi = None
        self._prepared = False
        self._prepare_pending = False
        self._prepare_attempts = 0
        self._prepare_retry_at = 0.0
        self._prepare_gen += 1
        self.reset_zoom()
        self.canvas.delete("all")
        self._set_state("未连接", C_IDLE)
        self.lbl_serial.config(text=f"ID {self.serial}" if self.serial else "")
        self.lbl_status.config(text="相机已关闭", fg=C_IDLE)
        self.lbl_coord.config(text="坐标: (---, ---)   深度: ---")
        self.btn_retry.config(state="disabled")

    def frame_callback(self, frame_set):
        now = time.monotonic()
        prev = self.last_arrival
        self.last_arrival = now
        if prev is not None:
            self.arrival_ms = (now - prev) * 1000.0
            self.arrival_history.append(self.arrival_ms)
            if len(self.arrival_history) > 10:
                del self.arrival_history[:-10]
        self.frames_received += 1
        self.last_frame_wall = now
        try:
            self.frame_queue.put_nowait((now, frame_set))
        except queue.Full:
            self.frames_dropped += 1

    def _clear_queue(self):
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _reset_frame_stats(self):
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_arrival = None
        self.arrival_ms = None
        self.arrival_history = []
        self.last_render_wall = time.monotonic()
        self.last_frame_wall = self.last_render_wall
        self.stall_notified = False
        self.recovering = False
        self.recover_count = 0
        self._recover_result = None
        self._recover_gen += 1

    def pump_frame(self):
        if not self.is_running or self.cam is None:
            return
        try:
            arrival, fs = self.frame_queue.get_nowait()
        except queue.Empty:
            self._check_stall()
            return
        try:
            ok = self._render_frame(fs, arrival)
        except Exception as e:
            self.lbl_status.config(text=f"显示错误：{e}", fg=C_ERR)
            return
        if not ok:
            return
        self.last_render_wall = time.monotonic()
        self.stall_notified = False
        self.recover_count = 0
        self.frame_seq += 1
        self._maybe_auto_prepare()
        self._refresh_status()

    def _render_frame(self, fs, arrival=None):
        depth_image = fs.GetImage(vcam.SensorType.Depth)
        if not depth_image.IsValid():
            return False
        self.depth_data = depth_image.to_numpy().astype(np.float32) * self.depth_scale
        self.depth_h, self.depth_w = self.depth_data.shape[:2]
        if self.show_height and self.calibrated:
            vis_data = self.compute_height_map()
        else:
            vis_data = self.depth_data
        vis = self._make_vis(vis_data)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        self._preview = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        self.src_w = self.depth_w
        self.src_h = self.depth_h
        if self.fx is None:
            try:
                K = np.asarray(depth_image.calib_info().intrinsic.matrix, dtype=np.float64)
                self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
            except Exception:
                self.fx = self.fy = None
        self._redraw_image()
        return True

    @property
    def show_height(self):
        return bool(self.app.show_height.get())

    def compute_height_map(self):
        xs = np.arange(self.depth_w, dtype=np.float32)
        ys = np.arange(self.depth_h, dtype=np.float32)
        plane = self.plane_a * xs[None, :] + self.plane_b * ys[:, None] + self.plane_c
        height = self.depth_data - plane
        invalid = ~np.isfinite(height) | (self.depth_data <= 0)
        height[invalid] = 0
        return height

    def _make_vis(self, vis_data):
        if self.show_height and self.calibrated:
            return cv2.normalize(vis_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if self.max_mm > self.min_mm:
            vis = (vis_data - self.min_mm) * (255.0 / (self.max_mm - self.min_mm))
            np.clip(vis, 0.0, 255.0, out=vis)
            np.nan_to_num(vis, copy=False, nan=0.0)
            return vis.astype(np.uint8)
        return cv2.normalize(vis_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def _redraw_image(self):
        if self._preview is None:
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
        resized = cv2.resize(self._preview, (new_w, new_h), interpolation=interp)
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

    def _update_roi_overlay(self):
        if not self.roi or self.depth_w <= 0:
            if self._roi_item is not None:
                self.canvas.delete(self._roi_item)
                self._roi_item = None
            return
        x0, y0, x1, y1 = self.roi
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

    def on_mouse_wheel(self, event, delta=None):
        if self._preview is None:
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
        self._redraw_image()

    def reset_zoom(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._redraw_image()

    def on_mouse_move(self, event):
        if self.depth_data is None:
            self.lbl_coord.config(text="坐标: (---, ---)   深度: ---")
            return
        x = int((event.x - self.offset_x) / self.scale)
        y = int((event.y - self.offset_y) / self.scale)
        if 0 <= x < self.depth_w and 0 <= y < self.depth_h:
            value = float(self.depth_data[y, x])
            text = f"坐标: ({x}, {y})   深度: {value:.1f} mm"
            if self.calibrated:
                height = value - (self.plane_a * x + self.plane_b * y + self.plane_c)
                text += f"   高度: {height:.1f} mm"
            self.lbl_coord.config(text=text)
        else:
            self.lbl_coord.config(text="坐标: (---, ---)   深度: ---")

    def _refresh_status(self):
        if self.cam is None:
            return
        if self.not_ready_reason():
            self._set_state("准备中", C_WARN)
        else:
            self._set_state("就绪", C_OK)
        parts = []
        if self.arrival_ms is not None:
            fps = 1000.0 / self.arrival_ms if self.arrival_ms > 0 else 0.0
            parts.append(f"帧间隔 {self.arrival_ms:.0f} ms ≈ {fps:.1f} fps")
            parts.append(f"收 {self.frames_received} 帧")
            if self.frames_dropped:
                parts.append(f"丢 {self.frames_dropped} 帧")
        else:
            parts.append("等待首帧")
        if self._prepare_pending:
            parts.append("正在自动计算范围/拟合基准面…")
        elif self.calibrated:
            parts.append(f"基准面 z={self.plane_a:.5f}x+{self.plane_b:.5f}y+{self.plane_c:.2f}")
        else:
            parts.append("未标定基准面")
        if self.roi:
            parts.append(f"范围 {self.roi[0]},{self.roi[1]} → {self.roi[2]},{self.roi[3]}"
                         f"（{'自动' if self.roi_mode == 'auto' else '手动'}）")
        else:
            parts.append("范围未设置")
        self.lbl_status.config(text="　".join(parts),
                               fg=C_WARN if self.not_ready_reason() else "#555")

    def not_ready_reason(self):
        if self.cam is None:
            return "相机未打开"
        if not self.is_running:
            return "采集未启动"
        if self.depth_data is None:
            return "等待首帧"
        if self._prepare_pending:
            return "正在准备"
        if not self.calibrated:
            return "未标定基准面"
        if not self.roi:
            return "计算范围未设置"
        return None

    def _maybe_auto_prepare(self):
        if self._prepared or self._prepare_pending or self.depth_data is None:
            return
        if self.serial is None or time.monotonic() < self._prepare_retry_at:
            return
        self._prepare_pending = True
        depth = np.array(self.depth_data, copy=True)
        plane = (self.plane_a, self.plane_b, self.plane_c) if self.calibrated else None
        roi = self.roi
        mode = self.roi_mode
        auto_cal = self.auto_calibrate
        gen = self._prepare_gen
        threading.Thread(target=self._prepare_worker,
                         args=(gen, depth, plane, roi, mode, auto_cal), daemon=True).start()

    def _prepare_worker(self, gen, depth, plane, roi, mode, auto_cal):
        err = None
        new_roi = roi
        new_plane = None
        try:
            if mode == "auto" or not new_roi:
                new_roi = auto_roi_bbox(depth, plane) or clamp_roi(roi, depth.shape)
            else:
                new_roi = clamp_roi(new_roi, depth.shape)
            if new_roi is None:
                raise RuntimeError("自动确定计算范围失败")
            if auto_cal or plane is None:
                if not auto_cal and plane is None:
                    new_plane = None
                else:
                    fit = fit_plane_ransac(depth, new_roi, step=2, max_iter=400)
                    if fit is None:
                        raise RuntimeError("基准面拟合失败")
                    new_plane = (float(fit[0]), float(fit[1]), float(fit[2]))
        except Exception as e:
            err = str(e)
        self.app.ui_after(lambda: self._prepare_done(gen, new_roi, new_plane, err))

    def _prepare_done(self, gen, roi, plane, err):
        if gen != self._prepare_gen:
            return
        self._prepare_pending = False
        if self.app.closed:
            return
        if err:
            self._prepare_attempts += 1
            self.lbl_status.config(text=f"自动准备失败：{err}", fg=C_ERR)
            self.app.log(f"{self.name} 自动准备失败：{err}", "err")
            if self._prepare_attempts < 3:
                self._prepare_retry_at = time.monotonic() + 3.0
            else:
                self._prepared = True
            return
        self._prepared = True
        self._prepare_retry_at = 0.0
        self.roi = roi
        cfg = self.camera_cfg
        if cfg is not None:
            cut = dict(cfg.get("cut") or cfgmod.DEFAULT_CUT)
            cut["roi_mode"] = self.roi_mode
            cut["roi"] = list(roi) if roi else None
            cfg["cut"] = cut
            if plane is not None:
                self.plane_a, self.plane_b, self.plane_c = plane
                self.calibrated = True
                cfg["calibration"] = {
                    "type": "plane_xy",
                    "coefficients": [float(plane[0]), float(plane[1]), float(plane[2])],
                    "unit": "mm",
                    "roi": list(roi) if roi else None,
                    "inlier_ratio": None,
                    "calibrated_at": cfgmod.now_str(),
                }
            self.app.store.mark_dirty()
            self.app.save_config()
        self.app.log(f"{self.name} 已自动确定计算范围 {roi}"
                     + (f"，基准面 z={plane[0]:.5f}x+{plane[1]:.5f}y+{plane[2]:.2f}"
                        if plane is not None else "，保留原基准面"), "ok")
        self._refresh_status()
        self._update_roi_overlay()

    def _check_stall(self):
        if self._recover_result is not None:
            self._finish_recover()
            return
        if self.recovering:
            return
        ref = self.last_frame_wall or self.last_render_wall
        if ref is None:
            return
        gap = time.monotonic() - ref
        if gap >= STALL_RECOVER_S:
            if self.recover_count < RECOVER_MAX_ATTEMPTS:
                self._set_state("自动恢复", C_WARN)
                self._start_recover()
            elif not self.stall_notified:
                self.stall_notified = True
                self._set_state("断流", C_ERR)
                self.lbl_status.config(text="相机断流且自动恢复多次失败，请检查供电/网线", fg=C_ERR)
            return
        if self.stall_notified or gap < STALL_WARN_S:
            return
        self.stall_notified = True
        self.lbl_status.config(text=f"已 {gap:.1f}s 没有新帧（相机可能断流）", fg=C_WARN)

    def _start_recover(self):
        self.recovering = True
        self.recover_count += 1
        self._recover_gen += 1
        self._recover_result = None
        gen = self._recover_gen
        threading.Thread(target=self._recover_worker,
                         args=(gen, self.serial, self.cam), daemon=True).start()

    def _recover_worker(self, gen, serial, old):
        err = None
        newcam = None
        cam = None
        try:
            try:
                old.StopCapture()
            except Exception:
                pass
            try:
                old.Disconnect()
            except Exception:
                pass
            time.sleep(0.5)
            cam = vcam.CameraFactory.GetCameraBySerialNumber(serial)
            st = cam.Connect()
            if not st:
                raise RuntimeError(f"重连失败：{st.message()}")
            m, st = cam.GetFeature("AcquisitionMode")
            if st:
                m.SetValue(2)
            cam.SetSensorEnabled(vcam.SensorType.Depth, True)
            cam.RegisterFrameSetCallback(self.frame_callback)
            if not cam.StartCapture():
                raise RuntimeError("重连后启动采集失败")
            newcam = cam
        except Exception as e:
            err = str(e)
            if cam is not None and newcam is None:
                try:
                    cam.StopCapture()
                    cam.Disconnect()
                except Exception:
                    pass
        if gen != self._recover_gen:
            if newcam is not None:
                try:
                    newcam.StopCapture()
                    newcam.Disconnect()
                except Exception:
                    pass
            return
        self._recover_result = (gen, err, newcam)

    def _finish_recover(self):
        gen, err, newcam = self._recover_result
        self._recover_result = None
        if gen != self._recover_gen:
            if newcam is not None:
                try:
                    newcam.StopCapture()
                    newcam.Disconnect()
                except Exception:
                    pass
            return
        self.recovering = False
        if err or newcam is None:
            self._set_state("恢复失败", C_ERR)
            self.lbl_status.config(text=f"自动恢复失败：{err}", fg=C_ERR)
            return
        self.cam = newcam
        self._clear_queue()
        self.last_frame_wall = time.monotonic()
        self.last_render_wall = self.last_frame_wall
        self.stall_notified = False
        self._set_state("采集中", C_OK)
        self.lbl_status.config(text=f"相机已自动重连（第 {self.recover_count} 次），等待新帧…",
                               fg=C_WARN)

    def panel_data(self):
        return {
            "serial": self.serial,
            "depth": self.depth_data,
            "plane": ((self.plane_a, self.plane_b, self.plane_c)
                      if self.calibrated else None),
            "calibrated": bool(self.calibrated),
            "running": bool(self.is_running),
            "frame_id": self.frame_seq,
            "fx": self.fx,
            "fy": self.fy,
            "a_max_mm2": self.a_max_mm2,
            "roi": list(self.roi) if self.roi else None,
        }

    def apply_fitted_plane(self, coeffs):
        a, b, c = (float(v) for v in coeffs)
        self.plane_a, self.plane_b, self.plane_c = a, b, c
        self.calibrated = True
        cfg = self.app.store.camera(self.serial) if self.serial else None
        if cfg is not None:
            cfg["calibration"] = {
                "type": "plane_xy",
                "coefficients": [a, b, c],
                "unit": "mm",
                "roi": list(self.roi) if self.roi else None,
                "inlier_ratio": None,
                "calibrated_at": cfgmod.now_str(),
            }
        self.app.log(f"{self.name} 基准面已由 Modbus 校准", "ok")
        self._refresh_status()


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
        self.serials = [str(c.serial_number) for c in app.cam_infos]
        self.drafts = {}
        self.sections = [{}, {}]
        self._closing = False
        self._build()
        self._init_sections()
        self._refresh_previews()
        self.win.protocol("WM_DELETE_WINDOW", self.close)

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
        s["photo"] = None
        s["fit"] = None
        s["drag"] = None

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
        ttk.Button(roi_row, text="自动计算", command=lambda i=index: self._auto_roi(i)).pack(
            side="left", padx=(12, 2))
        ttk.Button(roi_row, text="清除", command=lambda i=index: self._clear_roi(i)).pack(
            side="left", padx=2)

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
        self.var_auto_modbus = tk.BooleanVar(
            value=bool(self.app.store.app.get("auto_start_modbus", True)))
        ttk.Checkbutton(box, text="开机自动连接 485，等待 PLC 触发",
                        variable=self.var_auto_modbus).grid(row=0, column=0, columnspan=4,
                                                           sticky="w", pady=(0, 6))
        mcfg = dict(self.app.store.app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        self.var_port = tk.StringVar(value=str(mcfg.get("port", "")))
        self.var_baud = tk.StringVar(value=str(mcfg.get("baudrate", 9600)))
        self.var_slave = tk.StringVar(value=str(mcfg.get("slave_addr", 1)))
        self.var_base = tk.StringVar(value=str(mcfg.get("addr_base", 0)))
        self.var_timeout = tk.StringVar(value=str(mcfg.get("timeout_s", 3.0)))

        fields = (("串口:", self.var_port, 18), ("波特率:", self.var_baud, 8),
                  ("站号:", self.var_slave, 5), ("地址基址(0/1):", self.var_base, 4),
                  ("测量超时(s):", self.var_timeout, 6))
        for i, (label, var, width) in enumerate(fields):
            ttk.Label(box, text=label).grid(row=1 + i // 2, column=(i % 2) * 2,
                                            sticky="w", pady=2)
            ttk.Entry(box, textvariable=var, width=width).grid(
                row=1 + i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(2, 16), pady=2)

        run = ttk.LabelFrame(parent, text="显示", padding=8)
        run.pack(fill="x", padx=4, pady=6)
        ttk.Label(run, text="刷新间隔(ms):").pack(side="left")
        ms = int(self.app.store.app.get("display_interval_ms", DISPLAY_INTERVAL_MS))
        self.var_refresh = tk.StringVar(value=str(ms) if str(ms) in REFRESH_CHOICES
                                        else str(DISPLAY_INTERVAL_MS))
        ttk.Combobox(run, textvariable=self.var_refresh, state="readonly", width=6,
                     values=REFRESH_CHOICES).pack(side="left", padx=4)

        note = tk.Label(
            parent,
            text=("说明：自动计算范围会在启动后根据木块轮廓生成矩形范围并保存；\n"
                  "手动框选会覆盖自动范围。自动校准平面使用这个范围拟合基准面。"),
            justify="left", fg="#555", font=UI_FONT)
        note.pack(fill="x", padx=8, pady=(4, 0))

    def _init_sections(self):
        used = set()
        for i in range(2):
            s = self.sections[i]
            sel = ""
            panel = self.app.panels[i]
            if panel.serial in self.serials and panel.serial not in used:
                sel = panel.serial
            else:
                for serial in self.serials:
                    if serial not in used:
                        sel = serial
                        break
            s["serial"] = sel
            if sel:
                used.add(sel)
            s["combo"].set(sel)
            self._load_section(i)

    def _draft(self, serial):
        if serial not in self.drafts:
            cfg = self.app.store.camera(serial)
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
            }
        return self.drafts[serial]

    def _load_section(self, index):
        s = self.sections[index]
        serial = s["serial"]
        enabled = bool(serial)
        for key in ("combo",):
            s[key].config(state="readonly")
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
        old = s.get("serial") or ""
        if old:
            self._capture_section(index)
        new = s["combo"].get()
        s["serial"] = new
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
        if panel.serial != serial or panel.depth_data is None:
            messagebox.showwarning("提示", "该窗口还没有这台相机的画面，请等相机打开后再试", parent=self.win)
            return
        plane = (panel.plane_a, panel.plane_b, panel.plane_c) if panel.calibrated else None
        roi = auto_roi_bbox(panel.depth_data, plane)
        if roi is None:
            messagebox.showwarning("提示", "自动计算范围失败，请改用手动框选", parent=self.win)
            return
        d = self._draft(serial)
        d["roi"] = list(roi)
        d["roi_mode"] = "auto"
        s["var_roi_mode"].set("auto")
        self._update_roi_label(index)
        self._update_preview(index)

    def _calibrate_now(self, index):
        s = self.sections[index]
        serial = s.get("serial") or ""
        if not serial:
            return
        panel = self.app.panels[index]
        if panel.serial != serial or panel.depth_data is None:
            messagebox.showwarning("提示", "该窗口还没有这台相机的画面，请等相机打开后再试", parent=self.win)
            return
        self._capture_section(index)
        d = self._draft(serial)
        plane = (panel.plane_a, panel.plane_b, panel.plane_c) if panel.calibrated else None
        roi = d.get("roi") or auto_roi_bbox(panel.depth_data, plane)
        if roi is None:
            messagebox.showwarning("提示", "无法确定校准范围", parent=self.win)
            return
        s["lbl_cal"].config(text="正在拟合…", fg=C_WARN)
        depth = np.array(panel.depth_data, copy=True)
        threading.Thread(target=self._calibrate_worker,
                         args=(index, serial, depth, roi), daemon=True).start()

    def _calibrate_worker(self, index, serial, depth, roi):
        err = None
        fit = None
        try:
            fit = fit_plane_ransac(depth, clamp_roi(roi, depth.shape), step=2, max_iter=400)
            if fit is None:
                raise RuntimeError("拟合失败")
        except Exception as e:
            err = str(e)
        self.app.ui_after(lambda: self._calibrate_done(index, serial, fit, err))

    def _calibrate_done(self, index, serial, fit, err):
        if self._closing:
            return
        s = self.sections[index]
        if err or fit is None:
            s["lbl_cal"].config(text=f"校准失败：{err}", fg=C_ERR)
            return
        d = self._draft(serial)
        d["plane"] = (float(fit[0]), float(fit[1]), float(fit[2]))
        if s.get("serial") == serial:
            s["lbl_cal"].config(
                text=f"已标定：z={fit[0]:.5f}x+{fit[1]:.5f}y+{fit[2]:.2f}", fg=C_OK)

    def _update_preview(self, index):
        s = self.sections[index]
        canvas = s["canvas"]
        canvas.delete("all")
        s["photo"] = None
        s["fit"] = None
        serial = s.get("serial") or ""
        panel = self.app.panels[index]
        if not serial:
            canvas.create_text(10, 10, anchor="nw", fill="#aaa",
                               text="未绑定相机")
            return
        if panel.serial != serial or panel._preview is None:
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
        resized = cv2.resize(panel._preview, (ow, oh),
                             interpolation=cv2.INTER_AREA if ow < iw else cv2.INTER_LINEAR)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        s["photo"] = photo
        s["fit"] = (scale, ox, oy, iw, ih)
        canvas.create_image(ox, oy, anchor="nw", image=photo)
        d = self._draft(serial)
        roi = d.get("roi")
        if roi:
            color = "#ffb000" if index == 0 else "#00c8ff"
            canvas.create_rectangle(ox + roi[0] * scale, oy + roi[1] * scale,
                                    ox + roi[2] * scale, oy + roi[3] * scale,
                                    outline=color, width=2)
        if self._closing:
            return

    def _refresh_previews(self):
        if self._closing:
            return
        if any(self.sections[i].get("drag") is not None for i in range(2)):
            self.win.after(500, self._refresh_previews)
            return
        for i in range(2):
            self._update_preview(i)
        self.win.after(500, self._refresh_previews)

    def _canvas_to_image(self, index, cx, cy):
        fit = self.sections[index].get("fit")
        if not fit:
            return None
        scale, ox, oy, iw, ih = fit
        x = int((cx - ox) / scale)
        y = int((cy - oy) / scale)
        x = max(0, min(iw - 1, x))
        y = max(0, min(ih - 1, y))
        return x, y

    def _drag_start(self, index, event):
        s = self.sections[index]
        if (s.get("serial") or "") == "" or s.get("fit") is None:
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
        a = self._canvas_to_image(index, min(x0c, event.x), min(y0c, event.y))
        b = self._canvas_to_image(index, max(x0c, event.x), max(y0c, event.y))
        if not a or not b:
            return
        serial = s.get("serial") or ""
        depth = self.app.panels[index].depth_data
        shape = depth.shape if depth is not None else (10 ** 6, 10 ** 6)
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

    def save(self):
        for i in range(2):
            if not self._capture_section(i):
                messagebox.showerror(
                    "参数错误", f"{LABELS[i]} 距离范围无效，请检查最小值、最大值和面积阈值。",
                    parent=self.win)
                return
        serials = [self.sections[i]["serial"] for i in range(2)]
        filled = [s for s in serials if s]
        if len(filled) != len(set(filled)):
            messagebox.showerror("配置错误", "A/B 两个窗口不能绑定同一台相机", parent=self.win)
            return
        try:
            baud = int(float(self.var_baud.get()))
            slave = int(float(self.var_slave.get()))
            base = int(float(self.var_base.get()))
            timeout = float(self.var_timeout.get())
            refresh = int(float(self.var_refresh.get()))
        except ValueError:
            messagebox.showerror("配置错误", "485 参数或刷新间隔必须是数字", parent=self.win)
            return
        if base not in (0, 1):
            messagebox.showerror("配置错误", "地址基址只能是 0 或 1", parent=self.win)
            return
        if baud <= 0 or not 1 <= slave <= 247 or timeout <= 0:
            messagebox.showerror(
                "配置错误", "波特率/超时时间必须大于 0，站号必须在 1~247 之间。",
                parent=self.win)
            return

        for serial in self.serials:
            cfg = self.app.store.camera(serial)
            cfg["panel"] = None

        for i, serial in enumerate(serials):
            if not serial:
                continue
            d = self._draft(serial)
            cfg = self.app.store.camera(serial)
            cfg["panel"] = i
            cfg["auto_calibrate"] = bool(d["auto_calibrate"])
            cfg["filter"] = {"min_mm": float(d["min_mm"]), "max_mm": float(d["max_mm"])}
            cut = dict(cfg.get("cut") or cfgmod.DEFAULT_CUT)
            cut["a_max_mm2"] = float(d["a_max_mm2"])
            cut["roi_mode"] = d["roi_mode"]
            cut["roi"] = list(d["roi"]) if d["roi"] else None
            cfg["cut"] = cut
            if d.get("plane") is not None:
                a, b, c = d["plane"]
                cfg["calibration"] = {
                    "type": "plane_xy",
                    "coefficients": [float(a), float(b), float(c)],
                    "unit": "mm",
                    "roi": list(d["roi"]) if d["roi"] else None,
                    "inlier_ratio": None,
                    "calibrated_at": cfgmod.now_str(),
                }

        m = dict(self.app.store.app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        m["port"] = self.var_port.get().strip()
        m["baudrate"] = baud
        m["slave_addr"] = slave
        m["addr_base"] = base
        m["timeout_s"] = timeout
        self.app.store.app["modbus"] = m
        self.app.store.app["auto_start_modbus"] = bool(self.var_auto_modbus.get())
        self.app.store.app["display_interval_ms"] = max(10, refresh)
        self.app.store.mark_dirty()
        if not self.app.save_config("配置已保存"):
            messagebox.showerror("保存失败", "配置文件写入失败，请检查目录权限", parent=self.win)
            return
        self.app.config_saved(serials)
        self.close()

    def close(self):
        self._closing = True
        try:
            self.win.destroy()
        except Exception:
            pass
        self.app.config_window = None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("双深度相机 A/B 面测量（PLC 触发）")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1320, sw - 60), min(900, sh - 80)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        root.minsize(1000, 650)
        self.closed = False
        self.config_window = None
        self.scanning = False
        self.cam_infos = []
        self._pending_opens = 0
        self._opened_once = False
        self._modbus_connecting = False
        self._modbus_gen = 0
        self._modbus_retry_at = 0.0
        self._modbus_fail_count = 0
        self._auto_scan_at = 0.0

        self.store = cfgmod.ConfigStore(CONFIG_PATH)
        self.has_config_file = self.store.load()
        self.unlock_report = None

        self.show_height = tk.BooleanVar(value=False)
        self.display_interval_ms = int(self.store.app.get(
            "display_interval_ms", DISPLAY_INTERVAL_MS))

        self.regs = mb.HoldingRegisters(addr_base=0)
        self.modbus_log = []
        self._log_queue = queue.Queue()
        self.modbus = ModbusService(
            self.regs, self.get_panel_data,
            compute_fn=compute_measure, calibrate_fn=calibrate_pair,
            log=self.log_modbus, timeout_s=3.0)
        self.modbus.on_calibrated = self.on_modbus_calibrated

        self._build_ui()
        self._apply_app_config()

        if vcam is None:
            self.log("未找到 vcamera 模块，无法扫描相机；仍可打开配置窗口", "err")
        else:
            try:
                vcam.CameraUtils.Init(True)
            except Exception as e:
                self.log(f"相机 SDK 初始化失败：{e}", "err")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.auto_scan)
        self.root.after(self.display_interval_ms, self._tick)

    def _build_ui(self):
        bar = ttk.Frame(self.root)
        bar.grid(row=0, column=0, sticky="we", padx=6, pady=(6, 2))
        ttk.Button(bar, text="重新扫描", command=self.rescan).pack(side="left")
        ttk.Button(bar, text="配置", command=self.open_config).pack(side="left", padx=6)
        self.btn_modbus = ttk.Button(bar, text="连接485", command=self.toggle_modbus)
        self.btn_modbus.pack(side="left", padx=(18, 2))
        ttk.Button(bar, text="寄存器监视", command=self.show_registers).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="高度图", variable=self.show_height).pack(side="left", padx=(18, 2))

        self.lbl_status = tk.Label(self.root, text="准备中…", anchor="w", font=UI_FONT,
                                   fg="#555")
        self.lbl_status.grid(row=1, column=0, sticky="we", padx=6)
        self.lbl_health = tk.Label(self.root, text="", anchor="w", font=UI_FONT, fg="#555")
        self.lbl_health.grid(row=2, column=0, sticky="we", padx=6)
        self.lbl_modbus = tk.Label(self.root, text="Modbus：未连接", anchor="w",
                                   font=UI_FONT, fg=C_IDLE)
        self.lbl_modbus.grid(row=3, column=0, sticky="we", padx=6)

        container = ttk.Frame(self.root)
        container.grid(row=4, column=0, sticky="nsew", padx=6, pady=6)
        container.rowconfigure(0, weight=1)
        for col in range(2):
            container.columnconfigure(col, weight=1)
        self.panels = []
        for i in range(2):
            panel = CameraView(self, container, i)
            panel.frame.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            self.panels.append(panel)

        logf = ttk.LabelFrame(self.root, text="运行日志", padding=(6, 4))
        logf.grid(row=5, column=0, sticky="nsew", padx=6, pady=(0, 6))
        holder = ttk.Frame(logf)
        holder.pack(fill="both", expand=True)
        self.txt_log = tk.Text(holder, height=6, wrap="none", font=MONO_FONT,
                               state="disabled", bg="#fbfbfb")
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt_log.pack(side="left", fill="both", expand=True)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(4, weight=1)
        self.root.rowconfigure(5, weight=0)

    def _apply_app_config(self):
        ms = int(self.store.app.get("display_interval_ms", DISPLAY_INTERVAL_MS))
        self.display_interval_ms = max(10, ms)
        mcfg = dict(self.store.app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        self.regs.addr_base = int(mcfg.get("addr_base", 0))
        self.modbus.timeout_s = float(mcfg.get("timeout_s", 3.0))
        if self.store.warnings:
            for w in self.store.warnings:
                self.log(f"配置告警：{w}", "warn")

    def log(self, msg, level="info"):
        line = f"{stamp()}  {msg}"
        print(line)
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

    def log_modbus(self, msg):
        self.modbus_log.append(f"{stamp()}  {msg}")
        if len(self.modbus_log) > 800:
            del self.modbus_log[:-800]
        self._log_queue.put(msg)

    def _drain_logs(self):
        while True:
            try:
                msg = self._log_queue.get_nowait()
            except queue.Empty:
                return
            self.log(msg)

    def ui_after(self, fn):
        try:
            self.root.after(0, fn)
        except Exception:
            pass

    def info_of(self, serial):
        for info in self.cam_infos:
            if str(info.serial_number) == str(serial):
                return info
        return None

    def get_panel_data(self):
        return {"A": self.panels[0].panel_data(), "B": self.panels[1].panel_data()}

    def on_modbus_calibrated(self, res):
        try:
            if "A" in res:
                self.panels[0].apply_fitted_plane(res["A"])
            if "B" in res:
                self.panels[1].apply_fitted_plane(res["B"])
            self.store.mark_dirty()
            self.save_config("基准面已由 Modbus 校准并保存")
        except Exception as e:
            self.log(f"Modbus 校准回填失败：{e}", "err")

    def save_config(self, note=""):
        ok = self.store.save()
        if not ok:
            if self.store.warnings:
                self.log(f"配置保存失败：{self.store.warnings[-1]}", "err")
            return False
        if note:
            self.lbl_status.config(text=note)
            self.log(note, "ok")
        return True

    def config_summary(self):
        r = self.unlock_report
        if r is None:
            return "配置未解锁"
        parts = [f"解锁 {len(r['unlocked'])} 台"]
        if r["missing"]:
            parts.append(f"{len(r['missing'])} 台无配置")
        if r["rejected"]:
            parts.append(f"{len(r['rejected'])} 台未授权")
        if self.store.locked:
            parts.append(f"{self.store.locked} 段打不开")
        return "配置：" + "，".join(parts)

    def auto_scan(self, manual=False):
        if self.closed or self.scanning:
            return
        if vcam is None:
            self.lbl_status.config(text="未找到 vcamera 模块，无法扫描相机", fg=C_ERR)
            return
        self.scanning = True
        self.lbl_status.config(text="正在后台扫描相机…", fg=C_WARN)
        self.log("开始扫描相机")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        err = None
        infos = []
        try:
            infos = list(vcam.CameraUtils.DiscoverCameras())
        except Exception as e:
            err = str(e)
        self.ui_after(lambda: self._scan_done(infos, err))

    def _scan_done(self, infos, err):
        self.scanning = False
        if self.closed:
            return
        if err:
            self.lbl_status.config(text=f"扫描失败：{err}，将自动重试", fg=C_ERR)
            self.log(f"扫描失败：{err}", "err")
            self._schedule_auto_scan()
            return
        if not infos:
            self.lbl_status.config(text=f"未发现相机，{int(SCAN_RETRY_S)} 秒后自动重试", fg=C_WARN)
            self.log("未发现相机，稍后自动重试", "warn")
            self._schedule_auto_scan()
            return

        self.cam_infos = infos
        self._auto_scan_at = 0.0
        names = [str(c.serial_number) for c in infos]
        self.log(f"发现 {len(infos)} 台相机：" + "、".join(names))
        self.store.reset_decoded()
        self.unlock_report = self.store.unlock(names)
        rep = self.unlock_report or {}
        if rep.get("rejected"):
            self.log("未授权相机已锁定：" + "、".join(rep["rejected"]), "warn")
        if rep.get("missing"):
            self.log(f"{len(rep['missing'])} 台相机无配置，使用默认参数并自动准备", "warn")
        if rep.get("failed"):
            self.log(f"{len(rep['failed'])} 台相机配置读取失败", "err")
        self._assign_panels(names)

        for i, panel in enumerate(self.panels):
            serial = panel.serial
            if serial:
                cfg = self.store.camera(serial)
                panel.bind_config(serial, cfg)
                panel.btn_retry.config(state="normal")

        self.lbl_status.config(text=self.config_summary(), fg=C_OK)
        self._open_all()
        self._update_health()

    def _assign_panels(self, names):
        assigned = {}
        claimed = set()
        for serial, cfg in self.store.cameras.items():
            p = cfg.get("panel")
            if isinstance(p, int) and 0 <= p < NUM_CAMERAS and p not in assigned:
                assigned[p] = serial
                claimed.add(serial)
        for i, panel in enumerate(self.panels):
            if i in assigned:
                panel.serial = assigned[i]
                continue
            free = [s for s in sorted(names) if s not in claimed]
            if free:
                serial = free[0]
                assigned[i] = serial
                claimed.add(serial)
                panel.serial = serial
                cfg = self.store.camera(serial)
                cfg["panel"] = i
                self.store.mark_dirty()
            else:
                panel.serial = None
        for i, panel in enumerate(self.panels):
            if panel.serial:
                self.log(f"{LABELS[i]} 面窗口 → 相机 {panel.serial}")
            else:
                panel.set_idle("未分配相机")
                self.log(f"{LABELS[i]} 面窗口没有可用相机", "warn")
        if self.store.dirty:
            self.save_config()

    def _open_all(self):
        targets = [p for p in self.panels if p.serial and p.cam is None and not p.opening]
        self._pending_opens = len(targets)
        self._opened_once = False
        if not targets:
            self._on_all_open_attempted()
            return
        for i, panel in enumerate(targets):
            info = self.info_of(panel.serial)
            if info is None:
                self._pending_opens -= 1
                continue
            self.root.after(80 * i, lambda p=panel, inf=info: p.open_async(inf))
        if self._pending_opens <= 0:
            self._on_all_open_attempted()

    def on_panel_open_done(self, panel):
        self._pending_opens = max(0, self._pending_opens - 1)
        if self._pending_opens == 0:
            self._on_all_open_attempted()

    def _on_all_open_attempted(self):
        self._opened_once = any(p.cam is not None for p in self.panels)
        if self._opened_once:
            self.lbl_status.config(text="相机已打开，等待首帧并自动准备测量…", fg=C_WARN)
            self.log("相机打开流程结束，等待自动校准/计算范围", "ok")
        else:
            self.lbl_status.config(text="没有相机被成功打开，请检查连接或点面板上的重试", fg=C_ERR)
            self.log("没有相机被成功打开", "err")

    def open_panel(self, index):
        panel = self.panels[index]
        info = self.info_of(panel.serial) if panel.serial else None
        if info is None:
            self.log(f"{panel.name} 没有可用的相机信息，请重新扫描", "warn")
            return
        panel.open_async(info)

    def rescan(self):
        if self.scanning:
            return
        self.log("手动重新扫描：先关闭已打开的相机")
        self.stop_modbus("重新扫描")
        for panel in self.panels:
            panel.close_camera()
            panel.set_idle("等待重新扫描")
        self.cam_infos = []
        self.store.reset_decoded()
        self.unlock_report = None
        self._opened_once = False
        self._pending_opens = 0
        self.auto_scan(manual=True)

    def _schedule_auto_scan(self):
        self._auto_scan_at = time.monotonic() + SCAN_RETRY_S

    def _apply_config_values(self):
        self.display_interval_ms = max(10, int(self.store.app.get(
            "display_interval_ms", DISPLAY_INTERVAL_MS)))
        mcfg = dict(self.store.app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        self.regs.addr_base = int(mcfg.get("addr_base", 0))
        self.modbus.timeout_s = float(mcfg.get("timeout_s", 3.0))

    def config_saved(self, serials):
        self._apply_config_values()
        changed = False
        for i, panel in enumerate(self.panels):
            serial = serials[i] if i < len(serials) else ""
            if panel.serial != serial:
                changed = True
            if serial:
                cfg = self.store.camera(serial)
                if panel.serial == serial:
                    panel.bind_config(serial, cfg)
                    panel._prepared = False
        self.stop_modbus("配置已更新")
        if changed or not any(p.cam for p in self.panels):
            self.rescan()
        else:
            self.log("配置已应用", "ok")

    def open_config(self):
        if self.config_window is not None:
            try:
                self.config_window.win.lift()
            except Exception:
                pass
            return
        self.config_window = ConfigWindow(self)

    def _tick(self):
        if self.closed:
            return
        for panel in self.panels:
            panel.pump_frame()
        self.modbus.poll()
        self._drain_logs()
        self._maybe_auto_scan()
        self._maybe_auto_modbus()
        self._update_health()
        self.root.after(self.display_interval_ms, self._tick)

    def _maybe_auto_scan(self):
        if not self._auto_scan_at or self.scanning:
            return
        if any(p.cam is not None or p.opening for p in self.panels):
            self._auto_scan_at = 0.0
            return
        if time.monotonic() >= self._auto_scan_at:
            self._auto_scan_at = 0.0
            self.auto_scan()

    def _update_health(self):
        if self.closed:
            return
        parts = []
        for i, panel in enumerate(self.panels):
            reason = panel.not_ready_reason() if panel.serial else "未分配"
            if not panel.serial:
                state = "未分配"
            elif reason is None:
                state = "就绪"
            elif panel.opening:
                state = "连接中"
            else:
                state = reason
            parts.append(f"{LABELS[i]}：{state}")
        self.lbl_health.config(text="　|　".join(parts),
                               fg=C_OK if all(p.serial and p.not_ready_reason() is None
                                              for p in self.panels) else C_WARN)

    def _auto_modbus_enabled(self):
        return bool(self.store.app.get("auto_start_modbus", True))

    def _maybe_auto_modbus(self):
        if self.closed or not self._auto_modbus_enabled():
            return
        if self.modbus.running or self._modbus_connecting:
            return
        if not any(p.cam is not None and p.is_running for p in self.panels):
            return
        if time.monotonic() < self._modbus_retry_at:
            return
        self.start_modbus(auto=True)

    def start_modbus(self, auto=False):
        if self._modbus_connecting or self.modbus.running:
            return
        cfg = dict(self.store.app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        port = str(cfg.get("port") or "").strip()
        if not port:
            self._modbus_failed("485 端口未配置", auto)
            return
        self._modbus_connecting = True
        self.lbl_modbus.config(text="Modbus：正在连接…", fg=C_WARN)
        args = (self._modbus_gen, port, int(cfg.get("baudrate", 9600)),
                int(cfg.get("slave_addr", 1)), int(cfg.get("addr_base", 0)))
        threading.Thread(target=self._modbus_worker, args=args, daemon=True).start()

    def _modbus_worker(self, gen, port, baud, slave, base):
        ok = False
        err = None
        try:
            self.modbus.stop()
            ok = bool(self.modbus.start(port, baud, slave, base))
            if not ok:
                err = self.modbus_log[-1] if self.modbus_log else "打开串口失败"
        except Exception as e:
            err = str(e)
        self.ui_after(lambda: self._modbus_started(gen, ok, err, port, baud, slave, base))

    def _modbus_started(self, gen, ok, err, port, baud, slave, base):
        self._modbus_connecting = False
        if gen != self._modbus_gen or self.closed:
            if ok:
                self.modbus.stop()
            return
        if not ok:
            self._modbus_failed(err or "打开串口失败", True)
            return
        self._modbus_fail_count = 0
        self._modbus_retry_at = 0.0
        self.btn_modbus.config(text="断开485")
        self.lbl_modbus.config(
            text=f"Modbus：已连接 {port} @{baud} 站号{slave} 基址{base}", fg=C_OK)
        self.log(f"485 已连接：{port} @{baud} 站号 {slave} 基址 {base}", "ok")

    def _modbus_failed(self, err, auto):
        self._modbus_connecting = False
        self._modbus_fail_count += 1
        self._modbus_retry_at = time.monotonic() + MODBUS_RETRY_S
        self.btn_modbus.config(text="连接485")
        self.lbl_modbus.config(text=f"Modbus：未连接（{err}）", fg=C_ERR)
        if self._modbus_fail_count == 1 or self._modbus_fail_count % 10 == 0:
            self.log(f"485 连接失败：{err}；{int(MODBUS_RETRY_S)} 秒后自动重试", "warn")

    def stop_modbus(self, why=""):
        self._modbus_gen += 1
        was = self.modbus.running
        try:
            self.modbus.stop()
        except Exception:
            pass
        if was:
            self.log(f"485 已断开（{why}）" if why else "485 已断开", "warn")
        self.btn_modbus.config(text="连接485")
        self.lbl_modbus.config(text="Modbus：未连接", fg=C_IDLE)

    def toggle_modbus(self):
        if self.modbus.running:
            self.stop_modbus("手动断开")
            return
        self._modbus_retry_at = 0.0
        self.start_modbus()

    def show_registers(self):
        s = self.regs.snapshot()
        base = self.regs.addr_base
        lines = ["寄存器一览（文档编号 / 线上地址 / 名称 / 值）", ""]
        for i in range(self.regs.size):
            rw = "R" if i in mb.REG_READONLY else "W"
            val = s[i]
            extra = ""
            if i == mb.REG_STATUS:
                extra = f"   ({mb.STATUS_NAMES.get(val, '?')})"
            elif i in (mb.REG_DIST_A, mb.REG_DIST_B):
                extra = f"   ({val / mb.DIST_SCALE:.1f} mm)"
            lines.append(f"  {40001 + i:<6} wire={i + base:<3} {rw}  "
                         f"{mb.REG_NAMES.get(i, '?'):<20} {val:<7}{extra}")
        lines += ["", f"距离分辨率：{mb.DIST_SCALE:.0f} → 0.1mm，"
                      f"量程 {mb.DIST_MIN_MM:.0f}~{mb.DIST_MAX_MM:.0f} mm",
                  f"连接状态：{'已连接' if self.modbus.running else '未连接'}"]
        if self.modbus.last_result:
            lines.append(f"上次下刀位置：A面 {self.modbus.last_result[0]:.1f} mm，"
                         f"B面 {self.modbus.last_result[1]:.1f} mm"
                         f"（周期 {self.modbus.cycle_ms or 0:.0f} ms）")
        if self.modbus.last_error:
            lines.append(f"上次故障：{self.modbus.last_error}")
        if self.modbus.slave is not None:
            st = self.modbus.slave.stats
            lines += ["", f"通讯统计：收 {st['rx']} 帧，发 {st['tx']} 帧，"
                          f"CRC 错 {st['crc_err']}，站号不符 {st['addr_skip']}，"
                          f"异常响应 {st['exc']}"]
        lines += ["", "PLC 写入记录（最近 20 条）："]
        for idx, val, ts in self.regs.write_log[-20:]:
            lines.append(f"  {ts}  40001+{idx} ({mb.REG_NAMES.get(idx, '?')}) = {val}")
        lines += ["", "通讯日志（最近 40 条）："]
        lines += ["  " + x for x in self.modbus_log[-40:]]
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

    def on_close(self):
        self.closed = True
        try:
            self.modbus.stop()
        except Exception:
            pass
        for panel in self.panels:
            panel.close_camera()
        try:
            if self.store.segment_count() or self.store.cameras or self.store.dirty:
                self.store.save()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()
