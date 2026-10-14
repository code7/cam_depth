"""双深度相机采集界面。

同时支持两台 VCamera 深度相机：各自独立开关、独立采集、独立缩放、
独立平面标定与高度图，并额外提供“全部开始 / 全部停止 / 全部获取一帧”的批量操作。

配置：所有和某台相机强相关的参数（标定平面、距离过滤、面板位置、采集参数）
按相机**序列号**存进加密配置文件 cam_config.enc，只有把对应相机接上、
读出它的序列号，才解得开属于它的那一段。
"""

import collections
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import vcamera as vcam
import cam_config as cfgmod
import cut_plane
import modbus_rtu as mb
from modbus_service import ModbusService


CANVAS_W = 600          # 单个相机画布的初始宽度
CANVAS_H = 450          # 单个相机画布的初始高度
GRAB_TIMEOUT = 10.0
DEPTH_TO_MM = 0.125
NUM_CAMERAS = 2         # 同时支持的深度相机数量

# 显示刷新：两路画面共用一个定时器，按下面这个间隔取最新帧并重绘。
# 间隔越小画面越流畅，但 Tk 重绘开销越大；深度相机本身出帧不快，80 ms（约 12 fps）足够。
DISPLAY_INTERVAL_MS = 80
REFRESH_CHOICES = ("30", "50", "80", "120", "200", "300")

# 超过这么久没有新帧就在状态栏给出断流提示
STALL_WARN_S = 3.0
# 超过这么久没有新帧就自动恢复相机。
# 实测：相机可能静默卡死（状态显示 Capturing 但一帧不出），而且
# StopCapture→StartCapture **救不回来**，必须 Disconnect→Connect 重连才有效。
STALL_RECOVER_S = 8.0
RECOVER_MAX_ATTEMPTS = 6

# 配置文件默认放在程序目录下的 config/
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, cfgmod.DEFAULT_FILE_NAME)


def value_to_python(v):
    """把 SDK 的 Value 包装类型转成可 JSON 序列化的基本类型。

    注意不能用 isinstance(v, bool) 判断：Value 是包装类型，
    isinstance 永远是假，会走到错误的转换分支并抛异常。
    必须用它自己的类型查询和访问器。
    """
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


def fit_plane_ransac(depth_data, roi, step=2, max_iter=300, inlier_thresh=4.0, min_inliers=100):
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


class CameraPanel:
    """一台深度相机的完整面板：设备开关、采集、显示、缩放、标定与高度图。"""

    def __init__(self, app, parent, index):
        self.app = app
        self.root = app.root
        self.index = index
        self.name = f"相机 {index + 1}"

        # ---- 相机与采集状态 ----
        self.cam = None
        self.serial = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.is_running = False
        self.is_grabbing = False

        # ---- 画面状态 ----
        self._photo = None       # Tk 图像，复用而不是每帧重建，避免闪烁
        self._img_item = None    # 画布上的图像项，只创建一次
        self._preview = None     # 当前帧的 RGB 预览图（numpy，全分辨率）
        self.src_w = 0           # 预览图原始宽高，用于坐标换算
        self.src_h = 0
        self.depth_data = None
        self.depth_w = 0
        self.depth_h = 0
        self.scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # ---- 标定状态 ----
        self.calibrated = False
        self.calib_mode = False
        self.show_height = tk.BooleanVar(value=False)
        self.roi = None
        self.drag_start = None
        self._sel_rect = None
        self.plane_a = 0.0
        self.plane_b = 0.0
        self.plane_c = 0.0

        # ---- 这台相机在配置里的参数（按序列号关联，解锁后填入） ----
        self.camera_cfg = None

        # ---- 内参（第一次拿到帧时缓存，用于把像素面积换算成 mm²） ----
        self.fx = None
        self.fy = None
        self.cut_test_running = False

        # ---- 帧序号：Modbus 服务靠它判断"触发之后有没有来新帧" ----
        self.frame_seq = 0
        # ---- 切平参数（按相机存）----
        self.a_max_mm2 = cfgmod.DEFAULT_CUT["a_max_mm2"]

        # ---- 帧间隔统计 ----
        # 注意线程边界：frame_callback 跑在 SDK 原生线程上，只允许读写这几个
        # 普通变量；任何 Tk 控件的更新都必须回到主线程（见 pump_frame）。
        self.frames_received = 0        # 相机回调送来的帧总数
        self.frames_dropped = 0         # 显示来不及消费、被丢掉的帧数
        self.last_arrival = None        # 上一帧到达时刻（time.monotonic）
        self.arrival_ms = None          # 最近一次「上一帧 → 这一帧」的间隔
        self.arrival_history = collections.deque(maxlen=10)
        self.last_render_wall = None    # 上一帧真正画到屏幕上的时刻
        self.stall_notified = False
        # ---- 断流自动恢复 ----
        self.last_frame_wall = None     # 上一次收到帧的时刻（回调线程写）
        self.recovering = False
        self.recover_count = 0
        self._recover_result = None

        # ---- 「这一帧到达到上屏」的耗时（毫秒）----
        self.latency_ms = None          # 最近一次的总耗时
        self.latency_history = collections.deque(maxlen=10)
        self.stage_wait_ms = None       # 到达 → 开始处理（在队列里等刷新节拍）
        self.stage_compute_ms = None    # 取图 / 换算 / 高度图 / 配色
        self.stage_show_ms = None       # 缩放 / 上屏

        self._build_ui(parent)

    # ------------------------------------------------------------------ 界面
    def _build_ui(self, parent):
        self.frame = ttk.LabelFrame(parent, text=self.name)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(3, weight=1)

        # 设备选择与开关
        top = ttk.Frame(self.frame)
        top.grid(row=0, column=0, sticky="we", padx=4, pady=(4, 0))

        ttk.Label(top, text="设备:").pack(side="left")
        self.combo = ttk.Combobox(top, state="readonly", width=22)
        self.combo.pack(side="left", padx=4)
        self.btn_open = ttk.Button(top, text="打开", command=self.open_camera)
        self.btn_open.pack(side="left", padx=2)
        self.btn_close = ttk.Button(top, text="关闭相机", command=self.close_camera, state="disabled")
        self.btn_close.pack(side="left", padx=2)

        # 采集控制
        ctrl = ttk.Frame(self.frame)
        ctrl.grid(row=1, column=0, sticky="we", padx=4, pady=4)

        self.btn_start = ttk.Button(ctrl, text="开始", command=self.start_capture, state="disabled")
        self.btn_start.pack(side="left", padx=2)
        self.btn_stop = ttk.Button(ctrl, text="停止", command=self.stop_capture, state="disabled")
        self.btn_stop.pack(side="left", padx=2)
        self.btn_grab = ttk.Button(ctrl, text="获取一帧", command=self.grab_one_frame, state="disabled")
        self.btn_grab.pack(side="left", padx=2)
        self.btn_calib = ttk.Button(ctrl, text="标定平面", command=self.start_calibration, state="disabled")
        self.btn_calib.pack(side="left", padx=2)
        self.btn_cuttest = ttk.Button(ctrl, text="切面测试", command=self.run_cut_test, state="disabled")
        self.btn_cuttest.pack(side="left", padx=2)
        self.chk_height = ttk.Checkbutton(ctrl, text="显示高度图", variable=self.show_height)
        self.chk_height.pack(side="left", padx=6)

        # 距离过滤：按相机存，因为两台安装距离往往不同
        flt = ttk.Frame(self.frame)
        flt.grid(row=2, column=0, sticky="we", padx=4, pady=(0, 4))

        ttk.Label(flt, text="最小距离(mm):").pack(side="left")
        self.ent_min = ttk.Entry(flt, width=8)
        self.ent_min.pack(side="left", padx=(2, 8))
        self.ent_min.insert(0, str(int(cfgmod.DEFAULT_FILTER["min_mm"])))

        ttk.Label(flt, text="最大距离(mm):").pack(side="left")
        self.ent_max = ttk.Entry(flt, width=8)
        self.ent_max.pack(side="left", padx=2)
        self.ent_max.insert(0, str(int(cfgmod.DEFAULT_FILTER["max_mm"])))

        ttk.Label(flt, text="最大单坑(mm²):").pack(side="left", padx=(12, 2))
        self.ent_amax = ttk.Entry(flt, width=8)
        self.ent_amax.pack(side="left", padx=2)
        self.ent_amax.insert(0, f"{self.a_max_mm2:.0f}")

        # 输入框失焦或回车时把过滤区间存进这台相机的配置
        for ent in (self.ent_min, self.ent_max):
            ent.bind("<FocusOut>", self.on_filter_changed)
            ent.bind("<Return>", self.on_filter_changed)
        self.ent_amax.bind("<FocusOut>", self.on_cut_changed)
        self.ent_amax.bind("<Return>", self.on_cut_changed)

        # 画布
        self.canvas = tk.Canvas(self.frame, width=CANVAS_W, height=CANVAS_H, bg="black",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel(e, -120))
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

        self.lbl_coord = tk.Label(self.frame, text="坐标: (---, ---)   深度: ---", anchor="w")
        self.lbl_coord.grid(row=4, column=0, sticky="we", padx=4)

        self.lbl_status = tk.Label(self.frame, text="未打开", anchor="w", fg="#555")
        self.lbl_status.grid(row=5, column=0, sticky="we", padx=4, pady=(0, 4))

    # ------------------------------------------------------------------ 配置
    def bind_camera_config(self, cfg: dict):
        """把从配置文件解出来的参数应用到本面板。"""
        self.camera_cfg = cfg

        flt = cfg.get("filter") or {}
        self.ent_min.delete(0, "end")
        self.ent_min.insert(0, f"{float(flt.get('min_mm', cfgmod.DEFAULT_FILTER['min_mm'])):.0f}")
        self.ent_max.delete(0, "end")
        self.ent_max.insert(0, f"{float(flt.get('max_mm', cfgmod.DEFAULT_FILTER['max_mm'])):.0f}")

        cal = cfg.get("calibration") or {}
        if cfgmod.is_calibrated(cfg):
            a, b, c = (float(v) for v in cal["coefficients"])
            self.plane_a, self.plane_b, self.plane_c = a, b, c
            self.calibrated = True
            self.roi = tuple(cal["roi"]) if cal.get("roi") else None
        else:
            self.calibrated = False
            self.roi = None

        cut = cfg.get("cut") or {}
        self.a_max_mm2 = float(cut.get("a_max_mm2", cfgmod.DEFAULT_CUT["a_max_mm2"]))
        self.ent_amax.delete(0, "end")
        self.ent_amax.insert(0, f"{self.a_max_mm2:.0f}")

    def on_cut_changed(self, _event=None):
        """最大单坑面积变化：写回这台相机的配置并存盘。"""
        try:
            v = float(self.ent_amax.get())
        except ValueError:
            return
        if self.serial is None:
            return
        self.a_max_mm2 = max(1.0, v)
        cfg = self.app.store.camera(self.serial)
        cut = dict(cfg.get("cut") or cfgmod.DEFAULT_CUT)
        cut["a_max_mm2"] = self.a_max_mm2
        cfg["cut"] = cut
        self.app.save_config(f"相机 {self.index + 1} 最大单坑面积已保存")

    def panel_data(self):
        """给 Modbus 服务用的面板快照。"""
        return {
            "serial": self.serial,
            "depth": self.depth_data,
            "plane": ((self.plane_a, self.plane_b, self.plane_c)
                      if self.calibrated else None),
            "calibrated": bool(self.calibrated),
            "running": bool(self.is_running),
            "frame_id": self.frame_seq,
            "fx": self.fx, "fy": self.fy,
            "a_max_mm2": self.a_max_mm2,
        }

    def apply_fitted_plane(self, coeffs):
        """Modbus 校准请求拟合出的基准面回填到本面板（不经过手动框选）。"""
        a, b, c = (float(v) for v in coeffs)
        self.plane_a, self.plane_b, self.plane_c = a, b, c
        self.calibrated = True
        if self.serial is not None:
            cfg = self.app.store.camera(self.serial)
            cfg["calibration"] = {
                "type": "plane_xy",
                "coefficients": [a, b, c],
                "unit": "mm",
                "roi": None,
                "inlier_ratio": None,
                "calibrated_at": cfgmod.now_str(),
            }
            self.camera_cfg = cfg
        self.lbl_status.config(text=f"基准面已由 Modbus 校准："
                                    f"d = {a:.5f}·x + {b:.5f}·y + {c:.2f}")

    def on_filter_changed(self, _event=None):
        """距离过滤输入框变化：写回这台相机的配置并存盘。"""
        rng = self.get_dist_range()
        if rng[0] is None or self.serial is None:
            return
        cfg = self.app.store.camera(self.serial)
        cfg["filter"] = {"min_mm": rng[0], "max_mm": rng[1]}
        self.app.save_config(f"相机 {self.index + 1} 距离过滤已保存")

    def get_dist_range(self):
        """返回 (min_mm, max_mm)；输入非法或区间反了就返回 (None, None)。"""
        try:
            mn = float(self.ent_min.get())
            mx = float(self.ent_max.get())
        except ValueError:
            return None, None
        return (mn, mx) if mx > mn else (None, None)

    # ------------------------------------------------------------------ 帧回调
    def frame_callback(self, frame_set):
        """SDK 原生线程回调：只记录间隔并尽快入队，绝不碰 Tk 控件。

        队列只有 1 个位置：显示来不及消费时，新帧直接丢掉并计数。
        所以「收到多少帧」和「画了多少帧」是两件事，界面会把两者都显示出来。
        """
        now = time.monotonic()
        prev = self.last_arrival
        self.last_arrival = now
        if prev is not None:
            self.arrival_ms = (now - prev) * 1000.0
            self.arrival_history.append(self.arrival_ms)
        self.frames_received += 1
        self.last_frame_wall = now
        # 连同到达时刻一起入队：显示端要算「这一帧从到达到上屏花了多久」。
        # 不能用另一个成员变量记时刻——回调线程可能在主线程取走帧之后、
        # 读取时刻之前又塞进新的一帧，那样量出来的耗时会偏小。
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

    # ------------------------------------------------------------------ 打开 / 关闭
    def open_camera(self):
        app = self.app
        if not app.cam_infos:
            messagebox.showwarning("提示", f"{self.name}：请先点击【扫描】")
            return

        idx = self.combo.current()
        if idx < 0 or idx >= len(app.cam_infos):
            messagebox.showwarning("提示", f"{self.name}：请选择设备")
            return

        info = app.cam_infos[idx]
        serial = str(info.serial_number)

        other = app.other_panel(self.index)
        if other is not None and other.cam is not None and other.serial == serial:
            messagebox.showwarning("提示", f"设备 {serial} 已被{other.name}打开，请为{self.name}选择另一台设备")
            return

        cam = vcam.CameraFactory.GetCameraBySerialNumber(info.serial_number)

        status = cam.Connect()
        if not status:
            hint = ""
            if "busy" in status.message().lower():
                hint = "（相机已被占用，请先关掉另一个正在运行的程序实例）"
            messagebox.showerror("打开失败",
                                 f"{self.name} 连接相机失败: {status.message()}{hint}")
            return

        try:
            acq_mode, status = cam.GetFeature("AcquisitionMode")
            if not status:
                raise RuntimeError(f"获取 AcquisitionMode 失败: {status.message()}")
            status = acq_mode.SetValue(2)
            if not status:
                raise RuntimeError(f"设置连续采集模式失败: {status.message()}")
        except Exception as e:
            cam.Disconnect()
            messagebox.showerror("打开失败", f"{self.name}: {e}")
            return

        status = cam.SetSensorEnabled(vcam.SensorType.Depth, True)
        if not status:
            cam.Disconnect()
            messagebox.showerror("打开失败", f"{self.name}: 使能深度传感器失败")
            return

        # 把配置里存的采集参数下发到相机（必须在 StartCapture 之前，
        # 实测采集期间写 feature 会返回 DeviceBusy）
        acq_warnings = self.apply_acquisition_config(cam)

        cam.RegisterFrameSetCallback(self.frame_callback)

        self.cam = cam
        self.serial = serial

        # 记住面板归属并记录实际生效的采集参数（写回加密配置）
        cfg = app.store.camera(serial)
        cfg["panel"] = self.index
        cfg["model"] = str(getattr(info, "model", "") or "")
        cfg["firmware"] = str(getattr(info, "firmware_version", "") or "")
        self.camera_cfg = cfg
        # 切到这台相机就载入它自己的过滤区间和标定参数
        self.bind_camera_config(cfg)
        app.store.apply_acquisition_from_camera(
            serial, self._read_acquisition_features(cam),
            self._read_image_mode(cam), acq_warnings)

        self._clear_queue()
        self.combo.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.btn_close.config(state="normal")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_grab.config(state="normal")
        self.btn_calib.config(state="normal", text="标定平面", command=self.start_calibration)
        self.btn_cuttest.config(state="normal")
        note = ""
        if self.calibrated:
            note = "，已载入标定参数"
        if acq_warnings:
            note += f"，采集参数有 {len(acq_warnings)} 条告警"
        self.lbl_status.config(text=f"已打开 {serial}{note}，请点击【开始】或【获取一帧】")
        app.save_config()
        app.update_global_buttons()

    # ------------------------------------------------------------------ 采集参数
    def _read_acquisition_features(self, cam) -> dict:
        """读回相机上实际生效的采集参数，用于存进配置。"""
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

    def apply_acquisition_config(self, cam) -> list:
        """把配置里的采集参数写到相机上，返回告警列表。

        单条写不进去只记告警、不阻断开机 —— 现场换一台不支持某参数的相机，
        程序仍然要能起来。
        """
        warnings = []
        cfg = self.camera_cfg
        if not cfg:
            return warnings

        acq = cfg.get("acquisition") or {}
        feats = acq.get("features") or {}
        for name, item in feats.items():
            try:
                feat, st = cam.GetFeature(name)
                if not st:
                    warnings.append(f"{name} 相机不支持，已跳过")
                    continue
                acc, _ = feat.GetAccessMode()
                if str(acc).split(".")[-1] not in ("Writable", "ReadWritable"):
                    warnings.append(f"{name} 只读，已跳过")
                    continue
                st = feat.SetValue(item["value"])
                if not st:
                    warnings.append(f"{name} 写入失败：{st.message()}")
                    continue
                val, vs = feat.GetValue()
                if vs and val != item["value"]:
                    warnings.append(f"{name} 实际生效为 {val}（要求 {item['value']}）")
            except Exception as e:
                warnings.append(f"{name} 写入异常：{e}")

        mode = acq.get("image_mode")
        if mode:
            try:
                modes, st = cam.GetImageModes(vcam.SensorType.Depth)
                if st and modes:
                    match = None
                    for m in modes:
                        if (int(m.width) == mode["width"] and int(m.height) == mode["height"]
                                and (not mode.get("pixel_format")
                                     or str(getattr(m, "pixel_format", "")).split(".")[-1]
                                     == mode["pixel_format"])):
                            match = m
                            break
                    if match is None:
                        warnings.append("配置里的图像模式当前相机不支持，已跳过")
                    else:
                        st = cam.SetImageMode(vcam.SensorType.Depth, match)
                        if not st:
                            warnings.append(f"图像模式写入失败：{st.message()}")
                else:
                    warnings.append("无法读取图像模式列表，已跳过")
            except Exception as e:
                warnings.append(f"图像模式写入异常：{e}")
        return warnings

    def close_camera(self):
        if self.cam is None:
            return

        self.is_running = False
        self.is_grabbing = False
        cam = self.cam
        self.cam = None
        self.serial = None
        try:
            cam.StopCapture()
        except Exception:
            pass
        try:
            cam.Disconnect()
        except Exception:
            pass

        self.depth_data = None
        self.depth_w = self.depth_h = 0
        self._preview = None
        self.src_w = self.src_h = 0
        self._photo = None
        self._img_item = None
        self.calibrated = False
        self.calib_mode = False
        self.show_height.set(False)
        self.roi = None
        self.reset_zoom()

        self.combo.config(state="readonly")
        self.btn_open.config(state="normal")
        self.btn_close.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.btn_grab.config(state="disabled")
        self.btn_calib.config(state="disabled", text="标定平面", command=self.start_calibration)
        self.btn_cuttest.config(state="disabled")
        self.canvas.delete("all")
        self.lbl_status.config(text="相机已关闭")
        self.lbl_coord.config(text="坐标: (---, ---)   深度: ---")
        self.app.update_global_buttons()

    # ------------------------------------------------------------------ 开始 / 停止
    def start_capture(self, quiet=False):
        if self.cam is None:
            if not quiet:
                messagebox.showwarning("提示", f"{self.name}：请先打开相机")
            return False
        if self.is_running or self.is_grabbing:
            return False

        self._clear_queue()
        if not self.cam.StartCapture():
            if not quiet:
                messagebox.showerror("开始失败", f"{self.name}：启动采集失败")
            return False

        self._reset_frame_stats()
        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_grab.config(state="disabled")
        self.lbl_status.config(text="采集中...  等待第一帧")
        self.app.update_global_buttons()
        # 显示由主窗口的统一节拍驱动（见 DepthCamApp._tick），这里不再各自起定时器
        return True

    def _reset_frame_stats(self):
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_arrival = None
        self.arrival_ms = None
        self.arrival_history.clear()
        # 断流计时的基准从"开始采集"算起。
        # 不能只依赖 last_render_wall：从没收到过帧的相机两个时间都是 None，
        # 那样 _check_stall 会直接返回，永远触发不了自动恢复。
        self.last_render_wall = time.monotonic()
        self.last_frame_wall = self.last_render_wall
        self.stall_notified = False
        self.recovering = False
        self.recover_count = 0
        self._recover_result = None
        self.latency_ms = None
        self.latency_history.clear()
        self.stage_wait_ms = None
        self.stage_compute_ms = None
        self.stage_show_ms = None

    def frame_status_text(self):
        """状态栏里的采集信息：帧间隔、平均间隔/帧率、收帧与丢帧数。"""
        if self.arrival_ms is None:
            return "采集中...  等待第一帧"
        text = f"采集中...   帧间隔 {self.arrival_ms:.0f} ms"
        if self.arrival_history:
            avg = sum(self.arrival_history) / len(self.arrival_history)
            fps = 1000.0 / avg if avg > 0 else 0.0
            text += f"（近 {len(self.arrival_history)} 帧均 {avg:.0f} ms ≈ {fps:.2f} fps）"
        text += f"   收 {self.frames_received} 帧"
        if self.frames_dropped:
            text += f" / 丢 {self.frames_dropped} 帧"
        return text

    def timing_text(self):
        """第二行：从帧到达，到（用标定平面算出的）画面真正上屏花了多久。"""
        if self.latency_ms is None:
            return None
        if self.show_height.get() and self.calibrated:
            label = "高度图耗时"
        else:
            label = "显示耗时"
        avg = (sum(self.latency_history) / len(self.latency_history)
               if self.latency_history else self.latency_ms)
        return (f"{label} {self.latency_ms:.0f} ms ＝ 等待 {self.stage_wait_ms:.0f}"
                f" + 计算 {self.stage_compute_ms:.0f}"
                f" + 上屏 {self.stage_show_ms:.0f}   （近 {len(self.latency_history)} 帧均 {avg:.0f} ms）")

    def status_text(self):
        """状态栏完整文字（采集信息 + 耗时拆解）。"""
        first = self.frame_status_text()
        second = self.timing_text()
        return first if second is None else f"{first}\n{second}"

    def stop_capture(self, quiet=False):
        if not self.is_running:
            return False

        self.is_running = False
        try:
            self.cam.StopCapture()
        except Exception as e:
            self.lbl_status.config(text=f"停止异常: {e}")

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_grab.config(state="normal")
        summary = f"已停止采集（共收 {self.frames_received} 帧"
        if self.arrival_history:
            avg = sum(self.arrival_history) / len(self.arrival_history)
            summary += f"，平均间隔 {avg:.0f} ms"
        if self.frames_dropped:
            summary += f"，丢弃 {self.frames_dropped} 帧"
        if self.latency_history:
            lat = sum(self.latency_history) / len(self.latency_history)
            summary += f"，上屏平均 {lat:.0f} ms"
        self.lbl_status.config(text=summary + "）")
        self.app.update_global_buttons()
        return True

    # ------------------------------------------------------------------ 获取一帧
    def grab_one_frame(self, quiet=False):
        if self.cam is None:
            if not quiet:
                messagebox.showwarning("提示", f"{self.name}：请先打开相机")
            return False
        if self.is_running:
            if not quiet:
                messagebox.showwarning("提示", f"{self.name}：正在采集中，请先点击【停止】")
            return False
        if self.is_grabbing:
            return False

        self.is_grabbing = True
        self.btn_grab.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.lbl_status.config(text="正在获取一帧...")
        self.app.update_global_buttons()
        self.root.update_idletasks()
        threading.Thread(target=self._grab_worker, daemon=True).start()
        return True

    def _grab_worker(self):
        error = None
        elapsed = None
        fs = None
        arrival = None
        try:
            self._clear_queue()
            if not self.cam.StartCapture():
                error = "启动采集失败"
            else:
                t0 = time.perf_counter()
                deadline = time.time() + GRAB_TIMEOUT
                while time.time() < deadline:
                    try:
                        arrival, fs = self.frame_queue.get(timeout=0.5)
                        break
                    except queue.Empty:
                        continue
                elapsed = (time.perf_counter() - t0) * 1000
                try:
                    self.cam.StopCapture()
                except Exception:
                    pass
        except Exception as e:
            error = str(e)
            try:
                self.cam.StopCapture()
            except Exception:
                pass
        self.root.after(0, lambda: self._grab_done(fs, elapsed, error, arrival))

    def _grab_done(self, fs, elapsed, error, arrival=None):
        self.is_grabbing = False
        self.btn_grab.config(state="normal")
        self.btn_start.config(state="normal")

        if error:
            messagebox.showerror("获取失败", f"{self.name}: {error}")
            self.lbl_status.config(text="获取一帧失败")
            self.app.update_global_buttons()
            return
        if fs is None:
            messagebox.showwarning("提示", f"{self.name}：超时（{GRAB_TIMEOUT:.0f}s）未获取到帧")
            self.lbl_status.config(text="获取一帧超时")
            self.app.update_global_buttons()
            return
        if not self._render_frame(fs, arrival):
            messagebox.showwarning("提示", f"{self.name}：帧无效")
            self.lbl_status.config(text="获取一帧失败：帧无效")
        else:
            text = f"已获取一帧（采集启动到收帧 {elapsed:.0f} ms）"
            if self.latency_ms is not None:
                text += f"；上屏耗时 {self.latency_ms:.0f} ms"
            self.lbl_status.config(text=text)
        self.app.update_global_buttons()

    # ------------------------------------------------------------------ 显示
    def _render_frame(self, fs, arrival=None):
        """处理一帧并上屏。

        arrival 是这一帧到达的 time.monotonic 时刻；给了它就会记录
        「到达到上屏」的总耗时，并拆成 等待 / 计算（含高度图）/ 上屏 三段。
        """
        t_start = time.perf_counter()
        depth_image = fs.GetImage(vcam.SensorType.Depth)
        if not depth_image.IsValid():
            return False

        self.depth_data = depth_image.to_numpy().astype(np.float32) * DEPTH_TO_MM
        self.depth_h, self.depth_w = self.depth_data.shape[:2]

        if self.show_height.get() and self.calibrated:
            vis_data = self.compute_height_map()
        else:
            vis_data = self.depth_data

        vis = self._make_vis(vis_data)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        # 保留 RGB 预览图（numpy），画布刷新时再用 cv2 缩放——比 PIL resize 快
        self._preview = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        self.src_w = self.depth_w
        self.src_h = self.depth_h
        t_computed = time.perf_counter()

        # 顺手缓存内参（只做一次），后面把像素面积换算成 mm² 要用
        if self.fx is None:
            try:
                K = np.asarray(depth_image.calib_info().intrinsic.matrix,
                               dtype=np.float64)
                self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
            except Exception:
                self.fx = self.fy = None

        self._redraw_image()
        t_shown = time.perf_counter()

        # 这一段耗时（取图 → 高度图算完 → 上屏）
        self.stage_compute_ms = (t_computed - t_start) * 1000.0
        self.stage_show_ms = (t_shown - t_computed) * 1000.0
        if arrival is not None:
            self.stage_wait_ms = (t_start - arrival) * 1000.0
            self.latency_ms = (t_shown - arrival) * 1000.0
            self.latency_history.append(self.latency_ms)
        return True

    def _make_vis(self, vis_data):
        if self.show_height.get() and self.calibrated:
            return self._make_vis_auto(vis_data)
        mn, mx = self.get_dist_range()
        if mn is not None:
            vis = (vis_data - mn) * (255.0 / (mx - mn))
            np.clip(vis, 0.0, 255.0, out=vis)
            np.nan_to_num(vis, copy=False, nan=0.0)
            return vis.astype(np.uint8)
        return self._make_vis_auto(vis_data)

    def _make_vis_auto(self, vis_data):
        return cv2.normalize(vis_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def _redraw_image(self):
        """按当前缩放把预览图画到画布上。

        画布上的图像项和 Tk 图像都会被复用：只在尺寸变化时重建 Tk 图像，
        其余情况用 PhotoImage.paste 原地更新。原来的“每帧 delete + create_image”
        会让画布反复重建显示对象，正是画面闪烁/刷新卡顿的来源。
        """
        if self._preview is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        if cw <= 1 or ch <= 1:
            cw, ch = CANVAS_W, CANVAS_H

        iw, ih = self.src_w, self.src_h
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

    def _on_canvas_resize(self):
        self._redraw_image()

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

    def on_mouse_wheel(self, event, delta=None):
        if self._preview is None:
            return
        if delta is None:
            delta = event.delta
        factor = 1.1 if delta > 0 else 1 / 1.1
        new_zoom = min(max(self.zoom * factor, 0.05), 20.0)
        if new_zoom == self.zoom:
            return

        # 以鼠标指针为中心缩放：保持鼠标下的图像点位置不变
        img_x = (event.x - self.offset_x) / self.scale
        img_y = (event.y - self.offset_y) / self.scale
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = CANVAS_W, CANVAS_H
        iw, ih = self.src_w, self.src_h

        self.zoom = new_zoom
        scale = min(cw / iw, ch / ih) * self.zoom
        self.scale = scale
        self.offset_x = event.x - img_x * scale
        self.offset_y = event.y - img_y * scale
        self._redraw_image()

    def reset_zoom(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._redraw_image()

    def pump_frame(self):
        """由主窗口的统一节拍调用：取队列里最新的一帧并重绘，没有新帧就不动。"""
        if not self.is_running:
            return
        try:
            arrival, fs = self.frame_queue.get_nowait()
        except queue.Empty:
            self._check_stall()
            return
        try:
            ok = self._render_frame(fs, arrival)
        except Exception as e:
            self.lbl_status.config(text=f"显示错误: {e}")
            return
        if not ok:
            return

        self.last_render_wall = time.monotonic()
        self.stall_notified = False
        self.recover_count = 0     # 真正出画面了才清空重试预算
        self.frame_seq += 1        # Modbus 服务靠它判断新帧
        self.lbl_status.config(text=self.status_text())

    def _check_stall(self):
        """没新帧时的处理：先提示，再自动恢复相机。

        实测相机可能静默卡死（相机状态仍是 Capturing，但回调一帧都不来），
        而且 StopCapture→StartCapture 救不回来，只有 Disconnect→Connect 有效，
        所以这里走的是完整重连。
        """
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
                self.lbl_status.config(
                    text=f"⚠ 已 {gap:.1f}s 无新帧，正在自动恢复相机"
                         f"（第 {self.recover_count + 1} 次）…")
                self._start_recover()
            elif not self.stall_notified:
                self.stall_notified = True
                self.lbl_status.config(
                    text=f"⚠ 相机断流且连续恢复 {RECOVER_MAX_ATTEMPTS} 次失败，"
                         f"请检查供电/网线后重新打开相机")
            return

        if self.stall_notified or gap < STALL_WARN_S:
            return
        self.stall_notified = True
        self.lbl_status.config(text=f"⚠ 采集中，但已 {gap:.1f} s 没有新帧（相机可能断流）")

    # ------------------------------------------------------------------ 断流恢复
    def _start_recover(self):
        self.recovering = True
        self.recover_count += 1
        self._recover_result = None
        threading.Thread(target=self._recover_worker, daemon=True).start()

    def _recover_worker(self):
        """断开重连并重启采集。跑在工作线程里，不碰 Tk 控件。"""
        err = None
        newcam = None
        old = self.cam
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
            cam = vcam.CameraFactory.GetCameraBySerialNumber(self.serial)
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
        self._recover_result = (err, newcam)

    def _finish_recover(self):
        """回到主线程收尾。"""
        err, newcam = self._recover_result
        self._recover_result = None
        self.recovering = False
        if err or newcam is None:
            self.lbl_status.config(text=f"⚠ 自动恢复失败：{err}")
            return
        self.cam = newcam
        self._clear_queue()
        n = self.recover_count
        self.last_frame_wall = time.monotonic()   # 重新开始计时，避免立刻又触发
        self.last_render_wall = self.last_frame_wall
        self.stall_notified = False
        # 注意：这里**不**清零 recover_count。重连成功不代表出帧正常，
        # 只有真的收到画面了（pump_frame）才清空重试预算，否则会无限重连。
        self.lbl_status.config(text=f"相机已自动重连（第 {n} 次），等待新帧…")

    def compute_height_map(self):
        # 用广播代替 np.indices，避免每帧分配两块 int64 全尺寸数组（2048x1536 时各 25MB）
        xs = np.arange(self.depth_w, dtype=np.float32)
        ys = np.arange(self.depth_h, dtype=np.float32)
        plane = self.plane_a * xs[None, :] + self.plane_b * ys[:, None] + self.plane_c
        height = self.depth_data - plane
        invalid = ~np.isfinite(height) | (self.depth_data <= 0)
        height[invalid] = 0
        return height

    # ------------------------------------------------------------------ 鼠标坐标
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

    # ------------------------------------------------------------------ 标定
    def start_calibration(self):
        if not self.calib_mode:
            self.calib_mode = True
            self.roi = None
            self.btn_calib.config(text="取消标定", command=self.cancel_calibration)
            self.lbl_status.config(text="标定模式：请在画布上拖拽框选平坦的基准面区域")
        else:
            self.cancel_calibration()

    def cancel_calibration(self):
        self.calib_mode = False
        if self._sel_rect is not None:
            self.canvas.delete(self._sel_rect)
            self._sel_rect = None
        self.btn_calib.config(text="标定平面", command=self.start_calibration)
        self.lbl_status.config(text="采集中..." if self.is_running else "已停止采集")

    def on_drag_start(self, event):
        if not self.calib_mode:
            return
        self.drag_start = (event.x, event.y)
        self._sel_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                      outline="red", width=2)

    def on_drag_move(self, event):
        if not self.calib_mode or self.drag_start is None:
            return
        self.canvas.coords(self._sel_rect, self.drag_start[0], self.drag_start[1], event.x, event.y)

    def on_drag_end(self, event):
        if not self.calib_mode or self.drag_start is None:
            return
        x0c, y0c = self.drag_start
        x1c, y1c = event.x, event.y
        self.drag_start = None

        if abs(x1c - x0c) < 10 or abs(y1c - y0c) < 10:
            self.canvas.delete(self._sel_rect)
            self._sel_rect = None
            return

        ix0, iy0 = self.canvas_to_image(min(x0c, x1c), min(y0c, y1c))
        ix1, iy1 = self.canvas_to_image(max(x0c, x1c), max(y0c, y1c))
        self.roi = (max(0, ix0), max(0, iy0), min(self.depth_w, ix1), min(self.depth_h, iy1))
        self.canvas.delete(self._sel_rect)
        self._sel_rect = None
        self.calib_mode = False
        self.btn_calib.config(text="标定平面", command=self.start_calibration)
        self.run_calibration()

    def canvas_to_image(self, cx, cy):
        x = int((cx - self.offset_x) / self.scale)
        y = int((cy - self.offset_y) / self.scale)
        return x, y

    def run_calibration(self):
        if self.depth_data is None or self.depth_w <= 0 or self.depth_h <= 0:
            messagebox.showwarning("提示", f"{self.name}：暂无图像，请先获取一帧或开始采集")
            return
        if self.roi is None:
            return
        x0, y0, x1, y1 = self.roi
        if x1 - x0 < 8 or y1 - y0 < 8:
            messagebox.showwarning("提示", f"{self.name}：标定区域过小")
            return

        self.btn_calib.config(state="disabled")
        self.lbl_status.config(text="正在拟合平面，请稍候...")
        data = np.array(self.depth_data)
        roi = self.roi
        threading.Thread(target=self._calib_worker, args=(data, roi), daemon=True).start()

    def _calib_worker(self, data, roi):
        error = None
        try:
            result = fit_plane_ransac(data, roi)
        except Exception as e:
            result = None
            error = str(e)
        self.root.after(0, lambda: self._calib_done(result, error))

    def _calib_done(self, result, error):
        self.btn_calib.config(state="normal")
        if error:
            messagebox.showerror("标定失败", f"{self.name}: {error}")
            self.lbl_status.config(text="标定失败")
            return
        if result is None:
            messagebox.showerror("标定失败", f"{self.name}：区域内有效深度点不足或拟合失败，请重新框选")
            self.lbl_status.config(text="标定失败，请重新框选")
            return

        a, b, c, ratio = result
        self.plane_a, self.plane_b, self.plane_c = a, b, c
        self.calibrated = True

        # 标定结果按序列号写回加密配置：只有这台相机接上才解得开
        saved = ""
        if self.serial is not None:
            cfg = self.app.store.camera(self.serial)
            cfg["calibration"] = {
                "type": "plane_xy",
                "coefficients": [float(a), float(b), float(c)],
                "unit": "mm",
                "roi": list(self.roi) if self.roi else None,
                "inlier_ratio": float(ratio),
                "calibrated_at": cfgmod.now_str(),
            }
            self.camera_cfg = cfg
            if self.app.save_config():
                saved = "，已保存到配置"
            else:
                saved = "，但配置保存失败"

        self.lbl_status.config(
            text=f"标定完成：d = {a:.5f}·x + {b:.5f}·y + {c:.2f}，内点率 {ratio:.1%}{saved}。"
                 f"勾选【显示高度图】查看高度")

    # ------------------------------------------------------------------ 切面测试
    def run_cut_test(self):
        """诊断按钮：报告基准面、噪声、以及不同 a_max 下算出的切深。

        这一步只是"看数据"，不参与生产流程 —— 用来在真机上决定
        a_max / 降噪 / 空洞填补这几个参数该怎么定。
        """
        if self.depth_data is None:
            messagebox.showwarning("提示", f"{self.name}：暂无图像，请先【开始】或【获取一帧】")
            return
        if self.cut_test_running:
            return

        self.cut_test_running = True
        self.btn_cuttest.config(state="disabled")
        self.lbl_status.config(text="正在做切面测试，请稍候...")
        self.root.update_idletasks()

        data = np.array(self.depth_data)
        coeffs = ((self.plane_a, self.plane_b, self.plane_c)
                  if self.calibrated else None)
        fx, fy = self.fx, self.fy
        threading.Thread(target=self._cut_test_worker,
                         args=(data, coeffs, fx, fy), daemon=True).start()

    def _cut_test_worker(self, data, coeffs, fx, fy):
        try:
            report = build_cut_test_report(data, coeffs, fx, fy, self.name)
            err = None
        except Exception as e:
            report, err = None, str(e)
        self.root.after(0, lambda: self._cut_test_done(report, err))

    def _cut_test_done(self, report, err):
        self.cut_test_running = False
        self.btn_cuttest.config(state="normal")
        if err:
            messagebox.showerror("切面测试失败", f"{self.name}: {err}")
            self.lbl_status.config(text="切面测试失败")
            return
        self.lbl_status.config(text="切面测试完成，结果见弹窗")
        self.app.show_report(f"{self.name} · 切面测试", report)


def build_cut_test_report(data_mm, coeffs, fx, fy, name="相机", pick_a_max=500.0):
    """生成切面测试报告文本。不依赖 Tk，便于单测。

    报告分两步走，对应现场的两个方案：
      方案 A  零凹坑：切到最低点，保证不留凹坑，但最费料（标尺）
      方案 B  容差切平：允许单个连通凹坑 ≤ a_max，换取厚度
    """
    lines = [f"===== {name} 切面测试 =====", ""]
    valid = cut_plane.valid_mask(data_mm)
    if not valid.any():
        raise RuntimeError("这一帧没有有效深度数据")
    h_img, w_img = data_mm.shape

    # 1) 基准面
    if coeffs is None:
        # 没标定过就临时在中心区域拟合一个，只为了把 a、b 的量级看出来
        roi = (w_img // 4, h_img // 4, 3 * w_img // 4, 3 * h_img // 4)
        res = fit_plane_ransac(data_mm, roi)
        if res is None:
            lines.append("基准面：（未标定，且临时拟合失败）")
            return "\n".join(lines)
        a, b, c, ratio = res
        lines.append("基准面：**临时拟合**（还没有标定，仅供参考）")
        lines.append(f"  区域 {roi}")
    else:
        a, b, c = coeffs
        ratio = float("nan")
        lines.append("基准面：来自已保存的标定")
    lines.append(f"  z = {a:.6f}·x + {b:.6f}·y + {c:.2f}")
    if ratio == ratio:
        lines.append(f"  内点率 {ratio:.1%}")
    slope = float(np.degrees(np.arctan(np.hypot(a, b))))
    lines.append(f"  面内坡度 {slope:.3f} 度"
                 f"（接近 0 说明基准面正对相机；明显不为 0 则现有线性拟合会有系统误差）")
    lines.append(f"  a、b 对整幅图的影响 |a|·W={abs(a)*w_img:.2f} mm，"
                 f"|b|·H={abs(b)*h_img:.2f} mm")
    lines.append("")

    plane = cut_plane.plane_from_coeffs((a, b, c), data_mm.shape)
    h = cut_plane.build_height(data_mm, plane)

    # 2) 有效点与面积
    if fx and fy:
        mm2_px = cut_plane.mm2_per_pixel(data_mm, fx, fy)
        lines.append(f"内参：fx={fx:.1f} fy={fy:.1f}  → 每像素 {mm2_px:.4f} mm²")
    else:
        mm2_px = 1.0
        lines.append("内参：**不可用**，面积按像素数计（a_max 的单位此时是像素）")

    lines.append(f"有效深度点：{valid.mean()*100:.1f}%"
                 f"   （立体相机在光滑木面上覆盖率可能偏低，需重点确认）")
    lines.append(f"有效面积：{valid.sum()*mm2_px/100:.1f} cm²")
    lines.append("")

    # 3) 噪声
    rng = cut_plane.diagnose(h, valid, mm2_px,
                             a_max_list=(50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0))
    lines.append(f"高度 h = 深度 − 基准面（近负远正）")
    lines.append(f"  均值 {rng['mean']:+.3f} mm   标准差 {rng['std']:.3f} mm")
    lines.append(f"  范围 {rng['min']:+.3f} ~ {rng['max']:+.3f} mm")
    lines.append(f"  中值滤波残差：稳健噪声估计(MAD) = {rng['noise_mad']:.3f} mm"
                 f"   ← 这个才是噪声水平")
    lines.append(f"                高频 std = {rng['hf_std']:.3f} mm"
                 f"   （含真实台阶/边缘，通常明显偏大，仅供参考）")
    lines.append(f"  （测量噪声直接决定方案A 的切深，也决定要不要先降噪）")
    lines.append("")
    lines.append("面积加权分位数（对应『总凹坑面积占比 ≤ α』的另一种口径，仅供参考）：")
    for al, v in rng["quantiles"].items():
        lines.append(f"  α={al:.0%} → 切深 {v:+.3f} mm")
    lines.append("")

    # 4) 方案 A：零凹坑
    plan_a = cut_plane.solve_no_pit(h, valid, mm2_px)
    lines.append("【方案 A】零凹坑 —— 切到最低点（最费料，作为标尺）")
    lines.append(f"  切深 δA = {plan_a['delta_mm']:+.3f} mm"
                 f"   （相对基准面的偏移，正值=比基准面切得更深）")
    lines.append(f"  切除体积 = {plan_a['cut_volume_mm3']/1000:.1f} cm³"
                 f"      残留凹坑 = 0 个")
    worst_robust = max(plan_a["robust"].values()) if plan_a["robust"] else plan_a["delta_mm"]
    gap = plan_a["delta_mm"] - worst_robust
    for p, v in sorted(plan_a["robust"].items(), reverse=True):
        lines.append(f"  稳健性：去掉最高 {1-p:.02%} 像素后 δ = {v:+.3f} mm"
                     f"（差 {plan_a['delta_mm']-v:.3f} mm）")
    lines.append(f"  最高点 0.2mm 内的像素只有 {plan_a['near_max_count']} 个")
    if gap > 0.3:
        lines.append(f"  ⚠ 稳健性差值 {gap:.3f} mm 偏大 —— δA 主要由极少数像素决定，"
                     f"很可能是在切噪声/坏点，建议先降噪再采信这个数")
    lines.append("")

    # 5) 方案 B：容差切平（扫描）
    lines.append("【方案 B】容差切平 —— 只要『单个连通凹坑面积 ≤ a_max』，其余凹坑允许保留")
    lines.append(f"  {'a_max(mm²)':>12} {'切深δB(mm)':>11} {'最大坑(mm²)':>12} "
                 f"{'凹坑占比':>9} {'坑数':>5} {'贴边':>5} {'二分':>5}")
    for a_max, r in rng["a_max_sweep"].items():
        if "error" in r:
            lines.append(f"  {a_max:>12.0f}   失败: {r['error']}")
            continue
        lines.append(f"  {a_max:>12.0f} {r['delta_mm']:>11.3f} "
                     f"{r['max_pit_area_mm2']:>12.1f} {r['pit_ratio']:>9.2%} "
                     f"{r['pit_count']:>5} {r['edge_pits']:>5} {r['iterations']:>5}")
    lines.append("")

    # 6) 两步对比：容差换来多少料
    ref = cut_plane.solve_cut_plane(h, valid, mm2_px, pick_a_max)
    saved_depth = plan_a["delta_mm"] - ref["delta_mm"]
    saved_vol = plan_a["cut_volume_mm3"] - ref["cut_volume_mm3"]
    lines.append(f"【对比】以 a_max = {pick_a_max:.0f} mm² 为例，容差换来的收益")
    lines.append(f"  方案A δA = {plan_a['delta_mm']:+.3f} mm   （零凹坑）")
    lines.append(f"  方案B δB = {ref['delta_mm']:+.3f} mm   "
                 f"（最大坑 {ref['max_pit_area_mm2']:.1f} mm²，"
                 f"凹坑占比 {ref['pit_ratio']:.2%}，共 {ref['pit_count']} 个）")
    lines.append(f"  省下厚度 = δA − δB = {saved_depth:+.3f} mm")
    lines.append(f"  省下木料 ≈ {saved_vol/1000:.1f} cm³")
    if saved_depth > 0 and gap > saved_depth:
        lines.append(f"  注：方案A 的稳健性差值({gap:.3f} mm)大于容差收益({saved_depth:.3f} mm)，"
                     f"说明当前数据里方案A 的数字还不可靠")
    lines.append("")

    lines.append("说明：")
    lines.append("  · 切深是相对基准面的偏移，正值表示比基准面切得更深。")
    lines.append("  · 贴边的坑面积被画面截断，数值不可信，选择 a_max 时要留意。")
    lines.append("  · 方案A 是理论最费料的上界；方案B 用 a_max 换取厚度，"
                 "a_max 越大省料越多、但留下的坑越大。")
    if valid.mean() < 0.5:
        lines.append("  · ⚠ 有效点不足 50%，空洞会把大坑切碎、使结果偏乐观，"
                     "建议先改善拍摄条件或加大空洞填补半径。")
    return "\n".join(lines)


class DepthCamApp:
    """主窗口：两台深度相机面板 + 全局的批量操作与配置管理。"""

    def __init__(self, root):
        self.root = root
        self.root.title(f"{NUM_CAMERAS} 路深度相机")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cam_infos = []
        self.panels = []
        self.display_interval_ms = DISPLAY_INTERVAL_MS

        # 配置：先只读容器。此时还不知道相机 ID，所以解不开任何一段。
        # 真正解密发生在扫描拿到序列号之后（见 unlock_config）。
        self.store = cfgmod.ConfigStore(CONFIG_PATH)
        self.has_config_file = self.store.load()
        self.unlock_report = None

        # Modbus / RS-485：PLC 是主站，本程序是从站
        self.regs = mb.HoldingRegisters(addr_base=0)
        self.modbus = ModbusService(
            self.regs, self.get_panel_data, log=self.log_modbus,
            timeout_s=3.0)
        self.modbus.on_calibrated = self.on_modbus_calibrated
        self.modbus_log = []

        self._build_ui()
        self._apply_app_config()

        vcam.CameraUtils.Init(True)
        self.root.after(self.display_interval_ms, self._tick)

    # ------------------------------------------------------------------ Modbus
    def log_modbus(self, msg):
        stamp = time.strftime("%H:%M:%S")
        self.modbus_log.append(f"{stamp}  {msg}")
        if len(self.modbus_log) > 500:
            del self.modbus_log[:-500]
        print(msg)

    def get_panel_data(self):
        """给 Modbus 服务的 A/B 面数据。A面=面板1，B面=面板2（按相机序列号绑定）。"""
        return {"A": self.panels[0].panel_data(),
                "B": self.panels[1].panel_data()}

    def on_modbus_calibrated(self, res):
        """PLC 触发的距离校准：把拟合出来的基准面回填到面板并存盘。"""
        try:
            if "A" in res:
                self.panels[0].apply_fitted_plane(res["A"])
            if "B" in res:
                self.panels[1].apply_fitted_plane(res["B"])
            self.save_config("基准面已由 Modbus 校准并保存")
        except Exception as e:
            self.log_modbus(f"[Modbus] 校准回填失败：{e}")

    def toggle_modbus(self):
        if self.modbus.running:
            self.modbus.stop()
            self.btn_modbus.config(text="连接485")
            self.lbl_modbus.config(text="Modbus：未连接", fg="#888")
            self.log_modbus("[Modbus] 已断开")
            return
        cfg = dict(self.store.app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        cfg["port"] = self.ent_port.get().strip() or cfg["port"]
        try:
            cfg["baudrate"] = int(float(self.ent_baud.get()))
            cfg["slave_addr"] = int(float(self.ent_slave.get()))
        except ValueError:
            messagebox.showwarning("提示", "波特率/站号必须是数字")
            return
        cfg["addr_base"] = int(self.cmb_base.get())
        self.store.app["modbus"] = cfg
        self.save_config()

        if not self.modbus.start(cfg["port"], cfg["baudrate"],
                                 cfg["slave_addr"], cfg["addr_base"]):
            err = self.modbus_log[-1] if self.modbus_log else "打开串口失败"
            self.lbl_modbus.config(text=f"Modbus：连接失败", fg="#c00")
            messagebox.showerror("485 连接失败", err)
            return
        self.btn_modbus.config(text="断开485")
        self.lbl_modbus.config(
            text=f"Modbus：已连接 {cfg['port']} @{cfg['baudrate']} "
                 f"站号{cfg['slave_addr']} 基址{cfg['addr_base']}",
            fg="#0a6")

    def show_registers(self):
        """寄存器监视窗口：现场对地址、看 PLC 写了什么。"""
        s = self.regs.snapshot()
        base = self.regs.addr_base
        L = ["寄存器一览（文档编号 / 线上地址 / 名称 / 值）", ""]
        for i in range(self.regs.size):
            rw = "R" if i in mb.REG_READONLY else "W"
            val = s[i]
            extra = ""
            if i == mb.REG_STATUS:
                extra = f"   ({mb.STATUS_NAMES.get(val, '?')})"
            elif i in (mb.REG_DIST_A, mb.REG_DIST_B):
                extra = f"   ({val/mb.DIST_SCALE:.1f} mm)"
            L.append(f"  {40001+i:<6} wire={i+base:<3} {rw}  "
                     f"{mb.REG_NAMES.get(i,'?'):<20} {val:<7}{extra}")
        L += ["", f"距离分辨率：{mb.DIST_SCALE:.0f} → 0.1mm，"
                  f"量程 {mb.DIST_MIN_MM:.0f}~{mb.DIST_MAX_MM:.0f} mm",
              f"连接状态：{'已连接' if self.modbus.running else '未连接'}"]
        if self.modbus.last_result:
            L.append(f"上次下刀位置：A面 {self.modbus.last_result[0]:.1f} mm，"
                     f"B面 {self.modbus.last_result[1]:.1f} mm"
                     f"（周期 {self.modbus.cycle_ms:.0f} ms）")
        if self.modbus.last_error:
            L.append(f"上次故障：{self.modbus.last_error}")
        if self.modbus.slave is not None:
            st = self.modbus.slave.stats
            L += ["", f"通讯统计：收 {st['rx']} 帧，发 {st['tx']} 帧，"
                      f"CRC 错 {st['crc_err']}，站号不符 {st['addr_skip']}，"
                      f"异常响应 {st['exc']}"]
            addrs = sorted(st["req_addrs"].items())
            if addrs:
                L.append("PLC 用过的线上地址（用来判断 40001 该配 0 还是 1）：")
                L.append("  " + "，".join(f"{a}×{n}" for a, n in addrs))
        L += ["", "PLC 写入记录（最近 20 条）："]
        for idx, val, ts in self.regs.write_log[-20:]:
            L.append(f"  {ts}  40001+{idx} ({mb.REG_NAMES.get(idx,'?')}) = {val}")
        L += ["", "通讯日志（最近 40 条）："]
        L += ["  " + x for x in self.modbus_log[-40:]]
        self.show_report("Modbus / RS-485 监视", "\n".join(L))

    # ------------------------------------------------------------------ 配置
    def _apply_app_config(self):
        """应用与具体相机无关的全局配置。"""
        ms = int(self.store.app.get("display_interval_ms", DISPLAY_INTERVAL_MS))
        self.display_interval_ms = max(10, ms)
        self.cmb_refresh.set(str(ms) if str(ms) in REFRESH_CHOICES else str(DISPLAY_INTERVAL_MS))

        if self.store.warnings:
            for w in self.store.warnings:
                print(f"[配置告警] {w}")

        if not self.has_config_file:
            state = "未找到配置文件，将使用默认参数"
        else:
            state = f"配置已读取（{self.store.segment_count()} 段），扫描到相机后解锁"
        self.lbl_status.config(text=state)

        # 485 参数
        mcfg = dict(self.store.app.get("modbus") or cfgmod.DEFAULT_MODBUS)
        self.ent_port.delete(0, "end")
        self.ent_port.insert(0, str(mcfg["port"]))
        self.ent_baud.delete(0, "end")
        self.ent_baud.insert(0, str(mcfg["baudrate"]))
        self.ent_slave.delete(0, "end")
        self.ent_slave.insert(0, str(mcfg["slave_addr"]))
        self.cmb_base.set(str(mcfg["addr_base"]))
        self.regs.addr_base = int(mcfg["addr_base"])
        self.modbus.timeout_s = float(mcfg.get("timeout_s", 3.0))

    def unlock_config(self, serials):
        """用扫描到的相机序列号解开配置，并把参数分发到各面板。

        这是"先拿相机 ID、再解密、再初始化"的核心环节。
        """
        self.store.reset_decoded()
        report = self.store.unlock(serials)
        self.unlock_report = report

        # 按配置里的 panel 字段稳定分配面板，避免发现顺序变化导致两路接反
        assigned = {}          # panel 索引 -> 序列号
        claimed = set()
        for serial, cfg in self.store.cameras.items():
            p = cfg.get("panel")
            if isinstance(p, int) and 0 <= p < len(self.panels) and p not in assigned:
                assigned[p] = serial
                claimed.add(serial)

        names = [str(c.serial_number) for c in self.cam_infos]
        for i, panel in enumerate(self.panels):
            serial = assigned.get(i)
            if serial is None:
                # 没有面板归属的相机（新相机）按序列号排序依次填充，
                # 保证同一批相机每次的分配结果一致
                free = [s for s in sorted(names) if s not in claimed]
                if not free:
                    continue
                serial = free[0]
                assigned[i] = serial
                claimed.add(serial)
            try:
                panel.combo.current(names.index(serial))
            except ValueError:
                continue
            panel.bind_camera_config(self.store.camera(serial))
            self.store.camera(serial)["panel"] = i

        return report

    def save_config(self, note=""):
        """把当前内存里的配置加密回写。"""
        ok = self.store.save()
        if not ok and self.store.warnings:
            self.lbl_status.config(text=f"配置保存失败：{self.store.warnings[-1]}")
            return False
        if note:
            self.lbl_status.config(text=note)
        return True

    def config_summary(self):
        """给状态栏用的一句话配置概览。"""
        if not self.has_config_file:
            return "无配置文件"
        r = self.unlock_report
        if not r:
            return f"配置 {self.store.segment_count()} 段未解锁"
        parts = [f"解锁 {len(r['unlocked'])} 台"]
        if r["missing"]:
            parts.append(f"{len(r['missing'])} 台无配置")
        if r["rejected"]:
            parts.append(f"{len(r['rejected'])} 台未授权")
        if self.store.locked:
            parts.append(f"{self.store.locked} 段打不开")
        return "配置：" + "，".join(parts)

    # ------------------------------------------------------------------ 显示节拍
    def show_report(self, title, text):
        """弹一个可滚动、可复制的文本窗口（比 messagebox 适合放长报告）。"""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("980x620")
        holder = ttk.Frame(win)
        holder.pack(fill="both", expand=True, padx=6, pady=6)
        txt = tk.Text(holder, wrap="none", font=("Consolas", 10))
        sb = ttk.Scrollbar(holder, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=(0, 8))
        return win

    def _tick(self):
        """两路画面共用的唯一刷新定时器。

        每 tick 最多给每个面板重绘一帧，队列里堆积的旧帧直接丢弃，
        因此刷新频率由这里决定，而不是由相机出帧速度决定。
        """
        for p in self.panels:
            p.pump_frame()
        self.modbus.poll()          # Modbus 请求也由这个节拍驱动
        self.root.after(self.display_interval_ms, self._tick)

    def set_refresh_interval(self, _event=None):
        try:
            ms = int(float(self.cmb_refresh.get()))
        except ValueError:
            return
        self.display_interval_ms = max(10, ms)
        self.store.app["display_interval_ms"] = self.display_interval_ms
        self.save_config(f"显示刷新间隔已设为 {self.display_interval_ms} ms")

    # ------------------------------------------------------------------ 界面
    def _build_ui(self):
        # 全局工具条
        bar = ttk.Frame(self.root)
        bar.grid(row=0, column=0, sticky="we", padx=5, pady=5)

        self.btn_scan = ttk.Button(bar, text="扫描", command=self.scan_cameras)
        self.btn_scan.pack(side="left")

        self.btn_start_all = ttk.Button(bar, text="全部开始", command=self.start_all, state="disabled")
        self.btn_start_all.pack(side="left", padx=(16, 2))
        self.btn_stop_all = ttk.Button(bar, text="全部停止", command=self.stop_all, state="disabled")
        self.btn_stop_all.pack(side="left", padx=2)
        self.btn_grab_all = ttk.Button(bar, text="全部获取一帧", command=self.grab_all, state="disabled")
        self.btn_grab_all.pack(side="left", padx=2)

        ttk.Label(bar, text="刷新(ms):").pack(side="left", padx=(16, 2))
        self.cmb_refresh = ttk.Combobox(bar, state="readonly", width=5, values=REFRESH_CHOICES)
        self.cmb_refresh.pack(side="left")
        self.cmb_refresh.set(str(DISPLAY_INTERVAL_MS))
        self.cmb_refresh.bind("<<ComboboxSelected>>", self.set_refresh_interval)

        self.lbl_config = tk.Label(self.root, text="", anchor="w", fg="#0a6")
        self.lbl_config.grid(row=1, column=0, sticky="we", padx=5)

        # 485 / Modbus
        bus = ttk.Frame(self.root)
        bus.grid(row=2, column=0, sticky="we", padx=5)

        ttk.Label(bus, text="485 端口:").pack(side="left")
        self.ent_port = ttk.Entry(bus, width=16)
        self.ent_port.pack(side="left", padx=(2, 8))
        ttk.Label(bus, text="波特率:").pack(side="left")
        self.ent_baud = ttk.Entry(bus, width=7)
        self.ent_baud.pack(side="left", padx=(2, 8))
        ttk.Label(bus, text="站号:").pack(side="left")
        self.ent_slave = ttk.Entry(bus, width=5)
        self.ent_slave.pack(side="left", padx=(2, 8))
        ttk.Label(bus, text="地址基址:").pack(side="left")
        self.cmb_base = ttk.Combobox(bus, state="readonly", width=3, values=("0", "1"))
        self.cmb_base.pack(side="left", padx=(2, 8))
        self.btn_modbus = ttk.Button(bus, text="连接485", command=self.toggle_modbus)
        self.btn_modbus.pack(side="left", padx=2)
        ttk.Button(bus, text="寄存器监视", command=self.show_registers).pack(side="left", padx=2)

        self.lbl_modbus = tk.Label(self.root, text="Modbus：未连接", anchor="w", fg="#888")
        self.lbl_modbus.grid(row=3, column=0, sticky="we", padx=5)

        self.lbl_status = tk.Label(self.root, text="请先点击【扫描】", anchor="w")
        self.lbl_status.grid(row=4, column=0, sticky="we", padx=5)

        # 两台相机的面板并排显示
        container = ttk.Frame(self.root)
        container.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        for col in range(NUM_CAMERAS):
            container.columnconfigure(col, weight=1)
        container.rowconfigure(0, weight=1)

        for i in range(NUM_CAMERAS):
            panel = CameraPanel(self, container, i)
            panel.frame.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            self.panels.append(panel)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(5, weight=1)
        self.root.bind("<Control-0>", lambda e: self.reset_all_zoom())

    # ------------------------------------------------------------------ 面板协作
    def other_panel(self, index):
        for p in self.panels:
            if p.index != index:
                return p
        return None

    def update_global_buttons(self):
        any_open = any(p.cam is not None for p in self.panels)
        any_idle = any(p.cam is not None and not p.is_running and not p.is_grabbing for p in self.panels)
        any_running = any(p.is_running for p in self.panels)

        self.btn_scan.config(state="disabled" if any_open else "normal")
        self.btn_start_all.config(state="normal" if any_idle else "disabled")
        self.btn_stop_all.config(state="normal" if any_running else "disabled")
        self.btn_grab_all.config(state="normal" if any_idle else "disabled")

    def reset_all_zoom(self):
        for p in self.panels:
            p.reset_zoom()

    # ------------------------------------------------------------------ 扫描
    def scan_cameras(self):
        self.btn_scan.config(state="disabled")
        self.lbl_status.config(text="扫描中，请稍候...")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            cam_infos = vcam.CameraUtils.DiscoverCameras()
            error = None
        except Exception as e:
            cam_infos = []
            error = str(e)
        self.root.after(0, lambda: self._scan_done(cam_infos, error))

    def _scan_done(self, cam_infos, error):
        if error:
            self.update_global_buttons()
            messagebox.showerror("扫描失败", error)
            self.lbl_status.config(text="扫描失败")
            return

        self.cam_infos = cam_infos
        if not cam_infos:
            for p in self.panels:
                p.combo["values"] = []
                p.combo.set("")
            self.lbl_config.config(text="")
            self.update_global_buttons()
            self.lbl_status.config(text="扫描完成：未发现设备")
            return

        names = [str(c.serial_number) for c in cam_infos]
        for p in self.panels:
            p.combo["values"] = names
            p.combo.current(0)

        self.update_global_buttons()

        # 拿到相机 ID 了，现在才解锁配置并按序列号把参数分发到各面板
        self.lbl_status.config(text="正在解锁配置...")
        self.root.update_idletasks()
        report = self.unlock_config(names)

        self.lbl_config.config(text=self.config_summary())

        msgs = []
        if len(cam_infos) < NUM_CAMERAS:
            msgs.append(f"只发现 {len(cam_infos)} 个设备（需要 {NUM_CAMERAS} 台才能双路同时采集）")
        else:
            msgs.append(f"发现 {len(cam_infos)} 个设备，请分别点击两个面板的【打开】")
        if report["rejected"]:
            msgs.append("未授权的相机：" + "、".join(report["rejected"]))
        for serial, why in report["failed"]:
            msgs.append(f"{serial} 配置解不开：{why}")
        if self.store.locked:
            msgs.append(f"另有 {self.store.locked} 段配置不属于当前相机，已原样保留")
        if report["missing"]:
            msgs.append("无配置（将用默认参数）：" + "、".join(report["missing"]))
        self.lbl_status.config(text="扫描完成：" + "；".join(msgs))

    # ------------------------------------------------------------------ 批量操作
    def start_all(self):
        targets = [p for p in self.panels if p.cam is not None and not p.is_running and not p.is_grabbing]
        if not targets:
            return
        failed = [p.name for p in targets if not p.start_capture(quiet=True)]
        if failed:
            messagebox.showerror("开始失败", "以下相机启动采集失败：" + "、".join(failed))
            return
        self.lbl_status.config(text=f"已启动 {len(targets)} 台相机采集")

    def stop_all(self):
        stopped = [p.name for p in self.panels if p.stop_capture(quiet=True)]
        if stopped:
            self.lbl_status.config(text=f"已停止 {len(stopped)} 台相机采集")

    def grab_all(self):
        targets = [p for p in self.panels if p.cam is not None and not p.is_running and not p.is_grabbing]
        if not targets:
            return
        for p in targets:
            p.grab_one_frame(quiet=True)
        self.lbl_status.config(text=f"正在为 {len(targets)} 台相机获取图像...")

    # ------------------------------------------------------------------ 关闭
    def on_close(self):
        for p in self.panels:
            p.close_camera()
        if self.store.dirty or self.store.segment_count() or self.store.cameras:
            self.store.save()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthCamApp(root)
    root.mainloop()
