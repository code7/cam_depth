"""双深度相机采集界面。

同时支持两台 VCamera 深度相机：各自独立开关、独立采集、独立缩放、
独立平面标定与高度图，并额外提供“全部开始 / 全部停止 / 全部获取一帧”的批量操作。
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import vcamera as vcam


CANVAS_W = 600          # 单个相机画布的初始宽度
CANVAS_H = 450          # 单个相机画布的初始高度
GRAB_TIMEOUT = 10.0
DEPTH_TO_MM = 0.125
NUM_CAMERAS = 2         # 同时支持的深度相机数量

# 显示刷新：两路画面共用一个定时器，按下面这个间隔取最新帧并重绘。
# 间隔越小画面越流畅，但 Tk 重绘开销越大；深度相机本身出帧不快，80 ms（约 12 fps）足够。
DISPLAY_INTERVAL_MS = 80
REFRESH_CHOICES = ("30", "50", "80", "120", "200", "300")


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

        self._build_ui(parent)

    # ------------------------------------------------------------------ 界面
    def _build_ui(self, parent):
        self.frame = ttk.LabelFrame(parent, text=self.name)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)

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
        self.chk_height = ttk.Checkbutton(ctrl, text="显示高度图", variable=self.show_height)
        self.chk_height.pack(side="left", padx=6)

        # 画布
        self.canvas = tk.Canvas(self.frame, width=CANVAS_W, height=CANVAS_H, bg="black",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 4))
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
        self.lbl_coord.grid(row=3, column=0, sticky="we", padx=4)

        self.lbl_status = tk.Label(self.frame, text="未打开", anchor="w", fg="#555")
        self.lbl_status.grid(row=4, column=0, sticky="we", padx=4, pady=(0, 4))

    # ------------------------------------------------------------------ 帧回调
    def frame_callback(self, frame_set):
        if not self.frame_queue.full():
            self.frame_queue.put_nowait(frame_set)

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
            messagebox.showerror("打开失败", f"{self.name} 连接相机失败: {status.message()}")
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

        cam.RegisterFrameSetCallback(self.frame_callback)

        self.cam = cam
        self.serial = serial
        self._clear_queue()
        self.combo.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.btn_close.config(state="normal")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_grab.config(state="normal")
        self.btn_calib.config(state="normal", text="标定平面", command=self.start_calibration)
        self.lbl_status.config(text=f"已打开 {serial}，请点击【开始】或【获取一帧】")
        app.update_global_buttons()

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

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_grab.config(state="disabled")
        self.lbl_status.config(text="采集中...")
        self.app.update_global_buttons()
        # 显示由主窗口的统一节拍驱动（见 DepthCamApp._tick），这里不再各自起定时器
        return True

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
        self.lbl_status.config(text="已停止采集")
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
        try:
            self._clear_queue()
            if not self.cam.StartCapture():
                error = "启动采集失败"
            else:
                t0 = time.perf_counter()
                deadline = time.time() + GRAB_TIMEOUT
                while time.time() < deadline:
                    try:
                        fs = self.frame_queue.get(timeout=0.5)
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
        self.root.after(0, lambda: self._grab_done(fs, elapsed, error))

    def _grab_done(self, fs, elapsed, error):
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
        if not self._render_frame(fs):
            messagebox.showwarning("提示", f"{self.name}：帧无效")
            self.lbl_status.config(text="获取一帧失败：帧无效")
        else:
            self.lbl_status.config(text=f"已获取一帧（{elapsed:.0f} ms）")
        self.app.update_global_buttons()

    # ------------------------------------------------------------------ 显示
    def _render_frame(self, fs):
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

        self._redraw_image()
        return True

    def _make_vis(self, vis_data):
        if self.show_height.get() and self.calibrated:
            return self._make_vis_auto(vis_data)
        mn, mx = self.app.get_dist_range()
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
            fs = self.frame_queue.get_nowait()
        except queue.Empty:
            return
        try:
            self._render_frame(fs)
        except Exception as e:
            self.lbl_status.config(text=f"显示错误: {e}")

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
        self.lbl_status.config(
            text=f"标定完成：d = {a:.5f}·x + {b:.5f}·y + {c:.2f}，内点率 {ratio:.1%}。勾选【显示高度图】查看高度")


class DepthCamApp:
    """主窗口：两台深度相机面板 + 全局的距离过滤与批量操作。"""

    def __init__(self, root):
        self.root = root
        self.root.title(f"{NUM_CAMERAS} 路深度相机")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cam_infos = []
        self.panels = []
        self.display_interval_ms = DISPLAY_INTERVAL_MS

        self._build_ui()

        vcam.CameraUtils.Init(True)
        self.root.after(self.display_interval_ms, self._tick)

    # ------------------------------------------------------------------ 显示节拍
    def _tick(self):
        """两路画面共用的唯一刷新定时器。

        每 tick 最多给每个面板重绘一帧，队列里堆积的旧帧直接丢弃，
        因此刷新频率由这里决定，而不是由相机出帧速度决定。
        """
        for p in self.panels:
            p.pump_frame()
        self.root.after(self.display_interval_ms, self._tick)

    def set_refresh_interval(self, _event=None):
        try:
            ms = int(float(self.cmb_refresh.get()))
        except ValueError:
            return
        self.display_interval_ms = max(10, ms)
        self.lbl_status.config(text=f"显示刷新间隔已设为 {self.display_interval_ms} ms")

    # ------------------------------------------------------------------ 界面
    def _build_ui(self):
        # 全局工具条
        bar = ttk.Frame(self.root)
        bar.grid(row=0, column=0, sticky="we", padx=5, pady=5)

        self.btn_scan = ttk.Button(bar, text="扫描", command=self.scan_cameras)
        self.btn_scan.pack(side="left")

        ttk.Label(bar, text="最小距离(mm):").pack(side="left", padx=(12, 2))
        self.ent_min = ttk.Entry(bar, width=8)
        self.ent_min.pack(side="left")
        self.ent_min.insert(0, "500")

        ttk.Label(bar, text="最大距离(mm):").pack(side="left", padx=(12, 2))
        self.ent_max = ttk.Entry(bar, width=8)
        self.ent_max.pack(side="left")
        self.ent_max.insert(0, "2000")

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

        self.lbl_status = tk.Label(self.root, text="请先点击【扫描】", anchor="w")
        self.lbl_status.grid(row=1, column=0, sticky="we", padx=5)

        # 两台相机的面板并排显示
        container = ttk.Frame(self.root)
        container.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        for col in range(NUM_CAMERAS):
            container.columnconfigure(col, weight=1)
        container.rowconfigure(0, weight=1)

        for i in range(NUM_CAMERAS):
            panel = CameraPanel(self, container, i)
            panel.frame.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            self.panels.append(panel)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
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

    def get_dist_range(self):
        try:
            mn = float(self.ent_min.get())
            mx = float(self.ent_max.get())
        except ValueError:
            return None, None
        return (mn, mx) if mx > mn else (None, None)

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
            self.update_global_buttons()
            self.lbl_status.config(text="扫描完成：未发现设备")
            return

        names = [str(c.serial_number) for c in cam_infos]
        for i, p in enumerate(self.panels):
            p.combo["values"] = names
            # 默认把第 i 台设备分配给第 i 个面板，避免两台相机选到同一台
            p.combo.current(i if i < len(names) else 0)

        self.update_global_buttons()
        if len(cam_infos) < NUM_CAMERAS:
            self.lbl_status.config(
                text=f"扫描完成：只发现 {len(cam_infos)} 个设备，需要 {NUM_CAMERAS} 台才能双路同时采集")
        else:
            self.lbl_status.config(
                text=f"扫描完成：发现 {len(cam_infos)} 个设备，请分别点击两个面板的【打开】")

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
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthCamApp(root)
    root.mainloop()
