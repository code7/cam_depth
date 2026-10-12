import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import vcamera as vcam


CANVAS_W = 800
CANVAS_H = 600
GRAB_TIMEOUT = 10.0
DEPTH_TO_MM = 0.125


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


class DepthCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("深度相机")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cam = None
        self.cam_infos = []
        self.frame_queue = queue.Queue(maxsize=1)
        self.is_running = False
        self.is_grabbing = False
        self._photo = None
        self._img_src = None
        self.depth_data = None
        self.depth_w = 0
        self.depth_h = 0
        self.scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.calibrated = False
        self.calib_mode = False
        self.show_height = tk.BooleanVar(value=False)
        self.roi = None
        self.drag_start = None
        self._sel_rect = None
        self.plane_a = 0.0
        self.plane_b = 0.0
        self.plane_c = 0.0

        self._build_ui()

        vcam.CameraUtils.Init(True)

    def _build_ui(self):
        # 第 0 行
        self.btn_scan = ttk.Button(self.root, text="扫描", command=self.scan_cameras)
        self.btn_scan.grid(row=0, column=0, padx=5, pady=5)

        self.combo = ttk.Combobox(self.root, state="readonly", width=28)
        self.combo.grid(row=0, column=1, padx=5, pady=5, sticky="we")

        self.btn_open = ttk.Button(self.root, text="打开", command=self.open_camera)
        self.btn_open.grid(row=0, column=2, padx=5, pady=5)

        self.btn_close = ttk.Button(self.root, text="关闭相机", command=self.close_camera, state="disabled")
        self.btn_close.grid(row=0, column=3, padx=5, pady=5)

        # 第 1 行
        self.btn_start = ttk.Button(self.root, text="开始", command=self.start_capture, state="disabled")
        self.btn_start.grid(row=1, column=0, padx=5, pady=5)

        self.btn_stop = ttk.Button(self.root, text="停止", command=self.stop_capture, state="disabled")
        self.btn_stop.grid(row=1, column=1, padx=5, pady=5)

        self.btn_grab = ttk.Button(self.root, text="获取一帧", command=self.grab_one_frame, state="disabled")
        self.btn_grab.grid(row=1, column=2, padx=5, pady=5)

        self.btn_calib = ttk.Button(self.root, text="标定平面", command=self.start_calibration, state="disabled")
        self.btn_calib.grid(row=1, column=3, padx=5, pady=5)

        self.chk_height = ttk.Checkbutton(self.root, text="显示高度图", variable=self.show_height)
        self.chk_height.grid(row=1, column=4, padx=5, pady=5)

        # 第 2 行：距离过滤
        tk.Label(self.root, text="最小距离(mm):").grid(row=2, column=0, sticky="e", padx=5)
        self.ent_min = ttk.Entry(self.root, width=8)
        self.ent_min.grid(row=2, column=1, sticky="w", padx=5)
        self.ent_min.insert(0, "500")

        tk.Label(self.root, text="最大距离(mm):").grid(row=2, column=2, sticky="e", padx=5)
        self.ent_max = ttk.Entry(self.root, width=8)
        self.ent_max.grid(row=2, column=3, sticky="w", padx=5)
        self.ent_max.insert(0, "2000")

        # 第 3 行
        self.lbl_status = tk.Label(self.root, text="请先扫描设备", anchor="w")
        self.lbl_status.grid(row=3, column=0, columnspan=5, sticky="we", padx=5)

        # 第 4 行
        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H, bg="black",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.grid(row=4, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel(e, -120))
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<Control-0>", lambda e: self.reset_zoom())
        self.root.bind("<Control-0>", lambda e: self.reset_zoom())

        # 第 5 行
        self.lbl_coord = tk.Label(self.root, text="坐标: (---, ---)   深度: ---", anchor="w")
        self.lbl_coord.grid(row=5, column=0, columnspan=5, sticky="we", padx=5, pady=(0, 5))

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(4, weight=1)

    # ------------------------------------------------------------------ 回调
    def frame_callback(self, frame_set):
        if not self.frame_queue.full():
            self.frame_queue.put_nowait(frame_set)

    def _clear_queue(self):
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

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
        self.btn_scan.config(state="normal")
        if error:
            messagebox.showerror("扫描失败", error)
            self.lbl_status.config(text="扫描失败")
            return

        self.cam_infos = cam_infos
        if not cam_infos:
            self.combo.set("")
            self.lbl_status.config(text="扫描完成：未发现设备")
            return

        names = [str(c.serial_number) for c in cam_infos]
        self.combo["values"] = names
        self.combo.current(0)
        self.lbl_status.config(text=f"扫描完成：发现 {len(cam_infos)} 个设备，请点击【打开】")

    # ------------------------------------------------------------------ 打开 / 关闭
    def open_camera(self):
        if not self.cam_infos:
            messagebox.showwarning("提示", "请先点击【扫描】")
            return

        idx = self.combo.current()
        if idx < 0:
            messagebox.showwarning("提示", "请选择设备")
            return

        info = self.cam_infos[idx]
        cam = vcam.CameraFactory.GetCameraBySerialNumber(info.serial_number)

        status = cam.Connect()
        if not status:
            messagebox.showerror("打开失败", f"连接相机失败: {status.message()}")
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
            messagebox.showerror("打开失败", f"{e}")
            return

        status = cam.SetSensorEnabled(vcam.SensorType.Depth, True)
        if not status:
            cam.Disconnect()
            messagebox.showerror("打开失败", "使能深度传感器失败")
            return

        cam.RegisterFrameSetCallback(self.frame_callback)

        self.cam = cam
        self._clear_queue()
        self.btn_scan.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.btn_close.config(state="normal")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_grab.config(state="normal")
        self.btn_calib.config(state="normal", text="标定平面", command=self.start_calibration)
        self.lbl_status.config(text="设备已打开，请点击【开始】或【获取一帧】")

    def close_camera(self):
        if self.cam is None:
            return

        self.is_running = False
        self.is_grabbing = False
        cam = self.cam
        self.cam = None
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
        self._img_src = None
        self.calibrated = False
        self.calib_mode = False
        self.show_height.set(False)

        self.btn_scan.config(state="normal")
        self.btn_open.config(state="normal")
        self.btn_close.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.btn_grab.config(state="disabled")
        self.btn_calib.config(state="disabled", text="标定平面", command=self.start_calibration)
        self.canvas.delete("all")
        self.lbl_status.config(text="相机已关闭")
        self.lbl_coord.config(text="坐标: (---, ---)   深度: ---")

    # ------------------------------------------------------------------ 开始 / 停止
    def start_capture(self):
        if self.cam is None:
            messagebox.showwarning("提示", "请先打开相机")
            return
        if self.is_running or self.is_grabbing:
            return

        self._clear_queue()
        if not self.cam.StartCapture():
            messagebox.showerror("开始失败", "启动采集失败")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_grab.config(state="disabled")
        self.lbl_status.config(text="采集中...")
        self.root.after(30, self.update_display)

    def stop_capture(self):
        if not self.is_running:
            return

        self.is_running = False
        try:
            self.cam.StopCapture()
        except Exception as e:
            self.lbl_status.config(text=f"停止异常: {e}")

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_grab.config(state="normal")
        self.lbl_status.config(text="已停止采集")

    # ------------------------------------------------------------------ 获取一帧
    def grab_one_frame(self):
        if self.cam is None:
            messagebox.showwarning("提示", "请先打开相机")
            return
        if self.is_running:
            messagebox.showwarning("提示", "正在采集中，请先点击【停止】")
            return
        if self.is_grabbing:
            return

        self.is_grabbing = True
        self.btn_grab.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.lbl_status.config(text="正在获取一帧...")
        self.root.update_idletasks()
        threading.Thread(target=self._grab_worker, daemon=True).start()

    def _grab_worker(self):
        error = None
        elapsed = None
        fs = None
        try:
            self._clear_queue()
            # t0 = time.perf_counter()
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
            messagebox.showerror("获取失败", error)
            self.lbl_status.config(text="获取一帧失败")
            return
        if fs is None:
            messagebox.showwarning("提示", f"超时（{GRAB_TIMEOUT:.0f}s）未获取到帧")
            self.lbl_status.config(text="获取一帧超时")
            return
        if not self._render_frame(fs):
            messagebox.showwarning("提示", "帧无效")

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
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        self._img_src = Image.fromarray(rgb)

        self._redraw_image()
        return True

    def _get_dist_range(self):
        try:
            mn = float(self.ent_min.get())
        except ValueError:
            return None, None
        try:
            mx = float(self.ent_max.get())
        except ValueError:
            return None, None
        return (mn, mx) if mx > mn else (None, None)

    def _make_vis(self, vis_data):
        if self.show_height.get() and self.calibrated:
            return self._make_vis_auto(vis_data)
        mn, mx = self._get_dist_range()
        if mn is not None:
            vis = (vis_data - mn) * (255.0 / (mx - mn))
            np.clip(vis, 0.0, 255.0, out=vis)
            np.nan_to_num(vis, copy=False, nan=0.0)
            return vis.astype(np.uint8)
        return self._make_vis_auto(vis_data)

    def _make_vis_auto(self, vis_data):
        return cv2.normalize(vis_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def _redraw_image(self):
        """Draw the current image onto the canvas with current zoom, clamped to keep it on-screen."""
        if self._img_src is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # Ensure we have actual dimensions
        if cw <= 1 or ch <= 1:
            cw, ch = CANVAS_W, CANVAS_H

        iw, ih = self._img_src.size
        fit = min(cw / iw, ch / ih)
        scale = fit * self.zoom
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        resized = self._img_src.resize((new_w, new_h), Image.BILINEAR)

        self.scale = scale
        self.offset_x, self.offset_y = self._clamp_offset(new_w, new_h, cw, ch)

        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("img")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw",
                                 image=self._photo, tags="img")

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
        if self._img_src is None:
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
        iw, ih = self._img_src.size

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

    def update_display(self):
        if not self.is_running:
            return

        try:
            fs = self.frame_queue.get_nowait()
            self._render_frame(fs)
        except queue.Empty:
            pass
        except Exception as e:
            self.lbl_status.config(text=f"显示错误: {e}")

        self.root.after(30, self.update_display)

    def compute_height_map(self):
        ys, xs = np.indices((self.depth_h, self.depth_w))
        plane = self.plane_a * xs + self.plane_b * ys + self.plane_c
        height = self.depth_data - plane
        invalid = ~np.isfinite(height) | (self.depth_data <= 0)
        height[invalid] = 0
        return height

    # ------------------------------------------------------------------ 鼠标坐标
    def on_mouse_move(self, event):
        if not self.is_running or self.depth_data is None:
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
        self.lbl_status.config(text="采集运行中..." if self.is_running else "设备已打开")

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
            messagebox.showwarning("提示", "暂无图像，请先获取一帧或开始采集")
            return
        if self.roi is None:
            return
        x0, y0, x1, y1 = self.roi
        if x1 - x0 < 8 or y1 - y0 < 8:
            messagebox.showwarning("提示", "标定区域过小")
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
            messagebox.showerror("标定失败", error)
            self.lbl_status.config(text="标定失败")
            return
        if result is None:
            messagebox.showerror("标定失败", "区域内有效深度点不足或拟合失败，请重新框选")
            self.lbl_status.config(text="标定失败，请重新框选")
            return

        a, b, c, ratio = result
        self.plane_a, self.plane_b, self.plane_c = a, b, c
        self.calibrated = True
        self.lbl_status.config(
            text=f"标定完成：d = {a:.5f}·x + {b:.5f}·y + {c:.2f}，内点率 {ratio:.1%}。勾选【显示高度图】查看高度")

    # ------------------------------------------------------------------ 关闭
    def on_close(self):
        self.close_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthCamApp(root)
    root.mainloop()