# -*- coding: utf-8 -*-
"""相机服务：独占深度相机 SDK，负责采集、预览流、ROI/基准面准备、测量与校准。

对外只通过 MQTT：
- 订阅 camera/req，提供 measure / calibrate / auto_roi / probe 等动作；
- 发布 camera/{A,B}/frame（JPEG 预览）和 camera/{A,B}/status；
- 发布 camera/inventory（发现到的相机与 A/B 绑定）。

不碰 Tk，不含任何界面代码。加 --simulate 可在没有相机 SDK 的机器上跑通全链路。
"""

import argparse
import base64
import copy
import math
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cam_config as cfgmod
from common import measure, topics
from common.mqtt_node import MqttNode

try:
    import vcamera as vcam
except ImportError:
    vcam = None

NUM_CAMERAS = 2
LABELS = ("A", "B")
DEFAULT_INTERVAL_MS = 80
LOGIC_TAG = "relative-delta-v1"


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# 现场相机约 1 FPS，且启动/重新开始采集后数秒才出首帧。
# 阈值不能按高帧率相机设置，否则会在首帧到达前误判断流并反复重连。
STALL_WARN_S = _env_float("CAMERA_STALL_WARN_S", 10.0)
STALL_RECOVER_S = _env_float("CAMERA_STALL_RECOVER_S", 25.0)
RECOVER_MAX_ATTEMPTS = 6
OPEN_RETRY_MAX = 10
OPEN_RETRY_BASE_S = 1.5
OPEN_RETRY_MAX_S = 5.0
SCAN_RETRY_S = 5.0
CONFIG_RETRY_S = 5.0
STATUS_PERIOD_S = 0.5
PREVIEW_MAX_W = 800
JPEG_QUALITY = 72


class SimCameraHandle:
    def StopCapture(self):
        return True

    def Disconnect(self):
        return True


class CameraWorker:
    def __init__(self, service, index):
        self.service = service
        self.index = index
        self.panel = LABELS[index]
        self.name = f"{self.panel} 面相机"

        self.cam = None
        self.serial = None
        self.cfg = None
        self.opening = False
        self.running = False
        self.want_close = False
        self.open_gen = 0
        self.last_error = None
        self.next_open_at = 0.0
        self.open_retry_count = 0

        self.lock = threading.RLock()
        self.frame_queue = queue.Queue(maxsize=1)
        self.depth = None
        self.depth_w = 0
        self.depth_h = 0
        self.depth_scale = cfgmod.DEFAULT_DEPTH_SCALE_MM
        self.seq = 0
        self.fx = None
        self.fy = None

        self.calibrated = False
        self.plane = (0.0, 0.0, 0.0)
        self.plane_center_mm = None
        self.roi = None
        self.roi_mode = "auto"
        self.auto_calibrate = True
        self.min_mm = cfgmod.DEFAULT_FILTER["min_mm"]
        self.max_mm = cfgmod.DEFAULT_FILTER["max_mm"]
        self.a_max_mm2 = cfgmod.DEFAULT_CUT["a_max_mm2"]

        self.frames_received = 0
        self.frames_dropped = 0
        self.last_arrival = None
        self.arrival_ms = None
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

        self._last_preview_seq = -1
        self._last_preview_wall = 0.0
        self._sim_next = 0.0
        self._sim_rng = np.random.default_rng(1000 + index)

    # -------------------------------------------------- 配置
    def bind_config(self, serial, cfg):
        cfg = copy.deepcopy(cfg or cfgmod.default_camera_config(serial))
        with self.lock:
            changed_serial = serial != self.serial
            self.serial = str(serial)
            if changed_serial:
                self.next_open_at = 0.0
                self.open_retry_count = 0
            self.cfg = cfg
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
                self.roi = measure.clamp_roi(raw_roi, (self.depth_h, self.depth_w))
            else:
                try:
                    self.roi = tuple(int(v) for v in raw_roi) if raw_roi else None
                except (TypeError, ValueError):
                    self.roi = None
            if cfgmod.is_calibrated(cfg):
                a, b, c = (float(v) for v in cal["coefficients"])
                self.plane = (a, b, c)
                center = cal.get("plane_center_mm")
                self.plane_center_mm = None if center is None else float(center)
                self.calibrated = True
            else:
                self.calibrated = False
                self.plane_center_mm = None
            self._prepare_gen += 1
            self._prepared = False
            self._prepare_pending = False
            self._prepare_attempts = 0
            self._prepare_retry_at = 0.0
            self.last_error = None if changed_serial else self.last_error

    def set_idle(self, serial=None):
        with self.lock:
            self.serial = serial
            self.next_open_at = 0.0
            self.open_retry_count = 0
            self.cfg = None
            self.depth = None
            self.depth_w = self.depth_h = 0
            self.calibrated = False
            self.plane_center_mm = None
            self.roi = None
            self._prepared = False
            self._prepare_pending = False
            self.last_error = None
            self._last_preview_seq = -1

    # -------------------------------------------------- 打开 / 关闭
    def open_async(self, reset_retry=False):
        with self.lock:
            if self.cam is not None or self.opening or not self.serial:
                return
            if reset_retry:
                self.open_retry_count = 0
            if vcam is None and not self.service.simulate:
                self.last_error = "未找到 vcamera 模块"
                self.service.node.log("err", f"{self.name} 无法打开：{self.last_error}")
                return
            self.want_close = False
            self.opening = True
            self.last_error = None
            self.next_open_at = 0.0
            self.open_gen += 1
            gen = self.open_gen
            serial = self.serial
            cfg = copy.deepcopy(self.cfg or {})
        threading.Thread(target=self._open_worker,
                         args=(gen, serial, cfg), daemon=True).start()

    def _open_worker(self, gen, serial, cfg):
        cam = None
        connected = False
        err = None
        warnings = []
        features = {}
        image_mode = None
        try:
            if self.service.simulate:
                time.sleep(0.2)
                cam = SimCameraHandle()
                features = {}
                image_mode = {"width": 640, "height": 480, "pixel_format": "sim"}
            else:
                info = self.service.info_of(serial)
                if info is None:
                    raise RuntimeError("扫描结果中没有这台相机")
                cam = vcam.CameraFactory.GetCameraBySerialNumber(serial)
                status = cam.Connect()
                if not status:
                    raise RuntimeError(status.message())
                connected = True
                self._set_continuous_acquisition(cam)
                status = cam.SetSensorEnabled(vcam.SensorType.Depth, True)
                if not status:
                    raise RuntimeError("使能深度传感器失败")
                warnings = self._apply_acquisition_config(cam, cfg)
                features = self._read_acquisition_features(cam)
                image_mode = self._read_image_mode(cam)
                with self.lock:
                    self._clear_queue()
                    self._reset_frame_stats()
                self._register_callbacks(cam)
                if not cam.StartCapture():
                    raise RuntimeError("启动采集失败")
        except Exception as e:
            err = str(e)
            if cam is not None:
                try:
                    if connected:
                        cam.StopCapture()
                except Exception:
                    pass
                try:
                    cam.Disconnect()
                except Exception:
                    pass
                cam = None
        self._open_done(gen, cam, err, warnings, features, image_mode)

    def _open_done(self, gen, cam, err, warnings, features, image_mode):
        with self.lock:
            stale = gen != self.open_gen or self.service.stopping
            want_close = self.want_close
            self.opening = False
            if stale or want_close:
                if cam is not None:
                    try:
                        cam.StopCapture()
                        cam.Disconnect()
                    except Exception:
                        pass
                return
            if err or cam is None:
                self.last_error = err or "打开失败"
                if self.open_retry_count < OPEN_RETRY_MAX:
                    self.open_retry_count += 1
                    delay = min(OPEN_RETRY_MAX_S,
                                OPEN_RETRY_BASE_S * self.open_retry_count)
                    self.next_open_at = time.monotonic() + delay
                    hint = ("；设备可能仍被上次进程或其他程序占用"
                            if "busy" in self.last_error.lower() else "")
                    self.service.node.log(
                        "warn", f"{self.name} 打开失败：{self.last_error}{hint}；"
                                f"{delay:.1f} 秒后自动重试"
                                f"（{self.open_retry_count}/{OPEN_RETRY_MAX}）")
                else:
                    self.service.node.log(
                        "err", f"{self.name} 打开失败：{self.last_error}；"
                               f"已重试 {OPEN_RETRY_MAX} 次，停止自动重试")
                return
            self.cam = cam
            self.running = True
            self.last_error = None
            self.next_open_at = 0.0
            self.open_retry_count = 0
            serial = self.serial
            cfg = self.cfg

        self.service.node.log("ok", f"{self.name} 已打开（{serial}），采集已启动")
        if warnings:
            self.service.node.log("warn", f"{self.name} 采集参数告警：" + "；".join(warnings))
        if cfg is not None and not self.service.simulate:
            info = self.service.info_of(serial) or {}
            patch = {
                "panel": self.index,
                "model": str(info.get("model", "") or ""),
                "firmware": str(info.get("firmware", "") or ""),
            }
            if features:
                patch["acquisition"] = {
                    "features": features,
                    "image_mode": image_mode,
                    "applied_at": cfgmod.now_str(),
                    "warnings": list(warnings or [])[:20],
                }
            self.service.persist_camera(serial, patch, save=False)
            self.service.config_save()

    def close(self):
        with self.lock:
            self.want_close = True
            self.open_gen += 1
            self.opening = False
            self.next_open_at = 0.0
            self.open_retry_count = 0
            self._recover_gen += 1
            self.recovering = False
            self._recover_result = None
            self.running = False
            cam, self.cam = self.cam, None
            self.last_error = None
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
        with self.lock:
            self.depth = None
            self.depth_w = self.depth_h = 0
            self.calibrated = False
            self.roi = None
            self._prepared = False
            self._prepare_pending = False
            self._prepare_attempts = 0
            self._prepare_retry_at = 0.0
            self._prepare_gen += 1
            self.stall_notified = False
            self._last_preview_seq = -1

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
        self.last_frame_wall = time.monotonic()
        self.stall_notified = False
        self.recovering = False
        self.recover_count = 0
        self._recover_result = None
        self._recover_gen += 1

    # -------------------------------------------------- 采集
    def frame_callback(self, frame_set):
        # 必须在 SDK 回调线程内立即取图并复制，不能把 FrameSet 留到
        # 主循环再取。FrameSet 持有 SDK 内部帧缓冲；跨线程长期持有会
        # 让低帧率相机的缓冲池耗尽，表现为只出首帧、之后回调停止。
        try:
            depth_image = frame_set.GetImage(vcam.SensorType.Depth)
            if not depth_image.IsValid():
                return
            depth = np.array(depth_image.to_numpy(), dtype=np.float32,
                             copy=True)
            depth *= self.depth_scale
            fx = fy = None
            if self.fx is None or self.fy is None:
                try:
                    K = np.asarray(depth_image.calib_info().intrinsic.matrix,
                                   dtype=np.float64)
                    fx, fy = float(K[0, 0]), float(K[1, 1])
                except Exception:
                    pass
        except Exception as e:
            with self.lock:
                self.last_error = f"取帧失败：{e}"
            return

        now = time.monotonic()
        prev = self.last_arrival
        self.last_arrival = now
        if prev is not None:
            self.arrival_ms = (now - prev) * 1000.0
        self.frames_received += 1
        self.last_frame_wall = now
        try:
            self.frame_queue.put_nowait((now, depth, fx, fy))
        except queue.Full:
            self.frames_dropped += 1

    def pump(self):
        if not self.running or self.cam is None:
            return
        if self.service.simulate:
            self._sim_tick()
            return
        try:
            arrival, depth, fx, fy = self.frame_queue.get_nowait()
        except queue.Empty:
            self._check_stall()
            return
        self._store_depth(depth, arrival, fx, fy)
        if self.frames_received == 1:
            self.service.node.log(
                "ok", f"{self.name} 已收到首帧 "
                      f"{depth.shape[1]}x{depth.shape[0]}")
        self.stall_notified = False
        self.recover_count = 0
        self._maybe_auto_prepare()

    def event_callback(self, code, param):
        if self.service.stopping:
            return
        name = str(code).split(".")[-1]
        level = "err" if name in ("Error", "Offlined") else "info"
        self.service.node.log(
            level, f"{self.name} 相机事件：{name} ({param})")

    def _set_continuous_acquisition(self, cam):
        acq_mode, status = cam.GetFeature("AcquisitionMode")
        if not status:
            raise RuntimeError(f"获取 AcquisitionMode 失败：{status.message()}")
        value = 2
        try:
            items, st = acq_mode.GetEnumItems()
            if st and items:
                options = {str(it.name).strip().lower(): int(it.value) for it in items}
                if "continuous" in options:
                    value = options["continuous"]
                else:
                    raise RuntimeError(
                        "AcquisitionMode 不支持 Continuous，可选："
                        + "、".join(f"{it.name}={it.value}" for it in items))
                self.service.node.log(
                    "info", f"{self.name} AcquisitionMode 选项："
                            + "、".join(f"{it.name}={it.value}" for it in items)
                            + f"，使用 Continuous={value}")
            else:
                self.service.node.log(
                    "warn", f"{self.name} 无法读取 AcquisitionMode 枚举，回退值 2")
        except RuntimeError:
            raise
        except Exception as e:
            self.service.node.log(
                "warn", f"{self.name} 读取 AcquisitionMode 枚举失败，回退值 2：{e}")
        status = acq_mode.SetValue(value)
        if not status:
            raise RuntimeError(f"设置连续采集失败：{status.message()}")

    def _register_callbacks(self, cam):
        cam.RegisterFrameSetCallback(self.frame_callback)
        register_event = getattr(cam, "RegisterCameraEventCallback", None)
        if register_event is not None:
            try:
                register_event(self.event_callback)
            except Exception as e:
                self.service.node.log("warn", f"{self.name} 注册相机事件回调失败：{e}")

    def _store_depth(self, depth, arrival, fx=None, fy=None):
        with self.lock:
            self.depth = depth
            self.depth_h, self.depth_w = depth.shape[:2]
            self.seq += 1
            if fx is not None:
                self.fx, self.fy = fx, fy
            self.last_frame_wall = arrival or time.monotonic()

    # -------------------------------------------------- 模拟相机
    def _sim_tick(self):
        now = time.monotonic()
        if now < self._sim_next:
            return
        self._sim_next = now + 0.1
        H, W = 480, 640
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rng = self._sim_rng
        h = np.zeros((H, W), np.float32)
        h += 6.0 * np.exp(-(((xx - 230) ** 2 + (yy - 190) ** 2) / (2 * 34.0 ** 2)))
        h += 3.5 * np.exp(-(((xx - 430) ** 2 + (yy - 300) ** 2) / (2 * 22.0 ** 2)))
        base = 1000.0 + 0.0008 * xx + 0.0004 * yy + (60.0 if self.index else 0.0)
        depth = base + h + rng.normal(0, 0.15, (H, W)).astype(np.float32)
        depth[rng.random((H, W)) < 0.02] = 0.0
        self._store_depth(depth, now, 600.0, 600.0)
        with self.lock:
            self.arrival_ms = 100.0
            self.frames_received += 1
        self.stall_notified = False
        self.recover_count = 0
        self._maybe_auto_prepare()

    # -------------------------------------------------- 预览
    def preview_due(self, now):
        interval = max(0.02, self.service.preview_interval)
        with self.lock:
            if not self.service.stream_enabled:
                return False
            if self.depth is None or self.seq == self._last_preview_seq:
                return False
            if now - self._last_preview_wall < interval:
                return False
            self._last_preview_wall = now
            self._last_preview_seq = self.seq
            depth = self.depth
            plane = self.plane
            calibrated = self.calibrated
            scale = self.depth_scale
            min_mm, max_mm = self.min_mm, self.max_mm
            show_height = self.service.show_height
            seq = self.seq
            serial = self.serial
            roi = list(self.roi) if self.roi else None
        if show_height and calibrated:
            xs = np.arange(depth.shape[1], dtype=np.float32)
            ys = np.arange(depth.shape[0], dtype=np.float32)
            vis = depth - (plane[0] * xs[None, :] + plane[1] * ys[:, None] + plane[2])
            vis[~np.isfinite(vis) | (depth <= 0)] = 0
            vis8 = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif max_mm > min_mm:
            vis8 = np.clip((depth - min_mm) * (255.0 / (max_mm - min_mm)), 0, 255)
            vis8 = np.nan_to_num(vis8, nan=0.0).astype(np.uint8)
        else:
            vis8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
        if colored.shape[1] > PREVIEW_MAX_W:
            ratio = PREVIEW_MAX_W / colored.shape[1]
            colored = cv2.resize(colored, (PREVIEW_MAX_W, max(1, int(colored.shape[0] * ratio))),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return False
        payload = {
            "panel": self.panel,
            "serial": serial,
            "seq": seq,
            "ts": time.time(),
            "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
            "preview_w": int(colored.shape[1]),
            "preview_h": int(colored.shape[0]),
            "depth_w": int(depth.shape[1]),
            "depth_h": int(depth.shape[0]),
            "depth_scale": scale,
            "show_height": bool(show_height and calibrated),
            "min_mm": min_mm,
            "max_mm": max_mm,
            "roi": roi,
            "plane": list(plane) if calibrated else None,
        }
        self.service.node.publish_json(topics.camera_frame(self.panel), payload, qos=0)
        return True

    # -------------------------------------------------- 自动准备
    def _maybe_auto_prepare(self):
        with self.lock:
            if (self._prepared or self._prepare_pending or self.depth is None
                    or not self.serial):
                return
            if time.monotonic() < self._prepare_retry_at:
                return
            self._prepare_pending = True
            depth = np.array(self.depth, copy=True)
            plane = tuple(self.plane) if self.calibrated else None
            roi = self.roi
            mode = self.roi_mode
            auto_cal = self.auto_calibrate
            gen = self._prepare_gen
        threading.Thread(target=self._prepare_worker,
                         args=(gen, depth, plane, roi, mode, auto_cal),
                         daemon=True).start()

    def _prepare_worker(self, gen, depth, plane, roi, mode, auto_cal):
        err = None
        new_roi = roi
        new_plane = None
        new_center = None
        try:
            if mode == "auto" or not new_roi:
                new_roi = measure.auto_roi_bbox(depth, plane) or measure.clamp_roi(roi, depth.shape)
            else:
                new_roi = measure.clamp_roi(new_roi, depth.shape)
            if new_roi is None:
                raise RuntimeError("自动确定计算范围失败")
            if auto_cal or plane is None:
                if not auto_cal and plane is None:
                    new_plane = None
                else:
                    fit = measure.fit_plane_ransac(depth, new_roi, step=2, max_iter=400)
                    if fit is None:
                        raise RuntimeError("基准面拟合失败")
                    new_plane = (float(fit[0]), float(fit[1]), float(fit[2]))
                    new_center = measure.plane_center(new_plane, depth.shape)
        except Exception as e:
            err = str(e)
        with self.lock:
            if gen != self._prepare_gen:
                return
            self._prepare_pending = False
            if err:
                self._prepare_attempts += 1
                self.last_error = f"自动准备失败：{err}"
                if self._prepare_attempts < 3:
                    self._prepare_retry_at = time.monotonic() + 3.0
                else:
                    self._prepared = True
                serial = self.serial
            else:
                self._prepared = True
                self._prepare_retry_at = 0.0
                self.roi = new_roi
                self.last_error = None
                if new_plane is not None:
                    self.plane = new_plane
                    self.plane_center_mm = new_center
                    self.calibrated = True
                serial = self.serial
                if self.cfg is not None:
                    cut = dict(self.cfg.get("cut") or cfgmod.DEFAULT_CUT)
                    cut["roi_mode"] = self.roi_mode
                    cut["roi"] = list(new_roi) if new_roi else None
                    self.cfg["cut"] = cut
                    if new_plane is not None:
                        self.cfg["calibration"] = {
                            "type": "plane_xy",
                            "coefficients": list(new_plane),
                            "plane_center_mm": new_center,
                            "unit": "mm",
                            "roi": list(new_roi) if new_roi else None,
                            "inlier_ratio": None,
                            "calibrated_at": cfgmod.now_str(),
                        }
        if err:
            self.service.node.log("err", f"{self.name} 自动准备失败：{err}")
            return
        self.service.node.log(
            "ok", f"{self.name} 已自动确定计算范围 {new_roi}"
                  + (f"，基准面 z={new_plane[0]:.5f}x+{new_plane[1]:.5f}y+{new_plane[2]:.2f}"
                     if new_plane is not None else "，保留原基准面"))
        patch = {"cut": {"roi_mode": self.roi_mode, "roi": list(new_roi)}}
        if new_plane is not None:
            patch["calibration"] = {
                "type": "plane_xy",
                "coefficients": list(new_plane),
                "plane_center_mm": new_center,
                "unit": "mm",
                "roi": list(new_roi),
                "inlier_ratio": None,
                "calibrated_at": cfgmod.now_str(),
            }
        self.service.persist_camera(serial, patch, save=True)

    # -------------------------------------------------- 断流恢复
    def _check_stall(self):
        if self._recover_result is not None:
            self._finish_recover()
            return
        if self.recovering or self.service.simulate:
            return
        ref = self.last_frame_wall
        if ref is None:
            return
        gap = time.monotonic() - ref
        if gap >= STALL_RECOVER_S:
            if self.recover_count < RECOVER_MAX_ATTEMPTS:
                self._start_recover()
            elif not self.stall_notified:
                self.stall_notified = True
                self.last_error = "相机断流且自动恢复多次失败，请检查供电/网线"
                self.service.node.log("err", f"{self.name} {self.last_error}")
            return
        if self.stall_notified or gap < STALL_WARN_S:
            return
        self.stall_notified = True
        if self.depth is None:
            self.service.node.log(
                "warn", f"{self.name} 已 {gap:.1f}s 未收到首帧，继续等待（低帧率相机）")
        else:
            self.service.node.log(
                "warn", f"{self.name} 已 {gap:.1f}s 没有新帧（可能断流）")

    def _start_recover(self):
        self.recovering = True
        self.recover_count += 1
        self._recover_gen += 1
        self._recover_result = None
        gen = self._recover_gen
        serial = self.serial
        old = self.cam
        threading.Thread(target=self._recover_worker,
                         args=(gen, serial, old), daemon=True).start()

    def _recover_worker(self, gen, serial, old):
        err = None
        newcam = None
        cam = None
        try:
            try:
                old.StopCapture()
                old.Disconnect()
            except Exception:
                pass
            time.sleep(0.5)
            info = self.service.info_of(serial)
            if info is None:
                raise RuntimeError("扫描结果中没有这台相机")
            cam = vcam.CameraFactory.GetCameraBySerialNumber(serial)
            st = cam.Connect()
            if not st:
                raise RuntimeError(f"重连失败：{st.message()}")
            self._set_continuous_acquisition(cam)
            cam.SetSensorEnabled(vcam.SensorType.Depth, True)
            with self.lock:
                self._clear_queue()
                self.frames_received = 0
                self.frames_dropped = 0
                self.last_arrival = None
                self.arrival_ms = None
                self.last_frame_wall = time.monotonic()
                self.stall_notified = False
            self._register_callbacks(cam)
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
            self.last_error = f"自动恢复失败：{err}"
            self.service.node.log("err", f"{self.name} {self.last_error}")
            return
        self.cam = newcam
        self.service.node.log(
            "ok", f"{self.name} 已自动重连（第 {self.recover_count} 次），等待新帧…")

    # -------------------------------------------------- 数据访问
    def snapshot_measure(self):
        with self.lock:
            if self.depth is None:
                raise RuntimeError(f"{self.panel} 面无图像")
            if not self.calibrated:
                raise RuntimeError(f"{self.panel} 面未标定基准面")
            if not self.running:
                raise RuntimeError(f"{self.panel} 面未在采集")
            return {
                "depth": np.array(self.depth, copy=True),
                "plane": tuple(self.plane),
                "plane_center_mm": self.plane_center_mm,
                "roi": list(self.roi) if self.roi else None,
                "fx": self.fx,
                "fy": self.fy,
                "a_max_mm2": self.a_max_mm2,
                "seq": self.seq,
            }

    def snapshot_calibration(self):
        with self.lock:
            if self.depth is None:
                raise RuntimeError(f"{self.panel} 面无图像")
            if not self.running:
                raise RuntimeError(f"{self.panel} 面未在采集")
            return {
                "depth": np.array(self.depth, copy=True),
                "plane": tuple(self.plane) if self.calibrated else None,
                "plane_center_mm": self.plane_center_mm,
                "roi": list(self.roi) if self.roi else None,
                "fx": self.fx,
                "fy": self.fy,
                "a_max_mm2": self.a_max_mm2,
                "seq": self.seq,
            }

    def current_seq(self):
        with self.lock:
            return self.seq

    def is_running(self):
        with self.lock:
            return bool(self.running and self.depth is not None)

    def apply_plane(self, coeffs, center_mm=None):
        with self.lock:
            self.plane = tuple(float(v) for v in coeffs)
            self.plane_center_mm = None if center_mm is None else float(center_mm)
            self.calibrated = True
            if self.cfg is not None:
                self.cfg["calibration"] = {
                    "type": "plane_xy",
                    "coefficients": list(self.plane),
                    "plane_center_mm": self.plane_center_mm,
                    "unit": "mm",
                    "roi": list(self.roi) if self.roi else None,
                    "inlier_ratio": None,
                    "calibrated_at": cfgmod.now_str(),
                }

    def probe(self, x, y):
        with self.lock:
            if self.depth is None or not (0 <= x < self.depth_w and 0 <= y < self.depth_h):
                return None
            value = float(self.depth[y, x])
            height = None
            if self.calibrated:
                height = value - (self.plane[0] * x + self.plane[1] * y + self.plane[2])
            return {"x": int(x), "y": int(y), "depth": value, "height": height,
                    "calibrated": self.calibrated}

    def status(self):
        with self.lock:
            ready = bool(
                self.serial and self.running and self.depth is not None
                and not self._prepare_pending and self.calibrated and self.roi)
            reason = None
            if not self.serial:
                state, kind = "未分配", "idle"
                reason = "未分配相机"
            elif self.recovering:
                state, kind = "自动恢复", "warn"
                reason = "自动恢复中"
            elif self.opening:
                state, kind = "正在连接", "warn"
                reason = "正在连接"
            elif not self.running:
                if self.last_error:
                    state, kind = "打开失败", "err"
                    reason = self.last_error
                else:
                    state, kind = "未连接", "idle"
                    reason = "相机未打开"
            elif self.depth is None:
                state, kind = "等待首帧", "warn"
                reason = "等待首帧"
            elif self.stall_notified:
                state, kind = "断流", "err"
                reason = self.last_error or "相机断流"
            elif self._prepare_pending:
                state, kind = "准备中", "warn"
                reason = "正在准备"
            elif not self.calibrated:
                state, kind = "未标定基准面", "warn"
                reason = "未标定基准面"
            elif not self.roi:
                state, kind = "计算范围未设置", "warn"
                reason = "计算范围未设置"
            else:
                state, kind = "就绪", "ok"
            fps = 1000.0 / self.arrival_ms if self.arrival_ms else None
            return {
                "panel": self.panel,
                "serial": self.serial,
                "state": state,
                "state_kind": kind,
                "ready": ready,
                "reason": reason,
                "opening": bool(self.opening),
                "running": bool(self.running),
                "recovering": bool(self.recovering),
                "prepared": bool(self._prepared),
                "prepare_pending": bool(self._prepare_pending),
                "calibrated": bool(self.calibrated),
                "plane": list(self.plane) if self.calibrated else None,
                "plane_center_mm": self.plane_center_mm,
                "roi": list(self.roi) if self.roi else None,
                "roi_mode": self.roi_mode,
                "a_max_mm2": self.a_max_mm2,
                "min_mm": self.min_mm,
                "max_mm": self.max_mm,
                "auto_calibrate": bool(self.auto_calibrate),
                "arrival_ms": self.arrival_ms,
                "fps": fps,
                "frames_received": self.frames_received,
                "frames_dropped": self.frames_dropped,
                "frame_seq": self.seq,
                "depth_w": self.depth_w,
                "depth_h": self.depth_h,
                "fx": self.fx,
                "fy": self.fy,
                "last_error": self.last_error,
                "simulate": bool(self.service.simulate),
                "ts": time.time(),
            }

    # -------------------------------------------------- SDK 采集参数
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
                py = measure.value_to_python(val)
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


class CameraService:
    def __init__(self, node, simulate=False):
        self.node = node
        self.simulate = simulate
        self.stopping = False
        self.workers = [CameraWorker(self, i) for i in range(NUM_CAMERAS)]
        self.infos = []
        self.serials = []
        self.configs = {}
        self.config_online = False
        self.app = cfgmod.default_app_config()
        self.unlocked = []
        self.persist_serials = set()
        self.preview_interval = DEFAULT_INTERVAL_MS / 1000.0
        self.stream_enabled = True
        self.show_height = False
        self.scanning = False
        self.scan_error = None
        self.rescan_at = 0.0
        self.config_retry_at = 0.0
        self._lock = threading.RLock()
        self._reload_lock = threading.Lock()
        self._last_inventory = None

        node.subscribe(topics.CAMERA_REQ, self._on_request)
        node.subscribe(topics.CONFIG_CHANGED, self._on_config_changed)
        if vcam is not None and not simulate:
            try:
                vcam.CameraUtils.Init(True)
            except Exception as e:
                node.log("err", f"相机 SDK 初始化失败：{e}")
        elif vcam is None and not simulate:
            node.log("err", "未找到 vcamera 模块，相机服务无法扫描相机")

    # -------------------------------------------------- 配置
    def config_request(self, action, data=None, timeout=5.0):
        return self.node.request_ok(topics.CONFIG_REQ, topics.CONFIG_RES,
                                    action, data, timeout)

    def _fetch_all(self):
        data = self.config_request("get_all", {}, timeout=5.0)
        self._apply_config_snapshot(data)
        return data

    def _apply_config_snapshot(self, data):
        with self._lock:
            self.config_online = True
            self.app = copy.deepcopy(data.get("app") or cfgmod.default_app_config())
            self.configs = copy.deepcopy(data.get("cameras") or {})
            report = data.get("report") or {}
            self.unlocked = list(report.get("unlocked") or [])
            self.persist_serials = set(self.unlocked) | set(
                report.get("missing") or [])
            self.preview_interval = max(0.01, int(self.app.get(
                "display_interval_ms", DEFAULT_INTERVAL_MS)) / 1000.0)

    def persist_camera(self, serial, patch, save=False):
        if self.simulate or not serial or not self.config_online:
            return None
        if serial not in self.persist_serials:
            return None
        try:
            return self.config_request("set_camera", {
                "serial": serial, "patch": patch, "save": save,
                "source": "camera", "notify": False}, timeout=8.0)
        except Exception as e:
            self.node.log("warn", f"写回配置失败（{serial}）：{e}")
            return None

    def config_save(self):
        if self.simulate or not self.config_online:
            return None
        try:
            return self.config_request("save", {"source": "camera"}, timeout=10.0)
        except Exception as e:
            self.node.log("warn", f"保存配置失败：{e}")
            return None

    def _unlock_configs(self):
        if self.simulate:
            with self._lock:
                self.config_online = True
            return
        if not self.serials:
            data = self._fetch_all()
            return data
        data = self.config_request("unlock", {"serials": self.serials}, timeout=8.0)
        self._apply_config_snapshot(data)
        self.node.log("info", f"已从配置服务载入 {len(self.configs)} 台相机配置")
        return data

    def _cfg_for(self, serial):
        cfg = self.configs.get(serial)
        if cfg is None:
            cfg = cfgmod.default_camera_config(serial)
        return copy.deepcopy(cfg)

    # -------------------------------------------------- 扫描
    def start_scan(self):
        if self.scanning or self.stopping:
            return
        self.scanning = True
        self.scan_error = None
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        err = None
        infos = []
        try:
            if self.simulate:
                infos = [
                    {"serial": "SIM0001", "model": "SIM-DEPTH", "firmware": "sim"},
                    {"serial": "SIM0002", "model": "SIM-DEPTH", "firmware": "sim"},
                ]
            elif vcam is None:
                raise RuntimeError("未找到 vcamera 模块")
            else:
                infos = []
                for info in vcam.CameraUtils.DiscoverCameras():
                    infos.append({
                        "serial": str(info.serial_number),
                        "model": str(getattr(info, "model", "") or ""),
                        "firmware": str(getattr(info, "firmware_version", "") or ""),
                    })
        except Exception as e:
            err = str(e)
        self._apply_scan(infos, err)

    def info_of(self, serial):
        for info in self.infos:
            if info["serial"] == serial:
                return info
        return None

    def _apply_scan(self, infos, err):
        self.scanning = False
        if self.stopping:
            return
        if err:
            self.scan_error = err
            self.node.log("err", f"扫描失败：{err}，{int(SCAN_RETRY_S)} 秒后重试")
            self.rescan_at = time.monotonic() + SCAN_RETRY_S
            self._publish_inventory()
            return
        if not infos:
            self.scan_error = "未发现相机"
            self.node.log("warn", f"未发现相机，{int(SCAN_RETRY_S)} 秒后重试")
            self.rescan_at = time.monotonic() + SCAN_RETRY_S
            self._publish_inventory()
            return

        self.infos = infos
        self.serials = [i["serial"] for i in infos]
        self.scan_error = None
        self.rescan_at = 0.0
        self.node.log("info", f"发现 {len(infos)} 台相机：" + "、".join(self.serials))
        try:
            self._unlock_configs()
        except Exception as e:
            self.config_online = False
            self.config_retry_at = time.monotonic() + CONFIG_RETRY_S
            self.node.log("warn", f"读取配置服务失败，先使用默认参数：{e}")
        self._assign_panels(rebind=True)
        self._publish_inventory()

    def _assign_panels(self, rebind=False):
        with self._lock:
            names = list(self.serials)
            assigned = {}
            claimed = set()
            for serial, cfg in self.configs.items():
                p = cfg.get("panel")
                if (isinstance(p, int) and 0 <= p < NUM_CAMERAS and p not in assigned
                        and serial in names):
                    assigned[p] = serial
                    claimed.add(serial)
            for i in range(NUM_CAMERAS):
                if i in assigned:
                    continue
                free = [s for s in sorted(names) if s not in claimed]
                if free:
                    assigned[i] = free[0]
                    claimed.add(free[0])
            for i, worker in enumerate(self.workers):
                serial = assigned.get(i)
                if worker.serial == serial and (serial is None or worker.running):
                    if rebind and serial:
                        worker.bind_config(serial, self._cfg_for(serial))
                    continue
                if worker.cam is not None or worker.opening:
                    worker.close()
                if serial:
                    worker.bind_config(serial, self._cfg_for(serial))
                    worker.open_async(reset_retry=True)
                    self.node.log("info", f"{LABELS[i]} 面窗口 → 相机 {serial}")
                else:
                    worker.set_idle()
                    self.node.log("warn", f"{LABELS[i]} 面窗口没有可用相机")
        self._persist_panel_binding()

    def _persist_panel_binding(self):
        if self.simulate or not self.config_online:
            return
        changed = False
        for i, worker in enumerate(self.workers):
            serial = worker.serial
            if not serial or serial not in self.persist_serials:
                continue
            cfg = self.configs.get(serial) or {}
            if cfg.get("panel") != i:
                cfg["panel"] = i
                self.configs[serial] = cfg
                self.persist_camera(serial, {"panel": i}, save=False)
                changed = True
        if changed:
            self.config_save()

    # -------------------------------------------------- 配置热更新
    def _on_config_changed(self, topic, obj):
        if not isinstance(obj, dict):
            return
        source = obj.get("source")
        if source in ("camera", "unlock"):
            return
        serials = [str(s) for s in (obj.get("serials") or [])]
        threading.Thread(target=self._reload_worker,
                         args=(serials,), daemon=True).start()

    def _reload_worker(self, serials):
        if not self._reload_lock.acquire(blocking=False):
            return
        serials = [str(s) for s in (serials or [])]
        try:
            data = self._fetch_all()
            new_cfgs = data.get("cameras") or {}
            panel_changed = False
            with self._lock:
                for serial in serials:
                    cfg = new_cfgs.get(serial)
                    if cfg is None:
                        continue
                    for worker in self.workers:
                        if worker.serial == serial:
                            worker.bind_config(serial, cfg)
                            if cfg.get("panel") not in (None, worker.index):
                                panel_changed = True
            if panel_changed:
                self._assign_panels()
            self.node.log("info", "配置已热更新")
        except Exception as e:
            self.node.log("warn", f"配置热更新失败：{e}")
        finally:
            self._reload_lock.release()

    # -------------------------------------------------- 主循环
    def pump(self):
        now = time.monotonic()
        for worker in self.workers:
            worker.pump()
            worker.preview_due(now)
            if (worker.serial and worker.cam is None and not worker.opening
                    and not worker.running and worker.next_open_at
                    and now >= worker.next_open_at):
                worker.open_async()
        if (not self.simulate and not self.config_online and self.serials
                and now >= self.config_retry_at):
            self.config_retry_at = now + CONFIG_RETRY_S
            try:
                self._unlock_configs()
                self._assign_panels(rebind=True)
                self._publish_inventory()
            except Exception as e:
                self.node.log("warn", f"配置服务仍不可用：{e}")
        if self.rescan_at and now >= self.rescan_at:
            self.rescan_at = 0.0
            self.start_scan()

    def _publish_inventory(self):
        panels = {w.panel: w.serial for w in self.workers}
        payload = {
            "ts": time.time(),
            "simulate": bool(self.simulate),
            "sdk": bool(vcam is not None or self.simulate),
            "error": self.scan_error,
            "cameras": list(self.infos),
            "panels": panels,
            "config_online": self.config_online,
        }
        self._last_inventory = payload
        self.node.publish_json(topics.CAMERA_INVENTORY, payload, qos=1, retain=True)

    def publish_status(self):
        for worker in self.workers:
            self.node.publish_json(topics.camera_status(worker.panel),
                                   worker.status(), qos=1, retain=True)

    # -------------------------------------------------- 请求
    def _on_request(self, topic, obj):
        if not isinstance(obj, dict):
            return
        rid = obj.get("id")
        if not obj.get("action"):
            self.node.reply(topics.CAMERA_RES, rid, False, error="缺少 action")
            return
        threading.Thread(target=self._handle, args=(rid, obj), daemon=True).start()

    def _handle(self, rid, req):
        action = req.get("action")
        try:
            if action == "ping":
                data = {"pong": time.time()}
            elif action == "state":
                data = {
                    "inventory": self._last_inventory or {},
                    "panels": {w.panel: w.status() for w in self.workers},
                    "app": copy.deepcopy(self.app),
                    "simulate": bool(self.simulate),
                }
            elif action == "inventory":
                data = self._last_inventory or {}
            elif action == "rescan":
                self.start_scan()
                data = {"started": True}
            elif action == "reload":
                self._reload_worker(self.serials)
                data = {"reloaded": True}
            elif action == "open":
                worker = self._worker(req)
                worker.open_async(reset_retry=True)
                data = {"panel": worker.panel}
            elif action == "close":
                worker = self._worker(req)
                worker.close()
                worker.set_idle(worker.serial)
                data = {"panel": worker.panel}
            elif action == "set_stream":
                self.stream_enabled = bool(req.get("enabled", True))
                for worker in self.workers:
                    worker._last_preview_seq = -1
                data = {"enabled": self.stream_enabled}
            elif action == "set_show_height":
                self.show_height = bool(req.get("enabled"))
                for worker in self.workers:
                    worker._last_preview_seq = -1
                data = {"enabled": self.show_height}
            elif action == "probe":
                worker = self._worker(req)
                data = {"probe": worker.probe(int(req.get("x", -1)),
                                              int(req.get("y", -1)))}
            elif action == "auto_roi":
                worker = self._worker(req)
                with worker.lock:
                    depth = None if worker.depth is None else np.array(worker.depth, copy=True)
                    plane = tuple(worker.plane) if worker.calibrated else None
                if depth is None:
                    raise RuntimeError("该面还没有图像")
                roi = measure.auto_roi_bbox(depth, plane)
                if roi is None:
                    raise RuntimeError("自动确定计算范围失败")
                data = {"roi": list(roi)}
            elif action == "fit_plane":
                worker = self._worker(req)
                with worker.lock:
                    depth = None if worker.depth is None else np.array(worker.depth, copy=True)
                if depth is None:
                    raise RuntimeError("该面还没有图像")
                roi = req.get("roi") or worker.roi
                roi = measure.clamp_roi(roi, depth.shape) or measure.auto_roi_bbox(depth)
                if roi is None:
                    raise RuntimeError("无法确定校准范围")
                fit = measure.fit_plane_ransac(depth, roi, step=2, max_iter=400)
                if fit is None:
                    raise RuntimeError("基准面拟合失败")
                data = {"plane": [float(fit[0]), float(fit[1]), float(fit[2])],
                        "plane_center_mm": measure.plane_center(
                            (fit[0], fit[1], fit[2]), depth.shape),
                        "inlier_ratio": float(fit[3]), "roi": list(roi)}
            elif action == "calibrate":
                data = self._do_calibrate(req)
            elif action == "measure":
                data = self._do_measure(req)
            else:
                raise ValueError(f"未知 action：{action}")
            self.node.reply(topics.CAMERA_RES, rid, True, data)
        except Exception as e:
            self.node.log("err", f"相机请求失败（{action}）：{e}")
            self.node.reply(topics.CAMERA_RES, rid, False, error=str(e))

    def _worker(self, req):
        panel = str(req.get("panel") or "").upper()
        if panel not in LABELS:
            raise ValueError(f"panel 必须是 A/B，收到 {panel!r}")
        return self.workers[LABELS.index(panel)]

    def _wait_fresh_frames(self, workers, delay_s, timeout_s, need_calibrated):
        delay = max(0.0, float(delay_s or 0.0))
        timeout = max(0.2, float(timeout_s or 3.0))
        t0 = time.monotonic()
        starts = {}
        for worker in workers:
            if not worker.serial:
                raise RuntimeError(f"{worker.panel} 面未分配相机")
            with worker.lock:
                if worker.depth is None:
                    raise RuntimeError(f"{worker.panel} 面无图像")
                if need_calibrated and not worker.calibrated:
                    raise RuntimeError(f"{worker.panel} 面未标定基准面")
                if not worker.running:
                    raise RuntimeError(f"{worker.panel} 面未在采集")
                interval = (worker.arrival_ms or 0.0) / 1000.0
                need = 1
                if delay > 0 and interval > 0:
                    need = max(1, int(math.ceil(delay / interval)))
                starts[worker.panel] = (worker.seq, worker.frames_received, need,
                                        worker.arrival_ms)

        deadline = t0 + delay + timeout
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(
                    f"等待相机新帧超时（延迟 {delay:.1f}s + 超时 {timeout:.1f}s）")
            if now - t0 >= delay:
                fresh = True
                for worker in workers:
                    start_seq, start_frames, need, _ = starts[worker.panel]
                    with worker.lock:
                        if (worker.seq <= start_seq
                                or worker.frames_received - start_frames < need
                                or worker.last_frame_wall is None
                                or worker.last_frame_wall < t0 + delay):
                            fresh = False
                            break
                if fresh:
                    received = {}
                    for worker in workers:
                        with worker.lock:
                            received[worker.panel] = (
                                worker.frames_received - starts[worker.panel][1])
                    return {
                        "wait_ms": (now - t0) * 1000.0,
                        "arrival_ms": {p: starts[p][3] for p in starts},
                        "need": {p: starts[p][2] for p in starts},
                        "received": received,
                    }
            for worker in workers:
                with worker.lock:
                    if not worker.running:
                        raise RuntimeError(f"{worker.panel} 面采集已停止")
            time.sleep(0.005)

    def _do_measure(self, req):
        timeout = float(req.get("timeout_s") or
                        (self.app.get("modbus") or {}).get("timeout_s", 3.0))
        delay = float(req.get("delay_s") or 0.0)
        workers = self.workers
        t0 = time.monotonic()
        wait_info = self._wait_fresh_frames(
            workers, delay, timeout, need_calibrated=True)

        t_snap = time.monotonic()
        snap = {w.panel: w.snapshot_measure() for w in workers}
        snap_ms = (time.monotonic() - t_snap) * 1000.0

        t_compute = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = {
                w.panel: pool.submit(
                    measure.solve_panel_detail, snap[w.panel], 500.0)
                for w in workers
            }
            details = {panel: fut.result() for panel, fut in futures.items()}
        compute_ms = (time.monotonic() - t_compute) * 1000.0
        da = details["A"]["relative_mm"]
        db = details["B"]["relative_mm"]
        for worker in workers:
            d = details[worker.panel]
            reference = d["saved_plane_center_mm"]
            if reference is None:
                reference = d["plane_center_mm"]
            self.node.log(
                "info", f"{worker.panel} 面：绝对 {d['absolute_mm']:.1f} mm，"
                        f"校准面 {reference:.1f} mm，"
                        f"输出相对 {d['relative_mm']:.1f} mm")
        cycle_ms = (time.monotonic() - t0) * 1000.0

        def _arr(panel):
            value = wait_info["arrival_ms"].get(panel)
            return "-" if value is None else f"{value:.0f}"

        self.node.log(
            "info",
            f"测量耗时 {cycle_ms:.0f} ms：等待取帧 {wait_info['wait_ms']:.0f} ms"
            f"（A 间隔 {_arr('A')} ms，需 {wait_info['need']['A']} 帧，"
            f"实收 {wait_info['received']['A']}；"
            f"B 间隔 {_arr('B')} ms，需 {wait_info['need']['B']} 帧，"
            f"实收 {wait_info['received']['B']}），"
            f"快照 {snap_ms:.0f} ms，计算 {compute_ms:.0f} ms")
        return {"A": float(da), "B": float(db), "cycle_ms": cycle_ms,
                "delay_s": delay, "wait_ms": wait_info["wait_ms"],
                "snap_ms": snap_ms, "compute_ms": compute_ms,
                "arrival_ms": wait_info["arrival_ms"],
                "frames_wait": wait_info["need"],
                "frames_received": wait_info["received"],
                "frame_ids": {w.panel: w.seq for w in workers}}

    def _do_calibrate(self, req):
        persist = bool(req.get("persist", False))
        timeout = float(req.get("timeout_s") or
                        (self.app.get("modbus") or {}).get("timeout_s", 3.0))
        delay = float(req.get("delay_s") or 0.0)
        self._wait_fresh_frames(self.workers, delay, timeout, need_calibrated=False)
        snap = {w.panel: w.snapshot_calibration() for w in self.workers}
        res = measure.calibrate_pair(snap["A"], snap["B"])
        out = {k: [float(v) for v in vals] for k, vals in res.items()}
        centers = {k: measure.plane_center(vals, snap[k]["depth"].shape)
                   for k, vals in res.items()}
        if persist:
            for worker in self.workers:
                center = centers[worker.panel]
                worker.apply_plane(out[worker.panel], center)
                self.persist_camera(worker.serial, {
                    "calibration": {
                        "type": "plane_xy",
                        "coefficients": out[worker.panel],
                        "plane_center_mm": center,
                        "unit": "mm",
                        "roi": list(worker.roi) if worker.roi else None,
                        "inlier_ratio": None,
                        "calibrated_at": cfgmod.now_str(),
                    }
                }, save=False)
            self.config_save()
            self.node.log("ok", "基准面已重新拟合并保存")
        return {"planes": out, "centers_mm": centers, "persisted": persist}

    def stop(self):
        self.stopping = True
        for worker in self.workers:
            worker.close()
            self.node.publish_json(
                topics.camera_status(worker.panel),
                {"panel": worker.panel, "online": False, "serial": worker.serial,
                 "state": "服务离线", "state_kind": "err", "ready": False,
                 "running": False, "ts": time.time()}, qos=1, retain=True)
        self.node.publish_json(topics.CAMERA_INVENTORY,
                               {"online": False, "error": "相机服务已离线",
                                "ts": time.time()}, qos=1, retain=True)


def main():
    parser = argparse.ArgumentParser(description="深度相机服务")
    parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--mqtt-port", type=int,
                        default=int(os.environ.get("MQTT_PORT", "1883")))
    parser.add_argument("--simulate", action="store_true",
                        help="无相机 SDK 时生成合成深度图，用于全链路联调")
    args = parser.parse_args()

    node = MqttNode("camera", args.mqtt_host, args.mqtt_port)
    node.start(wait=3.0)
    svc = CameraService(node, simulate=args.simulate)
    if args.simulate:
        node.log("warn", "模拟模式：使用合成深度图，不会写入配置文件")
    node.log("info", f"测量输出模式：相对校准面（{LOGIC_TAG}）")

    stop = threading.Event()

    def _sig(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)

    node.log("info", "相机服务启动，开始扫描相机")
    svc.start_scan()
    last_status = 0.0
    try:
        while not stop.is_set():
            svc.pump()
            now = time.monotonic()
            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                svc.publish_status()
            time.sleep(0.01)
    finally:
        svc.stop()
        node.stop()


if __name__ == "__main__":
    main()
