"""把 Modbus 寄存器表和两台相机的测量流程接起来。

时序（按现场确认的握手方式）
----------------------------
    PLC 写 40002 = 1
        → 我们**自动清 40002 = 0**（PLC 一看回 0 就知道被接受了）
        → 40001 = 1（测量中）
        → 等两台相机各来一帧**触发之后**的新帧
        → 算下刀位置
        → 写 40003 / 40004，40001 = 2（完成）
    出错 → 40001 = 3（故障），下次触发时自动清除

    PLC 写 40005 = 1
        → 自动清 40005 = 0
        → 用触发之后的新帧**自动拟合基准面**（不需要在界面上框选）
        → 成功 40001 = 2，失败 40001 = 3

线程边界
--------
串口收发在 Modbus 自己的线程里；测量计算放在工作线程；**只有主线程碰 Tk**。
所以 poll() 必须由主线程的显示节拍调用，计算结果通过 schedule() 回到主线程。
本模块不 import tkinter，全部外部依赖靠注入，因此可以脱机测试。
"""

import queue
import threading
import time

import numpy as np

import cut_plane
import modbus_rtu as mb


class ModbusService:
    """寄存器表 <-> 测量流程的桥接。

    注入的依赖（便于脱机测试）：
        get_panel_data()  返回 {"A": {...}, "B": {...}}，每项含
                          depth / plane / calibrated / running / frame_id
        compute_fn(A, B)  算下刀位置，返回 (dist_a_mm, dist_b_mm)
        calibrate_fn(A,B) 自动拟合基准面，返回 {"A": (a,b,c), "B": (a,b,c)}
        log(msg)          日志

    工作线程算完后**不直接碰 Tk**，而是把结果塞进队列，由主线程的 poll() 取走。
    （Tk 的 after() 从别的线程调用并不严格安全，用队列可以彻底绕开。）
    """

    ST_IDLE = "idle"
    ST_WAIT_FRAMES = "wait_frames"
    ST_BUSY = "busy"

    def __init__(self, registers, get_panel_data,
                 compute_fn=None, calibrate_fn=None,
                 log=None, timeout_s=3.0, a_max_mm2=500.0):
        self.regs = registers
        self.get_panel_data = get_panel_data
        self.compute_fn = compute_fn or self._default_compute
        self.calibrate_fn = calibrate_fn or self._default_calibrate
        self.log = log or (lambda msg: None)
        self.timeout_s = timeout_s
        self.a_max_mm2 = a_max_mm2

        self.slave = None
        self.on_calibrated = None     # 界面注入：校准成功后写回配置
        self._results = queue.Queue()
        self._state = self.ST_IDLE
        self._pending = None          # "measure" / "calibrate"
        self._deadline = 0.0
        self._seen = {}               # key -> 触发时的帧序号
        self._t0 = 0.0
        self.last_result = None
        self.last_error = None
        self.cycle_ms = None

    # -------------------------------------------------- 串口
    def start(self, port, baudrate=9600, slave_addr=1, addr_base=0):
        self.regs.addr_base = addr_base
        self.slave = mb.ModbusRTUSlave(self.regs, slave_addr=slave_addr,
                                       port=port, baudrate=baudrate,
                                       log=self.log)
        return self.slave.start()

    def stop(self):
        if self.slave is not None:
            self.slave.stop()
            self.slave = None
        self._state = self.ST_IDLE
        self._pending = None

    @property
    def running(self):
        return self.slave is not None and self.slave._running

    # -------------------------------------------------- 主节拍
    def poll(self):
        """由主线程的显示节拍调用。"""
        self._drain_results()
        if self._state == self.ST_IDLE:
            self._check_requests()
        elif self._state == self.ST_WAIT_FRAMES:
            self._check_frames()

    def _drain_results(self):
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                return
            kind = item[0]
            if kind == "measure":
                self._finish_measure(item[1], item[2], item[3])
            elif kind == "calibrate":
                self._finish_calibrate(item[1], item[2])
            elif kind == "error":
                self._finish_error(item[1])

    def _check_requests(self):
        # 触发测量
        if self.regs.get(mb.REG_TRIGGER) == 1:
            self.regs.set(mb.REG_TRIGGER, 0)          # 自动清零
            self._begin("measure")
            return
        # 距离校准
        if self.regs.get(mb.REG_CALIBRATE) == 1:
            self.regs.set(mb.REG_CALIBRATE, 0)
            self._begin("calibrate")

    def _begin(self, what):
        data = self.get_panel_data()
        bad = []
        for key in ("A", "B"):
            p = data.get(key) or {}
            if p.get("depth") is None:
                bad.append(f"{key}面无图像")
            elif not p.get("calibrated"):
                bad.append(f"{key}面未标定基准面")
            elif not p.get("running"):
                bad.append(f"{key}面未在采集")
        if bad:
            self._fault("；".join(bad))
            return

        self._pending = what
        self._seen = {k: int((data.get(k) or {}).get("frame_id", 0))
                      for k in ("A", "B")}
        self._deadline = time.monotonic() + self.timeout_s
        self._t0 = time.monotonic()
        self._state = self.ST_WAIT_FRAMES
        self.regs.set(mb.REG_STATUS, mb.ST_MEASURING)
        self.log(f"[Modbus] 收到{'测量' if what == 'measure' else '校准'}请求，"
                 f"等待相机新帧…")

    def _check_frames(self):
        if time.monotonic() > self._deadline:
            self._fault(f"等待相机新帧超时（{self.timeout_s:.1f}s）")
            return
        data = self.get_panel_data()
        for key in ("A", "B"):
            p = data.get(key) or {}
            if p.get("depth") is None:
                return
            if int(p.get("frame_id", 0)) <= self._seen.get(key, 0):
                return                                   # 还没来新帧

        # 拿到两台的新帧，快照后交给工作线程算
        snap = {}
        for key in ("A", "B"):
            p = data[key]
            snap[key] = {"depth": np.array(p["depth"]),
                         "plane": tuple(p["plane"]) if p.get("plane") else None,
                         "fx": p.get("fx"), "fy": p.get("fy")}
        self._state = self.ST_BUSY
        what = self._pending
        self.log(f"[Modbus] 已取得新帧，开始{'算下刀位置' if what == 'measure' else '拟合基准面'}…")
        threading.Thread(target=self._worker, args=(what, snap), daemon=True).start()

    def _worker(self, what, snap):
        try:
            if what == "measure":
                da, db = self.compute_fn(snap["A"], snap["B"], self.a_max_mm2)
                self._results.put(("measure", da, db, None))
            else:
                res = self.calibrate_fn(snap["A"], snap["B"])
                self._results.put(("calibrate", res, None))
        except Exception as e:
            self._results.put(("error", f"{what} 失败：{e}"))

    # -------------------------------------------------- 结果
    def _finish_measure(self, da, db, err):
        self.cycle_ms = (time.monotonic() - self._t0) * 1000.0
        if err:
            self._fault(err)
            return
        # 先校验量程：超了宁可报故障，也不能把截断后的错误位置发给机床
        for name, v in (("A面", da), ("B面", db)):
            if not (mb.DIST_MIN_MM <= v <= mb.DIST_MAX_MM):
                self._fault(f"{name}下刀位置 {v:.1f} mm 超出寄存器量程 "
                            f"({mb.DIST_MIN_MM:.1f}~{mb.DIST_MAX_MM:.1f} mm)")
                return
        self.regs.set(mb.REG_DIST_A, mb.encode_distance(da))
        self.regs.set(mb.REG_DIST_B, mb.encode_distance(db))
        self.regs.set(mb.REG_STATUS, mb.ST_DONE)
        self.last_result = (da, db)
        self.last_error = None
        self._state = self.ST_IDLE
        self._pending = None
        self.log(f"[Modbus] 测量完成：A面下刀 {da:.1f} mm，B面下刀 {db:.1f} mm，"
                 f"周期 {self.cycle_ms:.0f} ms")

    def _finish_calibrate(self, res, err):
        if err:
            self._fault(err)
            return
        self.regs.set(mb.REG_STATUS, mb.ST_DONE)
        self.last_error = None
        self._state = self.ST_IDLE
        self._pending = None
        self.log(f"[Modbus] 基准面已重新拟合：{res}")
        if self.on_calibrated is not None:
            try:
                self.on_calibrated(res)
            except Exception as e:
                self.log(f"[Modbus] 校准回写失败：{e}")

    def _finish_error(self, msg):
        self._fault(msg)

    def _fault(self, msg):
        self.regs.set(mb.REG_STATUS, mb.ST_FAULT)
        self.last_error = msg
        self._state = self.ST_IDLE
        self._pending = None
        self.log(f"[Modbus] 故障：{msg}")

    # -------------------------------------------------- 默认实现
    @staticmethod
    def _prepare(p, a_max_mm2):
        depth = p["depth"]
        valid = cut_plane.valid_mask(depth)
        if not valid.any():
            raise RuntimeError("没有有效深度数据")
        plane = cut_plane.plane_from_coeffs(p["plane"], depth.shape)
        h = cut_plane.build_height(depth, plane)
        fx, fy = p.get("fx"), p.get("fy")
        if fx and fy:
            mm2 = cut_plane.mm2_per_pixel(depth, fx, fy)
        else:
            mm2 = 1.0
        return depth, valid, h, mm2

    @classmethod
    def _default_compute(cls, a, b, a_max_mm2):
        out = {}
        for key, p in (("A", a), ("B", b)):
            # 最大单坑面积是按相机配的，优先用面板自己的值
            am = float(p.get("a_max_mm2") or a_max_mm2)
            depth, valid, h, mm2 = cls._prepare(p, am)
            r = cut_plane.solve_cut_plane(h, valid, mm2, am)
            # 下刀位置 = 基准面在图像中心处的深度 + 切深偏移
            H, W = depth.shape[:2]
            pa, pb, pc = p["plane"]
            center = pa * (W / 2.0) + pb * (H / 2.0) + pc
            out[key] = center + r["delta_mm"]
            p["_last"] = r
        return out["A"], out["B"]

    @classmethod
    def _default_calibrate(cls, a, b):
        """不用框选，直接在有效区域上拟合基准面。"""
        from DepthCamGUI import fit_plane_ransac
        res = {}
        for key, p in (("A", a), ("B", b)):
            depth = p["depth"]
            valid = cut_plane.valid_mask(depth)
            ys, xs = np.nonzero(valid)
            if ys.size < 1000:
                raise RuntimeError(f"{key}面有效点太少（{ys.size}）")
            roi = (int(xs.min()), int(ys.min()),
                   int(xs.max()) + 1, int(ys.max()) + 1)
            fit = fit_plane_ransac(depth, roi, step=2, max_iter=400)
            if fit is None:
                raise RuntimeError(f"{key}面基准面拟合失败")
            res[key] = (fit[0], fit[1], fit[2])
        return res
