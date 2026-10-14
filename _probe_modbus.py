"""脱机测试 Modbus 服务：触发 → 等新帧 → 算 → 写寄存器 的完整时序。"""
import struct
import time

import numpy as np

import modbus_rtu as mb
from modbus_service import ModbusService

REF = 1800.0          # 基准面在图像中心的深度
FX = FY = 2002.52


class FakePanels:
    """伪造两台相机的面板数据。"""

    def __init__(self):
        self.frame_id = 0
        self.calibrated = True
        self.running = True
        self.have_depth = True
        H, W = 768, 1024        # 小一点，测试快
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # 木块面：基准面 + 一个碗形凹坑（更深=离相机更远）
        self.depth = (REF + 3.0 * np.exp(
            -(((xx - 300) ** 2 + (yy - 400) ** 2) / (2 * 120.0 ** 2)))).astype(np.float32)
        self.plane = (0.0, 0.0, REF)     # 正对相机
        self.fx, self.fy = FX, FY

    def bump(self):
        self.frame_id += 1

    def data(self):
        out = {}
        for key in ("A", "B"):
            out[key] = {
                "depth": self.depth if self.have_depth else None,
                "plane": self.plane if self.calibrated else None,
                "calibrated": self.calibrated,
                "running": self.running,
                "frame_id": self.frame_id,
                "fx": self.fx, "fy": self.fy,
            }
        return out


def drive_frames(svc, panels, ticks=40):
    """模拟主线程节拍：每次 tick 调 poll()，中途让相机出新帧。"""
    for i in range(ticks):
        svc.poll()
        if i == 3:                  # 触发后相机才出新帧
            panels.bump()
        time.sleep(0.005)
    # 等工作线程算完
    for _ in range(400):
        svc.poll()
        if svc._state == ModbusService.ST_IDLE and svc.regs.get(mb.REG_STATUS) != mb.ST_MEASURING:
            break
        time.sleep(0.01)


def main():
    print("== Modbus 服务时序测试 ==")
    regs = mb.HoldingRegisters(addr_base=0)
    panels = FakePanels()
    logs = []
    svc = ModbusService(regs, panels.data, log=logs.append, timeout_s=2.0, a_max_mm2=200.0)

    # 1) 初始应处于待机
    assert regs.get(mb.REG_STATUS) == mb.ST_IDLE
    print(f"  1) 初始状态 = {regs.get(mb.REG_STATUS)}（待机）✓")

    # 2) 未采集时应报故障
    panels.running = False
    regs.set(mb.REG_TRIGGER, 1)
    svc.poll()
    assert regs.get(mb.REG_STATUS) == mb.ST_FAULT, regs.get(mb.REG_STATUS)
    assert regs.get(mb.REG_TRIGGER) == 0, "触发位应被自动清零"
    print(f"  2) 未采集时触发 → 状态 3(故障)，40002 已自动清零 ✓")
    print(f"     故障原因：{svc.last_error}")
    panels.running = True

    # 3) 未标定也应报故障
    panels.calibrated = False
    regs.set(mb.REG_TRIGGER, 1)
    svc.poll()
    assert regs.get(mb.REG_STATUS) == mb.ST_FAULT
    print(f"  3) 未标定时触发 → 故障：「{svc.last_error}」✓")
    panels.calibrated = True

    # 4) 正常触发：应经历 测量中(1) → 完成(2)
    seen = set()
    regs.set(mb.REG_TRIGGER, 1)
    svc.poll()
    seen.add(regs.get(mb.REG_STATUS))
    assert regs.get(mb.REG_STATUS) == mb.ST_MEASURING
    drive_frames(svc, panels)
    seen.add(regs.get(mb.REG_STATUS))
    assert regs.get(mb.REG_STATUS) == mb.ST_DONE, f"最终状态 {regs.get(mb.REG_STATUS)}"
    print(f"  4) 正常触发：状态经历 {sorted(seen)}（1=测量中 → 2=完成）✓")

    # 5) 下刀位置数值检查
    da = mb.decode_distance(regs.get(mb.REG_DIST_A))
    db = mb.decode_distance(regs.get(mb.REG_DIST_B))
    print(f"  5) 下刀位置：A面 {da:.1f} mm，B面 {db:.1f} mm（分辨率 0.1mm）")
    assert abs(da - db) < 1e-6, "两面数据应相同（伪造数据一样）"
    assert REF < da < REF + 3.5, f"下刀位置应在基准面之外一点，实际 {da}"
    print(f"     基准面 {REF:.1f} mm，凹坑最深 3.0 mm → 下刀位置落在合理区间 ✓")

    # 6) 距离编码分辨率（现场确认 0.1mm）
    raw = regs.get(mb.REG_DIST_A)
    assert mb.DIST_SCALE == 10.0, "现场要求 0.1mm 分辨率"
    assert abs(raw / mb.DIST_SCALE - da) < 1e-9
    print(f"     寄存器原始值 {raw}，×0.1mm = {raw/10.0:.1f} mm ✓"
          f"（量程 {mb.DIST_MIN_MM:.0f}~{mb.DIST_MAX_MM:.0f} mm）")

    # 6b) 超量程必须报故障，不能截断
    svc.compute_fn = lambda a, b, m: (100.0, 99999.0)
    regs.set(mb.REG_TRIGGER, 1)
    svc.poll()
    panels.bump()
    drive_frames(svc, panels, ticks=200)
    assert regs.get(mb.REG_STATUS) == mb.ST_FAULT, "超量程应报故障"
    assert "超出寄存器量程" in svc.last_error
    print(f"  6b) 超量程 → 故障而非截断：「{svc.last_error}」✓")
    svc.compute_fn = ModbusService._default_compute

    # 7) 校准请求：自动拟合，不需要框选
    fitted = {}

    def fake_calibrate(a, b):
        fitted["called"] = True
        return {"A": (0.0001, 0.0, 1799.5), "B": (0.0, 0.0002, 1798.8)}

    svc.calibrate_fn = fake_calibrate
    svc.on_calibrated = lambda res: fitted.update(res)
    regs.set(mb.REG_CALIBRATE, 1)
    svc.poll()
    assert regs.get(mb.REG_CALIBRATE) == 0, "校准位应被自动清零"
    assert regs.get(mb.REG_STATUS) == mb.ST_MEASURING
    panels.bump()
    drive_frames(svc, panels)
    assert regs.get(mb.REG_STATUS) == mb.ST_DONE
    assert fitted.get("called") and "A" in fitted and "B" in fitted
    print(f"  7) 校准请求：自动拟合成功，回写 {fitted['A']} / {fitted['B']} ✓")

    # 8) 等新帧超时
    regs.set(mb.REG_TRIGGER, 1)
    svc.poll()
    t0 = time.time()
    while svc._state != ModbusService.ST_IDLE and time.time() - t0 < 3:
        svc.poll()
        time.sleep(0.02)
    assert regs.get(mb.REG_STATUS) == mb.ST_FAULT
    print(f"  8) 相机不出新帧 → 超时故障：「{svc.last_error}」✓")

    # 9) 真实算法跑一遍（用完整分辨率会慢，这里只验证不抛异常）
    regs.set(mb.REG_TRIGGER, 1)
    svc.poll()
    panels.bump()
    drive_frames(svc, panels, ticks=200)
    assert regs.get(mb.REG_STATUS) == mb.ST_DONE, svc.last_error
    da2 = mb.decode_distance(regs.get(mb.REG_DIST_A))
    print(f"  9) 接真实 cut_plane 算法跑通：下刀位置 {da2:.1f} mm，"
          f"周期 {svc.cycle_ms:.0f} ms ✓")

    print("\n日志（最后 6 条）:")
    for line in logs[-6:]:
        print("   ", line)
    print("\n全部通过")


if __name__ == "__main__":
    main()
