"""Modbus RTU 从站（RS-485），用于和 PLC 交换测量结果。

角色
----
**PLC 是主站，本程序是从站**（PLC 主动读/写，我们被动应答）。这是工业现场
的常规接法，也和给出的寄存器表一致（有 W 寄存器说明是别人来写）。

寄存器映射（4xxxx 文档编号）
---------------------------
    40001  设备状态        R   0=待机 1=测量中 2=完成 3=故障
    40002  触发测量        W   1=测量
    40003  相机到A面木块距离 R   单位 mm
    40004  相机到B面木块距离 R   单位 mm
    40005  木块距离校准     W   1=距离校准

**地址基址是个必须现场确认的坑**：Modbus 规范里 40001 对应线上地址 0（从 0 起），
但很多 PLC/HMI 组态软件把 40001 当成线上地址 1（从 1 起）。搞错了就会"读得到但
全是 0"。所以这里做成可配置（addr_base），并且把所有收到的请求地址都记进日志，
现场拿日志一对就知道对方用的是哪种。

为什么不用 pymodbus
-------------------
目标机（树莓派）可能离线装不了包。RTU 从站协议本身很简单（CRC + 三种功能码），
自己实现还能把协议逻辑和串口 I/O 分开，从而**脱离硬件做完整测试**。
只用 pyserial（已经装好）负责串口本身。
"""

import struct
import threading
import time

try:
    import serial
except ImportError:      # 没有串口库时仍可用纯协议部分（测试/模拟）
    serial = None


# ------------------------------------------------------------------ 寄存器定义

REG_STATUS = 0        # 40001
REG_TRIGGER = 1       # 40002
REG_DIST_A = 2        # 40003
REG_DIST_B = 3        # 40004
REG_CALIBRATE = 4     # 40005
REG_COUNT = 5

REG_NAMES = {
    REG_STATUS: "设备状态",
    REG_TRIGGER: "触发测量",
    REG_DIST_A: "相机到A面木块距离",
    REG_DIST_B: "相机到B面木块距离",
    REG_CALIBRATE: "木块距离校准",
}
# 按寄存器表，这几个是只读的；被写时记警告但不拒绝（现场组态软件常常乱写）
REG_READONLY = (REG_STATUS, REG_DIST_A, REG_DIST_B)

ST_IDLE = 0
ST_MEASURING = 1
ST_DONE = 2
ST_FAULT = 3

STATUS_NAMES = {ST_IDLE: "待机", ST_MEASURING: "测量中",
                ST_DONE: "完成", ST_FAULT: "故障"}

# 距离寄存器是无符号还是带符号：现场距离为正，用无符号；若以后要传带符号的
# 偏移量（比如切深相对基准面可能为负），把这一个开关打开即可。
DIST_SIGNED = False
# 距离单位换算：寄存器值 = round(毫米 * DIST_SCALE)。
# 现场确认用 0.1mm 分辨率，所以是 10.0；量程随之变成 6553.5mm（6.5 米）。
DIST_SCALE = 10.0

# 由上面两个开关推出的量程。超出量程必须报故障，**绝不能截断**——
# 给机床一个被悄悄改过的下刀位置比直接报错危险得多。
DIST_MAX_MM = 32767 / DIST_SCALE if DIST_SIGNED else 65535 / DIST_SCALE
DIST_MIN_MM = -32768 / DIST_SCALE if DIST_SIGNED else 0.0


def encode_distance(mm_value, signed=DIST_SIGNED, scale=DIST_SCALE):
    """把毫米值编码成寄存器整数。"""
    v = int(round(float(mm_value) * scale))
    if signed:
        v = max(-32768, min(32767, v)) & 0xFFFF
    else:
        v = max(0, min(65535, v))
    return v


def decode_distance(raw, signed=DIST_SIGNED, scale=DIST_SCALE):
    v = raw - 65536 if (signed and raw >= 32768) else raw
    return v / scale


# ------------------------------------------------------------------ 协议基础

FC_READ_HOLDING = 3
FC_WRITE_SINGLE = 6
FC_WRITE_MULTIPLE = 16

EXC_ILLEGAL_FUNCTION = 1
EXC_ILLEGAL_ADDRESS = 2
EXC_ILLEGAL_VALUE = 3


def crc16(data: bytes) -> int:
    """Modbus CRC16（多项式 0xA001 反射，初值 0xFFFF）。"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def append_crc(pdu: bytes) -> bytes:
    return pdu + struct.pack("<H", crc16(pdu))


def check_crc(frame: bytes) -> bool:
    if len(frame) < 4:
        return False
    body, got = frame[:-2], struct.unpack("<H", frame[-2:])[0]
    return crc16(body) == got


# ------------------------------------------------------------------ 寄存器表

class HoldingRegisters:
    """线程安全的寄存器表。

    协议线程读它、界面线程写它，所以所有访问都加锁。
    内部下标 0..N-1 对应文档编号 40001..4000N；线上地址由 addr_base 换算。
    """

    def __init__(self, size=REG_COUNT, addr_base=0):
        self.size = size
        self.addr_base = addr_base
        self._regs = [0] * size
        self._lock = threading.RLock()
        self.write_log = []          # 最近被写的 (寄存器, 值, 时间)
        self.max_log = 50

    # ---- 线上地址 <-> 内部下标 ----
    def to_index(self, wire_addr):
        return wire_addr - self.addr_base

    def to_wire(self, index):
        return index + self.addr_base

    # ---- 原始访问 ----
    def read_block(self, wire_addr, count):
        """返回 (values, ok)。越界返回 (None, False)。"""
        with self._lock:
            start = self.to_index(wire_addr)
            if start < 0 or count <= 0 or start + count > self.size:
                return None, False
            return list(self._regs[start:start + count]), True

    def write_block(self, wire_addr, values):
        with self._lock:
            start = self.to_index(wire_addr)
            if start < 0 or not values or start + len(values) > self.size:
                return False
            for i, v in enumerate(values):
                self._regs[start + i] = int(v) & 0xFFFF
            now = time.strftime("%H:%M:%S")
            for i, v in enumerate(values):
                idx = start + i
                self.write_log.append((idx, int(v) & 0xFFFF, now))
            if len(self.write_log) > self.max_log:
                del self.write_log[:-self.max_log]
            return True

    # ---- 语义访问 ----
    def get(self, index):
        with self._lock:
            return self._regs[index]

    def set(self, index, value):
        with self._lock:
            self._regs[index] = int(value) & 0xFFFF

    def snapshot(self):
        with self._lock:
            return list(self._regs)


# ------------------------------------------------------------------ 从站

class ModbusRTUSlave:
    """RTU 从站。协议解析（handle_frame）与串口 I/O 分离，便于脱机测试。"""

    def __init__(self, registers: HoldingRegisters, slave_addr=1,
                 port=None, baudrate=19200, log=None):
        self.regs = registers
        self.slave_addr = slave_addr
        self.port_name = port
        self.baudrate = baudrate
        self.log = log or (lambda msg: None)
        self.ser = None
        self._running = False
        self._thread = None
        self.stats = {"rx": 0, "tx": 0, "crc_err": 0, "addr_skip": 0,
                      "exc": 0, "req_addrs": {}}

    # -------------------------------------------------- 协议
    def handle_frame(self, frame: bytes):
        """处理一个完整 RTU 帧，返回要发回的字节（None 表示不应答）。"""
        if len(frame) < 4:
            return None
        if not check_crc(frame):
            self.stats["crc_err"] += 1
            return None

        addr = frame[0]
        pdu = frame[1:-2]

        # 广播（地址 0）不回应；不是给我们的也不回应
        if addr == 0:
            self._apply(pdu, broadcast=True)
            return None
        if addr != self.slave_addr:
            self.stats["addr_skip"] += 1
            return None

        self.stats["rx"] += 1
        resp = self._apply(pdu, broadcast=False)
        if not resp:
            return None
        # 响应帧同样是 [从站地址][PDU][CRC]，别漏了地址字节
        return append_crc(bytes([self.slave_addr]) + resp)

    def _apply(self, pdu: bytes, broadcast: bool):
        """执行请求，返回响应 PDU（不含地址和 CRC）。"""
        fc = pdu[0]

        if fc == FC_READ_HOLDING:
            if len(pdu) != 5:
                return self._exc(fc, EXC_ILLEGAL_VALUE)
            start, count = struct.unpack(">HH", pdu[1:5])
            self._note_addr(start)
            if not (1 <= count <= 125):
                return self._exc(fc, EXC_ILLEGAL_VALUE)
            vals, ok = self.regs.read_block(start, count)
            if not ok:
                self.log(f"[Modbus] 读越界 线上地址 {start} 数量 {count}")
                return self._exc(fc, EXC_ILLEGAL_ADDRESS)
            body = b"".join(struct.pack(">H", v) for v in vals)
            return bytes([fc, len(body)]) + body

        if fc == FC_WRITE_SINGLE:
            if len(pdu) != 5:
                return self._exc(fc, EXC_ILLEGAL_VALUE)
            start, value = struct.unpack(">HH", pdu[1:5])
            self._note_addr(start)
            if not self.regs.write_block(start, [value]):
                self.log(f"[Modbus] 写越界 线上地址 {start}")
                return self._exc(fc, EXC_ILLEGAL_ADDRESS)
            idx = self.regs.to_index(start)
            tag = "只读寄存器" if idx in REG_READONLY else ""
            if tag:
                self.log(f"[Modbus] ⚠ PLC 写了{tag} "
                         f"{40001+idx}({REG_NAMES.get(idx, '?')}) = {value}")
            else:
                self.log(f"[Modbus] PLC 写 {40001+idx}"
                         f"({REG_NAMES.get(idx, '?')}) = {value}")
            return pdu                      # 正常回显请求

        if fc == FC_WRITE_MULTIPLE:
            if len(pdu) < 6:
                return self._exc(fc, EXC_ILLEGAL_VALUE)
            start, count, nbytes = struct.unpack(">HHB", pdu[1:6])
            self._note_addr(start)
            if nbytes != count * 2 or len(pdu) != 6 + nbytes:
                return self._exc(fc, EXC_ILLEGAL_VALUE)
            vals = list(struct.unpack(">" + "H" * count, pdu[6:6 + nbytes]))
            if not self.regs.write_block(start, vals):
                return self._exc(fc, EXC_ILLEGAL_ADDRESS)
            for i, v in enumerate(vals):
                idx = self.regs.to_index(start) + i
                self.log(f"[Modbus] PLC 写 {40001+idx}"
                         f"({REG_NAMES.get(idx, '?')}) = {v}")
            # 响应是 [功能码(1字节)][起始地址(2)][数量(2)]，功能码不能当 16 位打包
            return bytes([fc]) + struct.pack(">HH", start, count)

        self.log(f"[Modbus] 不支持的功能码 {fc}")
        return self._exc(fc, EXC_ILLEGAL_FUNCTION)

    def _exc(self, fc, code):
        self.stats["exc"] += 1
        return bytes([fc | 0x80, code])

    def _note_addr(self, wire_addr):
        """记录对方用过的地址，用于现场判断 addr_base 该取 0 还是 1。"""
        d = self.stats["req_addrs"]
        d[wire_addr] = d.get(wire_addr, 0) + 1

    # -------------------------------------------------- 串口
    def start(self):
        if self._running:
            return True
        if serial is None:
            self.log("[Modbus] 未安装 pyserial，无法打开串口")
            return False
        try:
            self.ser = serial.Serial(self.port_name, self.baudrate,
                                     bytesize=8, parity="N", stopbits=1,
                                     timeout=0.02)
        except Exception as e:
            self.log(f"[Modbus] 打开串口 {self.port_name} 失败：{e}")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.log(f"[Modbus] 从站已启动 {self.port_name} @{self.baudrate} "
                 f"站号={self.slave_addr} 地址基址={self.regs.addr_base}")
        return True

    def stop(self):
        self._running = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _serve(self):
        """按 3.5 字符静默间隔切分 RTU 帧。"""
        # 一帧最短 4 字节（地址+功能码+CRC），最长按 256 字节缓冲
        gap = max(0.002, 3.5 * 11.0 / self.baudrate)
        buf = bytearray()
        last_rx = 0.0
        while self._running:
            try:
                chunk = self.ser.read(256)
            except Exception as e:
                self.log(f"[Modbus] 串口读异常：{e}")
                break
            now = time.monotonic()
            if chunk:
                buf += chunk
                last_rx = now
                continue
            if buf and (now - last_rx) >= gap:
                frame = bytes(buf)
                buf.clear()
                resp = self.handle_frame(frame)
                if resp:
                    try:
                        self.ser.write(resp)
                        self.stats["tx"] += 1
                    except Exception as e:
                        self.log(f"[Modbus] 串口写异常：{e}")
                elif not check_crc(frame):
                    self.log(f"[Modbus] CRC 错，丢弃 {frame.hex(' ')}")


# ------------------------------------------------------------------ 自检

def _build(addr, pdu):
    return append_crc(bytes([addr]) + pdu)


def _selftest():
    print("== modbus_rtu 自检 ==")

    # 1) CRC 标准校验值
    chk = crc16(b"123456789")
    print(f"  1) CRC16('123456789') = 0x{chk:04X}（标准值 0x4B37）")
    assert chk == 0x4B37, "CRC 实现有误"
    # 自洽性：data+CRC 的 CRC 应为 0
    f = append_crc(b"123456789")
    assert crc16(f) == 0, "CRC 自洽性失败"
    print("     自洽性检查通过（data+CRC 的 CRC 为 0）")

    regs = HoldingRegisters(addr_base=0)
    slave = ModbusRTUSlave(regs, slave_addr=1)

    # 2) 读保持寄存器 FC03
    regs.set(REG_STATUS, ST_IDLE)
    regs.set(REG_DIST_A, 1234)
    regs.set(REG_DIST_B, 5678)
    req = _build(1, struct.pack(">BHH", FC_READ_HOLDING, 0, 5))
    resp = slave.handle_frame(req)
    assert resp is not None and check_crc(resp)
    assert resp[1] == FC_READ_HOLDING and resp[2] == 10
    vals = list(struct.unpack(">5H", resp[3:13]))
    print(f"  2) FC03 读 40001~40005 → {vals}")
    assert vals == [ST_IDLE, 0, 1234, 5678, 0], vals
    print("     数值与寄存器表一致 ✓")

    # 3) 写单个寄存器 FC06（触发测量）
    req = _build(1, struct.pack(">BHH", FC_WRITE_SINGLE, 1, 1))
    resp = slave.handle_frame(req)
    assert resp == req, "FC06 应回显原请求"
    assert regs.get(REG_TRIGGER) == 1
    print(f"  3) FC06 写 40002=1 → 回显正确，寄存器值 = {regs.get(REG_TRIGGER)}")

    # 4) 写单个寄存器 FC16（写多个，一次写 40005）
    req = _build(1, struct.pack(">BHHBH", FC_WRITE_MULTIPLE, 4, 1, 2, 1))
    resp = slave.handle_frame(req)
    assert resp is not None and resp[1] == FC_WRITE_MULTIPLE, resp.hex(" ")
    assert resp[2:6] == struct.pack(">HH", 4, 1), "响应应回显起始地址与数量"
    assert regs.get(REG_CALIBRATE) == 1
    print(f"  4) FC16 写 40005=1 → 响应回显起始地址/数量正确，"
          f"寄存器值 = {regs.get(REG_CALIBRATE)}")

    # 5) 越界地址 → 异常码 02
    req = _build(1, struct.pack(">BHH", FC_READ_HOLDING, 100, 2))
    resp = slave.handle_frame(req)
    assert resp[1] == (FC_READ_HOLDING | 0x80) and resp[2] == EXC_ILLEGAL_ADDRESS
    print(f"  5) 越界地址 → 异常码 0x{resp[2]:02X}(非法数据地址) ✓")

    # 6) 不支持的功能码 → 异常码 01
    req = _build(1, struct.pack(">BHH", 99, 0, 1))
    resp = slave.handle_frame(req)
    assert resp[1] == (99 | 0x80) and resp[2] == EXC_ILLEGAL_FUNCTION
    print(f"  6) 功能码 99 → 异常码 0x{resp[2]:02X}(非法功能) ✓")

    # 7) 站号不是我们 → 不应答
    req = _build(9, struct.pack(">BHH", FC_READ_HOLDING, 0, 1))
    assert slave.handle_frame(req) is None
    print("  7) 站号不匹配 → 不应答 ✓")

    # 8) 广播（站号 0）→ 执行但不应答
    before = regs.get(REG_TRIGGER)
    req = _build(0, struct.pack(">BHH", FC_WRITE_SINGLE, 1, 7))
    assert slave.handle_frame(req) is None
    assert regs.get(REG_TRIGGER) == 7, "广播写应生效"
    regs.set(REG_TRIGGER, before)
    print("  8) 广播写 → 生效但不应答 ✓")

    # 9) CRC 错 → 丢弃
    bad = bytearray(_build(1, struct.pack(">BHH", FC_READ_HOLDING, 0, 1)))
    bad[-1] ^= 0xFF
    assert slave.handle_frame(bytes(bad)) is None
    assert slave.stats["crc_err"] >= 1
    print("  9) CRC 错 → 丢弃并计数 ✓")

    # 10) addr_base=1 的换算（PLC 组态把 40001 当线上地址 1 的情况）
    regs1 = HoldingRegisters(addr_base=1)
    s1 = ModbusRTUSlave(regs1, slave_addr=1)
    # 40003 的内部下标是 2，addr_base=1 时线上地址 = 2+1 = 3
    req = _build(1, struct.pack(">BHH", FC_WRITE_SINGLE, 3, 4321))
    s1.handle_frame(req)
    assert regs1.get(REG_DIST_A) == 4321, "addr_base=1 时线上3 应落到 40003"
    # 线上地址 1 应落到 40001
    s1.handle_frame(_build(1, struct.pack(">BHH", FC_WRITE_SINGLE, 1, 5)))
    assert regs1.get(REG_STATUS) == 5
    print(" 10) addr_base=1：线上1→40001、线上3→40003 ✓"
          "（两种基址都可配，现场用日志确认对方用哪种）")

    # 11) 地址使用统计（现场排查基址用）
    print(f" 11) 对方用过的线上地址统计: {dict(sorted(slave.stats['req_addrs'].items()))}")

    # 12) 距离编解码
    for mm in (0, 1, 1234, 65535):
        assert abs(decode_distance(encode_distance(mm)) - mm) < 1e-9
    print(f" 12) 距离编解码往返一致（1mm 分辨率，量程 0~65535mm）")

    print("\n全部自检通过")
    print("\n寄存器表（文档编号 / 线上地址 / 名称）:")
    for i in range(regs.size):
        base = regs.addr_base
        print(f"  {40001+i:<6} wire={i+base:<3} {REG_NAMES.get(i,'?'):<20}"
              f" {'R' if i in REG_READONLY else 'W'}")


if __name__ == "__main__":
    _selftest()
