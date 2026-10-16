# -*- coding: utf-8 -*-
"""PLC 端模拟器（Windows + Tk）——用串口以 **Modbus RTU 主站** 身份和相机端联调。

角色与接线
----------
现场是「PLC 主站 + 相机端从站」（见 modbus_rtu.py 顶部说明）。本程序扮演 PLC：
在 Windows 上打开一个串口（USB 转 RS-485 接相机端，或 USB 转串口自环做台面测试），
按寄存器表主动读写，从而**在没有真正 PLC 的情况下**验证相机端的行为。

    Windows（本程序，主站）── USB-RS485 ── 相机端（DepthCamGUI，从站）

寄存器表（和 modbus_rtu.py 共用同一份定义，避免两边漂移）
    40001  设备状态          R   0=待机 1=测量中 2=完成 3=故障
    40002  触发测量          W   1=测量
    40003  相机到A面木块距离  R   单位 mm（寄存器值 = 毫米 × 10，即 0.1mm 分辨率）
    40004  相机到B面木块距离  R   单位 mm
    40005  木块距离校准      W   1=距离校准

握手时序（和 modbus_service.py 一致）
    写 40002=1 → 对方自动把 40002 清 0（PLC 一看回 0 就知道被受理）
               → 40001=1(测量中) → 算完写 40003/40004 → 40001=2(完成)
               出错 → 40001=3(故障)，故障原因**不占寄存器**，要看相机端日志
    写 40005=1 同上，但**不更新** 40003/40004（校准只重新拟合基准面）。

界面上的按钮
    触发测量   走完上面整条时序，最后把 A/B 两面的深度显示出来
    校准       写 40005=1，等对方重新拟合基准面
    读状态     只读 40001~40005，刷新寄存器表和深度显示
    连续监视   定时只读，让深度数据持续刷新
    自测模式   不接硬件：内置一个假从站按同样时序应答，用来验证本程序本身

为什么自己实现主站协议
----------------------
和从站一样，只用 pyserial，不依赖 pymodbus（现场机器未必装得上）。CRC 和三种
功能码直接复用 modbus_rtu.py 的实现，保证两边字节级一致。

线程边界
--------
**串口只在工作线程里被碰**。Tk 主线程通过 post() 投任务、通过 events 队列收结果，
既不出现两个线程同时读同一个串口，也不会在别的线程里碰 Tk。

命令行
------
    python plc_sim.py                 # 正常启动界面
    python plc_sim.py --list-ports    # 列出本机串口
    python plc_sim.py --selftest      # 脱机协议自检（不需要硬件、不开窗口）
    python plc_sim.py --smoke         # 起窗口 + 自测模式自动触发一次，然后退出
"""

import argparse
import math
import queue
import struct
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

try:
    import serial
    from serial.tools import list_ports
except ImportError:                     # 没装 pyserial 时界面仍能起，只是连不上
    serial = None
    list_ports = None


# ====================================================================== 协议定义

class _FallbackProtocol:
    """找不到 modbus_rtu.py 时的**应急最小副本**。

    字段一旦和 modbus_rtu.py 不一致就会联调失败，所以这种情况会记一条提示日志。
    这份副本只够串口模式用；自测模式需要真的从站类，会被禁用。
    """

    REG_STATUS, REG_TRIGGER, REG_DIST_A, REG_DIST_B, REG_CALIBRATE = 0, 1, 2, 3, 4
    REG_COUNT = 5
    REG_NAMES = {
        0: "设备状态", 1: "触发测量", 2: "相机到A面木块距离",
        3: "相机到B面木块距离", 4: "木块距离校准",
    }
    REG_READONLY = (0, 2, 3)

    ST_IDLE, ST_MEASURING, ST_DONE, ST_FAULT = 0, 1, 2, 3
    STATUS_NAMES = {0: "待机", 1: "测量中", 2: "完成", 3: "故障"}

    DIST_SCALE = 10.0

    @staticmethod
    def crc16(data):
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc & 0xFFFF

    def append_crc(self, pdu):
        return pdu + struct.pack("<H", self.crc16(pdu))

    def check_crc(self, frame):
        if len(frame) < 4:
            return False
        return self.crc16(frame[:-2]) == struct.unpack("<H", frame[-2:])[0]

    def encode_distance(self, mm_value):
        return max(0, min(65535, int(round(float(mm_value) * self.DIST_SCALE))))

    def decode_distance(self, raw):
        return raw / self.DIST_SCALE


def _load_protocol():
    try:
        import modbus_rtu as real
        return real, "modbus_rtu.py（与相机端共用同一份寄存器定义）"
    except ImportError:
        return (_FallbackProtocol(),
                "内置副本（没找到 modbus_rtu.py！请确认寄存器表/单位与相机端一致）")


mb, PROTO_SOURCE = _load_protocol()
HAS_SLAVE = hasattr(mb, "ModbusRTUSlave")      # 自测模式需要真的从站类

BAUD_CHOICES = ("1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200")
MONITOR_CHOICES = ("200", "500", "1000", "2000")
DEFAULT_SEQ_TIMEOUT = 8.0          # 触发/校准整条时序的等待上限（秒）
STALE_SECONDS = 3.0                # 超过这么久没读到新数据，就把深度标记为过期

UI_FONT = ("Microsoft YaHei UI", 10)
MONO_FONT = ("Consolas", 10)
BIG_FONT = ("Consolas", 24, "bold")

C_OK, C_ERR, C_WARN, C_IDLE = "#0a7a2f", "#c00000", "#b06000", "#888888"
C_TX, C_RX, C_TIME = "#1a4fa0", "#6a3fa0", "#7a7a7a"

EXC_NAMES = {1: "非法功能码", 2: "非法数据地址", 3: "非法数据值", 4: "从站设备故障"}

# 寄存器表里用的短名：现场笔记本常常是 1366×768 + 放大显示，全名会把列挤爆。
SHORT_NAMES = {0: "设备状态", 1: "触发测量", 2: "A面距离", 3: "B面距离", 4: "距离校准"}

FC_READ_HOLDING, FC_WRITE_SINGLE = 3, 6


def stamp():
    """带毫秒的时间戳：报文级日志靠它排序。"""
    return time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"


class LinkError(Exception):
    """通信/协议层错误（超时、CRC 错、异常响应…），消息直接给人看。"""


# ====================================================================== 传输层

class SerialTransport:
    """真串口。只被工作线程使用。"""

    def __init__(self, port, baud, read_timeout=0.02):
        self.ser = serial.Serial(port, baud, bytesize=8, parity="N",
                                 stopbits=1, timeout=read_timeout)

    def read(self, n):
        return self.ser.read(n)

    def write(self, data):
        return self.ser.write(data)

    def reset_input_buffer(self):
        self.ser.reset_input_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


class LoopTransport:
    """不经过串口的假传输：把帧直接喂给 VirtualPeer，用于自测模式。"""

    def __init__(self, peer):
        self.peer = peer
        self._rx = bytearray()

    def read(self, n):
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def write(self, data):
        resp = self.peer.exchange(bytes(data))
        if resp:
            self._rx += resp

    def reset_input_buffer(self):
        self._rx.clear()

    def close(self):
        pass


class VirtualPeer:
    """自测用的假从站：完全按 modbus_service.py 的握手时序应答。

    它**不做真正的测量**，只在收到触发后 busy_ms 毫秒把 40003/40004 写成一个
    缓慢漂移的值，好让界面上的深度刷新看得出在动。目的是验证本程序（主站）
    的时序、CRC、状态机，而不是验证算法。
    """

    def __init__(self, addr_base=0, slave_addr=1, busy_ms=350, log=None):
        self.regs = mb.HoldingRegisters(addr_base=addr_base)
        self.slave = mb.ModbusRTUSlave(self.regs, slave_addr=slave_addr, log=log)
        self.busy_ms = busy_ms
        self._t0 = 0.0
        self._mode = None
        self._n = 0

    def exchange(self, frame):
        self._advance()
        resp = self.slave.handle_frame(frame)
        self._check_request()
        self._advance()
        return resp

    def _check_request(self):
        if self.regs.get(mb.REG_TRIGGER) == 1:
            self.regs.set(mb.REG_TRIGGER, 0)          # 自动清零，和真从站一样
            self._begin("measure")
        elif self.regs.get(mb.REG_CALIBRATE) == 1:
            self.regs.set(mb.REG_CALIBRATE, 0)
            self._begin("calibrate")

    def _begin(self, mode):
        self._mode = mode
        self._t0 = time.monotonic()
        self.regs.set(mb.REG_STATUS, mb.ST_MEASURING)

    def _advance(self):
        if self._mode is None:
            return
        if self.regs.get(mb.REG_STATUS) != mb.ST_MEASURING:
            return
        if (time.monotonic() - self._t0) * 1000.0 < self.busy_ms:
            return
        if self._mode == "measure":
            self._n += 1
            a = 1800.0 + 6.0 * math.sin(self._n / 3.0)
            b = 1795.0 + 6.0 * math.cos(self._n / 3.0)
            self.regs.set(mb.REG_DIST_A, mb.encode_distance(a))
            self.regs.set(mb.REG_DIST_B, mb.encode_distance(b))
        self.regs.set(mb.REG_STATUS, mb.ST_DONE)
        self._mode = None


# ====================================================================== 主站

class PlcLink:
    """Modbus RTU 主站 + 串口。所有串口动作都在自己的工作线程里顺序执行。"""

    def __init__(self, events, slave_addr=1, addr_base=0, timeout=1.0,
                 retries=1, seq_timeout=DEFAULT_SEQ_TIMEOUT, poll_interval=0.08):
        self._events = events
        self.slave_addr = slave_addr
        self.addr_base = addr_base
        self.timeout = timeout              # 单帧等待响应的超时
        self.retries = retries
        self.seq_timeout = seq_timeout
        self.poll_interval = poll_interval

        self._jobs = queue.Queue()
        self._busy = False
        self._connected = False
        self._transport = None
        self._mode = None
        self.peer = None                    # 仅自测模式：假从站，供自检断言
        self.stats = {"req": 0, "timeout": 0, "crc": 0, "exc": 0, "last_ms": 0.0}
        self._poll_fail = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -------------------------------------------------- 线程/任务
    def post(self, fn):
        """从 Tk 线程投一个任务给工作线程。fn 里**不能碰 Tk**。"""
        self._jobs.put(fn)

    def idle(self):
        """工作线程手上没活（用于连续监视不堆任务）。"""
        return not self._busy and self._jobs.empty()

    def is_busy(self):
        return self._busy

    @property
    def connected(self):
        return self._connected

    def _run(self):
        while True:
            fn = self._jobs.get()
            if fn is None:
                break
            self._busy = True
            try:
                fn()
            except LinkError as e:                 # 各时序自己也会兜，这里是保底
                self._log(f"通信失败：{e}", "err")
            except Exception as e:                 # 不要让工作线程死掉
                self._log(f"内部错误：{type(e).__name__}: {e}", "err")
            finally:
                self._busy = False
                self._emit_stats()

    # -------------------------------------------------- 事件/日志
    def _log(self, msg, level="info"):
        self._events.put(("log", level, f"{stamp()}  {msg}"))

    def _emit_stats(self):
        self._events.put(("stats", dict(self.stats)))

    def wire(self, index):
        """内部下标 → 线上地址。地址基址搞错就会"读得到但全是 0"。"""
        return index + self.addr_base

    def _require(self):
        if not self._connected or self._transport is None:
            raise LinkError("串口未连接")

    # -------------------------------------------------- 连接
    def do_connect(self, mode, port, baud, slave_addr, addr_base):
        # 整个流程包在 try 里：无论如何都要发出一个 conn 事件，
        # 否则界面上"连接"按钮会一直停在禁用状态。
        try:
            self.do_disconnect(silent=True)
            self.slave_addr = int(slave_addr)
            self.addr_base = int(addr_base)
            self._poll_fail = 0
            if mode == "loop":
                if not HAS_SLAVE:
                    raise LinkError("自测模式需要同目录的 modbus_rtu.py（现在用的是内置副本）")
                self.peer = VirtualPeer(addr_base=self.addr_base,
                                        slave_addr=self.slave_addr,
                                        log=lambda m: self._log("假从站 " + m, "rx"))
                self._transport = LoopTransport(self.peer)
                desc = "自测模式（内置虚拟从站，不接硬件）"
            else:
                if serial is None:
                    raise LinkError("没有安装 pyserial（pip install pyserial）")
                self._transport = SerialTransport(port, int(baud))
                desc = f"{port} @{baud} 8N1"
        except Exception as e:
            self._transport, self.peer = None, None
            self._connected = False
            self._events.put(("conn", {"ok": False, "msg": f"打开串口失败：{e}"}))
            self._log(f"打开串口失败：{e}", "err")
            return
        self._connected = True
        self._mode = mode
        self._events.put(("conn", {"ok": True, "msg": desc, "mode": mode,
                                   "port": port, "baud": baud,
                                   "slave": self.slave_addr, "base": self.addr_base}))
        self._log(f"已连接：{desc}；站号 {self.slave_addr}，地址基址 {self.addr_base}"
                  f"（40001 → 线上地址 {self.addr_base}）", "ok")

    def do_disconnect(self, silent=False):
        if self._transport is not None:
            self._transport.close()
        self._transport, self.peer = None, None
        was, self._connected, self._mode = self._connected, False, None
        if not silent:
            self._events.put(("conn", {"ok": False, "msg": "已断开"}))
            if was:
                self._log("已断开串口", "warn")

    def shutdown(self):
        self.do_disconnect(silent=True)
        self._jobs.put(None)

    # -------------------------------------------------- 报文收发
    def _build(self, fc, tail):
        return mb.append_crc(bytes([self.slave_addr, fc]) + tail)

    def _frame_len(self, buf):
        """按功能码判断响应总长度；None = 还不够判断。"""
        fc = buf[1]
        if fc & 0x80:
            return 5                                     # 异常响应
        if fc == FC_READ_HOLDING:
            return None if len(buf) < 3 else 3 + buf[2] + 2
        if fc in (FC_WRITE_SINGLE, 16):
            return 8
        raise LinkError(f"响应里出现没请求过的功能码 {fc}")

    def _read_response(self):
        buf = bytearray()
        deadline = time.monotonic() + self.timeout
        while True:
            if len(buf) >= 2:
                n = self._frame_len(buf)
                if n is not None and len(buf) >= n:
                    return bytes(buf[:n])
            if time.monotonic() >= deadline:
                raise LinkError("等待响应超时")
            chunk = self._transport.read(256)
            if chunk:
                buf += chunk
                # 丢掉帧头之前的杂字节（总线上可能残留上半帧）
                while buf and buf[0] != self.slave_addr:
                    buf.pop(0)

    def _transact(self, req, retries=None):
        """发一帧收一帧。超时/CRC 错会重试；异常响应直接抛出（对方确实答了）。"""
        self._require()
        if retries is None:
            retries = self.retries
        last = None
        for attempt in range(retries + 1):
            if attempt:
                self._log(f"第 {attempt} 次重试…", "warn")
            t0 = time.monotonic()
            try:
                self._transport.reset_input_buffer()
                self._transport.write(req)
                self._log(f"TX  {req.hex(' ')}", "tx")
                frame = self._read_response()
            except LinkError as e:
                last = e
                self.stats["timeout"] += 1
                self._log(f"RX  无有效响应（{time.monotonic() - t0:.2f}s）：{e}", "warn")
                continue
            self.stats["req"] += 1
            self.stats["last_ms"] = (time.monotonic() - t0) * 1000.0
            self._log(f"RX  {frame.hex(' ')}  ({self.stats['last_ms']:.0f} ms)", "rx")
            if not mb.check_crc(frame):
                self.stats["crc"] += 1
                last = LinkError(f"响应 CRC 错：{frame.hex(' ')}")
                self._log(str(last), "err")
                continue
            if frame[0] != self.slave_addr:
                last = LinkError(f"响应站号 {frame[0]} 不是 {self.slave_addr}")
                continue
            if frame[1] & 0x80:
                code = frame[2]
                self.stats["exc"] += 1
                raise LinkError(f"从站返回异常码 {code}（{EXC_NAMES.get(code, '未知')}）")
            return frame
        raise last or LinkError("通信失败")

    def _read_holding(self, index, count):
        req = self._build(FC_READ_HOLDING, struct.pack(">HH", self.wire(index), count))
        frame = self._transact(req)
        if frame[1] != FC_READ_HOLDING or frame[2] != count * 2:
            raise LinkError(f"读响应不符合预期：{frame.hex(' ')}")
        return list(struct.unpack(">" + "H" * count, frame[3:3 + count * 2]))

    def _read_reg(self, index):
        return self._read_holding(index, 1)[0]

    def _write_reg(self, index, value):
        req = self._build(FC_WRITE_SINGLE, struct.pack(">HH", self.wire(index), value))
        frame = self._transact(req)
        if frame != req:                        # FC06 正常响应就是原样回显
            raise LinkError(f"写响应回显不一致：{frame.hex(' ')}")
        self._log(f"写 {40001 + index}({mb.REG_NAMES.get(index, '?')}) = {value}", "info")

    def _write_request_bit(self, index, name):
        """写触发位/校准位 = 1。

        这里**不能盲目重发**：丢的如果只是"响应"，对方其实已经执行了，重发就会
        让它多测一次。所以超时后先查 40001——状态一动就说明这一刀已被受理。
        查不到（链路不通）时也不重发：宁可报错让人确认，也不要凭空多触发一次。
        """
        req = self._build(FC_WRITE_SINGLE, struct.pack(">HH", self.wire(index), 1))
        try:
            frame = self._transact(req, retries=0)
        except LinkError as e:
            self._log(f"写 {name} 没收到响应（{e}），先查 40001 确认对方收到没有…", "warn")
            verdict = self._reconcile()
            if verdict is True:
                self._log(f"40001 已经动了 → {name} 确实被受理，不重发（避免多测一次）", "ok")
                return
            if verdict is None:
                raise LinkError(f"写 {name} 后链路不通，无法确认对方是否受理；"
                                f"**不重发**，请人工确认状态后再操作") from e
            self._log(f"40001 一直停在待机 → {name} 应该没送到，重发一次", "warn")
            frame = self._transact(req, retries=0)
        if frame != req:
            raise LinkError(f"写响应回显不一致：{frame.hex(' ')}")
        self._log(f"写 {40001 + index}({mb.REG_NAMES.get(index, '?')}) = 1", "info")

    def _reconcile(self):
        """查 40001 判断刚才的请求有没有被受理。

        True=已受理，False=确认没送到，None=读不到（链路不通，无法判断）。
        """
        deadline = time.monotonic() + max(0.4, self.timeout)
        while time.monotonic() < deadline:
            try:
                if self._read_reg(mb.REG_STATUS) != mb.ST_IDLE:
                    return True
            except LinkError:
                return None
            time.sleep(0.05)
        return False

    # -------------------------------------------------- 时序
    def _wait_clear(self, index, name):
        """等对方把请求位自动清 0 —— 这是"请求已被受理"的唯一凭据。"""
        deadline = time.monotonic() + max(1.0, self.timeout)
        while time.monotonic() < deadline:
            if self._read_reg(index) == 0:
                self._log(f"{name} 已被对方清 0，请求已受理", "ok")
                return True
            time.sleep(0.05)
        self._log(f"{name} 一直没被清 0：对方可能没在采集，或站号/地址基址不对；"
                  f"继续看状态…", "warn")
        return False

    def _wait_terminal(self, title):
        """轮询 40001 直到 完成(2) 或 故障(3)，状态变化才记日志。"""
        deadline = time.monotonic() + self.seq_timeout
        last = None
        while True:
            st = self._read_reg(mb.REG_STATUS)
            if st != last:
                level = "ok" if st == mb.ST_DONE else ("err" if st == mb.ST_FAULT else "info")
                self._log(f"40001(设备状态) = {st} {mb.STATUS_NAMES.get(st, '未知')}", level)
                self._events.put(("status", st))
                last = st
            if st in (mb.ST_DONE, mb.ST_FAULT):
                return st
            if time.monotonic() >= deadline:
                raise LinkError(f"等待{title}结果超时（{self.seq_timeout:.1f}s，"
                                f"40001 一直停在 {last}）")
            time.sleep(self.poll_interval)

    def do_trigger(self):
        self._sequence("触发测量", mb.REG_TRIGGER, "40002", measure=True)

    def do_calibrate(self):
        self._sequence("校准", mb.REG_CALIBRATE, "40005", measure=False)

    def _sequence(self, title, reg, name, measure):
        self._events.put(("seq", True))
        try:
            self._require()
            self._log(f"———— {title}：写 {name} = 1 ————", "info")
            t0 = time.monotonic()
            try:
                self._write_request_bit(reg, name)
                self._wait_clear(reg, name)
                st = self._wait_terminal(title)
            except LinkError as e:
                self._log(f"{title}失败：{e}", "err")
                return
            cycle = (time.monotonic() - t0) * 1000.0
            note = None
            if st == mb.ST_FAULT:
                # 寄存器表里没有故障码，这是现场排查的常见坑，所以这里说清楚
                note = "40001=3(故障)：故障原因不占寄存器，请看相机端日志"
                self._log(f"{title}：设备报故障（40001=3），本次没有有效数据", "err")
            else:
                self._log(f"{title}完成，端到端耗时 {cycle:.0f} ms", "ok")
                if not measure:
                    note = "校准不更新 40003/40004，下面深度仍是上次测量值"
            try:
                vals = self._read_holding(0, mb.REG_COUNT)
            except LinkError as e:
                self._log(f"读取寄存器失败：{e}", "err")
                return
            self._publish(vals, cycle if st == mb.ST_DONE else None, note)
        finally:
            self._events.put(("seq", False))

    def do_read_all(self, quiet=False):
        try:
            self._require()
            vals = self._read_holding(0, mb.REG_COUNT)
        except LinkError as e:
            self._poll_fail += 1
            if not quiet or self._poll_fail == 1 or self._poll_fail % 25 == 0:
                self._log(f"读状态失败（连续 {self._poll_fail} 次）：{e}", "err")
            return
        self._poll_fail = 0
        self._publish(vals, None, None, quiet=quiet)

    def do_read_all_quiet(self):
        self.do_read_all(quiet=True)

    def _publish(self, vals, cycle_ms, note, quiet=False):
        a_raw, b_raw = vals[mb.REG_DIST_A], vals[mb.REG_DIST_B]
        a_mm, b_mm = mb.decode_distance(a_raw), mb.decode_distance(b_raw)
        self._events.put(("snapshot", {"values": vals, "ts": time.time(),
                                       "cycle_ms": cycle_ms, "note": note}))
        if not quiet:
            self._log(f"A面深度 {a_mm:.1f} mm（40003={a_raw}）　"
                      f"B面深度 {b_mm:.1f} mm（40004={b_raw}）", "ok")
        if note:
            self._log(note, "warn")


# ====================================================================== 界面

LOG_COLORS = {"info": "#222222", "ok": C_OK, "warn": C_WARN, "err": C_ERR,
              "tx": C_TX, "rx": C_RX}


class App:
    """Tk 界面。只做两件事：把按钮变成任务投给 PlcLink，把事件画到界面上。"""

    def __init__(self, root, auto_selftest=False):
        self.root = root
        root.title("PLC 端模拟器（Modbus RTU 主站）— A/B 相机深度联调")
        # 按屏幕实际大小定窗口，别在 1366×768 的现场笔记本上比屏幕还大
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1120, sw - 100), min(820, sh - 100)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        root.minsize(940, 560)

        self.events = queue.Queue()
        self.link = PlcLink(self.events)
        self.connected = False
        self.seq_running = False
        self.last_ts = 0.0
        self.stale_marked = False
        self.last_vals = None
        self.last_depth = None
        self._next_poll_at = 0.0

        self._build_ui()
        self._refresh_ports(log=False)
        self._log(f"寄存器定义来源：{PROTO_SOURCE}", "info" if HAS_SLAVE else "warn")
        self._log("用法：选串口 → 连接 → 点【触发测量】，最后看下面的 A/B 深度数据。"
                  "没硬件就勾【自测模式】。", "info")

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind("<F5>", lambda e: self.on_trigger())
        root.bind("<F6>", lambda e: self.on_calibrate())
        root.after(60, self._tick)
        if auto_selftest:
            self.var_selftest.set(True)

    # -------------------------------------------------- 构建界面
    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}

        # --- 串口工具条
        top = ttk.LabelFrame(self.root, text="串口 / 站参数", padding=(8, 4))
        top.pack(fill="x", **pad)

        ttk.Label(top, text="串口:").pack(side="left")
        self.cmb_port = ttk.Combobox(top, width=10, values=())
        self.cmb_port.pack(side="left", padx=(2, 4))
        ttk.Button(top, text="刷新端口", command=self.on_refresh_ports).pack(side="left")

        ttk.Label(top, text="  波特率:").pack(side="left")
        self.cmb_baud = ttk.Combobox(top, width=7, values=BAUD_CHOICES)
        self.cmb_baud.set("9600")
        self.cmb_baud.pack(side="left", padx=(2, 6))

        ttk.Label(top, text="站号:").pack(side="left")
        self.ent_slave = ttk.Entry(top, width=4)
        self.ent_slave.insert(0, "1")
        self.ent_slave.pack(side="left", padx=(2, 6))

        ttk.Label(top, text="地址基址:").pack(side="left")
        self.cmb_base = ttk.Combobox(top, state="readonly", width=3, values=("0", "1"))
        self.cmb_base.set("0")
        self.cmb_base.pack(side="left", padx=(2, 6))

        self.btn_conn = ttk.Button(top, text="连接", command=self.on_connect)
        self.btn_conn.pack(side="left", padx=(8, 2))

        self.var_selftest = tk.BooleanVar(value=False)
        self.chk_self = ttk.Checkbutton(top, text="自测模式",
                                        variable=self.var_selftest,
                                        command=self.on_selftest_toggle)
        self.chk_self.pack(side="left", padx=(10, 2))

        # --- 连接状态
        self.lbl_conn = tk.Label(self.root, text="未连接", anchor="w", fg=C_IDLE,
                                 font=UI_FONT)
        self.lbl_conn.pack(fill="x", padx=10)

        # --- 操作按钮
        ops = ttk.LabelFrame(self.root, text="操作", padding=(8, 4))
        ops.pack(fill="x", **pad)
        self.btn_trigger = ttk.Button(ops, text="触发测量 (F5)", command=self.on_trigger,
                                      state="disabled", width=16)
        self.btn_trigger.pack(side="left")
        self.btn_cal = ttk.Button(ops, text="校准 (F6)", command=self.on_calibrate,
                                  state="disabled", width=12)
        self.btn_cal.pack(side="left", padx=4)
        self.btn_read = ttk.Button(ops, text="读状态", command=self.on_read,
                                   state="disabled", width=10)
        self.btn_read.pack(side="left", padx=4)

        self.var_monitor = tk.BooleanVar(value=False)
        ttk.Checkbutton(ops, text="连续监视", variable=self.var_monitor).pack(side="left", padx=(14, 2))
        self.cmb_mon = ttk.Combobox(ops, state="readonly", width=5, values=MONITOR_CHOICES)
        self.cmb_mon.set("500")
        self.cmb_mon.pack(side="left")
        ttk.Label(ops, text="ms").pack(side="left", padx=(2, 10))

        self.var_frames = tk.BooleanVar(value=False)
        ttk.Checkbutton(ops, text="显示收发报文", variable=self.var_frames).pack(side="left")
        ttk.Button(ops, text="清空日志", command=self.on_clear_log).pack(side="left", padx=8)

        # --- 中部与日志放进可拖动的分隔窗：屏幕小的时候用户自己分配空间
        split = ttk.Panedwindow(self.root, orient="vertical")
        split.pack(fill="both", expand=True, **pad)

        mid = ttk.Frame(split)
        mid.columnconfigure(0, weight=5)
        mid.columnconfigure(1, weight=4)
        mid.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(mid, text="A/B 相机深度数据（来自 40003 / 40004）",
                              padding=(10, 6))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.depth = {}
        cards = ttk.Frame(left)
        cards.pack(fill="x")
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        for col, (key, title, reg) in enumerate(
                (("A", "A 面相机深度", 40003), ("B", "B 面相机深度", 40004))):
            card = ttk.LabelFrame(cards, text=f"{title}（{reg}）", padding=(8, 4))
            card.grid(row=0, column=col, sticky="nsew",
                      padx=(0, 6) if col == 0 else (6, 0))
            val = tk.Label(card, text="—", font=BIG_FONT, fg=C_IDLE, anchor="w")
            val.pack(fill="x")
            raw = tk.Label(card, text="原始值 —", font=MONO_FONT, fg="#666", anchor="w")
            raw.pack(fill="x")
            upd = tk.Label(card, text="未读取", font=MONO_FONT, fg="#666", anchor="w")
            upd.pack(fill="x")
            self.depth[key] = {"val": val, "raw": raw, "upd": upd}

        self.lbl_dev = tk.Label(left, text="设备状态：—", anchor="w", font=UI_FONT, fg=C_IDLE)
        self.lbl_dev.pack(fill="x", pady=(8, 0))
        self.lbl_cycle = tk.Label(left, text="端到端耗时：—", anchor="w", font=UI_FONT)
        self.lbl_cycle.pack(fill="x")
        self.lbl_note = tk.Label(left, text="", anchor="w", justify="left", fg=C_WARN,
                                 wraplength=560, font=UI_FONT)
        self.lbl_note.pack(fill="x", pady=(4, 0))

        right = ttk.LabelFrame(mid, text="寄存器表（实时）", padding=(6, 4))
        right.grid(row=0, column=1, sticky="nsew")
        cols = ("doc", "wire", "name", "rw", "value", "decoded")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=5)
        for c, t, w in (("doc", "文档", 58), ("wire", "线上", 48),
                        ("name", "名称", 78), ("rw", "读写", 40),
                        ("value", "值", 52), ("decoded", "解码", 96)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, minwidth=36, stretch=(c in ("name", "decoded")),
                             anchor="center" if c != "name" else "w")
        self.tree.pack(fill="both", expand=True)
        for idx in range(mb.REG_COUNT):
            self.tree.insert("", "end", iid=str(idx),
                             values=(40001 + idx, "—", SHORT_NAMES.get(idx, "?"),
                                     "R" if idx in mb.REG_READONLY else "W", "—", "—"))

        # --- 日志
        logf = ttk.LabelFrame(split, text="日志（双击拖动中间的分隔条可调整高度）",
                              padding=(6, 4))
        holder = ttk.Frame(logf)
        holder.pack(fill="both", expand=True)
        self.txt = tk.Text(holder, wrap="none", font=MONO_FONT, height=8,
                           state="disabled", bg="#fbfbfb")
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)
        for tag, color in LOG_COLORS.items():
            self.txt.tag_configure(tag, foreground=color)

        # 加进分隔窗（必须在两个子窗都建好之后）
        split.add(mid, weight=2)
        split.add(logf, weight=3)      # 富余的高度多给日志一点

        # --- 底部统计
        self.lbl_stats = tk.Label(self.root, text="请求 0　超时 0　CRC 错 0　异常 0　"
                                                  "末帧 — ms", anchor="w", fg="#555",
                                  font=MONO_FONT)
        self.lbl_stats.pack(fill="x", padx=10, pady=(0, 6))

    # -------------------------------------------------- 日志
    def _log(self, text, level="info"):
        self._append_log(f"{stamp()}  {text}", level)

    def _append_log(self, text, level="info"):
        at_end = self.txt.yview()[1] > 0.999
        self.txt.configure(state="normal")
        self.txt.insert("end", text + "\n", level)
        lines = int(self.txt.index("end-1c").split(".")[0])
        if lines > 3000:                       # 长时间跑不要吃内存
            self.txt.delete("1.0", f"{lines - 3000}.0")
        self.txt.configure(state="disabled")
        if at_end:
            self.txt.see("end")

    def on_clear_log(self):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")

    # -------------------------------------------------- 串口
    def _refresh_ports(self, log=True):
        if list_ports is None:
            ports = []
        else:
            ports = list(list_ports.comports())
        self.cmb_port.configure(values=[p.device for p in ports])
        if ports and not self.cmb_port.get():
            self.cmb_port.set(ports[0].device)
        if log:
            if ports:
                self._log("本机串口：" + "；".join(
                    f"{p.device} {p.description}" for p in ports), "info")
            else:
                self._log("没有发现串口（USB 转串口驱动装了吗？也可手填 COM 号）", "warn")

    def on_refresh_ports(self):
        self._refresh_ports(log=True)

    def on_selftest_toggle(self):
        on = self.var_selftest.get()
        state = "disabled" if on else "normal"
        for w in (self.cmb_port, self.cmb_baud, self.cmb_base):
            w.configure(state=state)
        self.ent_slave.configure(state=state)
        self._log("已切到自测模式：不打开串口，用内置虚拟从站按同样时序应答" if on
                  else "已退出自测模式", "warn" if on else "info")
        if self.connected:          # 换模式必须重连，否则还挂在原来的链路上
            self.link.post(self.link.do_disconnect)
            self._log("模式变了，已断开；请重新点【连接】", "warn")

    def on_connect(self):
        if self.connected:
            self.link.post(self.link.do_disconnect)
            return
        mode = "loop" if self.var_selftest.get() else "serial"
        try:
            slave = int(self.ent_slave.get())
            base = int(self.cmb_base.get())
        except ValueError:
            self._log("站号/地址基址必须是整数", "err")
            return
        baud = self.cmb_baud.get().strip() or "9600"
        port = self.cmb_port.get().strip()
        if mode == "serial" and not port:
            self._log("请先选或填一个串口（如 COM3）", "err")
            return
        self._log(f"正在打开 {'虚拟从站' if mode == 'loop' else port} …", "info")
        self.btn_conn.configure(state="disabled")
        self.link.post(lambda: self.link.do_connect(mode, port, baud, slave, base))

    # -------------------------------------------------- 按钮
    def on_trigger(self):
        if not self._can_operate():
            return
        self.link.post(self.link.do_trigger)

    def on_calibrate(self):
        if not self._can_operate():
            return
        self.link.post(self.link.do_calibrate)

    def on_read(self):
        if not self.connected:
            return
        self.link.post(self.link.do_read_all)

    def _can_operate(self):
        if not self.connected:
            self._log("请先连接串口", "warn")
            return False
        if self.seq_running:
            self._log("上一次时序还没结束，请稍等", "warn")
            return False
        return True

    # -------------------------------------------------- 主节拍
    def _tick(self):
        for _ in range(300):
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle(ev)
        self._update_buttons()
        self._update_stale()
        self._maybe_poll()
        self.root.after(60, self._tick)

    def _handle(self, ev):
        kind = ev[0]
        if kind == "log":
            level, text = ev[1], ev[2]
            if level in ("tx", "rx") and not self.var_frames.get():
                return
            self._append_log(text, level)
        elif kind == "conn":
            self._on_conn(ev[1])
        elif kind == "snapshot":
            self._on_snapshot(ev[1])
        elif kind == "stats":
            s = ev[1]
            self.lbl_stats.configure(
                text=f"请求 {s['req']}　超时 {s['timeout']}　CRC 错 {s['crc']}　"
                     f"异常 {s['exc']}　末帧 {s['last_ms']:.0f} ms")
        elif kind == "status":
            self._set_dev_status(ev[1])
        elif kind == "seq":
            self.seq_running = bool(ev[1])

    def _on_conn(self, d):
        self.btn_conn.configure(state="normal")
        if d.get("ok"):
            self.connected = True
            self.stale_marked = False
            self.btn_conn.configure(text="断开")
            self.lbl_conn.configure(
                text=f"已连接：{d['msg']}　站号 {d['slave']}　地址基址 {d['base']}"
                     f"（40001 → 线上 {d['base']}）", fg=C_OK)
            if self.var_selftest.get():
                self.lbl_conn.configure(fg=C_WARN)
            self._render_regs(self.last_vals)      # 线上地址列要跟着基址变
        else:
            self.connected = False
            self.seq_running = False
            self.btn_conn.configure(text="连接")
            self.lbl_conn.configure(text=d.get("msg", "未连接"), fg=C_IDLE)
            # 断开后必须把深度抹掉：留着旧值最容易被误当成新数据
            self._clear_depth("未连接：没有数据")
            self._render_regs(None)

    def _clear_depth(self, note):
        self.last_ts = 0.0
        self.stale_marked = False
        self.last_depth = None
        for key in ("A", "B"):
            self.depth[key]["val"].configure(text="—", fg=C_IDLE)
            self.depth[key]["raw"].configure(text="原始值 —")
            self.depth[key]["upd"].configure(text="未读取")
        self.lbl_dev.configure(text="设备状态：—", fg=C_IDLE)
        self.lbl_cycle.configure(text="端到端耗时：—")
        self.lbl_note.configure(text=note, fg=C_WARN)

    def _on_snapshot(self, d):
        vals, cycle, note = d["values"], d["cycle_ms"], d["note"]
        self.last_vals = vals
        self.last_ts = d["ts"]
        self.stale_marked = False
        self._render_regs(vals)
        st = vals[mb.REG_STATUS]
        self._set_dev_status(st)

        fault = (st == mb.ST_FAULT)
        for key, idx in (("A", mb.REG_DIST_A), ("B", mb.REG_DIST_B)):
            raw = vals[idx]
            mm = mb.decode_distance(raw)
            self.depth[key]["val"].configure(text=f"{mm:.1f} mm",
                                             fg=C_ERR if fault else C_OK)
            self.depth[key]["raw"].configure(text=f"原始值 {raw}（÷{mb.DIST_SCALE:g}）")
        self.last_depth = (mb.decode_distance(vals[mb.REG_DIST_A]),
                           mb.decode_distance(vals[mb.REG_DIST_B]))
        if cycle is not None:
            self.lbl_cycle.configure(text=f"端到端耗时：{cycle:.0f} ms"
                                          f"（写触发位→读到完成）")
        self.lbl_note.configure(text=note or "", fg=C_WARN)

    def _set_dev_status(self, st):
        color = {mb.ST_DONE: C_OK, mb.ST_FAULT: C_ERR,
                 mb.ST_MEASURING: C_TX}.get(st, C_IDLE)
        self.lbl_dev.configure(text=f"设备状态：{mb.STATUS_NAMES.get(st, '未知')}（{st}）",
                               fg=color)

    def _render_regs(self, vals):
        base = int(self.cmb_base.get() or 0)
        for idx in range(mb.REG_COUNT):
            v = "—" if vals is None else vals[idx]
            self.tree.item(str(idx), values=(
                40001 + idx, idx + base, SHORT_NAMES.get(idx, "?"),
                "R" if idx in mb.REG_READONLY else "W",
                v, "—" if vals is None else self._decode(idx, vals[idx])))

    @staticmethod
    def _decode(idx, v):
        if idx == mb.REG_STATUS:
            return mb.STATUS_NAMES.get(v, f"未知({v})")
        if idx in (mb.REG_DIST_A, mb.REG_DIST_B):
            return f"{mb.decode_distance(v):.1f} mm"
        return {0: "空闲", 1: "请求中"}.get(v, f"未定义({v})")

    def _update_buttons(self):
        free = self.connected and not self.seq_running
        state = "normal" if free else "disabled"
        for b in (self.btn_trigger, self.btn_cal):
            b.configure(state=state)
        self.btn_read.configure(state="normal" if self.connected else "disabled")

    def _update_stale(self):
        """刷新"多久之前读到的"，并在连续监视掉线时把深度标灰。

        一次性触发拿到的是一次测量的结果，放久了也不会变成错的，所以**只有勾了
        连续监视**才对"不新鲜"报警；否则只老实显示数据年龄。
        """
        if not self.connected or self.last_ts == 0.0:
            return
        age = time.time() - self.last_ts
        for key in ("A", "B"):
            self.depth[key]["upd"].configure(
                text="读取于 " + time.strftime("%H:%M:%S", time.localtime(self.last_ts))
                     + f"（{age:.0f} 秒前）")
        if not self.var_monitor.get() or age <= STALE_SECONDS or self.stale_marked:
            return
        self.stale_marked = True
        for key in ("A", "B"):
            self.depth[key]["val"].configure(fg=C_IDLE)
        self.lbl_note.configure(
            text=f"连续监视：已有 {age:.0f} 秒没读到新数据，显示的是上次的值", fg=C_WARN)

    def _maybe_poll(self):
        if not (self.var_monitor.get() and self.connected and not self.seq_running):
            return
        if not self.link.idle() or time.time() < self._next_poll_at:
            return
        try:
            interval = max(0.1, int(self.cmb_mon.get()) / 1000.0)
        except ValueError:
            interval = 0.5
        self._next_poll_at = time.time() + interval
        self.link.post(self.link.do_read_all_quiet)

    def _on_close(self):
        try:
            self.link.shutdown()
        except Exception:
            pass
        self.root.destroy()


# ====================================================================== 命令行

def run_gui():
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


def run_smoke(seconds=8.0):
    """起窗口 + 自测模式自动跑一次触发，用来验证界面接线。"""
    root = tk.Tk()
    app = App(root, auto_selftest=True)
    result = {"depth": None}

    def connect():
        app.on_connect()

    def trigger():
        app.on_trigger()

    def finish():
        result["depth"] = app.last_depth
        result["vals"] = app.last_vals
        result["note"] = app.lbl_note.cget("text")
        result["dev"] = app.lbl_dev.cget("text")
        root.destroy()

    root.after(200, connect)
    root.after(900, trigger)
    root.after(int(seconds * 1000), finish)
    root.mainloop()

    print("== GUI smoke ==")
    print("  设备状态 :", result["dev"])
    print("  寄存器   :", result["vals"])
    print("  A/B 深度 :", result["depth"])
    print("  提示     :", result["note"])
    ok = (result["depth"] is not None
          and 1700.0 < result["depth"][0] < 1900.0
          and 1700.0 < result["depth"][1] < 1900.0)
    print("  结果     :", "通过" if ok else "失败")
    return 0 if ok else 1


def run_selftest():
    """脱机自检协议层：不需要硬件、不开窗口。"""
    print("== plc_sim 主站协议自检 ==")
    fails = []

    def check(name, cond, extra=""):
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}{('  ' + str(extra)) if extra else ''}")
        if not cond:
            fails.append(name)

    if not HAS_SLAVE:
        print("  没找到 modbus_rtu.py，自测模式不可用")
        return 1

    events = queue.Queue()
    link = PlcLink(events, retries=1, timeout=0.4, seq_timeout=6.0)
    logs = []

    def drain():
        while True:
            try:
                ev = events.get_nowait()
            except queue.Empty:
                return
            if ev[0] == "log":
                logs.append(ev[2])

    # 1) 连接（虚拟从站）
    link.do_connect("loop", "COM_TEST", "9600", 1, 0)
    drain()
    check("自测模式连接成功", link.connected)

    # 2) 触发测量：走完 40002 → 40001=1 → 40001=2 → 读回深度
    t0 = time.monotonic()
    link.do_trigger()
    drain()
    cycle = (time.monotonic() - t0) * 1000
    st = link.peer.regs.get(mb.REG_STATUS)
    check("触发后设备状态 = 完成(2)", st == mb.ST_DONE, f"实际 {st}")
    check("40002 触发位被对方清零", link.peer.regs.get(mb.REG_TRIGGER) == 0)
    a_mm = mb.decode_distance(link.peer.regs.get(mb.REG_DIST_A))
    b_mm = mb.decode_distance(link.peer.regs.get(mb.REG_DIST_B))
    check("A/B 深度落在模拟区间内", 1700 < a_mm < 1900 and 1700 < b_mm < 1900,
          f"A={a_mm:.1f} B={b_mm:.1f} mm")
    check("端到端耗时接近虚拟从站的 busy 时间（>300ms）", cycle > 300, f"{cycle:.0f} ms")
    check("全程没有 CRC 错", link.stats["crc"] == 0)
    check("全程没有超时", link.stats["timeout"] == 0, link.stats)

    # 3) 读状态（不触发）
    before = link.stats["req"]
    link.do_read_all()
    drain()
    check("读状态发出过一次请求", link.stats["req"] == before + 1)
    check("读到的 A/B 距离与寄存器一致",
          abs(mb.decode_distance(link.peer.regs.get(mb.REG_DIST_A)) - a_mm) < 1e-9)

    # 4) 校准：不更新 40003/40004
    link.do_calibrate()
    drain()
    check("校准后状态 = 完成(2)", link.peer.regs.get(mb.REG_STATUS) == mb.ST_DONE)
    check("40005 校准位被对方清零", link.peer.regs.get(mb.REG_CALIBRATE) == 0)
    check("校准不改变 40003/40004",
          abs(mb.decode_distance(link.peer.regs.get(mb.REG_DIST_A)) - a_mm) < 1e-9)

    # 5) 异常响应（越界地址）应抛出可读错误，且计入 exc
    exc_ok = False
    try:
        link._write_reg(99, 1)
    except LinkError as e:
        exc_ok = "异常码 2" in str(e)
    check("越界地址 → 抛出「异常码 2」", exc_ok)
    check("异常响应已计数", link.stats["exc"] >= 1)

    # 6) 站号不匹配：从站不应答 → 超时（现场最容易踩的坑）
    link.slave_addr = 9
    link.timeout = 0.15
    t0 = time.monotonic()
    timeout_ok = False
    try:
        link._read_reg(mb.REG_STATUS)
    except LinkError as e:
        timeout_ok = "超时" in str(e) and (time.monotonic() - t0) < 2.0
    link.slave_addr = 1
    link.timeout = 0.4
    check("站号写错 → 超时报错（不会静默返回 0）", timeout_ok)

    # 7) CRC 错应被发现并重试，最后报错
    real_transport = link._transport

    class Corrupt:
        def read(self, n):
            data = real_transport.read(n)
            if data:
                b = bytearray(data)
                b[-1] ^= 0xFF               # 破坏 CRC
                return bytes(b)
            return data

        def write(self, d):
            return real_transport.write(d)

        def reset_input_buffer(self):
            real_transport.reset_input_buffer()

        def close(self):
            pass

    link._transport = Corrupt()
    crc_before = link.stats["crc"]
    crc_ok = False
    try:
        link._read_reg(mb.REG_STATUS)
    except LinkError as e:
        crc_ok = "CRC" in str(e)
    link._transport = real_transport
    check("CRC 错响应被识别", crc_ok and link.stats["crc"] > crc_before)

    # 8) 写触发的「响应」丢了：不能盲目重发，否则对方会多测一次
    link.do_disconnect(silent=True)
    link.do_connect("loop", "COM_TEST", "9600", 1, 0)
    drain()

    class DropWriteResponse:
        """第一次写请求的响应被丢掉，但帧其实已经送到从站并被执行了。"""

        def __init__(self, inner):
            self.inner, self.armed = inner, True

        def write(self, d):
            return self.inner.write(d)

        def read(self, n):
            if self.armed:
                self.armed = False
                self.inner.reset_input_buffer()
                return b""
            return self.inner.read(n)

        def reset_input_buffer(self):
            self.inner.reset_input_buffer()

        def close(self):
            pass

    real2 = link._transport
    link._transport = DropWriteResponse(real2)
    ok = True
    try:
        link._write_request_bit(mb.REG_TRIGGER, "40002")
    except LinkError as e:
        ok = False
        print("     抛出：", e)
    link._transport = real2
    drain()
    # 假从站只在收到帧时才推进状态，所以要一边读一边等它跑完
    for _ in range(200):
        if link.peer.regs.get(mb.REG_STATUS) == mb.ST_DONE and link.peer._n:
            break
        link.do_read_all(quiet=True)
        time.sleep(0.02)
    check("响应丢失时靠 40001 确认已受理，不重发", ok and link.peer._n == 1,
          f"测量次数 {link.peer._n}（应为 1）")

    # 9) 写触发真的没送到：查 40001 确认后应当重发
    link.do_disconnect(silent=True)
    link.do_connect("loop", "COM_TEST", "9600", 1, 0)
    drain()

    class DropFirstWrite:
        """第一次写请求根本没送到从站（帧丢了），响应自然也丢。"""

        def __init__(self, inner):
            self.inner, self.armed = inner, True

        def write(self, d):
            if self.armed:
                self.armed = False
                return len(d)
            return self.inner.write(d)

        def read(self, n):
            return self.inner.read(n)

        def reset_input_buffer(self):
            self.inner.reset_input_buffer()

        def close(self):
            pass

    real3 = link._transport
    link._transport = DropFirstWrite(real3)
    ok = True
    try:
        link._write_request_bit(mb.REG_TRIGGER, "40002")
    except LinkError as e:
        ok = False
        print("     抛出：", e)
    link._transport = real3
    drain()
    for _ in range(200):
        if link.peer.regs.get(mb.REG_STATUS) == mb.ST_DONE and link.peer._n:
            break
        link.do_read_all(quiet=True)
        time.sleep(0.02)
    check("写确实丢了时应当重发一次并成功", ok and link.peer._n == 1,
          f"测量次数 {link.peer._n}（应为 1）")

    # 10) addr_base=1 的换算（组态软件把 40001 当线上地址 1 的情况）
    link.do_disconnect(silent=True)
    exc_before = link.stats["exc"]
    link.do_connect("loop", "COM_TEST", "9600", 1, 1)
    drain()
    link.do_read_all()
    drain()
    check("addr_base=1 时线上地址 = 下标+1，读得到数据",
          link.peer.regs.addr_base == 1 and link.stats["exc"] == exc_before,
          f"exc={link.stats['exc']}")
    link.do_trigger()
    drain()
    check("addr_base=1 下触发同样成功",
          link.peer.regs.get(mb.REG_STATUS) == mb.ST_DONE)

    # 11) 断开后应拒绝动作
    link.do_disconnect(silent=True)
    check("断开后 connected=False", not link.connected)
    refused = False
    try:
        link._read_reg(mb.REG_STATUS)
    except LinkError as e:
        refused = "未连接" in str(e)
    check("断开后读寄存器会被拒绝", refused)

    print("\n  日志尾部：")
    for line in logs[-8:]:
        print("   ", line)
    print(f"\n{'全部通过' if not fails else '失败项：' + '，'.join(fails)}")
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="PLC 端模拟器（Modbus RTU 主站）")
    ap.add_argument("--list-ports", action="store_true", help="列出本机串口后退出")
    ap.add_argument("--selftest", action="store_true", help="脱机协议自检（不开窗口）")
    ap.add_argument("--smoke", action="store_true", help="起窗口自测跑一次后退出")
    ap.add_argument("--smoke-seconds", type=float, default=8.0)
    args = ap.parse_args(argv)

    if args.list_ports:
        if list_ports is None:
            print("没有安装 pyserial")
            return 1
        for p in list_ports.comports():
            print(f"{p.device}\t{p.description}")
        return 0
    if args.selftest:
        return run_selftest()
    if args.smoke:
        return run_smoke(args.smoke_seconds)
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
