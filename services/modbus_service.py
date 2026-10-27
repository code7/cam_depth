# -*- coding: utf-8 -*-
"""Modbus 服务：独占 RS-485 串口，作为 RTU 从站与 PLC 握手。

测量/校准算法不在本进程，收到 PLC 触发后通过 MQTT 请求相机服务，
再把结果写入保持寄存器。配置（串口、站号、超时）来自配置服务。
"""

import argparse
import copy
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cam_config as cfgmod
import modbus_rtu as mb
from common import topics
from common.mqtt_node import MqttNode

CONFIG_RETRY_S = 5.0
MODBUS_RETRY_S = 15.0
REGISTER_PERIOD_S = 0.25
STATUS_PERIOD_S = 0.5
LOGIC_TAG = "relative-delta-v1"


class ModbusBridge:
    def __init__(self, node):
        self.node = node
        self.regs = mb.HoldingRegisters(addr_base=0)
        self.slave = None
        self.app = cfgmod.default_app_config()
        self.config_online = False
        self.last_error = "相机未就绪，等待 A/B 初始化"
        self.last_result = None
        self.cycle_ms = None
        self._connecting = False
        self._gen = 0
        self._need_restart = False
        self._reload_requested = True
        self._config_retry_at = 0.0
        self._modbus_retry_at = 0.0
        self._fail_count = 0
        self._manual_off = False
        self._busy = False
        self._busy_lock = threading.RLock()
        self._last_register_pub = 0.0
        self._last_status_pub = 0.0
        self._stop = False
        self._camera_ready = {panel: False for panel in topics.PANELS}
        self._initial_status_done = False
        self.regs.set(mb.REG_STATUS, mb.ST_FAULT)

        node.subscribe(topics.MODBUS_REQ, self._on_request)
        node.subscribe(topics.CONFIG_CHANGED, self._on_config_changed)
        for panel in topics.PANELS:
            node.subscribe(topics.camera_status(panel), self._on_camera_status)
        node.log("warn", "相机未就绪，40001=3；等待 A/B 初始化")

    # -------------------------------------------------- 配置
    def _fetch_config(self):
        data = self.node.request_ok(topics.CONFIG_REQ, topics.CONFIG_RES,
                                    "get_all", {}, timeout=5.0)
        self.config_online = True
        self._apply_app(data.get("app"))
        return data

    def _apply_app(self, app):
        old = self.app.get("modbus") or {}
        old_auto = bool(self.app.get("auto_start_modbus", True))
        new_app = copy.deepcopy(app or cfgmod.default_app_config())
        self.app = new_app
        new = new_app.get("modbus") or {}
        new_auto = bool(new_app.get("auto_start_modbus", True))
        if new != old or new_auto != old_auto:
            self._need_restart = True
        if new_auto != old_auto:
            self._manual_off = False

    def _on_config_changed(self, topic, obj):
        if isinstance(obj, dict):
            self._reload_requested = True
            self._config_retry_at = 0.0

    def _on_camera_status(self, topic, obj):
        panel = obj.get("panel") if isinstance(obj, dict) else None
        if panel not in self._camera_ready:
            parts = str(topic).rsplit("/", 2)
            panel = parts[-2] if len(parts) >= 2 else None
        if panel not in self._camera_ready:
            return
        ready = bool(
            isinstance(obj, dict) and obj.get("ready")
            and obj.get("online", True) is not False)
        with self._busy_lock:
            changed = self._camera_ready[panel] != ready
            self._camera_ready[panel] = ready
            if changed:
                self.node.log(
                    "info" if ready else "warn",
                    f"{panel} 面相机{'已就绪' if ready else '未就绪'}")
            self._sync_initial_status_locked()

    def _sync_initial_status_locked(self):
        if self._initial_status_done or self._busy:
            return
        if all(self._camera_ready.values()):
            self._initial_status_done = True
            self.last_error = None
            self.regs.set(mb.REG_STATUS, mb.ST_IDLE)
            self.node.log("ok", "A/B 相机已就绪，40001=0（待机）")

    @property
    def modbus_cfg(self):
        return self.app.get("modbus") or cfgmod.DEFAULT_MODBUS

    @property
    def running(self):
        return self.slave is not None and getattr(self.slave, "_running", False)

    # -------------------------------------------------- RTU 串口
    def start_rtu(self, auto=False):
        if self._connecting or self.running:
            return
        cfg = self.modbus_cfg
        port = str(cfg.get("port") or "").strip()
        if not port:
            self._failed("485 端口未配置")
            return
        self._connecting = True
        self._gen += 1
        gen = self._gen
        args = (gen, port, int(cfg.get("baudrate", 9600)),
                int(cfg.get("slave_addr", 1)), int(cfg.get("addr_base", 0)))
        threading.Thread(target=self._start_worker, args=args, daemon=True).start()

    def _start_worker(self, gen, port, baud, addr, base):
        ok = False
        err = None
        slave = None
        try:
            self.regs.addr_base = base
            slave = mb.ModbusRTUSlave(
                self.regs, slave_addr=addr, port=port, baudrate=baud,
                log=lambda msg: self.node.log("info", msg))
            ok = bool(slave.start())
            if not ok:
                err = "打开串口失败"
        except Exception as e:
            err = str(e)
        if gen != self._gen:
            if slave is not None:
                slave.stop()
            return
        self._connecting = False
        if not ok:
            if slave is not None:
                slave.stop()
            self._failed(err or "打开串口失败")
            return
        self.slave = slave
        self._fail_count = 0
        self._modbus_retry_at = 0.0
        self.last_error = None
        self.node.log("ok", f"485 已连接：{port} @{baud} 站号 {addr} 基址 {base}")
        self._publish_status(force=True)

    def _failed(self, err):
        self._connecting = False
        self._fail_count += 1
        self.last_error = str(err)
        self._modbus_retry_at = time.monotonic() + MODBUS_RETRY_S
        if self._fail_count == 1 or self._fail_count % 10 == 0:
            self.node.log("warn", f"485 连接失败：{err}；"
                                  f"{int(MODBUS_RETRY_S)} 秒后自动重试")
        self._publish_status(force=True)

    def stop_rtu(self, why=""):
        self._gen += 1
        was = self.slave is not None
        slave, self.slave = self.slave, None
        if slave is not None:
            try:
                slave.stop()
            except Exception:
                pass
        self._connecting = False
        if was:
            self.node.log("warn", f"485 已断开（{why}）" if why else "485 已断开")
        self._publish_status(force=True)

    # -------------------------------------------------- 主循环
    def tick(self):
        now = time.monotonic()
        if (self.config_online and self._reload_requested
                and now >= self._config_retry_at):
            self._config_retry_at = now + CONFIG_RETRY_S
            self._reload_requested = False
            try:
                self._fetch_config()
            except Exception as e:
                self.config_online = False
                self._reload_requested = True
                self.node.log("warn", f"读取配置失败：{e}")
        elif not self.config_online and now >= self._config_retry_at:
            self._config_retry_at = now + CONFIG_RETRY_S
            try:
                self._fetch_config()
                self._reload_requested = False
                self.node.log("info", "配置服务已连接")
            except Exception:
                self.config_online = False

        if self._need_restart:
            self._need_restart = False
            should_run = (self.slave is not None
                          or (bool(self.app.get("auto_start_modbus", True))
                              and not self._manual_off))
            if self.slave is not None:
                self.stop_rtu("配置已更新")
            if should_run:
                self._modbus_retry_at = 0.0

        if (self.config_online and self.app.get("auto_start_modbus", True)
                and not self._manual_off and self.slave is None
                and not self._connecting and now >= self._modbus_retry_at):
            self.start_rtu(auto=True)

        self.poll()
        if now - self._last_register_pub >= REGISTER_PERIOD_S:
            self._last_register_pub = now
            self._publish_registers()
        if now - self._last_status_pub >= STATUS_PERIOD_S:
            self._publish_status()

    def poll(self):
        if not self.running:
            return
        with self._busy_lock:
            if self._busy:
                return
            if self.regs.get(mb.REG_TRIGGER) == 1:
                self.regs.set(mb.REG_TRIGGER, 0)
                self._begin("measure")
            elif self.regs.get(mb.REG_CALIBRATE) == 1:
                self.regs.set(mb.REG_CALIBRATE, 0)
                self._begin("calibrate")

    def _begin(self, what):
        rejected = False
        with self._busy_lock:
            if not all(self._camera_ready.values()):
                self._fault("相机未就绪，无法开始任务")
                rejected = True
            else:
                self._initial_status_done = True
                self._busy = True
        if rejected:
            self._publish_status(force=True)
            return
        self.regs.set(mb.REG_STATUS, mb.ST_MEASURING)
        self._publish_status(force=True)
        delay = max(0.0, float(self.modbus_cfg.get("measure_delay_s", 2.0)))
        suffix = f"，延迟 {delay:.1f}s 取帧" if delay > 0 else ""
        self.node.log(
            "info", f"[Modbus] 收到{'测量' if what == 'measure' else '校准'}"
                    f"请求{suffix}")
        threading.Thread(target=self._run, args=(what,), daemon=True).start()

    def _run(self, what):
        t0 = time.monotonic()
        try:
            timeout = float(self.modbus_cfg.get("timeout_s", 3.0))
            delay = max(0.0, float(self.modbus_cfg.get("measure_delay_s", 2.0)))
            if what == "measure":
                data = self.node.request_ok(
                    topics.CAMERA_REQ, topics.CAMERA_RES, "measure",
                    {"timeout_s": timeout, "delay_s": delay},
                    timeout=timeout + delay + 5.0)
                da = float(data["A"])
                db = float(data["B"])
                for name, value in (("A面", da), ("B面", db)):
                    if abs(value) > mb.DIST_ABS_MAX_MM:
                        raise RuntimeError(
                            f"{name}相对距离 {value:.1f} mm 超过 "
                            f"{mb.DIST_ABS_MAX_MM / 1000.0:.0f}m，数据无效")
                    if not (mb.DIST_MIN_MM <= value <= mb.DIST_MAX_MM):
                        raise RuntimeError(
                            f"{name}相对距离 {value:.1f} mm 超出寄存器量程 "
                            f"({mb.DIST_MIN_MM:.1f}~{mb.DIST_MAX_MM:.1f} mm)")
                self.regs.set(mb.REG_DIST_A, mb.encode_distance(da))
                self.regs.set(mb.REG_DIST_B, mb.encode_distance(db))
                self.regs.set(mb.REG_STATUS, mb.ST_DONE)
                self.cycle_ms = (time.monotonic() - t0) * 1000.0
                self.last_result = (da, db)
                self.last_error = None
                timing = ""
                if data.get("wait_ms") is not None:
                    timing = (
                        f"，等待取帧 {float(data['wait_ms']):.0f} ms"
                        f"，计算 {float(data.get('compute_ms', 0.0)):.0f} ms")
                self.node.log(
                    "ok", f"[Modbus] 测量完成：A面相对 {da:.1f} mm，"
                          f"B面相对 {db:.1f} mm，周期 {self.cycle_ms:.0f} ms"
                          f"{timing}")
            else:
                data = self.node.request_ok(
                    topics.CAMERA_REQ, topics.CAMERA_RES, "calibrate",
                    {"persist": True, "timeout_s": timeout, "delay_s": delay},
                    timeout=timeout + delay + 10.0)
                self.regs.set(mb.REG_STATUS, mb.ST_DONE)
                self.last_error = None
                self.cycle_ms = (time.monotonic() - t0) * 1000.0
                self.node.log("ok", f"[Modbus] 基准面已重新拟合：{data.get('planes')}")
        except Exception as e:
            self._fault(str(e))
        finally:
            with self._busy_lock:
                self._busy = False
            self._publish_status(force=True)

    def _fault(self, msg):
        self.regs.set(mb.REG_STATUS, mb.ST_FAULT)
        self.last_error = str(msg)
        self.node.log("err", f"[Modbus] 故障：{msg}")

    # -------------------------------------------------- 发布
    def _registers_payload(self):
        snap = self.regs.snapshot()
        stats = dict(self.slave.stats) if self.slave is not None else {}
        with self._busy_lock:
            camera_ready = dict(self._camera_ready)
            initial_status_done = self._initial_status_done
        return {
            "ts": time.time(),
            "running": bool(self.running),
            "connecting": bool(self._connecting),
            "port": self.modbus_cfg.get("port"),
            "baudrate": self.modbus_cfg.get("baudrate"),
            "slave_addr": self.modbus_cfg.get("slave_addr"),
            "addr_base": self.regs.addr_base,
            "timeout_s": self.modbus_cfg.get("timeout_s"),
            "measure_delay_s": self.modbus_cfg.get("measure_delay_s"),
            "registers": snap,
            "names": {str(k): v for k, v in mb.REG_NAMES.items()},
            "readonly": list(mb.REG_READONLY),
            "write_log": [list(x) for x in self.regs.write_log],
            "stats": stats,
            "last_result": list(self.last_result) if self.last_result else None,
            "last_error": self.last_error,
            "cycle_ms": self.cycle_ms,
            "camera_ready": camera_ready,
            "camera_ready_all": all(camera_ready.values()),
            "initial_status_done": initial_status_done,
            "status_code": snap[mb.REG_STATUS] if snap else 0,
            "status_name": mb.STATUS_NAMES.get(snap[mb.REG_STATUS], "?") if snap else "?",
            "dist_scale": mb.DIST_SCALE,
            "dist_signed": mb.DIST_SIGNED,
            "dist_min": mb.DIST_MIN_MM,
            "dist_max": mb.DIST_MAX_MM,
            "dist_abs_max": mb.DIST_ABS_MAX_MM,
        }

    def _publish_registers(self):
        self.node.publish_json(topics.MODBUS_REGISTERS, self._registers_payload(),
                               qos=0, retain=False)

    def _publish_status(self, force=False):
        payload = self._registers_payload()
        payload["online"] = True
        self._last_status_pub = time.monotonic()
        self.node.publish_json(topics.MODBUS_STATUS, payload, qos=1, retain=True)

    def _on_request(self, topic, obj):
        if not isinstance(obj, dict):
            return
        rid = obj.get("id")
        action = obj.get("action")
        threading.Thread(target=self._handle, args=(rid, action), daemon=True).start()

    def _handle(self, rid, action):
        try:
            if action == "ping":
                data = {"pong": time.time()}
            elif action == "state":
                data = self._registers_payload()
            elif action == "connect":
                self._manual_off = False
                self._modbus_retry_at = 0.0
                self.start_rtu()
                data = {"connecting": True}
            elif action == "disconnect":
                self._manual_off = True
                self.stop_rtu("手动断开")
                data = {"running": False}
            elif action == "reconnect":
                self._manual_off = False
                self.stop_rtu("手动重连")
                self._modbus_retry_at = 0.0
                self.start_rtu()
                data = {"connecting": True}
            else:
                raise ValueError(f"未知 action：{action}")
            self.node.reply(topics.MODBUS_RES, rid, True, data)
        except Exception as e:
            self.node.reply(topics.MODBUS_RES, rid, False, error=str(e))

    def stop(self):
        self._stop = True
        self.stop_rtu("服务退出")
        self.node.publish_json(topics.MODBUS_STATUS,
                               {"online": False, "ts": time.time()},
                               qos=1, retain=True)


def main():
    parser = argparse.ArgumentParser(description="Modbus RTU 从站服务")
    parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--mqtt-port", type=int,
                        default=int(os.environ.get("MQTT_PORT", "1883")))
    args = parser.parse_args()

    node = MqttNode("modbus", args.mqtt_host, args.mqtt_port)
    node.start(wait=3.0)
    svc = ModbusBridge(node)
    node.log("info", f"Modbus 服务启动（输出相对校准面 {LOGIC_TAG}）")

    stop = threading.Event()

    def _sig(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)
    try:
        while not stop.is_set():
            svc.tick()
            time.sleep(0.02)
    finally:
        svc.stop()
        node.stop()


if __name__ == "__main__":
    main()
