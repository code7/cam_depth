# -*- coding: utf-8 -*-
"""配置服务：唯一持有加密配置文件 cam_config.enc 的进程。

其他进程（相机服务、UI）只能通过 MQTT 的 config/req 访问配置，不允许直接读写文件。
"""

import argparse
import copy
import os
import signal
import threading
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cam_config as cfgmod

from common import topics
from common.mqtt_node import MqttNode

DEFAULT_CONFIG = os.path.join(ROOT, "config", cfgmod.DEFAULT_FILE_NAME)


def _deep_merge(base, patch):
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return copy.deepcopy(patch)
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class ConfigService:
    def __init__(self, node, path):
        self.node = node
        self.path = path
        self.store = cfgmod.ConfigStore(path)
        self.file_exists = self.store.load()
        self._lock = threading.RLock()
        self._unlocked = []
        self._last_report = None
        self.node.subscribe(topics.CONFIG_REQ, self._on_request)
        self.node.log("info", f"配置文件：{path}（存在={self.file_exists}，"
                              f"段数={self.store.segment_count()}）")
        self._publish_ready()

    # -------------------------------------------------- 请求入口
    def _on_request(self, topic, obj):
        if not isinstance(obj, dict):
            return
        rid = obj.get("id")
        action = obj.get("action")
        if not action:
            self.node.reply(topics.CONFIG_RES, rid, False, error="缺少 action")
            return
        threading.Thread(target=self._handle, args=(rid, obj), daemon=True).start()

    def _handle(self, rid, req):
        action = req.get("action")
        source = str(req.get("source") or "unknown")
        try:
            with self._lock:
                if action == "ping":
                    data = {"pong": time.time()}
                elif action == "get_all":
                    data = self._snapshot()
                elif action == "get_camera":
                    serial = self._serial(req)
                    data = {"camera": copy.deepcopy(self.store.camera(serial))}
                elif action == "unlock":
                    data = self._unlock(req.get("serials") or [], req.get("force"))
                elif action == "set_camera":
                    data = self._set_camera(req, source)
                elif action == "set_app":
                    data = self._set_app(req, source)
                elif action == "save":
                    data = self._save(source)
                elif action == "status":
                    data = self._snapshot(include_cameras=False)
                else:
                    raise ValueError(f"未知 action：{action}")
            self.node.reply(topics.CONFIG_RES, rid, True, data)
        except Exception as e:
            self.node.log("err", f"配置请求失败（{action}）：{e}")
            self.node.reply(topics.CONFIG_RES, rid, False, error=str(e))

    # -------------------------------------------------- 动作
    @staticmethod
    def _serial(req):
        serial = str(req.get("serial") or "").strip()
        if not serial:
            raise ValueError("缺少 serial")
        return serial

    def _snapshot(self, include_cameras=True):
        data = {
            "file_exists": self.file_exists,
            "path": self.path,
            "segments": self.store.segment_count(),
            "locked": self.store.locked,
            "dirty": bool(self.store.dirty),
            "warnings": list(self.store.warnings),
            "app": copy.deepcopy(self.store.app),
            "unlocked": list(self._unlocked),
            "report": copy.deepcopy(self._last_report),
            "ts": time.time(),
        }
        if include_cameras:
            data["cameras"] = {s: copy.deepcopy(c) for s, c in self.store.cameras.items()}
        return data

    def _unlock(self, serials, force=False):
        serials = [str(s) for s in serials if str(s)]
        with self._lock:
            if set(serials) == set(self._unlocked) and not force:
                return self._snapshot()
            old_cameras = {s: copy.deepcopy(self.store.cameras[s])
                           for s in serials if s in self.store.cameras}
            old_seg = dict(self.store.segment_of)
            self.store.reset_decoded()
            rep = self.store.unlock(serials)
            for serial, cfg in old_cameras.items():
                self.store.cameras.setdefault(serial, cfg)
                if serial in old_seg and serial not in self.store.segment_of:
                    self.store.segment_of[serial] = old_seg[serial]
            for serial in serials:
                self.store.camera(serial)
            self._unlocked = serials
            self._last_report = rep
            self.node.log(
                "info",
                f"配置解锁：成功 {len(rep['unlocked'])}，缺配置 {len(rep['missing'])}，"
                f"未授权 {len(rep['rejected'])}，失败 {len(rep['failed'])}")
            self._publish_ready()
            self.node.publish_json(topics.CONFIG_CHANGED,
                                   {"source": "unlock", "serials": serials,
                                    "ts": time.time()}, qos=1)
            return self._snapshot()

    def _set_camera(self, req, source):
        serial = self._serial(req)
        if (self.store.authorized is not None
                and serial not in self.store.authorized):
            raise ValueError(f"未授权序列号，不能写入配置：{serial}")
        patch = req.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("patch 必须是对象")
        base = copy.deepcopy(self.store.camera(serial))
        merged = _deep_merge(base, patch)
        cfg, warnings = cfgmod.validate_camera_config(merged, serial)
        if cfg.get("panel") not in (None, 0, 1):
            cfg["panel"] = None
            warnings.append("panel 只支持 0/1/None，已改为 None")
        cfg["serial"] = serial
        self.store.cameras[serial] = cfg
        self.store.mark_dirty()
        if req.get("save", False):
            if not self.store.save():
                raise RuntimeError(self.store.warnings[-1]
                                   if self.store.warnings else "配置写入失败")
        if req.get("notify", True):
            self._notify_change(source, [serial])
        return {"camera": copy.deepcopy(cfg), "warnings": warnings,
                "saved": bool(req.get("save", False)),
                "snapshot": self._snapshot(include_cameras=False)}

    def _set_app(self, req, source):
        patch = req.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("patch 必须是对象")
        merged = _deep_merge(self.store.app, patch)

        try:
            display = int(merged.get("display_interval_ms", 80))
        except (TypeError, ValueError):
            display = 80
        merged["display_interval_ms"] = max(10, min(5000, display))
        merged["auto_start_modbus"] = bool(merged.get("auto_start_modbus", True))

        m = dict(cfgmod.DEFAULT_MODBUS)
        m.update(merged.get("modbus") or {})
        try:
            m["baudrate"] = max(1200, min(4000000, int(m.get("baudrate", 9600))))
        except (TypeError, ValueError):
            m["baudrate"] = 9600
        try:
            m["slave_addr"] = max(1, min(247, int(m.get("slave_addr", 1))))
        except (TypeError, ValueError):
            m["slave_addr"] = 1
        try:
            m["addr_base"] = 1 if int(m.get("addr_base", 0)) else 0
        except (TypeError, ValueError):
            m["addr_base"] = 0
        try:
            m["timeout_s"] = max(0.2, min(60.0, float(m.get("timeout_s", 3.0))))
        except (TypeError, ValueError):
            m["timeout_s"] = 3.0
        try:
            m["measure_delay_s"] = max(
                0.0, min(60.0, float(m.get("measure_delay_s", 2.0))))
        except (TypeError, ValueError):
            m["measure_delay_s"] = 2.0
        m["port"] = str(m.get("port") or "").strip()
        merged["modbus"] = m
        self.store.app = merged
        self.store.mark_dirty()
        if req.get("save", False):
            if not self.store.save():
                raise RuntimeError(self.store.warnings[-1]
                                   if self.store.warnings else "配置写入失败")
        if req.get("notify", True):
            self._notify_change(source, [])
        return {"app": copy.deepcopy(merged), "saved": bool(req.get("save", False)),
                "snapshot": self._snapshot(include_cameras=False)}

    def _save(self, source):
        ok = self.store.save()
        if not ok:
            raise RuntimeError(self.store.warnings[-1]
                               if self.store.warnings else "配置写入失败")
        self.file_exists = True
        self._publish_ready()
        self._notify_change(source, list(self.store.cameras))
        return {"saved": True, "snapshot": self._snapshot(include_cameras=False)}

    def _notify_change(self, source, serials):
        self.node.publish_json(
            topics.CONFIG_CHANGED,
            {"source": source, "serials": list(serials), "ts": time.time()},
            qos=1)
        self.node.publish_json(topics.CONFIG_READY, self._ready_payload(),
                               qos=1, retain=True)

    def _publish_ready(self):
        self.node.publish_json(topics.CONFIG_READY, self._ready_payload(),
                               qos=1, retain=True)

    def _ready_payload(self):
        return {"online": True, "file_exists": self.file_exists,
                "segments": self.store.segment_count(),
                "locked": self.store.locked, "dirty": bool(self.store.dirty),
                "unlocked": list(self._unlocked), "ts": time.time()}

    def stop(self):
        with self._lock:
            try:
                if self.store.dirty:
                    self.store.save()
            except Exception as e:
                self.node.log("err", f"退出保存配置失败：{e}")
        self.node.publish_json(topics.CONFIG_READY, {"online": False,
                                                     "ts": time.time()},
                               qos=1, retain=True)


def main():
    parser = argparse.ArgumentParser(description="深度相机配置服务")
    parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--mqtt-port", type=int,
                        default=int(os.environ.get("MQTT_PORT", "1883")))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    node = MqttNode("config", args.mqtt_host, args.mqtt_port)
    node.start(wait=3.0)
    svc = ConfigService(node, args.config)

    stop = threading.Event()

    def _sig(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)
    try:
        while not stop.wait(1.0):
            pass
    finally:
        svc.stop()
        node.stop()


if __name__ == "__main__":
    main()
