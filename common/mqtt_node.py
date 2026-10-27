# -*- coding: utf-8 -*-
"""轻量 MQTT 节点：连接管理、主题订阅、JSON 请求/响应。

设计目标：
- 服务进程只依赖本机 127.0.0.1:1883，broker 由现场提供；
- 请求方阻塞等待响应，响应方在独立线程里处理，互不拖住网络线程；
- 断线自动重连，重连后自动恢复订阅。
"""

import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeout

import paho.mqtt.client as mqtt

from . import topics


class MqttError(RuntimeError):
    pass


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"无法序列化的类型：{type(o)!r}")


def dumps(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                      default=_json_default).encode("utf-8")


def loads(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


class MqttNode:
    def __init__(self, name, host="127.0.0.1", port=1883, keepalive=30):
        self.name = name
        self.host = host
        self.port = int(port)
        self.keepalive = int(keepalive)
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"camdepth-{name}-{os.getpid()}-{uuid.uuid4().hex[:6]}",
            protocol=mqtt.MQTTv311)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.connected = threading.Event()
        self._subs = []
        self._extra_subs = set()
        self._pending = {}
        self._lock = threading.RLock()
        self._started = False
        self._last_connect_warn = 0.0

    # -------------------------------------------------- 生命周期
    def start(self, wait=5.0) -> bool:
        if self._started:
            return self.connected.is_set()
        self._started = True
        self.client.connect_async(self.host, self.port, self.keepalive)
        self.client.loop_start()
        return self.connected.wait(wait)

    def wait_connected(self, timeout=5.0) -> bool:
        return self.connected.wait(timeout)

    def stop(self):
        self._started = False
        try:
            self.client.disconnect()
        except Exception:
            pass
        try:
            self.client.loop_stop()
        except Exception:
            pass
        self.connected.clear()
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for fut, _ in pending:
            if not fut.done():
                fut.set_exception(MqttError("MQTT 节点已关闭"))

    # -------------------------------------------------- 订阅
    def subscribe(self, topic, callback):
        """callback(topic, payload_obj)。payload 不是 JSON 时 obj 为 None。"""
        with self._lock:
            self._subs.append((topic, callback))
        if self.connected.is_set():
            self.client.subscribe(topic, qos=1)
        return callback

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.connected.set()
            with self._lock:
                subs = list(self._subs)
                extras = set(self._extra_subs)
            for topic, _ in subs:
                client.subscribe(topic, qos=1)
            for topic in extras:
                client.subscribe(topic, qos=1)
            self.log("info", f"MQTT 已连接 {self.host}:{self.port}")
        else:
            self.connected.clear()
            now = time.monotonic()
            if now - self._last_connect_warn > 10:
                self._last_connect_warn = now
                self.log("warn", f"MQTT 连接失败：{reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.connected.clear()
        self.log("warn", f"MQTT 断开（{reason_code}），自动重连中…")

    def _on_message(self, client, userdata, msg):
        obj = loads(msg.payload)
        if isinstance(obj, dict):
            rid = obj.get("id")
            if rid is not None:
                with self._lock:
                    item = self._pending.get(rid)
                if item is not None and msg.topic == item[1]:
                    fut = item[0]
                    with self._lock:
                        self._pending.pop(rid, None)
                    if not fut.done():
                        fut.set_result(obj)
                    return
        with self._lock:
            subs = list(self._subs)
        for pattern, callback in subs:
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    callback(msg.topic, obj)
                except Exception as e:
                    self.log("err", f"消息处理异常 {msg.topic}: {e}")

    # -------------------------------------------------- 发布
    def publish_json(self, topic, obj, qos=0, retain=False):
        if not self.connected.is_set():
            return False
        try:
            info = self.client.publish(topic, dumps(obj), qos=qos, retain=retain)
            return info.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception:
            return False

    def publish_raw(self, topic, payload, qos=0, retain=False):
        if not self.connected.is_set():
            return False
        try:
            info = self.client.publish(topic, payload, qos=qos, retain=retain)
            return info.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception:
            return False

    # -------------------------------------------------- 请求 / 响应
    def request(self, req_topic, res_topic, action, data=None, timeout=3.0):
        if not isinstance(data, dict):
            data = {}
        rid = uuid.uuid4().hex
        payload = {"id": rid, "action": action, "ts": time.time()}
        payload.update(data)

        with self._lock:
            self._extra_subs.add(res_topic)
        if self.connected.is_set():
            self.client.subscribe(res_topic, qos=1)

        fut = Future()
        with self._lock:
            self._pending[rid] = (fut, res_topic)

        if not self.wait_connected(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise MqttError(f"MQTT 未连接（{self.host}:{self.port}）")
        if not self.publish_json(req_topic, payload, qos=1):
            with self._lock:
                self._pending.pop(rid, None)
            raise MqttError(f"请求发布失败：{req_topic}")
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            with self._lock:
                self._pending.pop(rid, None)
            raise MqttError(f"请求超时：{action}（{timeout:.1f}s）")

    def request_ok(self, req_topic, res_topic, action, data=None, timeout=3.0):
        resp = self.request(req_topic, res_topic, action, data, timeout)
        if not resp.get("ok"):
            raise MqttError(resp.get("error") or f"{action} 失败")
        return resp.get("data")

    def reply(self, res_topic, rid, ok, data=None, error=None):
        if not rid:
            return
        obj = {"id": rid, "ok": bool(ok), "ts": time.time()}
        if data is not None:
            obj["data"] = data
        if error:
            obj["error"] = str(error)
        self.publish_json(res_topic, obj, qos=1)

    # -------------------------------------------------- 日志
    def log(self, level, msg):
        line = f"[{self.name}] {msg}"
        print(line, flush=True)
        self.publish_json(topics.log_topic(self.name),
                          {"ts": time.time(), "service": self.name,
                           "level": level, "msg": str(msg)}, qos=0)
