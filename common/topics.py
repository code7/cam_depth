# -*- coding: utf-8 -*-
"""MQTT 主题与消息约定。

所有消息使用 UTF-8 JSON；请求/响应统一为::

    请求  {"id": "...", "action": "...", ...}
    响应  {"id": "...", "ok": true/false, "data": {...}, "error": "..."}

帧消息较大，使用 base64 包装 JPEG，避免 MQTT 二进制头信息问题。
"""

NS = "camdepth/v1"

CONFIG_REQ = f"{NS}/config/req"
CONFIG_RES = f"{NS}/config/res"
CONFIG_CHANGED = f"{NS}/config/changed"
CONFIG_READY = f"{NS}/config/ready"

CAMERA_REQ = f"{NS}/camera/req"
CAMERA_RES = f"{NS}/camera/res"
CAMERA_INVENTORY = f"{NS}/camera/inventory"
CAMERA_STATUS = f"{NS}/camera/{{panel}}/status"
CAMERA_FRAME = f"{NS}/camera/{{panel}}/frame"

MODBUS_REQ = f"{NS}/modbus/req"
MODBUS_RES = f"{NS}/modbus/res"
MODBUS_STATUS = f"{NS}/modbus/status"
MODBUS_REGISTERS = f"{NS}/modbus/registers"

LOG = f"{NS}/log/{{service}}"

PANELS = ("A", "B")


def camera_status(panel: str) -> str:
    return CAMERA_STATUS.format(panel=panel)


def camera_frame(panel: str) -> str:
    return CAMERA_FRAME.format(panel=panel)


def log_topic(service: str) -> str:
    return LOG.format(service=service)
