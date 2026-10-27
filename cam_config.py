"""按相机 ID 绑定解密的配置文件。

设计要点
--------
1. 文件分「段」存储，每台相机一段，段之间互不影响。
2. 每段的密钥 = PBKDF2-HMAC-SHA256(该相机的序列号, 随机盐, 20 万轮)，
   所以必须把对应相机接上、从硬件读出序列号，才解得开这一段。
3. 文件里**不出现任何明文序列号，也不出现可逆的序列号指纹**。
   否则拿到文件的人能自己算出密钥，"绑定到硬件"就名存实亡了。
   代价是"这段属于哪台相机"只能靠**逐个试解密**确定 —— 两台相机最多 2×2 次尝试。
4. 解不开的段原样保留、原样回写。这样只插一台相机做维修时，
   不会把另一台相机的配置抹掉。
5. **只用标准库**。目标机（树莓派）可能离线，不希望为了配置文件再引入依赖。

加密构造
--------
标准库没有 AES，所以用标准构造自己拼：PBKDF2 派生密钥，HMAC-SHA256 做 PRF
的计数器模式产生密钥流，再用 HMAC-SHA256 做「先加密后认证」（encrypt-then-MAC）。
校验标签用 hmac.compare_digest 常量时间比较，且**先验签再解密**。
配置文件只有几 KB，纯 Python 的速度完全够用。

注意：这是"混淆级"保护 —— 能防住只有配置文件的人、能防手改，
但防不住同时拿到硬件和程序源码的人。要防伪造需另加非对称签名（见 PLAN 中的授权层）。
"""

import hashlib
import hmac
import json
import os
import secrets
import shutil
import struct
import time

# ----------------------------------------------------------------- 常量

MAGIC = b"DCFG"
FMT_VERSION = 1          # 文件容器格式版本
SCHEMA_VERSION = 1       # 配置内容版本

PBKDF2_ITERATIONS = 200_000
SALT_LEN = 16
NONCE_LEN = 16
KEY_LEN = 32
TAG_LEN = 32

# 只允许这两台相机作为解密种子。这里以后会由树莓派授权层替代/补充。
AUTHORIZED_SERIALS = ("207000187846", "207000187862")

DEFAULT_FILTER = {"min_mm": 500.0, "max_mm": 2000.0}
DEFAULT_DEPTH_SCALE_MM = 0.125
DEFAULT_DISPLAY_INTERVAL_MS = 80

# 切平算法的参数（按相机独立）
DEFAULT_CUT = {"a_max_mm2": 500.0, "gap_px": 3, "roi_mode": "auto", "roi": None}

# Modbus / RS-485（整机共用，不分相机）。
# 现场确认：9600 8N1、站号 1、树莓派上一般是 /dev/ttySC0
DEFAULT_MODBUS = {
    "enabled": False,
    "port": "/dev/ttySC0",
    "baudrate": 9600,
    "slave_addr": 1,
    "addr_base": 0,          # 40001 对应线上地址 0 还是 1，现场用日志确认
    "timeout_s": 3.0,
    "measure_delay_s": 2.0,  # PLC 触发后等待多久再取新帧（补偿图像链路延迟）
}

# 需要持久化、并在打开相机时重新下发的采集参数
ACQ_FEATURES = ("AcquisitionFrameRateEnable", "AcquisitionFrameRate")

DEFAULT_FILE_NAME = "cam_config.enc"


class ConfigError(Exception):
    """配置相关错误基类。"""


class ConfigAuthError(ConfigError):
    """解不开：序列号不对，或文件被改过。"""


class ConfigFormatError(ConfigError):
    """文件容器结构损坏。"""


# ----------------------------------------------------------------- 加解密

def _aad() -> bytes:
    """附加认证数据：把文件魔数和格式版本绑进标签。"""
    return MAGIC + bytes([FMT_VERSION])


def _derive_keys(password: str, salt: bytes):
    """由口令派生 (加密密钥, 认证密钥)。"""
    okm = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS,
        dklen=KEY_LEN * 2)
    return okm[:KEY_LEN], okm[KEY_LEN:]


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """HMAC-SHA256 计数器模式密钥流。"""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(key, b"ks" + nonce + struct.pack(">Q", counter),
                        hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def _xor(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def encrypt_segment(plaintext: bytes, password: str) -> bytes:
    """加密一段，返回 salt|nonce|len|密文|标签。"""
    salt = secrets.token_bytes(SALT_LEN)
    nonce = secrets.token_bytes(NONCE_LEN)
    enc_key, mac_key = _derive_keys(password, salt)
    cipher = _xor(plaintext, _keystream(enc_key, nonce, len(plaintext)))
    tag = hmac.new(mac_key, _aad() + salt + nonce + cipher,
                   hashlib.sha256).digest()
    return salt + nonce + struct.pack(">I", len(cipher)) + cipher + tag


def decrypt_segment(blob: bytes, password: str) -> bytes:
    """解密一段；口令不对或内容被改则抛 ConfigAuthError。"""
    if len(blob) < SALT_LEN + NONCE_LEN + 4 + TAG_LEN:
        raise ConfigFormatError("段长度不足，文件可能被截断")

    salt = blob[:SALT_LEN]
    nonce = blob[SALT_LEN:SALT_LEN + NONCE_LEN]
    pos = SALT_LEN + NONCE_LEN
    (ct_len,) = struct.unpack(">I", blob[pos:pos + 4])
    pos += 4

    if len(blob) != pos + ct_len + TAG_LEN:
        raise ConfigFormatError("段长度与内容不符，文件可能被截断")

    cipher = blob[pos:pos + ct_len]
    tag = blob[pos + ct_len:]

    enc_key, mac_key = _derive_keys(password, salt)
    expect = hmac.new(mac_key, _aad() + salt + nonce + cipher,
                      hashlib.sha256).digest()
    # 先验签，再解密：常量时间比较，避免时序侧信道
    if not hmac.compare_digest(expect, tag):
        raise ConfigAuthError("校验失败：口令不对或文件被篡改")

    return _xor(cipher, _keystream(enc_key, nonce, ct_len))


# ----------------------------------------------------------------- 容器

class ConfigFile:
    """只管二进制容器的读写，不关心内容语义。"""

    def __init__(self, segments=None):
        # 每个元素是一整段原始字节，未知段原样保留
        self.segments = list(segments or [])

    @classmethod
    def parse(cls, blob: bytes) -> "ConfigFile":
        if len(blob) < 8:
            raise ConfigFormatError("文件太短")
        if blob[:4] != MAGIC:
            raise ConfigFormatError("文件标识不匹配（不是本程序的配置文件）")
        version = blob[4]
        if version != FMT_VERSION:
            raise ConfigFormatError(
                f"容器格式版本不支持：文件为 {version}，程序支持 {FMT_VERSION}")
        count = blob[5]
        segments = []
        pos = 8
        for _ in range(count):
            if pos + SALT_LEN + NONCE_LEN + 4 > len(blob):
                raise ConfigFormatError("段头被截断")
            (ct_len,) = struct.unpack(">I", blob[pos + SALT_LEN + NONCE_LEN:
                                                 pos + SALT_LEN + NONCE_LEN + 4])
            total = SALT_LEN + NONCE_LEN + 4 + ct_len + TAG_LEN
            if pos + total > len(blob):
                raise ConfigFormatError("段内容被截断")
            segments.append(blob[pos:pos + total])
            pos += total
        if pos != len(blob):
            raise ConfigFormatError("文件尾部有多余数据")
        return cls(segments)

    def to_bytes(self) -> bytes:
        if len(self.segments) > 255:
            raise ConfigFormatError("段数超过 255")
        head = MAGIC + bytes([FMT_VERSION, len(self.segments)]) + b"\x00\x00"
        return head + b"".join(self.segments)


# ----------------------------------------------------------------- 默认值

def now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def default_app_config() -> dict:
    return {"display_interval_ms": DEFAULT_DISPLAY_INTERVAL_MS,
            "auto_start_modbus": True,
            "modbus": dict(DEFAULT_MODBUS)}


def default_camera_config(serial: str) -> dict:
    return {
        "serial": serial,
        "panel": None,                 # None = 由扫描结果决定；0/1 = 固定到左/右
        "auto_calibrate": True,        # 启动后自动用计算范围拟合基准面
        "model": "",
        "firmware": "",
        "calibration": {
            "type": "plane_xy",
            "coefficients": None,      # [a, b, c]，None 表示未标定
            "plane_center_mm": None,   # 标定时图像中心到基准面的距离
            "unit": "mm",
            "roi": None,
            "inlier_ratio": None,
            "calibrated_at": None,
        },
        "filter": dict(DEFAULT_FILTER),
        "depth": {"scale_mm": DEFAULT_DEPTH_SCALE_MM},
        "cut": dict(DEFAULT_CUT),
        "acquisition": {
            "features": {},            # {名字: {"type": ..., "value": ...}}
            "image_mode": None,        # {"width":..,"height":..,"pixel_format":..}
            "applied_at": None,
            "warnings": [],
        },
    }


def is_calibrated(cfg: dict) -> bool:
    co = (cfg.get("calibration") or {}).get("coefficients")
    return isinstance(co, (list, tuple)) and len(co) == 3


# ----------------------------------------------------------------- 校验

def _num(value, default, lo=None, hi=None):
    """把值收敛成有限浮点数；不合法则返回默认值并报告。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default, False
    if f != f or f in (float("inf"), float("-inf")):
        return default, False
    if lo is not None and f < lo:
        return default, False
    if hi is not None and f > hi:
        return default, False
    return f, True


def validate_camera_config(raw, serial: str):
    """把解出来的 dict 收敛成合法配置，返回 (配置, 告警列表)。

    现场配置文件可能被改坏、或者是老版本写的，这里一律"能救则救、
    救不了就用默认值并留下告警"，绝不让一条坏参数导致程序起不来。
    """
    cfg = default_camera_config(serial)
    warn = []
    if not isinstance(raw, dict):
        return cfg, ["配置内容不是对象，已全部使用默认值"]

    cfg["model"] = str(raw.get("model") or "")
    cfg["firmware"] = str(raw.get("firmware") or "")

    panel = raw.get("panel")
    if panel is None:
        cfg["panel"] = None
    elif isinstance(panel, int) and 0 <= panel < 8:
        cfg["panel"] = panel
    else:
        warn.append(f"panel 取值非法（{panel!r}），已忽略")

    auto_cal = raw.get("auto_calibrate", True)
    cfg["auto_calibrate"] = bool(auto_cal)

    # 标定
    cal = raw.get("calibration") or {}
    if isinstance(cal, dict):
        co = cal.get("coefficients")
        if co is None:
            pass
        elif (isinstance(co, (list, tuple)) and len(co) == 3):
            vals, ok = [], True
            for v in co:
                f, good = _num(v, 0.0, -1e6, 1e6)
                vals.append(f)
                ok = ok and good
            if ok:
                cfg["calibration"].update({
                    "type": str(cal.get("type") or "plane_xy"),
                    "coefficients": vals,
                    "unit": str(cal.get("unit") or "mm"),
                })
                center, ok_center = _num(cal.get("plane_center_mm"), None,
                                         -1e6, 1e6)
                cfg["calibration"]["plane_center_mm"] = (
                    center if ok_center else None)
                roi = cal.get("roi")
                if (isinstance(roi, (list, tuple)) and len(roi) == 4):
                    try:
                        cfg["calibration"]["roi"] = [int(v) for v in roi]
                    except (TypeError, ValueError):
                        warn.append("标定 roi 非法，已忽略")
                ratio, ok = _num(cal.get("inlier_ratio"), None, 0.0, 1.0)
                cfg["calibration"]["inlier_ratio"] = ratio if ok else None
                cfg["calibration"]["calibrated_at"] = cal.get("calibrated_at")
            else:
                warn.append("标定系数超出合理范围，已按未标定处理")
        else:
            warn.append("标定系数格式非法，已按未标定处理")

    # 距离过滤
    flt = raw.get("filter") or {}
    if isinstance(flt, dict):
        mn, ok1 = _num(flt.get("min_mm"), DEFAULT_FILTER["min_mm"], 1.0, 1e6)
        mx, ok2 = _num(flt.get("max_mm"), DEFAULT_FILTER["max_mm"], 1.0, 1e6)
        if not (ok1 and ok2) or mx <= mn:
            warn.append(f"距离过滤区间非法（{flt.get('min_mm')}~{flt.get('max_mm')}），已用默认值")
            mn, mx = DEFAULT_FILTER["min_mm"], DEFAULT_FILTER["max_mm"]
        cfg["filter"] = {"min_mm": mn, "max_mm": mx}

    # 深度换算
    dep = raw.get("depth") or {}
    if isinstance(dep, dict):
        sc, ok = _num(dep.get("scale_mm"), DEFAULT_DEPTH_SCALE_MM, 1e-6, 1000.0)
        if not ok:
            warn.append("深度换算系数非法，已用默认值")
        cfg["depth"] = {"scale_mm": sc}

    # 切平参数
    cut = raw.get("cut") or {}
    if isinstance(cut, dict):
        am, ok = _num(cut.get("a_max_mm2"), DEFAULT_CUT["a_max_mm2"], 1.0, 1e7)
        if not ok:
            warn.append("最大单坑面积非法，已用默认值")
        try:
            gp = int(cut.get("gap_px", DEFAULT_CUT["gap_px"]))
        except (TypeError, ValueError):
            gp = DEFAULT_CUT["gap_px"]
        gp = max(0, min(50, gp))
        mode = str(cut.get("roi_mode") or DEFAULT_CUT["roi_mode"])
        if mode not in ("manual", "auto"):
            warn.append(f"ROI 模式 {mode!r} 不识别，已改为 {DEFAULT_CUT['roi_mode']}")
            mode = DEFAULT_CUT["roi_mode"]
        roi = cut.get("roi")
        clean_roi = None
        if roi is not None:
            if isinstance(roi, (list, tuple)) and len(roi) == 4:
                try:
                    vals = [int(round(float(v))) for v in roi]
                    if vals[2] > vals[0] and vals[3] > vals[1] and min(vals) >= 0:
                        clean_roi = vals
                    else:
                        warn.append("计算范围 roi 非法，已忽略")
                except (TypeError, ValueError):
                    warn.append("计算范围 roi 非法，已忽略")
            else:
                warn.append("计算范围 roi 格式非法，已忽略")
        cfg["cut"] = {"a_max_mm2": am, "gap_px": gp, "roi_mode": mode, "roi": clean_roi}

    # 采集参数
    acq = raw.get("acquisition") or {}
    if isinstance(acq, dict):
        feats = acq.get("features")
        if isinstance(feats, dict):
            clean = {}
            for name, item in feats.items():
                if not isinstance(item, dict):
                    warn.append(f"采集参数 {name} 格式非法，已忽略")
                    continue
                if "value" not in item:
                    warn.append(f"采集参数 {name} 缺少 value，已忽略")
                    continue
                clean[str(name)] = {"type": str(item.get("type") or ""),
                                    "value": item["value"]}
            cfg["acquisition"]["features"] = clean
        im = acq.get("image_mode")
        if isinstance(im, dict) and "width" in im and "height" in im:
            try:
                cfg["acquisition"]["image_mode"] = {
                    "width": int(im["width"]),
                    "height": int(im["height"]),
                    "pixel_format": str(im.get("pixel_format") or ""),
                }
            except (TypeError, ValueError):
                warn.append("图像模式非法，已忽略")
        elif im is not None:
            warn.append("图像模式格式非法，已忽略")
        cfg["acquisition"]["applied_at"] = acq.get("applied_at")
        old_warn = acq.get("warnings")
        if isinstance(old_warn, list):
            cfg["acquisition"]["warnings"] = [str(w) for w in old_warn][:20]

    return cfg, warn


# ----------------------------------------------------------------- 存储

class ConfigStore:
    """配置文件的完整生命周期：读取容器 → 用相机 ID 解锁 → 修改 → 原子回写。"""

    def __init__(self, path: str, authorized=AUTHORIZED_SERIALS):
        self.path = path
        self.authorized = tuple(authorized) if authorized else None
        self.file = ConfigFile()
        self.cameras = {}          # serial -> 已解开的配置
        self.app = default_app_config()
        self.segment_of = {}       # serial -> 段索引，回写时原地替换
        self.locked = 0            # 解不开的段数
        self.warnings = []
        self.loaded = False
        self.dirty = False

    # -------------------------------------------------- 读取
    def load(self) -> bool:
        """只读容器，不解密（此时还不知道相机 ID）。返回文件是否存在。"""
        self.loaded = True
        self.warnings = []
        if not os.path.exists(self.path):
            self.file = ConfigFile()
            return False
        try:
            with open(self.path, "rb") as f:
                blob = f.read()
        except OSError as e:
            self.warnings.append(f"配置文件无法读取：{e}")
            self.file = ConfigFile()
            return False

        try:
            self.file = ConfigFile.parse(blob)
        except ConfigFormatError as e:
            self.warnings.append(f"配置文件损坏（{e}），已忽略并会重建")
            self.file = ConfigFile()
            return False
        return True

    def segment_count(self) -> int:
        return len(self.file.segments)

    def reset_decoded(self):
        """清空上次解锁的结果，容器本身原样保留（重新扫描时调用）。"""
        self.cameras = {}
        self.segment_of = {}
        self.locked = 0
        self.warnings = []

    # -------------------------------------------------- 解锁
    def unlock(self, serials) -> dict:
        """用当前接着的相机序列号逐段试解。

        返回 {"unlocked": [...], "rejected": [...], "missing": [...], "failed": [...]}
        """
        unlocked, rejected, failed = [], [], []
        pending = list(range(len(self.file.segments)))

        for serial in serials:
            serial = str(serial)
            if self.authorized is not None and serial not in self.authorized:
                rejected.append(serial)
                continue
            matched = None
            for idx in pending:
                try:
                    plain = decrypt_segment(self.file.segments[idx], serial)
                except ConfigAuthError:
                    continue
                except ConfigFormatError as e:
                    failed.append((serial, str(e)))
                    break
                matched = idx
                break

            if matched is None:
                continue
            try:
                raw = json.loads(plain.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                failed.append((serial, f"内容不是合法 JSON：{e}"))
                continue

            cfg, warn = validate_camera_config(raw, serial)
            self.cameras[serial] = cfg
            self.segment_of[serial] = matched
            pending.remove(matched)
            unlocked.append(serial)
            for w in warn:
                self.warnings.append(f"[{serial}] {w}")

            app = raw.get("app")
            if isinstance(app, dict):
                iv, ok = _num(app.get("display_interval_ms"),
                              DEFAULT_DISPLAY_INTERVAL_MS, 10, 5000)
                if ok:
                    self.app["display_interval_ms"] = int(iv)
                if "auto_start_modbus" in app:
                    self.app["auto_start_modbus"] = bool(app["auto_start_modbus"])
                mbus = app.get("modbus")
                if isinstance(mbus, dict):
                    m = dict(self.app.get("modbus") or DEFAULT_MODBUS)
                    m["enabled"] = bool(mbus.get("enabled", m["enabled"]))
                    if mbus.get("port"):
                        m["port"] = str(mbus["port"])
                    for key, lo, hi in (("baudrate", 1200, 4000000),
                                        ("slave_addr", 1, 247),
                                        ("addr_base", 0, 1),
                                        ("timeout_s", 0.2, 60.0),
                                        ("measure_delay_s", 0.0, 60.0)):
                        v, ok2 = _num(mbus.get(key), m[key], lo, hi)
                        if ok2:
                            m[key] = (v if key in ("timeout_s", "measure_delay_s")
                                      else int(v))
                    self.app["modbus"] = m

        self.locked = len(pending)
        missing = [s for s in serials
                   if str(s) not in self.cameras and str(s) not in rejected]
        return {"unlocked": unlocked, "rejected": rejected,
                "missing": missing, "failed": failed}

    # -------------------------------------------------- 访问
    def camera(self, serial: str) -> dict:
        """取某台相机的配置，没有就按默认值建一个（不自动存盘）。"""
        serial = str(serial)
        if serial not in self.cameras:
            self.cameras[serial] = default_camera_config(serial)
        return self.cameras[serial]

    def panelled(self):
        """返回 {panel: serial}，只含显式指定了面板的相机。"""
        out = {}
        for serial, cfg in self.cameras.items():
            p = cfg.get("panel")
            if isinstance(p, int):
                out[p] = serial
        return out

    def mark_dirty(self):
        self.dirty = True

    # -------------------------------------------------- 回写
    def build_payload(self, serial: str) -> bytes:
        cfg = self.camera(serial)
        payload = {"schema_version": SCHEMA_VERSION,
                   "app": dict(self.app),
                   "saved_at": now_str()}
        payload.update(cfg)
        return json.dumps(payload, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")

    def save(self) -> bool:
        """原子回写。解不开的段原样保留。"""
        segments = list(self.file.segments)
        # 用列表副本迭代：build_payload 内部会调用 camera()，不能边遍历边改字典
        for serial in list(self.cameras):
            blob = encrypt_segment(self.build_payload(serial), serial)
            idx = self.segment_of.get(serial)
            if idx is None or idx >= len(segments):
                self.segment_of[serial] = len(segments)
                segments.append(blob)
            else:
                segments[idx] = blob

        data = ConfigFile(segments).to_bytes()
        try:
            _atomic_write(self.path, data)
        except OSError as e:
            self.warnings.append(f"配置写入失败：{e}")
            return False

        self.file = ConfigFile(segments)
        self.dirty = False
        return True

    # -------------------------------------------------- 快照
    def apply_acquisition_from_camera(self, serial: str, features: dict,
                                      image_mode, warnings=None):
        """把相机上实际生效的采集参数记进配置（用于以后原样下发）。"""
        acq = self.camera(serial)["acquisition"]
        if features:
            acq["features"] = features
        if image_mode is not None:
            acq["image_mode"] = image_mode
        acq["applied_at"] = now_str()
        acq["warnings"] = list(warnings or [])[:20]
        self.mark_dirty()


def _atomic_write(path: str, data: bytes):
    """写入临时文件 → fsync → 备份旧文件 → 原子替换。

    刻意不用 tempfile.mkstemp：它建文件失败时会用随机名无限重试
    （Windows 上 TMP_MAX 是 2^31-1），一旦被安全软件/沙箱拦成"失败"，
    就会表现成整个程序卡死。这里用确定的名字，项目里只有一个进程写配置，
    不存在并发冲突。
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    tmp = path + ".new"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # 某些文件系统不支持 fsync，不影响正确性
            pass

    if os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except OSError:
            pass

    os.replace(tmp, path)


# ----------------------------------------------------------------- 自检

def _selftest():
    print("== cam_config 自检 ==")
    # 刻意在工作区（本文件所在目录）内建测试目录：真实的配置目录也在程序目录下，
    # 而且能避开某些受限环境下"临时目录子目录不可写"的问题。
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cfgtest_run")
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, DEFAULT_FILE_NAME)

    store = ConfigStore(path)
    assert store.load() is False, "文件不存在时应返回 False"

    cfg = store.camera("207000187846")
    cfg["calibration"].update({"coefficients": [0.001, -0.002, 1580.5],
                               "roi": [1, 2, 3, 4], "inlier_ratio": 0.93,
                               "calibrated_at": now_str()})
    cfg["filter"] = {"min_mm": 700.0, "max_mm": 1800.0}
    cfg["panel"] = 0
    store.camera("207000187862")["panel"] = 1
    store.app["display_interval_ms"] = 120
    store.apply_acquisition_from_camera(
        "207000187846",
        {"AcquisitionFrameRate": {"type": "Float64", "value": 1.0}},
        {"width": 2048, "height": 1536, "pixel_format": "Depth"})
    assert store.save(), "写入失败"

    raw = open(path, "rb").read()
    for sn in ("207000187846", "207000187862"):
        assert sn.encode() not in raw, f"文件里出现了明文序列号 {sn}！"
    print(f"  1) 写入 OK，{len(raw)} 字节，文件内无明文序列号")

    # 正确 ID 能解开
    s2 = ConfigStore(path)
    assert s2.load() is True
    res = s2.unlock(["207000187846", "207000187862"])
    assert sorted(res["unlocked"]) == ["207000187846", "207000187862"], res
    c = s2.camera("207000187846")
    assert c["calibration"]["coefficients"] == [0.001, -0.002, 1580.5]
    assert c["filter"]["min_mm"] == 700.0
    assert c["panel"] == 0
    assert s2.app["display_interval_ms"] == 120
    assert s2.camera("207000187862")["panel"] == 1
    assert c["acquisition"]["image_mode"]["width"] == 2048
    print("  2) 正确 ID 解锁 OK，全部字段往返一致")

    # 单台相机只解开自己那段，另一段保留
    s3 = ConfigStore(path)
    s3.load()
    res3 = s3.unlock(["207000187846"])
    assert res3["unlocked"] == ["207000187846"], res3
    assert s3.locked == 1
    s3.camera("207000187846")["filter"] = {"min_mm": 111.0, "max_mm": 999.0}
    assert s3.save()
    s4 = ConfigStore(path)
    s4.load()
    res4 = s4.unlock(["207000187846", "207000187862"])
    assert sorted(res4["unlocked"]) == ["207000187846", "207000187862"], res4
    assert s4.camera("207000187846")["filter"]["min_mm"] == 111.0
    assert s4.camera("207000187862")["panel"] == 1, "未知段没有被保留"
    print("  3) 只插一台时 OK：另一台那段被原样保留，未被抹掉")

    # 错误 ID 解不开
    s5 = ConfigStore(path)
    s5.load()
    res5 = s5.unlock(["999999999999"])
    assert res5["unlocked"] == [], res5
    assert s5.locked == 2
    print("  4) 未授权 ID 解锁 OK：一段都解不开")

    # 白名单拦截
    s6 = ConfigStore(path)
    s6.load()
    res6 = s6.unlock(["111111111111"])
    assert res6["rejected"] == ["111111111111"], res6
    print("  5) 白名单 OK：非授权 ID 直接被拒")

    # 关掉白名单，验证真正拦住它的是密钥本身而不是白名单
    s5b = ConfigStore(path, authorized=None)
    s5b.load()
    res5b = s5b.unlock(["123456789012", "999999999999"])
    assert res5b["unlocked"] == [], res5b
    assert s5b.locked == 2, s5b.locked
    print("  5b) 关掉白名单也解不开 OK：是密钥在拦，不是白名单在拦")

    # --- 篡改检测 ---
    original = open(path, "rb").read()
    cf = ConfigFile.parse(original)
    seg0_len = len(cf.segments[0])
    tag_start = 8 + seg0_len - TAG_LEN      # 第 1 段标签的起点
    ct_start = 8 + SALT_LEN + NONCE_LEN + 4  # 第 1 段密文正文起点

    def tamper(offset, name, expect):
        b = bytearray(original)
        b[offset] ^= 0x01
        bad = os.path.join(tmpdir, name)
        with open(bad, "wb") as f:
            f.write(bytes(b))
        s = ConfigStore(bad, authorized=None)
        parsed = s.load()
        r = s.unlock(["207000187846"])
        # 核心不变量：改过之后一台都解不开
        assert r["unlocked"] == [], f"{name}: 篡改未被发现！{r}"
        if expect == "format":
            # 结构类篡改可能在解析阶段就被拦下，也可能落到解锁阶段的 failed
            assert (not parsed) or r["failed"], \
                f"{name}: 应报格式错误，实际 parsed={parsed} r={r}"
        where = "解析阶段" if not parsed else ("解锁阶段" if r["failed"] else "认证失败")
        print(f"  · {name} @偏移 {offset} 被拒绝（{where}）")

    tamper(tag_start, "tag_tamper.enc", "auth")          # 改认证标签
    tamper(ct_start + 2, "ct_tamper.enc", "auth")        # 改密文正文
    tamper(8 + SALT_LEN + NONCE_LEN, "len_tamper.enc", "format")  # 改长度字段
    tamper(8, "salt_tamper.enc", "auth")                 # 改盐
    print("  6) 篡改检测 OK：标签/正文/长度/盐 四种改动全部被拒")

    # 坏文件
    badf = os.path.join(tmpdir, "broken.enc")
    open(badf, "wb").write(b"NOTACONFIGFILE" * 3)
    s9 = ConfigStore(badf)
    assert s9.load() is False
    assert s9.warnings, "坏文件应留下告警"
    print("  8) 损坏文件 OK：不抛异常，留告警并重建")

    # 非法参数收敛
    s10 = ConfigStore(path)
    s10.load()
    s10.unlock(["207000187846"])
    cfg10, warn10 = validate_camera_config(
        {"filter": {"min_mm": 9999, "max_mm": 1},
         "calibration": {"coefficients": [1, 2]},
         "depth": {"scale_mm": -5},
         "panel": 77}, "X")
    assert cfg10["filter"]["min_mm"] == DEFAULT_FILTER["min_mm"], cfg10["filter"]
    assert cfg10["calibration"]["coefficients"] is None
    assert cfg10["depth"]["scale_mm"] == DEFAULT_DEPTH_SCALE_MM
    assert cfg10["panel"] is None
    assert len(warn10) >= 3, warn10
    print(f"  9) 非法参数收敛 OK：{len(warn10)} 条告警，全部回退默认值")

    # 性能
    import time as _t
    t0 = _t.perf_counter()
    encrypt_segment(b"x" * 2000, "207000187846")
    dt = (_t.perf_counter() - t0) * 1000
    print(f" 10) 单段加密耗时 {dt:.0f} ms（PBKDF2 {PBKDF2_ITERATIONS} 轮）")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\n全部自检通过")


if __name__ == "__main__":
    _selftest()
