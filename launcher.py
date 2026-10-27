# -*- coding: utf-8 -*-
"""统一启动器：按顺序拉起配置、相机、Modbus、UI 四个进程。

也可只启动部分服务，例如：:

    python launcher.py --only config,camera --simulate
    python launcher.py --simulate
"""

import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODULES = {
    "config": "services.config_service",
    "camera": "services.camera_service",
    "modbus": "services.modbus_service",
    "ui": "frontend.ui_main",
}


def _popen(name, host, port, simulate, ui_smoke):
    module = MODULES[name]
    cmd = [sys.executable, "-u", "-m", module,
           "--mqtt-host", host, "--mqtt-port", str(port)]
    if name == "camera" and simulate:
        cmd.append("--simulate")
    if name == "ui" and ui_smoke:
        cmd.extend(["--smoke", str(ui_smoke)])
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(cmd, cwd=ROOT, env=env, creationflags=flags)


def _ask_stop(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="深度相机系统统一启动器")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--only", default="config,camera,modbus,ui",
                        help="逗号分隔：config,camera,modbus,ui")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--simulate", action="store_true",
                        help="相机服务使用合成深度图")
    parser.add_argument("--ui-smoke", type=int, default=0,
                        help="UI 运行 N 秒后自动退出（联调用）")
    args = parser.parse_args()

    order = [n.strip() for n in args.only.split(",") if n.strip()]
    if args.no_ui and "ui" in order:
        order.remove("ui")
    bad = [n for n in order if n not in MODULES]
    if bad:
        parser.error(f"未知服务：{', '.join(bad)}")

    procs = []
    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    for name in order:
        if stopping:
            break
        proc = _popen(name, args.mqtt_host, args.mqtt_port, args.simulate,
                      args.ui_smoke)
        procs.append((name, proc))
        print(f"[launcher] {name} 已启动 pid={proc.pid}", flush=True)
        if name in ("config", "camera"):
            time.sleep(0.6)

    if not procs:
        return 0

    exit_code = 0
    try:
        while not stopping:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[launcher] {name} 已退出 code={code}", flush=True)
                    exit_code = code
                    stopping = True
                    break
            time.sleep(0.5)
    finally:
        for name, proc in reversed(procs):
            _ask_stop(proc)
        deadline = time.time() + 5
        for name, proc in reversed(procs):
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.1, deadline - time.time()))
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        print("[launcher] 已全部停止", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
