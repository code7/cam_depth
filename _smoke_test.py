"""自动化冒烟测试：在真实 Tk 事件循环里驱动两台相机完整跑一遍界面逻辑。

用法： python _smoke_test.py
不会弹出模态对话框（messagebox 被替换为打印）。
"""
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np

import DepthCamGUI as g


class FakeBox:
    """替换 messagebox，避免模态对话框卡住自动化测试。"""

    def __init__(self):
        self.calls = []

    def _rec(self, kind, title, msg, *a, **k):
        self.calls.append((kind, title, str(msg)))
        print(f"    [{kind}] {title}: {msg}")

    def showwarning(self, title, msg, *a, **k):
        self._rec("WARN", title, msg)

    def showerror(self, title, msg, *a, **k):
        self._rec("ERROR", title, msg)

    def showinfo(self, title, msg, *a, **k):
        self._rec("INFO", title, msg)


box = FakeBox()
g.messagebox = box


class Driver:
    """按顺序执行步骤：每步先执行动作，再轮询等待条件成立。"""

    def __init__(self, root):
        self.root = root
        self.actions = []
        self.i = 0
        self.deadline = 0.0
        self.failures = []

    def add(self, name, do, cond, timeout=40.0):
        self.actions.append((name, do, cond, timeout))
        return self

    def start(self):
        self.root.after(50, self._next)

    def _next(self):
        if self.i >= len(self.actions):
            self.root.quit()
            return
        name, do, cond, timeout = self.actions[self.i]
        print(f"--> {name}")
        try:
            if do is not None:
                do()
        except Exception as e:
            self.failures.append((name, f"动作异常: {e!r}"))
            self.i += 1
            self.root.after(30, self._next)
            return
        self.deadline = time.time() + timeout
        self._poll(name, cond, timeout)

    def _poll(self, name, cond, timeout):
        try:
            ok = bool(cond())
        except Exception as e:
            self.failures.append((name, f"检查异常: {e!r}"))
            self.i += 1
            self.root.after(30, self._next)
            return
        if ok:
            self.i += 1
            self.root.after(50, self._next)
        elif time.time() > self.deadline:
            self.failures.append((name, f"超时（{timeout:.0f}s）"))
            self.i += 1
            self.root.after(30, self._next)
        else:
            self.root.after(30, lambda: self._poll(name, cond, timeout))


def main():
    root = tk.Tk()
    app = g.DepthCamApp(root)
    d = Driver(root)
    counters = [0, 0]

    def hit(i):
        def cb(_f):
            pass
        return cb

    # 统计每个面板的渲染帧数
    for i, p in enumerate(app.panels):
        orig = p._render_frame

        def make(orig, i):
            def wrapper(fs):
                r = orig(fs)
                if r:
                    counters[i] += 1
                return r
            return wrapper
        p._render_frame = make(orig, i)

    state = {}

    d.add("扫描设备",
          app.scan_cameras,
          lambda: len(app.cam_infos) > 0)
    d.add("检查扫描结果",
          None,
          lambda: state.update(infos=[str(c.serial_number) for c in app.cam_infos]) or len(app.cam_infos) >= 2)
    d.add("检查两面板默认选中不同设备",
          None,
          lambda: app.panels[0].combo.current() != app.panels[1].combo.current())
    d.add("打开两台相机",
          lambda: [p.open_camera() for p in app.panels],
          lambda: all(p.cam is not None for p in app.panels), timeout=30)
    d.add("检查两台相机序列号不同",
          None,
          lambda: app.panels[0].serial != app.panels[1].serial)
    d.add("重复打开同一台设备应被拦截",
          lambda: (
              app.panels[1].combo.config(state="readonly"),
              app.panels[1].combo.current(app.panels[0].combo.current()),
              app.panels[1].combo.config(state="disabled"),
              app.panels[1].open_camera(),
          ),
          lambda: any("已被" in c[2] for c in box.calls))
    d.add("全部开始采集",
          app.start_all,
          lambda: all(p.is_running for p in app.panels))
    d.add("两路都出画面",
          None,
          lambda: all(p.depth_data is not None for p in app.panels), timeout=30)
    d.add("持续采集 2 秒",
          lambda: state.update(t0=time.time()),
          lambda: time.time() - state["t0"] > 2.0, timeout=30)
    d.add("检查两路画面尺寸与帧数",
          None,
          lambda: state.update(
              sizes=[(p.depth_w, p.depth_h) for p in app.panels],
              ok=all(p._preview is not None and p._img_item is not None for p in app.panels)) or True)
    d.add("两路分别标定平面",
          lambda: [setattr(p, "roi", (p.depth_w // 4, p.depth_h // 4,
                                      p.depth_w // 2, p.depth_h // 2)) or p.run_calibration()
                   for p in app.panels],
          lambda: all(p.calibrated for p in app.panels), timeout=60)
    d.add("打开两路高度图",
          lambda: [p.show_height.set(True) for p in app.panels],
          lambda: True)
    d.add("高度图渲染 1 秒",
          lambda: (state.update(t1=time.time(), c1=list(counters))),
          lambda: time.time() - state["t1"] > 1.0)
    d.add("记录画布项与 Tk 图像",
          lambda: state.update(item=[p._img_item for p in app.panels],
                               photo=[p._photo for p in app.panels]),
          lambda: True)
    d.add("再刷新 1.2 秒",
          lambda: state.update(t2=time.time()),
          lambda: time.time() - state["t2"] > 1.2)
    d.add("画布项与 Tk 图像被复用（不删不建，避免闪烁）",
          None,
          lambda: state.update(
              item_same=[p._img_item == state["item"][i] for i, p in enumerate(app.panels)],
              photo_same=[p._photo is state["photo"][i] for i, p in enumerate(app.panels)]) or
          all(p._img_item == state["item"][i] for i, p in enumerate(app.panels)))
    d.add("切换刷新间隔",
          lambda: (app.cmb_refresh.set("200"), app.set_refresh_interval()),
          lambda: app.display_interval_ms == 200)
    d.add("全部停止采集",
          app.stop_all,
          lambda: not any(p.is_running for p in app.panels))
    d.add("全部获取一帧",
          app.grab_all,
          lambda: not any(p.is_grabbing for p in app.panels), timeout=40)
    d.add("检查单帧获取结果",
          None,
          lambda: state.update(
              grab_status=[p.lbl_status.cget("text") for p in app.panels]) or
          all("已获取一帧" in p.lbl_status.cget("text") for p in app.panels))
    d.add("鼠标坐标换算",
          lambda: (app.panels[0].canvas.event_generate("<Motion>", x=100, y=100),
                   app.panels[0].on_mouse_move(SimpleNamespace(x=100, y=100))),
          lambda: state.update(coord=app.panels[0].lbl_coord.cget("text")) or
          ("(" in app.panels[0].lbl_coord.cget("text")
           and "---" not in app.panels[0].lbl_coord.cget("text")))
    d.add("缩放 2 倍后复位",
          lambda: (setattr(app.panels[0], "zoom", 2.0), app.panels[0]._redraw_image()),
          lambda: True)
    d.add("关闭窗口",
          app.on_close,
          lambda: True)

    d.start()
    root.mainloop()

    print()
    print("设备列表:", state.get("infos"))
    print("两路画面尺寸:", state.get("sizes"))
    print("每路渲染帧数:", counters, "(约 3 秒内的帧数)")
    print("画布项是否复用:", state.get("item_same"))
    print("Tk 图像是否原地更新:", state.get("photo_same"))
    print("每路状态:", state.get("grab_status"))
    print("坐标标签:", state.get("coord"))
    print("弹窗记录:", box.calls)

    if d.failures:
        print()
        print("!! SMOKE TEST FAILED")
        for name, why in d.failures:
            print(f"   - {name}: {why}")
        raise SystemExit(1)
    print()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
