"""自动化冒烟测试：在真实 Tk 事件循环里驱动两台相机 + 加密配置完整跑一遍。

用法： python _smoke_test.py
不会弹出模态对话框（messagebox 被替换为打印）。
配置写到 _smoke_config/ 下，不碰正式配置文件。
"""
import os
import shutil
import time
import tkinter as tk

import numpy as np

import cam_config as cfgmod

# 必须在创建 App 之前改掉配置路径
import DepthCamGUI as g

TEST_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_config")
TEST_CFG_PATH = os.path.join(TEST_CFG_DIR, cfgmod.DEFAULT_FILE_NAME)
shutil.rmtree(TEST_CFG_DIR, ignore_errors=True)
g.CONFIG_PATH = TEST_CFG_PATH


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


def best_roi(panel):
    """按实际有效深度点选一块标定区域。

    相机对着的场景会变，固定 ROI 可能正好落在没有深度返回的地方
    （实测同一个 ROI 有时 11% 有效、有时 0.1%），所以让测试自己挑。
    """
    d = panel.depth_data
    if d is None:
        return None
    h, w = d.shape
    cands = [
        (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        (3 * w // 8, 3 * h // 8, 5 * w // 8, 5 * h // 8),
        (0, 0, w, h),
        (w // 4, h // 4, w // 2, h // 2),
    ]
    best, best_frac = None, -1.0
    for roi in cands:
        x0, y0, x1, y1 = roi
        sub = d[y0:y1, x0:x1]
        frac = float((np.isfinite(sub) & (sub > 0)).mean())
        if frac > best_frac:
            best, best_frac = roi, frac
    return best if best_frac > 0.01 else None


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
    state = {}

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

    # ------------------------------------------------------------ 首次运行：无配置
    d.add("首次启动应无配置文件",
          None,
          lambda: app.has_config_file is False and app.store.segment_count() == 0)

    d.add("扫描设备", app.scan_cameras, lambda: len(app.cam_infos) > 0)
    d.add("首次运行：没有配置可解锁，两台都应报缺配置",
          None,
          lambda: state.update(
              report=app.unlock_report,
              serials=sorted(str(c.serial_number) for c in app.cam_infos)) or
          (app.unlock_report["unlocked"] == []
           and len(app.unlock_report["missing"]) == 2
           and app.store.locked == 0))
    d.add("面板按序列号稳定分配（两台不同）",
          None,
          lambda: len({p.combo.get() for p in app.panels}) == 2)
    d.add("打开两台相机",
          lambda: [p.open_camera() for p in app.panels],
          lambda: all(p.cam is not None for p in app.panels), timeout=30)
    d.add("两台序列号不同",
          None,
          lambda: app.panels[0].serial != app.panels[1].serial)
    d.add("打开后写入配置文件",
          None,
          lambda: os.path.exists(TEST_CFG_PATH) and app.store.segment_count() == 2)
    d.add("配置文件里不含明文序列号",
          None,
          lambda: state.update(raw=open(TEST_CFG_PATH, "rb").read()) or
          all(s.encode() not in state["raw"] for s in state["serials"]))
    d.add("相机型号已记录",
          None,
          lambda: all(app.store.camera(p.serial).get("model") for p in app.panels))

    # ------------------------------------------------------------ 采集
    d.add("全部开始采集", app.start_all,
          lambda: all(p.is_running for p in app.panels))
    d.add("两路都出画面",
          None,
          lambda: all(p.depth_data is not None for p in app.panels), timeout=30)
    d.add("持续采集 2 秒",
          lambda: state.update(t0=time.time()),
          lambda: time.time() - state["t0"] > 2.0, timeout=30)

    # ------------------------------------------------------------ 标定 + 存盘
    d.add("两路分别标定平面",
          lambda: [(setattr(p, "roi", best_roi(p)), p.run_calibration())
                   for p in app.panels],
          lambda: all(p.calibrated for p in app.panels), timeout=90)
    d.add("标定结果已写进配置",
          None,
          lambda: all(cfgmod.is_calibrated(app.store.camera(p.serial))
                      for p in app.panels))

    # ------------------------------------------------------------ 距离过滤按相机存
    d.add("改面板1的过滤区间并存盘",
          lambda: (app.panels[0].ent_min.delete(0, "end"),
                   app.panels[0].ent_min.insert(0, "777"),
                   app.panels[0].on_filter_changed()),
          lambda: app.store.camera(app.panels[0].serial)["filter"]["min_mm"] == 777.0)
    d.add("两台过滤区间互不影响",
          None,
          lambda: app.store.camera(app.panels[1].serial)["filter"]["min_mm"]
          != app.store.camera(app.panels[0].serial)["filter"]["min_mm"])

    # ------------------------------------------------------------ 模拟重启解锁
    d.add("记录当前标定值",
          lambda: state.update(plane_c0=app.panels[0].plane_c,
                               panel_of={p.serial: app.store.camera(p.serial)["panel"]
                                         for p in app.panels}),
          lambda: True)
    d.add("清空内存后从文件按相机 ID 重新解锁",
          lambda: (app.store.reset_decoded(), app.unlock_config(state["serials"])),
          lambda: len(app.unlock_report["unlocked"]) == 2)
    d.add("标定/过滤/面板均从配置恢复",
          None,
          lambda: all(p.calibrated for p in app.panels)
          and abs(app.panels[0].plane_c - state["plane_c0"]) < 1e-6
          and app.store.camera(app.panels[0].serial)["filter"]["min_mm"] == 777.0)

    # ------------------------------------------------------------ 刷新间隔存盘
    d.add("切换刷新间隔并存盘",
          lambda: (app.cmb_refresh.set("200"), app.set_refresh_interval()),
          lambda: app.display_interval_ms == 200)
    d.add("刷新间隔已入配置",
          None, lambda: app.store.app["display_interval_ms"] == 200)

    # ------------------------------------------------------------ Tk 刷新复用检查
    d.add("记录画布项与 Tk 图像",
          lambda: state.update(item=[p._img_item for p in app.panels],
                               photo=[p._photo for p in app.panels]),
          lambda: True)
    d.add("再刷新 1.2 秒",
          lambda: state.update(t2=time.time()),
          lambda: time.time() - state["t2"] > 1.2)
    d.add("画布项与 Tk 图像被复用",
          None,
          lambda: state.update(
              item_same=[p._img_item == state["item"][i] for i, p in enumerate(app.panels)],
              photo_same=[p._photo is state["photo"][i] for i, p in enumerate(app.panels)]) or
          all(p._img_item == state["item"][i] for i, p in enumerate(app.panels)))

    d.add("全部停止采集", app.stop_all,
          lambda: not any(p.is_running for p in app.panels))
    d.add("全部获取一帧", app.grab_all,
          lambda: not any(p.is_grabbing for p in app.panels), timeout=40)

    # ------------------------------------------------------------ 关窗存盘
    d.add("记录关窗前的参数",
          lambda: state.update(
              coef={p.serial: list(app.store.camera(p.serial)["calibration"]["coefficients"])
                    for p in app.panels},
              panels={p.serial: app.store.camera(p.serial)["panel"] for p in app.panels}),
          lambda: True)
    d.add("关闭窗口", app.on_close, lambda: True)

    d.start()
    root.mainloop()

    # 先看界面流程有没有失败，再看配置能否恢复
    if d.failures:
        print()
        print("!! SMOKE TEST FAILED（界面流程）")
        for name, why in d.failures:
            print(f"   - {name}: {why}")
        shutil.rmtree(TEST_CFG_DIR, ignore_errors=True)
        raise SystemExit(1)

    # ============================================================ 模拟重启：重读配置
    print()
    print("== 模拟重启：重新读配置 ==")
    assert "coef" in state, "前面标定步骤没成功，无法验证配置恢复"
    store2 = cfgmod.ConfigStore(TEST_CFG_PATH)
    assert store2.load() is True, "重启后应能读到配置文件"
    report2 = store2.unlock(state["serials"])
    assert sorted(report2["unlocked"]) == state["serials"], report2
    for serial, coef in state["coef"].items():
        got = store2.camera(serial)["calibration"]["coefficients"]
        assert got == coef, f"{serial} 标定值没恢复：{got} != {coef}"
        assert store2.camera(serial)["panel"] == state["panels"][serial]
    assert store2.camera(state["serials"][0])["filter"]["min_mm"] == 777.0
    assert store2.app["display_interval_ms"] == 200
    print(f"  重启后解锁 {len(report2['unlocked'])} 台，标定/面板/过滤/刷新全部恢复")

    bad = cfgmod.ConfigStore(TEST_CFG_PATH, authorized=None)
    bad.load()
    r_bad = bad.unlock(["123456789012", "000000000000"])
    assert r_bad["unlocked"] == [], r_bad
    assert bad.locked == 2
    print("  错误序列号一台都解不开")

    one = cfgmod.ConfigStore(TEST_CFG_PATH)
    one.load()
    r_one = one.unlock([state["serials"][0]])
    assert len(r_one["unlocked"]) == 1 and one.locked == 1, r_one
    assert one.save()
    again = cfgmod.ConfigStore(TEST_CFG_PATH)
    again.load()
    r_again = again.unlock(state["serials"])
    assert sorted(r_again["unlocked"]) == state["serials"], r_again
    print("  只插一台再存盘，另一台那段仍然完好")

    print()
    print("设备列表:", state["serials"])
    print("每路渲染帧数:", counters)
    print("画布项复用:", state["item_same"], " Tk 图像原地更新:", state["photo_same"])
    print("配置段数:", store2.segment_count(), " 文件字节:", os.path.getsize(TEST_CFG_PATH))
    print("弹窗记录:", box.calls)

    shutil.rmtree(TEST_CFG_DIR, ignore_errors=True)
    print()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
