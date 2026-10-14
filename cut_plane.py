"""面积容差切平面求解（最大连通凹坑面积约束）。

工艺背景
--------
木块放在平台上，深度相机水平拍摄待切的那个面。事先用切刀走一刀，
切出来的平面就是**基准面**（也是本程序里的"标定平面"）。切刀始终平行于
基准面进刀，所以这是一个**纯平移**问题：只需在基准面法向上求一个偏移量 δ。

坐标与符号约定（关键，搞反会静默算错）
--------------------------------------
    h = 深度 − 基准面深度          靠相机为负，远离为正

切刀削掉的是朝相机凸出来的部分，所以切面定在 h = δ 时：

    被切掉的区域 = {h < δ}          凸出来的，被削平
    残留的凹坑   = {h > δ}          没切到的，留在切面上

目标：在满足凹坑约束的前提下**切得越少越好** → 取满足约束的**最小** δ。

约束（按现场确认的口径）
------------------------
只约束**单个连通凹坑的最大面积** ≤ a_max（绝对值，mm²）；不限总面积占比、
不限坑深、不限坑数量。因为 max_comp(δ) 关于 δ 单调不增，可行域是区间
[δ*, +∞)，δ* 就是分界点 —— 二分即可，不需要迭代优化。

注意：这与"面积分位数"不是一回事。分位数对应的是"总凹坑面积占比 ≤ α"的
约束；本模块用的是连通域约束，数学上是两个不同的问题。总面积占比只作为
结果显示，不参与求解。

面积口径
--------
面积按「像素数 × 每像素 mm²」计算，标量面元。前提是基准面大致正对相机
（Z 在整幅图上近似常数）—— 现场安装正是如此。如果以后发现基准面明显倾斜，
需要改成逐像素面元 Z²/(fx·fy)。
"""

import time

import cv2
import numpy as np

# 无效深度造成的裂缝：凹坑掩码闭运算半径（像素）。太小补不上立体匹配的空洞，
# 太大会把相邻的两个坑并成一个。现场需要按实际空洞尺寸调。
DEFAULT_GAP_FILL_PX = 3
# 二分精度（mm）
DEFAULT_TOL_MM = 0.01
# 自动分割时认为属于木块面的高度带宽（mm）
DEFAULT_BLOCK_BAND_MM = 60.0


# ------------------------------------------------------------------ 基础

def plane_from_coeffs(coeffs, shape):
    """把标定系数 (a, b, c) 展开成整幅图的基准面深度：z = a·x + b·y + c。"""
    a, b, c = coeffs
    h, w = shape[:2]
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    return a * xs + b * ys + c


def build_height(depth_mm, plane_depth):
    """带符号高度：h = 深度 − 基准面深度（靠相机为负、远离为正）。"""
    return (depth_mm - plane_depth).astype(np.float32, copy=False)


def valid_mask(depth_mm):
    return np.isfinite(depth_mm) & (depth_mm > 0)


def mm2_per_pixel(depth_mm, fx, fy):
    """每像素对应的物方面积（mm²）。取中位深度算一个标量。"""
    d = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
    z = float(np.median(d)) if d.size else 0.0
    return (z / float(fx)) * (z / float(fy))


def close_mask(mask, gap_px):
    """闭运算填补无效像素造成的细缝，避免把一个大坑拆成几块。"""
    if gap_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px * 2 + 1,) * 2)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


# ------------------------------------------------------------------ 连通域

def component_stats(mask, mm2_px):
    """连通域统计。返回 (labels, 每个连通域面积 mm², 个数)。0 号是背景。"""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    if not m.any():
        return np.zeros(mask.shape, np.int32), np.zeros(1), 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return labels, np.zeros(1), 0
    # CC_STAT_AREA 是像素数；乘以标量面元即为 mm²，比 bincount 加权快得多
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64) * mm2_px
    areas[0] = 0.0
    return labels, areas, n - 1


def max_component_area(h, valid, delta, mm2_px, gap_px):
    """在切深 delta 下，最大的单个连通凹坑面积（mm²）。"""
    pit = (h > delta) & valid
    if not pit.any():
        return 0.0, None
    pit = close_mask(pit, gap_px)
    labels, areas, n = component_stats(pit, mm2_px)
    if n == 0:
        return 0.0, (labels, areas, n)
    return float(areas[1:].max()), (labels, areas, n)


# ------------------------------------------------------------------ 求解

def solve_cut_plane(h, valid, mm2_px, a_max_mm2,
                    gap_px=DEFAULT_GAP_FILL_PX,
                    tol_mm=DEFAULT_TOL_MM,
                    progress=None):
    """求满足「最大连通凹坑面积 ≤ a_max」的最小切深 δ。

    全程在**同一分辨率**上判断（不做降采样粗搜）：降采样会改变连通结构，
    粗搜出来的 δ 拿到全分辨率复核可能不满足约束，是个隐蔽的错误来源。
    二分次数由区间宽度/精度决定，实测 10mm 区间、0.01mm 精度只要约 10 次。
    """
    hv = h[valid]
    if hv.size == 0:
        raise ValueError("没有有效深度数据")
    if mm2_px <= 0:
        raise ValueError("面元标量必须为正")

    lo = float(hv.min())
    hi = float(hv.max())

    # 切到最低点仍不满足 → 说明 a_max 比最大的坑还小，只能切到最深
    a_lo, _ = max_component_area(h, valid, lo - 1e-6, mm2_px, gap_px)
    if a_lo <= a_max_mm2:
        delta, iters = lo, 0
    else:
        iters = 0
        while hi - lo > tol_mm and iters < 80:
            mid = 0.5 * (lo + hi)
            a_mid, _ = max_component_area(h, valid, mid, mm2_px, gap_px)
            if a_mid <= a_max_mm2:
                hi = mid
            else:
                lo = mid
            iters += 1
            if progress is not None:
                progress(iters, lo, hi, a_mid)
        delta = hi

    # 出结果（同一分辨率、同一 gap，所以必然与判断一致）
    max_area, packed = max_component_area(h, valid, delta, mm2_px, gap_px)
    labels, areas, n_comp = packed if packed else (np.zeros(h.shape, np.int32),
                                                   np.zeros(1), 0)
    pit = (h > delta) & valid

    total_px = int(valid.sum())
    pit_px = int(pit.sum())
    removed = (h < delta) & valid
    cut_volume = float(((delta - h)[removed]).sum() * mm2_px)

    # 贴边的坑：面积被测量区域截断，不可信，单独标出来
    edge_labels = set()
    for sl in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        edge_labels.update(int(v) for v in np.unique(sl) if v > 0)

    # 凹坑区域里有多少像素是无效数据：太高说明数据太稀，结论不可信
    pit_box = close_mask(pit, max(gap_px, 3))
    invalid_in_pit = int((pit_box & ~valid).sum())
    box_px = int(pit_box.sum())

    return {
        "delta_mm": float(delta),
        "pit_ratio": pit_px / total_px if total_px else 0.0,
        "max_pit_depth_mm": float(hv.max() - delta),
        "max_pit_area_mm2": max_area,
        "pit_count": int(n_comp),
        "edge_pits": len(edge_labels),
        "cut_volume_mm3": cut_volume,
        "total_area_mm2": total_px * mm2_px,
        "pit_area_mm2": pit_px * mm2_px,
        "pit_mask": pit,
        "labels": labels,
        "component_areas": areas,
        "iterations": iters,
        "invalid_in_pit_ratio": (invalid_in_pit / box_px) if box_px else 0.0,
        "satisfied": max_area <= a_max_mm2 * 1.0001,
    }


def cut_volume(h, valid, delta, mm2_px):
    """切到 delta 时切除的体积（mm³）：Σ max(δ − h, 0) · 面元。"""
    removed = valid & (h < delta)
    return float((delta - h[removed]).sum() * mm2_px)


# ------------------------------------------------------------------ 方案 A：零凹坑

def solve_no_pit(h, valid, mm2_px, robust_pcts=(0.999, 0.9999)):
    """方案 A：切到最低点，保证一个凹坑都不留 —— 最费料的理论上界。

    切面取 δ = max(h)：此时 {h > δ} 为空，凹坑数 0。
    它的价值不只是"另一个选项"：它是标尺，δA − δB 就是容差换来的省料量。

    注意 max(h) 是**单像素统计**，深度图上的坏点会把切深顶得很深，
    所以这里同时给出稳健性参考，让操作员看得出结果是不是被几个坏点带偏的。
    """
    hv = h[valid]
    if hv.size == 0:
        raise ValueError("没有有效深度数据")

    delta = float(hv.max())
    order = np.argsort(hv)
    n = hv.size

    robust = {}
    for p in robust_pcts:
        i = min(int(np.ceil(p * n)) - 1, n - 1)
        robust[p] = float(hv[order][i])

    return {
        "mode": "no_pit",
        "delta_mm": delta,
        "pit_ratio": 0.0,
        "max_pit_area_mm2": 0.0,
        "max_pit_depth_mm": 0.0,
        "pit_count": 0,
        "edge_pits": 0,
        "cut_volume_mm3": cut_volume(h, valid, delta, mm2_px),
        "total_area_mm2": float(valid.sum() * mm2_px),
        "robust": robust,
        "near_max_count": int((hv > delta - 0.2).sum()),
        "iterations": 0,
        "satisfied": True,
    }


def solve_both(h, valid, mm2_px, a_max_mm2,
               gap_px=DEFAULT_GAP_FILL_PX, tol_mm=DEFAULT_TOL_MM):
    """把两个方案一起算出来，并给出容差省下的料。"""
    plan_a = solve_no_pit(h, valid, mm2_px)
    plan_b = solve_cut_plane(h, valid, mm2_px, a_max_mm2, gap_px, tol_mm)
    return {
        "no_pit": plan_a,
        "tolerance": plan_b,
        "saved_depth_mm": plan_a["delta_mm"] - plan_b["delta_mm"],
        "saved_volume_mm3": plan_a["cut_volume_mm3"] - plan_b["cut_volume_mm3"],
    }


# ------------------------------------------------------------------ 自动分割

def auto_block_roi(h, valid, band_mm=DEFAULT_BLOCK_BAND_MM,
                   min_area_frac=0.02, gap_px=DEFAULT_GAP_FILL_PX):
    """自动把木块面从背景里分出来（A8 的自动选项）。

    思路：取基准面附近一个高度带内的有效点，取最大连通域，再填内部空洞。
    前提是相机能看到木块轮廓、且背景离基准面足够远。
    """
    cand = valid & (np.abs(h) <= band_mm)
    if not cand.any():
        return np.zeros(cand.shape, bool)
    cand = close_mask(cand, gap_px)
    _, areas, n = component_stats(cand, 1.0)
    if n == 0:
        return np.zeros(cand.shape, bool)
    # 重新标注以便取最大域（用像素数即可）
    m = np.ascontiguousarray(cand.astype(np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return np.zeros(cand.shape, bool)
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    roi = labels == best
    if roi.sum() < min_area_frac * h.size:
        return np.zeros(cand.shape, bool)
    cnts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    return filled > 0


# ------------------------------------------------------------------ 诊断

def diagnose(h, valid, mm2_px, a_max_list=(50.0, 100.0, 200.0, 500.0, 1000.0)):
    """给测试按钮用的诊断：噪声量级 + 不同 a_max 对应的切深。"""
    hv = h[valid].astype(np.float32)
    out = {
        "count": int(hv.size),
        "valid_ratio": float(valid.mean()),
        "mean": float(hv.mean()),
        "std": float(hv.std()),
        "min": float(hv.min()),
        "max": float(hv.max()),
        "mm2_per_px": float(mm2_px),
        "total_area_mm2": float(valid.sum() * mm2_px),
        "quantiles": {},
        "a_max_sweep": {},
    }
    # 面积加权分位数：只作参考（对应"总面积占比 ≤ α"的另一种口径）
    order = np.argsort(hv)
    cum = np.arange(1, hv.size + 1, dtype=np.float64) / hv.size
    for al in (0.01, 0.05, 0.10):
        i = int(np.searchsorted(cum, 1.0 - al))
        out["quantiles"][al] = float(hv[order][min(i, hv.size - 1)])

    # 高频成分 / 噪声估计。
    # 不能用「std(原图) − std(滤波后)」：表面本身就有很大幅度的宏观起伏，
    # 那个差值会被结构淹掉，算出来接近 0，看起来像"没有噪声"，是错的。
    # 正确做法是直接看中值滤波残差。
    filled = np.where(valid, h, 0.0).astype(np.float32)
    med = cv2.medianBlur(filled, 5)
    resid = (med - h)[valid].astype(np.float32)
    out["hf_std"] = float(resid.std())
    out["noise_mad"] = float(1.4826 * np.median(np.abs(resid)))

    for a in a_max_list:
        try:
            r = solve_cut_plane(h, valid, mm2_px, a)
            out["a_max_sweep"][a] = {
                "delta_mm": r["delta_mm"],
                "max_pit_area_mm2": r["max_pit_area_mm2"],
                "pit_ratio": r["pit_ratio"],
                "pit_count": r["pit_count"],
                "edge_pits": r["edge_pits"],
                "iterations": r["iterations"],
            }
        except Exception as e:
            out["a_max_sweep"][a] = {"error": str(e)}
    return out


# ------------------------------------------------------------------ 自检

def _synth(seed=7, hole_ratio=0.08, noise=0.15):
    """合成一块带凹坑的面：基准面平，几个大小深浅不同的碗形凹坑。"""
    rng = np.random.default_rng(seed)
    H, W = 1536, 2048
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    h = np.zeros((H, W), np.float32)
    h += 9.0 * np.exp(-(((xx - 700) ** 2 + (yy - 800) ** 2) / (2 * 150.0 ** 2)))
    h += 4.0 * np.exp(-(((xx - 1400) ** 2 + (yy - 500) ** 2) / (2 * 80.0 ** 2)))
    h += 3.0 * np.exp(-(((xx - 1200) ** 2 + (yy - 1100) ** 2) / (2 * 60.0 ** 2)))
    if noise:
        h += rng.normal(0, noise, h.shape).astype(np.float32)
    valid = np.ones_like(h, bool)
    if hole_ratio:
        valid &= rng.random(h.shape) >= hole_ratio
    return h, valid


def _selftest():
    print("== cut_plane 自检 ==")
    MM2_PX = 0.836          # 1831mm 处、fx=fy=2002.5 时实测值
    h, valid = _synth()
    print(f"  合成面 {h.shape[1]}x{h.shape[0]}，有效点 {valid.mean()*100:.1f}%，"
          f"总面积 {valid.sum()*MM2_PX/100:.0f} cm²")
    print(f"  高度: 均值 {h[valid].mean():.2f}  标准差 {h[valid].std():.2f}  "
          f"范围 {h[valid].min():.2f} ~ {h[valid].max():.2f} mm")

    # 1) 单调性
    ds = np.linspace(float(h[valid].min()), float(h[valid].max()), 25)
    prev, mono = None, True
    for d in ds:
        a, _ = max_component_area(h, valid, float(d), MM2_PX, DEFAULT_GAP_FILL_PX)
        if prev is not None and a > prev + 1e-6:
            mono = False
        prev = a
    print(f"\n  1) max_comp(δ) 单调性: {'OK' if mono else '违反！'}")
    assert mono, "单调性不成立，二分法不能用"

    # 2) a_max 扫描
    print(f"\n  2) a_max 扫描")
    print(f"     {'a_max(mm²)':>11} {'切深δ(mm)':>10} {'最大坑(mm²)':>12} "
          f"{'凹坑占比':>9} {'坑数':>4} {'贴边':>4} {'二分':>4}")
    deltas = []
    for a_max in (50.0, 100.0, 200.0, 500.0, 1000.0, 3000.0):
        r = solve_cut_plane(h, valid, MM2_PX, a_max)
        deltas.append(r["delta_mm"])
        assert r["satisfied"], f"a_max={a_max} 约束没满足：{r['max_pit_area_mm2']:.1f}"
        print(f"     {a_max:>11.0f} {r['delta_mm']:>10.3f} "
              f"{r['max_pit_area_mm2']:>12.1f} {r['pit_ratio']:>9.2%} "
              f"{r['pit_count']:>4} {r['edge_pits']:>4} {r['iterations']:>4}")
    assert deltas == sorted(deltas, reverse=True), f"切深没随 a_max 单调: {deltas}"
    print("     约束全部满足，切深随 a_max 单调递减 ✓")

    # 3) 闭运算的作用
    a_max = 500.0
    r_gap = solve_cut_plane(h, valid, MM2_PX, a_max, gap_px=DEFAULT_GAP_FILL_PX)
    r_nogap = solve_cut_plane(h, valid, MM2_PX, a_max, gap_px=0)
    print(f"\n  3) 空洞裂缝的影响（a_max={a_max:.0f} mm²）")
    print(f"     闭运算 {DEFAULT_GAP_FILL_PX}px : δ={r_gap['delta_mm']:+.3f} mm  "
          f"最大坑 {r_gap['max_pit_area_mm2']:.1f} mm²  坑数 {r_gap['pit_count']}")
    print(f"     不闭运算         : δ={r_nogap['delta_mm']:+.3f} mm  "
          f"最大坑 {r_nogap['max_pit_area_mm2']:.1f} mm²  坑数 {r_nogap['pit_count']}")
    print(f"     切深差异 {r_nogap['delta_mm']-r_gap['delta_mm']:+.3f} mm"
          f"（不闭运算会把坑切碎，少切了但实际留下的坑更大）")

    # 4) 性能
    t0 = time.perf_counter()
    r = solve_cut_plane(h, valid, MM2_PX, 200.0)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  4) 单次求解 {dt:.0f} ms（{r['iterations']} 次二分，全分辨率）")

    # 5) 诊断
    t0 = time.perf_counter()
    d = diagnose(h, valid, MM2_PX)
    print(f"\n  5) diagnose() {(time.perf_counter()-t0)*1000:.0f} ms   "
          f"σ={d['std']:.3f} mm  高频std={d['hf_std']:.3f} mm  "
          f"噪声MAD={d['noise_mad']:.3f} mm")
    print("     （合成数据实际加了 0.15mm 噪声，MAD 估计应接近该量级）")
    for al, v in d["quantiles"].items():
        print(f"     (1-{al:.0%}) 分位 = {v:.3f} mm")

    # 6) 方向检查
    print("\n  6) 符号方向检查")
    a0, _ = max_component_area(h, valid, 0.0, MM2_PX, DEFAULT_GAP_FILL_PX)
    a1, _ = max_component_area(h, valid, r_gap["delta_mm"], MM2_PX, DEFAULT_GAP_FILL_PX)
    print(f"     切在基准面 h=0   : 最大坑 {a0:.1f} mm²  ← 远超 a_max={a_max:.0f}")
    print(f"     切在 δ={r_gap['delta_mm']:+.3f} : 最大坑 {a1:.1f} mm²  ✓")
    print("     => 凹坑是 {h > δ}，取满足约束的最小 δ，方向正确")

    # 7) 自动分割（合成数据里没有背景，只验证不崩且能选出主体）
    roi = auto_block_roi(h, valid)
    print(f"\n  7) 自动分割选取区域占比 {roi.mean()*100:.1f}%（合成数据无背景，应接近全图）")

    # 8) 方案 A（零凹坑）与方案 B 的对比
    print("\n  8) 方案 A / B 对比")
    a_max_cmp = 500.0
    both = solve_both(h, valid, MM2_PX, a_max_cmp)
    pa, pb = both["no_pit"], both["tolerance"]
    print(f"     方案A 零凹坑(切到最低点) : δ={pa['delta_mm']:+.3f} mm  "
          f"切除体积 {pa['cut_volume_mm3']/1000:.1f} cm³  凹坑 0 个")
    print(f"     方案B 容差(a_max={a_max_cmp:.0f})  : δ={pb['delta_mm']:+.3f} mm  "
          f"切除体积 {pb['cut_volume_mm3']/1000:.1f} cm³  "
          f"最大坑 {pb['max_pit_area_mm2']:.1f} mm²  凹坑 {pb['pit_count']} 个")
    print(f"     容差省下 : {both['saved_depth_mm']:+.3f} mm 厚，"
          f"约 {both['saved_volume_mm3']/1000:.1f} cm³ 木料")
    assert pa["delta_mm"] >= pb["delta_mm"] - 1e-9, "方案A 必须切得不比方案B 浅"
    assert pa["pit_count"] == 0 and pb["satisfied"]
    print(f"     方案A 稳健性参考：去掉最高 0.1% 像素后 δ="
          f"{pa['robust'][0.999]:+.3f} mm（差 {pa['delta_mm']-pa['robust'][0.999]:.3f} mm）")
    print(f"                       最高点 0.2mm 内的像素只有 {pa['near_max_count']} 个"
          f"（很少说明结果由个别坏点决定，应先降噪）")
    print("     单调关系正确 ✓")

    print("\n全部自检通过")


if __name__ == "__main__":
    _selftest()
