# -*- coding: utf-8 -*-
"""测量算法公共实现。

从旧单机版 DepthCamMain.py 抽出，相机服务和 Modbus 服务共用。
流程：RANSAC 拟合基准面 → 高度图 → 切平面求解 → 图像中心相对基准面的距离。
"""

import numpy as np

import cut_plane


def value_to_python(v):
    try:
        if v.IsBool():
            return bool(v.as_bool())
        if v.IsString():
            return str(v.as_string())
        if v.IsFloat() or v.IsDouble():
            return float(v.as_float())
        if v.IsArray() or v.IsDictionary():
            return None
        return int(v.as_number())
    except Exception:
        return None


def fit_plane_ransac(depth_data, roi, step=2, max_iter=300,
                     inlier_thresh=4.0, min_inliers=100):
    x0, y0, x1, y1 = roi
    sub = depth_data[y0:y1:step, x0:x1:step]
    h, w = sub.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xs = (xs * step + x0).astype(np.float64).ravel()
    ys = (ys * step + y0).astype(np.float64).ravel()
    zs = sub.ravel().astype(np.float64)

    valid = np.isfinite(zs) & (zs > 0)
    pts = np.column_stack([xs[valid], ys[valid], zs[valid]])
    n = len(pts)
    if n < 3:
        return None

    rng = np.random.default_rng()
    best_cnt = 0
    best_inliers = None
    for _ in range(max_iter):
        idx = rng.choice(n, 3, replace=False)
        A = np.column_stack([pts[idx, 0], pts[idx, 1], np.ones(3)])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        abc = np.linalg.solve(A, pts[idx, 2])
        residual = np.abs(pts[:, 2] - (abc[0] * pts[:, 0] + abc[1] * pts[:, 1] + abc[2]))
        inliers = residual < inlier_thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers
            if cnt > n * 0.95:
                break

    if best_inliers is None or best_cnt < min_inliers:
        return None

    inliers = best_inliers
    for _ in range(3):
        ip = pts[inliers]
        if len(ip) < 3:
            break
        A = np.column_stack([ip[:, 0], ip[:, 1], np.ones(len(ip))])
        abc, _, _, _ = np.linalg.lstsq(A, ip[:, 2], rcond=None)
        residual = np.abs(pts[:, 2] - (abc[0] * pts[:, 0] + abc[1] * pts[:, 1] + abc[2]))
        inliers = residual < inlier_thresh

    a, b, c = abc
    inlier_ratio = int(inliers.sum()) / n
    return a, b, c, inlier_ratio


def clamp_roi(roi, shape):
    if not roi:
        return None
    h, w = shape[:2]
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def valid_bbox(valid):
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_from_mask(mask, min_pixels=200):
    if mask is None or int(mask.sum()) < min_pixels:
        return None
    return valid_bbox(mask)


def auto_roi_bbox(depth, plane=None):
    valid = cut_plane.valid_mask(depth)
    if not valid.any():
        return None
    if plane is None:
        box = valid_bbox(valid)
        fit = fit_plane_ransac(depth, box, step=4, max_iter=240, min_inliers=150)
        if fit is not None:
            plane = (fit[0], fit[1], fit[2])
        else:
            plane = (0.0, 0.0, float(np.median(depth[valid])))
    plane_depth = cut_plane.plane_from_coeffs(plane, depth.shape)
    h = cut_plane.build_height(depth, plane_depth)
    mask = cut_plane.auto_block_roi(h, valid)
    return bbox_from_mask(mask) or valid_bbox(valid)


def restrict_valid(valid, roi):
    if not roi:
        return valid
    x0, y0, x1, y1 = roi
    valid[:y0] = False
    valid[y1:] = False
    valid[:, :x0] = False
    valid[:, x1:] = False
    return valid


def plane_center(plane, shape):
    H, W = shape[:2]
    return plane[0] * (W / 2.0) + plane[1] * (H / 2.0) + plane[2]


def solve_panel_detail(p, default_a_max=500.0):
    depth = p["depth"]
    plane = p.get("plane")
    if plane is None:
        raise RuntimeError("未标定基准面")
    fx, fy = p.get("fx"), p.get("fy")
    mm2 = cut_plane.mm2_per_pixel(depth, fx, fy) if fx and fy else 1.0
    roi = clamp_roi(p.get("roi"), depth.shape)
    if roi:
        x0, y0, x1, y1 = roi
        pad = max(0, int(cut_plane.DEFAULT_GAP_FILL_PX))
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1 = min(depth.shape[1], x1 + pad)
        py1 = min(depth.shape[0], y1 + pad)
        crop = depth[py0:py1, px0:px1]
        valid = np.zeros(crop.shape, dtype=bool)
        valid[y0 - py0:y1 - py0, x0 - px0:x1 - px0] = cut_plane.valid_mask(
            depth[y0:y1, x0:x1])
        if not valid.any():
            raise RuntimeError("计算范围内没有有效深度数据")
        xs = np.arange(px0, px1, dtype=np.float32)[None, :]
        ys = np.arange(py0, py1, dtype=np.float32)[:, None]
        plane_depth = plane[0] * xs + plane[1] * ys + plane[2]
        h = cut_plane.build_height(crop, plane_depth)
    else:
        valid = cut_plane.valid_mask(depth)
        if not valid.any():
            raise RuntimeError("没有有效深度数据")
        plane_depth = cut_plane.plane_from_coeffs(plane, depth.shape)
        h = cut_plane.build_height(depth, plane_depth)
    a_max = float(p.get("a_max_mm2") or default_a_max)
    result = cut_plane.solve_cut_plane(h, valid, mm2, a_max)
    center = plane_center(plane, depth.shape)
    relative = float(result["delta_mm"])
    saved = p.get("plane_center_mm")
    return {
        "relative_mm": relative,
        "absolute_mm": center + relative,
        "plane_center_mm": center,
        "saved_plane_center_mm": None if saved is None else float(saved),
    }


def solve_panel(p, default_a_max=500.0):
    return solve_panel_detail(p, default_a_max)["relative_mm"]


def compute_measure(a, b, a_max_mm2=500.0):
    return solve_panel(a, a_max_mm2), solve_panel(b, a_max_mm2)


def calibrate_pair(a, b):
    out = {}
    for key, p in (("A", a), ("B", b)):
        depth = p["depth"]
        roi = clamp_roi(p.get("roi"), depth.shape) or auto_roi_bbox(depth, p.get("plane"))
        if roi is None:
            raise RuntimeError(f"{key}面无法确定有效区域")
        fit = fit_plane_ransac(depth, roi, step=2, max_iter=400)
        if fit is None:
            raise RuntimeError(f"{key}面基准面拟合失败")
        out[key] = (fit[0], fit[1], fit[2])
    return out
