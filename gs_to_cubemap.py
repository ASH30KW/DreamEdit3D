#!/usr/bin/env python3
"""将高斯泼溅 PLY 渲染为 cubemap 六面图 (纯 numpy)"""

import numpy as np
from plyfile import PlyData
from PIL import Image
import os

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199

FACE_DEFS = {
    'posx': (0, +1, 2, -1, 1, -1),
    'negx': (0, -1, 2, +1, 1, -1),
    'posy': (1, +1, 0, +1, 2, +1),
    'negy': (1, -1, 0, +1, 2, -1),
    'posz': (2, +1, 0, +1, 1, -1),
    'negz': (2, -1, 0, -1, 1, -1),
}


def load_ply(path):
    print(f"加载 {path} ...")
    ply = PlyData.read(path)
    v = ply['vertex']
    n = len(v['x'])

    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1).astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-np.array(v['opacity'], dtype=np.float32))))

    scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32)
    scales = np.exp(scales)
    max_scale = np.max(scales, axis=1)

    props = [p.name for p in v.properties]
    f_rest = None
    if 'f_rest_0' in props:
        rest_count = sum(1 for p in props if p.startswith('f_rest_'))
        f_rest = np.stack([v[f'f_rest_{i}'] for i in range(rest_count)], axis=-1).astype(np.float32)

    print(f"  {n} 个高斯, SH rest: {f_rest.shape[1] if f_rest is not None else 0} 维")
    return xyz, f_dc, f_rest, opacity, max_scale


def eval_sh(f_dc, f_rest, dirs):
    color = SH_C0 * f_dc + 0.5
    if f_rest is not None and f_rest.shape[1] >= 9:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        color += (-SH_C1 * y * f_rest[:, 0:3]
                 + SH_C1 * z * f_rest[:, 3:6]
                 - SH_C1 * x * f_rest[:, 6:9])
    return np.clip(color, 0, 1).astype(np.float32)


def normalized_conv_fill(color_acc, weight_acc, radius=6):
    """用积分图做归一化卷积，平滑填充空洞"""
    h, w = weight_acc.shape
    r = radius

    def integral_box_sum(arr, r):
        pad = np.pad(arr, [(r+1, r), (r+1, r)], mode='constant')
        ii = np.cumsum(np.cumsum(pad, axis=0), axis=1)
        return (ii[2*r+1:2*r+1+h, 2*r+1:2*r+1+w]
              - ii[:h, 2*r+1:2*r+1+w]
              - ii[2*r+1:2*r+1+h, :w]
              + ii[:h, :w])

    w_sum = integral_box_sum(weight_acc, r)
    safe_w = np.maximum(w_sum, 1e-10)

    result = color_acc.copy()
    gaps = weight_acc == 0
    for c in range(3):
        cw_sum = integral_box_sum(color_acc[:, :, c] * weight_acc, r)
        result[:, :, c][gaps] = cw_sum[gaps] / safe_w[gaps]

    return result


def render_face(xyz, colors, opacity, max_scale, face_name, size):
    d_idx, d_sign, sc_idx, sc_sign, tc_idx, tc_sign = FACE_DEFS[face_name]

    depth = xyz[:, d_idx] * d_sign
    mask = (depth > 0.01) & (opacity > 0.05)

    d = depth[mask]
    sc = xyz[:, sc_idx][mask] * sc_sign
    tc = xyz[:, tc_idx][mask] * tc_sign
    col = colors[mask]
    opa = opacity[mask]
    ms = max_scale[mask]

    focal = size / 2.0
    px = (focal * sc / d + size / 2.0).astype(np.int32)
    py = (focal * tc / d + size / 2.0).astype(np.int32)

    # 每个高斯的屏幕半径 (2-sigma)
    screen_r = np.clip(focal * ms * 2.0 / d, 1.0, 20.0)

    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    px, py, col, opa, screen_r = px[valid], py[valid], col[valid], opa[valid], screen_r[valid]

    color_acc = np.zeros((size, size, 3), dtype=np.float64)
    weight_acc = np.zeros((size, size), dtype=np.float64)

    # 按半径分桶处理：小半径用固定 kernel，大半径单独处理
    w_col = col * opa[:, None]

    # 小半径 (r < 3): 5x5 kernel
    small = screen_r < 3.0
    if small.any():
        spx, spy = px[small], py[small]
        sw_col, sopa = w_col[small], opa[small]
        sr = screen_r[small]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                dist2 = dx * dx + dy * dy
                npx, npy = spx + dx, spy + dy
                inv = (npx >= 0) & (npx < size) & (npy >= 0) & (npy < size)
                gw = np.exp(-dist2 / (sr[inv] ** 2 + 0.5))
                np.add.at(color_acc, (npy[inv], npx[inv]), sw_col[inv] * gw[:, None])
                np.add.at(weight_acc, (npy[inv], npx[inv]), sopa[inv] * gw)

    # 中等半径 (3 <= r < 8): 动态 kernel
    med = (screen_r >= 3.0) & (screen_r < 8.0)
    if med.any():
        mpx, mpy = px[med], py[med]
        mw_col, mopa = w_col[med], opa[med]
        mr = screen_r[med]
        for dy in range(-7, 8):
            for dx in range(-7, 8):
                dist2 = dx * dx + dy * dy
                if dist2 > 49:
                    continue
                npx, npy = mpx + dx, mpy + dy
                inv = (npx >= 0) & (npx < size) & (npy >= 0) & (npy < size)
                gw = np.exp(-dist2 / (mr[inv] ** 2 + 0.5))
                gw_mask = gw > 0.01
                if not gw_mask.any():
                    continue
                inv2 = np.where(inv)[0][gw_mask]
                gw = gw[gw_mask]
                npx2, npy2 = (mpx + dx)[inv2], (mpy + dy)[inv2]
                np.add.at(color_acc, (npy2, npx2), mw_col[inv2] * gw[:, None])
                np.add.at(weight_acc, (npy2, npx2), mopa[inv2] * gw)

    # 大半径 (r >= 8): 动态 kernel
    big = screen_r >= 8.0
    if big.any():
        bpx, bpy = px[big], py[big]
        bw_col, bopa = w_col[big], opa[big]
        br = screen_r[big]
        max_r = min(int(np.ceil(br.max())), 20)
        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                dist2 = dx * dx + dy * dy
                if dist2 > max_r * max_r:
                    continue
                npx, npy = bpx + dx, bpy + dy
                inv = (npx >= 0) & (npx < size) & (npy >= 0) & (npy < size)
                if not inv.any():
                    continue
                gw = np.exp(-dist2 / (br[inv] ** 2 + 0.5))
                gw_mask = gw > 0.01
                if not gw_mask.any():
                    continue
                inv2 = np.where(inv)[0][gw_mask]
                gw = gw[gw_mask]
                npx2, npy2 = (bpx + dx)[inv2], (bpy + dy)[inv2]
                np.add.at(color_acc, (npy2, npx2), bw_col[inv2] * gw[:, None])
                np.add.at(weight_acc, (npy2, npx2), bopa[inv2] * gw)

    # 归一化
    nonzero = weight_acc > 0
    color_acc[nonzero] /= weight_acc[nonzero, None]

    # 归一化卷积填充空洞
    color_acc = normalized_conv_fill(color_acc, nonzero.astype(np.float64), radius=8)

    return (np.clip(color_acc, 0, 1) * 255).astype(np.uint8)


def main():
    import sys
    ply_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ai/Desktop/Regal_Palace_Ballroom.ply"
    name = os.path.splitext(os.path.basename(ply_path))[0]
    out_dir = f"/home/ai/Desktop/{name}_cubemap"
    size = 1024

    os.makedirs(out_dir, exist_ok=True)
    xyz, f_dc, f_rest, opacity, max_scale = load_ply(ply_path)

    dirs = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True).clip(min=1e-8)
    colors = eval_sh(f_dc, f_rest, dirs)

    # 3DGS 用 Y-down (COLMAP), cubemap 用 Y-up, 翻转 Y 轴
    xyz = xyz.copy()
    xyz[:, 1] = -xyz[:, 1]

    print(f"\n分辨率: {size}x{size}")

    faces = {}
    for fname in FACE_DEFS:
        print(f"  渲染 {fname} ...", end=' ', flush=True)
        img = render_face(xyz, colors, opacity, max_scale, fname, size)
        faces[fname] = img
        Image.fromarray(img).save(f"{out_dir}/{fname}.jpg", quality=95)
        print("ok")

    s = size
    cross = np.zeros((3 * s, 4 * s, 3), dtype=np.uint8)
    cross[0:s, s:2*s] = faces['posy']
    cross[s:2*s, 0:s] = faces['negx']
    cross[s:2*s, s:2*s] = faces['posz']
    cross[s:2*s, 2*s:3*s] = faces['posx']
    cross[s:2*s, 3*s:4*s] = faces['negz']
    cross[2*s:3*s, s:2*s] = faces['negy']
    Image.fromarray(cross).save(f"{out_dir}/cross_preview.jpg", quality=90)
    print(f"\n完成！输出: {out_dir}/")
    print(f"十字预览: {out_dir}/cross_preview.jpg")


if __name__ == "__main__":
    main()
