import numpy as np
from PIL import Image

size = 512
face_names = ["+X (右)", "-X (左)", "+Y (上)", "-Y (下)", "+Z (前)", "-Z (后)"]
face_dirs = [
    lambda u, v: np.stack([ np.ones_like(u), -v, -u], axis=-1),  # +X
    lambda u, v: np.stack([-np.ones_like(u), -v,  u], axis=-1),  # -X
    lambda u, v: np.stack([ u,  np.ones_like(u), v], axis=-1),   # +Y
    lambda u, v: np.stack([ u, -np.ones_like(u), -v], axis=-1),  # -Y
    lambda u, v: np.stack([ u, -v,  np.ones_like(u)], axis=-1),  # +Z
    lambda u, v: np.stack([-u, -v, -np.ones_like(u)], axis=-1),  # -Z
]

def procedural_env(d):
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    dy = d[..., 1]
    t = dy * 0.5 + 0.5
    sky = np.stack([
        0.3 + 0.4 * t,
        0.3 + 0.45 * t,
        0.35 + 0.5 * t,
    ], axis=-1)
    ground_mask = dy < 0.0
    blend = np.clip((dy + 0.5) / 0.5, 0, 1)
    ground = np.stack([np.full_like(dy, 0.2)] * 3, axis=-1)
    sky = np.where(ground_mask[..., None], ground * (1 - blend[..., None]) + sky * blend[..., None], sky)

    # 顶部柔光
    top = np.clip((dy - 0.7) / 0.3, 0, 1)
    sky += np.stack([2.0 * top, 1.9 * top, 1.8 * top], axis=-1)

    # 前方补光
    front = np.array([0.0, 0.3, 1.0])
    front = front / np.linalg.norm(front)
    dot_f = np.clip(np.sum(d * front, axis=-1), 0, 1) ** 8
    sky += np.stack([1.0 * dot_f * 0.5, 1.0 * dot_f * 0.5, 1.1 * dot_f * 0.5], axis=-1)

    # 侧面补光
    side1 = np.array([1.0, 0.2, 0.0])
    side1 = side1 / np.linalg.norm(side1)
    dot_s1 = np.clip(np.sum(d * side1, axis=-1), 0, 1) ** 6
    sky += np.stack([0.8 * dot_s1 * 0.3, 0.7 * dot_s1 * 0.3, 0.6 * dot_s1 * 0.3], axis=-1)

    side2 = np.array([-1.0, 0.2, 0.0])
    side2 = side2 / np.linalg.norm(side2)
    dot_s2 = np.clip(np.sum(d * side2, axis=-1), 0, 1) ** 6
    sky += np.stack([0.6 * dot_s2 * 0.3, 0.7 * dot_s2 * 0.3, 0.8 * dot_s2 * 0.3], axis=-1)

    # tone mapping + gamma
    sky = sky / (sky + 1.0)
    sky = np.power(np.clip(sky, 0, 1), 1.0 / 2.2)
    return sky

u = np.linspace(-1, 1, size)
v = np.linspace(-1, 1, size)
uu, vv = np.meshgrid(u, v)

images = []
for i, dir_fn in enumerate(face_dirs):
    d = dir_fn(uu, vv)
    color = procedural_env(d)
    img = (np.clip(color, 0, 1) * 255).astype(np.uint8)
    images.append(img)

# 拼成一张 3x2 的预览图
rows = []
for r in range(2):
    row = np.hstack([images[r * 3 + c] for c in range(3)])
    rows.append(row)
grid = np.vstack(rows)

out = Image.fromarray(grid)
out.save("/home/ai/Desktop/cubemap_preview.png")
print(f"已保存: /home/ai/Desktop/cubemap_preview.png ({grid.shape[1]}x{grid.shape[0]})")
print("排列: +X(右) -X(左) +Y(上) / -Y(下) +Z(前) -Z(后)")
