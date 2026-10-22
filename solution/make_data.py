#!/usr/bin/env python3
"""Build the published pairs and the held-out set. Not shipped to the agent."""
import os, shutil, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finish import finish, auto_gain, srgb_to_lin

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PUB_SIZES = [(768, 512), (1024, 768), (640, 960), (1152, 768), (896, 896), (1536, 1024)]
HELD_SIZES = [(960, 640), (768, 512), (1024, 768), (683, 1024), (1280, 853),
              (1152, 768), (640, 960), (1536, 1024), (896, 896), (1000, 667)]


def smooth_noise(rng, sigma, W, H):
    n = gaussian_filter(rng.standard_normal((H, W)), sigma * W / 768.0, mode="wrap")
    return n / (n.std() + 1e-9)


def photo(seed, size):
    W, H = size
    k = W / 768.0
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    base = rng.uniform(0.0, 1.0, 3)
    top = rng.uniform(0.05, 1.0, 3)
    t = (yy / H)[..., None]
    img = top * (1 - t) + base * t
    for s, a in ((90, 0.30), (30, 0.16), (8, 0.07), (2, 0.025)):
        img += (a * smooth_noise(rng, s, W, H))[..., None] * rng.uniform(0.5, 1.2, 3)
    for _ in range(rng.integers(5, 10)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        rx, ry = rng.uniform(30, 200) * k, rng.uniform(30, 160) * k
        m = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1.0
        col = rng.uniform(0, 1, 3)
        if rng.random() < 0.35:
            col = np.clip(col * 1.6 - 0.3, 0, 1)
        img[m] = col * (0.8 + 0.1 * smooth_noise(rng, 20, W, H)[m][:, None])
    for _ in range(rng.integers(1, 4)):
        bw, bh = int(rng.integers(60, 120) * k), int(rng.integers(40, 100) * k)
        x0, y0 = rng.integers(0, W - bw), rng.integers(0, H - bh)
        img[y0:y0 + bh, x0:x0 + bw] = rng.uniform(0, 1, 3)
    bias = rng.choice([-0.22, -0.1, 0.0, 0.0, 0.14])
    img = np.clip(img + bias, 0, 1)
    if seed % 5 == 0:
        img = np.clip(img * 1.3, 0, 1)
    if seed % 7 == 0:
        img = img * 0.35
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def hsv_to_rgb(h, s, v):
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    out = np.zeros(h.shape + (3,))
    for k, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[m] = np.stack([r[m], g[m], b[m]], -1)
    return out


def flat(lv, W=768, H=512):
    return np.full((H, W, 3), lv, np.uint8)


def point(W, H):
    a = np.zeros((H, W, 3), np.uint8)
    a[H // 2, W // 2] = 255
    return a


def edge_v(W, H):
    a = np.full((H, W, 3), 40, np.uint8)
    a[:, W // 2:] = 215
    return a


def charts():
    out = {}
    W, H = 768, 512
    xs = np.linspace(0, 1, W)
    ramp = np.tile(xs[None, :], (H, 1))
    out["chart_ramp"] = np.repeat((ramp * 255 + 0.5).astype(np.uint8)[..., None], 3, 2)
    out["chart_gradient"] = np.repeat(((0.45 + 0.10 * ramp) * 255 + 0.5).astype(np.uint8)[..., None], 3, 2)
    for lv in (24, 64, 192, 235):
        out[f"chart_flat{lv}"] = flat(lv)
    out["chart_flat128_large"] = flat(128, 1536, 1024)   # grain field covers every held-out coordinate
    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 128, 0), (128, 0, 255), (0, 200, 90), (230, 40, 90), (20, 90, 220), (200, 200, 30)]
    p = np.zeros((H, W, 3), np.uint8)
    for k, c in enumerate(cols):
        r, q = divmod(k, 4)
        p[r * H // 3:(r + 1) * H // 3, q * W // 4:(q + 1) * W // 4] = c
    out["chart_patches"] = p
    hh = np.tile(xs[None, :], (H, 1))
    ss = np.tile(np.linspace(1, 0.3, H)[:, None], (1, W))
    out["chart_hue"] = (hsv_to_rgb(hh, ss, np.full_like(hh, 0.9)) * 255 + 0.5).astype(np.uint8)
    out["chart_point"] = point(W, H)
    out["chart_point_large"] = point(1536, 1024)
    out["chart_edge_v"] = edge_v(W, H)
    out["chart_edge_v_large"] = edge_v(1280, 960)
    yy, xx = np.mgrid[0:H, 0:W]
    ang = np.deg2rad(30)
    d = (xx - W / 2) * np.cos(ang) + (yy - H / 2) * np.sin(ang)
    e2 = np.full((H, W, 3), 40, np.uint8)
    e2[d > 0] = 215
    out["chart_edge_30"] = e2
    return out


def save(a, path):
    Image.fromarray(a, "RGB").save(path)


def main():
    ch = charts()
    assert len(ch) == 14, len(ch)
    items = list(ch.items()) + [(f"photo{i:02d}", photo(1000 + i, PUB_SIZES[i % len(PUB_SIZES)])) for i in range(26)]
    pairs = os.path.join(ROOT, "environment", "data", "pairs")
    held = os.path.join(ROOT, "tests", "heldout")
    for d in (pairs, held):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for k, (name, a) in enumerate(items):
        save(a, os.path.join(pairs, f"{k:02d}_src.png"))
        save(finish(a), os.path.join(pairs, f"{k:02d}_out.png"))
    # most held-out images sit in the unclamped part of the exposure rule, where the
    # exact statistic matters; one or two sit on each clamp so every branch is exercised
    want = ["mid", "mid", "mid", "mid", "mid", "mid", "mid", "low", "high", "mid"]
    for k in range(10):
        seed = 5000 + k * 1000
        while True:
            a = photo(seed, HELD_SIZES[k])
            g = auto_gain(srgb_to_lin(a / 255.0))
            cat = "low" if g <= 0.7001 else "high" if g >= 1.5999 else "mid"
            if cat == want[k] and (cat != "mid" or 0.8 <= g <= 1.45):
                break
            seed += 1
        save(a, os.path.join(held, f"{k:02d}_src.png"))
        save(finish(a), os.path.join(held, f"{k:02d}_ref.png"))


if __name__ == "__main__":
    main()
