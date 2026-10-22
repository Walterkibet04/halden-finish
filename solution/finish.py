#!/usr/bin/env python3
"""Reference implementation of the Halden finish (oracle solution)."""
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.int64)

LUMA = (0.2126, 0.7152, 0.0722)


def srgb_to_lin(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(v):
    v = np.clip(v, 0.0, None)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def luma(img):
    return img[..., 0] * LUMA[0] + img[..., 1] * LUMA[1] + img[..., 2] * LUMA[2]


def auto_gain(lin):
    """Image-wide exposure: pull the geometric-mean luma towards a fixed target."""
    L = (lin[..., 0] * LUMA[0] + lin[..., 1] * LUMA[1] + lin[..., 2] * LUMA[2]).astype(np.float64)
    gm = np.exp(np.mean(np.log(L + 1e-4)))
    return float(np.clip(0.17 / gm, 0.70, 1.60))


def bloom(lin, h, w):
    """Highlight halation: bright luma spills into a warm, wide glow (linear light)."""
    L = lin[..., 0] * LUMA[0] + lin[..., 1] * LUMA[1] + lin[..., 2] * LUMA[2]
    hl = np.clip(L - 0.65, 0.0, None)
    glow = gaussian_filter(hl, sigma=0.04 * max(h, w), mode="constant", truncate=4.0)
    tint = np.array([1.0, 0.55, 0.35], dtype=lin.dtype)
    return lin + 0.80 * glow[..., None] * tint


def tone(lin):
    x = lin
    knee = 0.70
    hi = x > knee
    x = np.where(hi, knee + 0.30 * np.tanh((x - knee) / 0.30), x)
    return 0.006 + x * (1.0 - 0.006)


def vignette(lin, h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2.0)
    f = 1.0 - 0.38 * r ** 2.5
    return lin * f[..., None]


def local_contrast(g):
    h, w, _ = g.shape
    L = luma(g)
    Lb = gaussian_filter(L, sigma=0.012 * max(h, w), mode="reflect", truncate=4.0)
    return g + 0.55 * (L - Lb)[..., None]


def saturate(g):
    L = luma(g)[..., None]
    out = L + (g - L) * 1.28
    return np.clip(out, 0.0, 1.0)


def hash_u32(h, w, c, seed=0x9E3779B1):
    y = np.arange(h, dtype=np.uint32)[:, None, None]
    x = np.arange(w, dtype=np.uint32)[None, :, None]
    ch = np.arange(c, dtype=np.uint32)[None, None, :]
    with np.errstate(over="ignore"):
        v = (x * np.uint32(73856093)) ^ (y * np.uint32(19349663)) ^ (ch * np.uint32(83492791)) ^ np.uint32(seed)
        v = v ^ (v >> np.uint32(16))
        v = v * np.uint32(0x7FEB352D)
        v = v ^ (v >> np.uint32(15))
        v = v * np.uint32(0x846CA68B)
        v = v ^ (v >> np.uint32(16))
    return v


def grain(g, dt):
    h, w, c = g.shape
    v = hash_u32(h, w, c)
    u = ((v >> np.uint32(8)).astype(np.float64) / 16777216.0 * 2.0 - 1.0).astype(dt)
    L = luma(g)[..., None]
    amp = 0.020 * (0.25 + 0.75 * 4.0 * L * (1.0 - L))
    return g + u * amp


def quantise(g, dt):
    h, w, _ = g.shape
    thr = ((BAYER8[np.arange(h)[:, None] % 8, np.arange(w)[None, :] % 8] + 0.5) / 64.0).astype(dt)
    q = np.floor(np.clip(g, 0.0, 1.0) * dt(255.0) + thr[..., None])
    return np.clip(q, 0, 255).astype(np.uint8)


def finish(arr_u8, dt=np.float32):
    h, w, _ = arr_u8.shape
    g = (arr_u8.astype(dt) / dt(255.0))
    lin = srgb_to_lin(g).astype(dt)
    lin = (lin * dt(auto_gain(lin))).astype(dt)
    lin = bloom(lin, h, w).astype(dt)
    lin = tone(lin).astype(dt)
    lin = vignette(lin, h, w).astype(dt)
    g = lin_to_srgb(lin).astype(dt)
    g = local_contrast(g).astype(dt)
    g = saturate(g).astype(dt)
    g = grain(g, dt).astype(dt)
    return quantise(g, dt)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    arr = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(finish(arr), "RGB").save(dst)


if __name__ == "__main__":
    main()
