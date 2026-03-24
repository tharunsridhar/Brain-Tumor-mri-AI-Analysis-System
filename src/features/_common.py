from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

PIX_MM = 0.5
MM2_CM2 = 0.01


def ensure_mask(mask):
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError('mask must be a 2D array')
    return (arr > 0).astype(np.uint8)


def bbox_from_mask(mask):
    mask = ensure_mask(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    return {
        'x': x0,
        'y': y0,
        'w': w,
        'h': h,
        'width_cm': round(w * PIX_MM / 10.0, 2),
        'height_cm': round(h * PIX_MM / 10.0, 2),
    }


def border_pixel_count(mask):
    mask = ensure_mask(mask)
    padded = np.pad(mask, 1)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    interior = mask & up & down & left & right
    border = mask & (~interior)
    return int(border.sum())


def load_env_file(env_path):
    values = {}
    path = Path(env_path)
    if not path.exists():
        return values
    for line in path.read_text(encoding='utf-8').splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def sample_history():
    return [
        {'filename': 'sample_case.jpg', 'area_cm2': 3.2, 'timestamp': '2026-03-20T10:00:00'},
        {'filename': 'sample_case.jpg', 'area_cm2': 4.1, 'timestamp': '2026-03-21T10:00:00'},
        {'filename': 'other_case.jpg', 'area_cm2': 1.4, 'timestamp': '2026-03-19T10:00:00'},
    ]


def sample_mask(shape=(128, 128), center=(64, 64), radius_y=18, radius_x=24):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    norm = ((yy - center[0]) / float(radius_y)) ** 2 + ((xx - center[1]) / float(radius_x)) ** 2
    return (norm <= 1.0).astype(np.uint8)


def sample_heatmap(mask):
    mask = ensure_mask(mask)
    heat = mask.astype(np.float32) * 0.8
    heat += np.roll(mask.astype(np.float32), 6, axis=1) * 0.15
    return np.clip(heat, 0.0, 1.0)


def sample_image(shape=(128, 128, 3)):
    h, w, c = shape
    x = np.linspace(20, 220, w, dtype=np.float32)
    y = np.linspace(30, 180, h, dtype=np.float32)[:, None]
    base = (x + y) / 2.0
    image = np.repeat(base[:, :, None], c, axis=2)
    return np.clip(image, 0, 255).astype(np.uint8)


def entropy_from_probs(scores: Iterable[float]):
    probs = np.asarray(list(scores), dtype=np.float32)
    total = float(probs.sum())
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())
