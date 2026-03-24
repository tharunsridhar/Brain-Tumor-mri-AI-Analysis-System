from __future__ import annotations

import numpy as np

from app.pipeline.utils import blend_overlay, mask_to_rgb, normalize_gray


def segment_scan(image_np: np.ndarray, predicted_label: str) -> dict[str, np.ndarray]:
    gray = normalize_gray(image_np)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = max(h * 0.18, 1.0), max(w * 0.22, 1.0)
    ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    bright_region = gray >= np.quantile(gray, 0.7)
    mask = np.logical_and(ellipse, bright_region).astype(np.uint8)
    if predicted_label == 'no_tumor':
        mask = np.zeros_like(mask, dtype=np.uint8)
    overlay = blend_overlay(image_np, mask_to_rgb(mask))
    return {'mask': mask, 'mask_overlay': overlay}
