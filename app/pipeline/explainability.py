from __future__ import annotations

import numpy as np

from app.pipeline.utils import blend_overlay, mask_to_rgb, normalize_gray


def generate_heatmap(image_np: np.ndarray, seg_mask: np.ndarray) -> dict[str, np.ndarray]:
    gray = normalize_gray(image_np)
    heat = 0.65 * gray + 0.35 * seg_mask.astype(np.float32)
    heat = np.clip(heat, 0.0, 1.0)

    overlay = np.zeros_like(image_np, dtype=np.uint8)
    overlay[..., 0] = (heat * 255).astype(np.uint8)
    overlay[..., 1] = (np.clip(1.0 - np.abs(heat - 0.5) * 2.0, 0.0, 1.0) * 180).astype(np.uint8)
    overlay[..., 2] = (np.clip(1.0 - heat, 0.0, 1.0) * 90).astype(np.uint8)

    return {
        'heatmap': overlay,
        'overlay': blend_overlay(image_np, overlay, alpha=0.45),
        'focus_overlay': blend_overlay(image_np, mask_to_rgb(seg_mask, color=(255, 180, 0)), alpha=0.25),
        'heat_values': heat,
    }
