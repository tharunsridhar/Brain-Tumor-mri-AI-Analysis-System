from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(image_source) -> np.ndarray:
    image = Image.open(image_source).convert('RGB')
    return np.asarray(image, dtype=np.uint8)


def normalize_gray(image_np: np.ndarray) -> np.ndarray:
    gray = image_np.astype(np.float32).mean(axis=2)
    gray -= gray.min()
    peak = float(gray.max())
    if peak > 0:
        gray /= peak
    return gray


def mask_to_rgb(mask: np.ndarray, color=(255, 64, 64)) -> np.ndarray:
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    overlay[mask > 0] = np.asarray(color, dtype=np.uint8)
    return overlay


def blend_overlay(image_np: np.ndarray, overlay_np: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    base = image_np.astype(np.float32)
    overlay = overlay_np.astype(np.float32)
    mixed = (1.0 - alpha) * base + alpha * overlay
    return np.clip(mixed, 0, 255).astype(np.uint8)


def ensure_reports_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
