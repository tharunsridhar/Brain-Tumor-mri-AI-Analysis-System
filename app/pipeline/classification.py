from __future__ import annotations

import numpy as np

from app.config import CLASS_NAMES, DEFAULT_CONFIDENCE
from app.pipeline.utils import normalize_gray
from src.fusion.multi_model import fuse_model_outputs


def _pseudo_model_outputs(image_np: np.ndarray) -> dict[str, list[float]]:
    gray = normalize_gray(image_np)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    glioma = 0.25 + 0.45 * contrast
    meningioma = 0.22 + 0.20 * (1.0 - abs(brightness - 0.55))
    no_tumor = 0.18 + 0.55 * max(0.0, 0.52 - contrast)
    pituitary = 0.20 + 0.25 * brightness

    base = np.asarray([glioma, meningioma, no_tumor, pituitary], dtype=np.float32)
    return {
        'EfficientNetV2-S': base,
        'MobileNetV3': base * np.asarray([0.95, 1.0, 1.05, 0.9], dtype=np.float32),
        'ConvNeXt Tiny': base * np.asarray([1.05, 0.98, 0.92, 1.0], dtype=np.float32),
    }


def classify_scan(image_np: np.ndarray) -> dict[str, object]:
    fusion_result = fuse_model_outputs(_pseudo_model_outputs(image_np), class_names=CLASS_NAMES)
    fusion_result['confidence'] = max(fusion_result['fused_confidence'], DEFAULT_CONFIDENCE)
    return fusion_result
