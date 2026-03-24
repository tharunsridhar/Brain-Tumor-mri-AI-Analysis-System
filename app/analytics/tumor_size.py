from __future__ import annotations

import math
import numpy as np

from app.analytics.common import PIX_MM, MM2_CM2, bbox_from_mask, ensure_mask


def estimate_size(mask):
    mask = ensure_mask(mask)
    tumor_pixels = int(mask.sum())
    total_pixels = int(mask.shape[0] * mask.shape[1])
    area_mm2 = tumor_pixels * (PIX_MM ** 2)
    area_cm2 = area_mm2 * MM2_CM2
    diameter_cm = (2.0 * math.sqrt(area_mm2 / math.pi)) / 10.0 if tumor_pixels else 0.0
    tumor_percent = (tumor_pixels / max(total_pixels * 0.60, 1)) * 100.0
    volume_cm3 = area_cm2 * 0.5
    return {
        'tumor_pixels': tumor_pixels,
        'area_mm2': round(area_mm2, 2),
        'area_cm2': round(area_cm2, 4),
        'diameter_cm': round(diameter_cm, 3),
        'tumor_percent': round(tumor_percent, 2),
        'volume_cm3': round(volume_cm3, 3),
        'bbox': bbox_from_mask(mask),
    }


def run_demo():
    from app.analytics.common import sample_mask
    return estimate_size(sample_mask())


if __name__ == '__main__':
    print(run_demo())

