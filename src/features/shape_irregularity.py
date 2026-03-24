from __future__ import annotations

import math
import numpy as np

from src.features._common import PIX_MM, border_pixel_count, ensure_mask


def analyze_shape(mask):
    mask = ensure_mask(mask)
    area = int(mask.sum())
    if area == 0:
        return None
    border = border_pixel_count(mask)
    if border == 0:
        return None
    perimeter = float(border)
    irregularity_raw = 1.0 - (4.0 * math.pi * area) / max(perimeter ** 2, 1.0)
    irregularity = round(min(max(irregularity_raw, 0.0), 1.0), 3)
    compactness = round((perimeter ** 2) / max(4.0 * math.pi * area, 1.0), 3)
    ys, xs = np.where(mask > 0)
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    convexity = round(area / max(width * height, 1.0), 3)
    eccentricity = 0.0
    major = max(width, height)
    minor = min(width, height)
    if major > 0:
        eccentricity = round(math.sqrt(max(0.0, 1.0 - (minor / major) ** 2)), 3)
    border_def = 'Poorly defined' if irregularity > 0.5 else ('Moderately defined' if irregularity > 0.3 else 'Well-defined')
    roughness = 'High' if irregularity > 0.5 else ('Moderate' if irregularity > 0.3 else 'Low')
    return {
        'irregularity': irregularity,
        'compactness': compactness,
        'convexity': convexity,
        'eccentricity': eccentricity,
        'border_def': border_def,
        'roughness': roughness,
    }


def mass_effect(mask):
    mask = ensure_mask(mask)
    h, w = mask.shape
    mid = w // 2
    left = int(mask[:, :mid].sum())
    right = int(mask[:, mid:].sum())
    total = left + right
    laterality = 'None'
    shift_mm = 0.0
    if total > 0:
        ratio = left / float(total)
        if ratio > 0.6:
            laterality = 'Left hemisphere'
        elif ratio < 0.4:
            laterality = 'Right hemisphere'
        else:
            laterality = 'Bilateral/Midline'
        shift_mm = round(abs(left - right) * (PIX_MM ** 2) / max(h * PIX_MM, 1e-6), 2)
    compression = 'Moderate' if shift_mm > 5 else ('Mild' if shift_mm > 2 else 'None')
    return {
        'laterality': laterality,
        'shift_mm': shift_mm,
        'compression': compression,
        'sulcal': 'Present' if shift_mm > 2 else 'Absent',
    }


def run_demo():
    from src.features._common import sample_mask
    mask = sample_mask(center=(64, 78))
    return {'shape': analyze_shape(mask), 'mass_effect': mass_effect(mask)}


if __name__ == '__main__':
    print(run_demo())
