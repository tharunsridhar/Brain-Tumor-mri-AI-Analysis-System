from __future__ import annotations

import numpy as np

from app.analytics.common import ensure_mask


def overlap_metrics(heatmap, seg_mask, heat_threshold=0.60):
    heat = np.asarray(heatmap, dtype=np.float32)
    lesion = ensure_mask(seg_mask)
    if heat.shape != lesion.shape:
        raise ValueError('heatmap and seg_mask must have the same shape')
    heat_region = (heat >= float(heat_threshold)).astype(np.uint8)
    inter = int(np.logical_and(heat_region, lesion).sum())
    union = int(np.logical_or(heat_region, lesion).sum())
    hot_pixels = int(heat_region.sum())
    iou = inter / union if union else 0.0
    inside = inter / hot_pixels if hot_pixels else 0.0
    return {
        'overlap_score': round(float(iou), 4),
        'attention_inside_lesion_percent': round(float(inside * 100.0), 2),
        'explainability_consistency': round(float((iou + inside) / 2.0), 4),
    }


def run_demo():
    from app.analytics.common import sample_heatmap, sample_mask
    mask = sample_mask()
    heatmap = sample_heatmap(mask)
    return overlap_metrics(heatmap, mask)


if __name__ == '__main__':
    print(run_demo())

