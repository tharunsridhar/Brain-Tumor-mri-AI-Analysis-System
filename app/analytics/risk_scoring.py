from __future__ import annotations


def reliability_and_risk(label, confidence, agreement, quality_score, size_info, shape_info, mass_info, overlap_score):
    if label == 'no_tumor':
        reliability = (0.35 * confidence + 0.25 * agreement + 0.20 * quality_score + 0.20 * overlap_score)
        return {
            'severity': 'None',
            'risk': 'None',
            'clinical_priority': 'Routine',
            'reliability_score': round(float(reliability), 4),
            'progression_risk': '0%',
            'score': 0.0,
        }
    shape_irregularity = shape_info['irregularity'] if shape_info else 0.0
    severity_index = (
        min(size_info['area_cm2'] / 25.0, 1.0) * 0.25
        + min(size_info['diameter_cm'] / 6.0, 1.0) * 0.15
        + min(size_info['tumor_percent'] / 30.0, 1.0) * 0.15
        + min(size_info['volume_cm3'] / 20.0, 1.0) * 0.10
        + min(shape_irregularity, 1.0) * 0.15
        + min(mass_info['shift_mm'] / 10.0, 1.0) * 0.10
        + min(overlap_score, 1.0) * 0.10
    )
    reliability = (0.30 * confidence + 0.25 * agreement + 0.20 * quality_score + 0.25 * overlap_score)
    severity = 'Severe' if severity_index >= 0.70 else ('Moderate' if severity_index >= 0.40 else 'Mild')
    risk = 'High' if severity_index >= 0.75 else ('Moderate' if severity_index >= 0.45 else 'Low')
    priority = 'Urgent' if severity_index >= 0.75 else ('Priority' if severity_index >= 0.45 else 'Routine')
    progression_risk = f"{int(severity_index * 100)}%" if severity_index >= 0.70 else (f"{int(severity_index * 80)}%" if severity_index >= 0.40 else f"{int(severity_index * 60)}%")
    return {
        'severity': severity,
        'risk': risk,
        'clinical_priority': priority,
        'reliability_score': round(float(reliability), 4),
        'progression_risk': progression_risk,
        'score': round(float(severity_index), 3),
    }


def run_demo():
    from app.analytics.tumor_size import estimate_size
    from app.analytics.shape_irregularity import analyze_shape, mass_effect
    from app.analytics.explainability_overlap import overlap_metrics
    from app.analytics.common import sample_heatmap, sample_mask

    mask = sample_mask(center=(60, 76))
    heatmap = sample_heatmap(mask)
    size_info = estimate_size(mask)
    shape_info = analyze_shape(mask)
    mass_info = mass_effect(mask)
    overlap = overlap_metrics(heatmap, mask)
    return reliability_and_risk('glioma', 0.82, 0.67, 0.74, size_info, shape_info, mass_info, overlap['overlap_score'])


if __name__ == '__main__':
    print(run_demo())

