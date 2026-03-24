from __future__ import annotations

from collections import Counter


def summarize_history(history):
    counts = Counter(row.get('predicted_label', 'unknown') for row in history)
    total_cases = len(history)
    latest = history[-1] if history else None
    return {
        'total_cases': total_cases,
        'label_counts': dict(counts),
        'latest_case': latest,
    }


def latest_cases(history, limit=5):
    return list(history[-limit:])[::-1]


def run_demo():
    history = [
        {'filename': 'scan_a.jpg', 'predicted_label': 'glioma', 'area_cm2': 4.1},
        {'filename': 'scan_b.jpg', 'predicted_label': 'meningioma', 'area_cm2': 2.4},
        {'filename': 'scan_c.jpg', 'predicted_label': 'glioma', 'area_cm2': 5.0},
    ]
    return {'summary': summarize_history(history), 'latest': latest_cases(history, limit=2)}


if __name__ == '__main__':
    print(run_demo())
