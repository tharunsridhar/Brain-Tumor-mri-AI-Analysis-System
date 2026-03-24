from __future__ import annotations


def compare_with_prior(history, filename, current_area):
    matches = [row for row in history if row.get('filename') == filename]
    prior_area = None
    for row in reversed(matches):
        area = row.get('area_cm2')
        if isinstance(area, (int, float)):
            prior_area = float(area)
            break
    if prior_area is None or prior_area <= 0:
        return {'prior_available': False, 'change_percent': None, 'progression_flag': 'Unknown'}
    change = ((float(current_area) - prior_area) / prior_area) * 100.0
    return {
        'prior_available': True,
        'change_percent': round(float(change), 2),
        'progression_flag': 'Progression' if change >= 20 else ('Regression' if change <= -20 else 'Stable'),
    }


def run_demo():
    from src.features._common import sample_history
    return compare_with_prior(sample_history(), 'sample_case.jpg', 5.1)


if __name__ == '__main__':
    print(run_demo())
