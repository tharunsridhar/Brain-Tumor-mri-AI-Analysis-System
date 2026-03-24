from __future__ import annotations

from tests.conftest import client, make_heatmap_png, make_mask_png, make_test_png


def test_health_endpoint_returns_status():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert 'classifiers_loaded' in data


def test_analyze_endpoint_returns_quality_and_mask_metrics():
    files = {
        'image': ('scan.png', make_test_png(), 'image/png'),
        'mask': ('mask.png', make_mask_png(), 'image/png'),
        'heatmap': ('heatmap.png', make_heatmap_png(), 'image/png'),
    }
    response = client.post('/analyze', files=files)
    assert response.status_code == 200
    data = response.json()
    assert data['quality']['quality_score'] >= 0
    assert data['size_info']['tumor_pixels'] > 0
    assert data['overlap']['overlap_score'] >= 0


def test_predict_endpoint_fails_honestly_when_models_are_missing():
    response = client.post('/predict', files={'image': ('scan.png', make_test_png(), 'image/png')})
    assert response.status_code == 503
