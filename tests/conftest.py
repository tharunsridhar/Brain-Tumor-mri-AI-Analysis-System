from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def make_test_png(width: int = 32, height: int = 32, value: int = 120) -> bytes:
    data = np.full((height, width, 3), value, dtype=np.uint8)
    image = Image.fromarray(data, mode='RGB')
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def make_mask_png(width: int = 32, height: int = 32) -> bytes:
    data = np.zeros((height, width), dtype=np.uint8)
    data[8:24, 10:22] = 255
    image = Image.fromarray(data, mode='L')
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def make_heatmap_png(width: int = 32, height: int = 32) -> bytes:
    data = np.zeros((height, width), dtype=np.uint8)
    data[8:24, 10:22] = 220
    image = Image.fromarray(data, mode='L')
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


with TestClient(app) as _client:
    client = _client
