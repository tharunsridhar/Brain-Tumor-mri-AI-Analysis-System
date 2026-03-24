from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.core.exceptions import InvalidImageError


def load_image_array(payload: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive branch
        raise InvalidImageError("Uploaded file is not a valid image") from exc
    return np.asarray(image)


def load_mask_array(payload: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(payload)).convert("L")
    except Exception as exc:  # pragma: no cover - defensive branch
        raise InvalidImageError("Uploaded mask is not a valid image") from exc
    return (np.asarray(image) > 0).astype("uint8")


def load_heatmap_array(payload: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(payload)).convert("L")
    except Exception as exc:  # pragma: no cover - defensive branch
        raise InvalidImageError("Uploaded heatmap is not a valid image") from exc
    return np.asarray(image, dtype="float32") / 255.0
