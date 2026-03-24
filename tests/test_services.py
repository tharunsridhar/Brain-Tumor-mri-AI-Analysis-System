from __future__ import annotations

import pytest

from app.core.exceptions import InvalidImageError
from app.services.analysis_service import parse_history
from app.services.image_io import load_image_array


def test_parse_history_requires_array():
    with pytest.raises(ValueError):
        parse_history('{"bad": true}')


def test_invalid_image_raises_error():
    with pytest.raises(InvalidImageError):
        load_image_array(b'not-an-image')
