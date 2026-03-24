from __future__ import annotations


class NeuroScanError(Exception):
    """Base application error."""


class InvalidImageError(NeuroScanError):
    """Raised when the uploaded image cannot be parsed."""


class ModelUnavailableError(NeuroScanError):
    """Raised when a required model file is not configured or cannot be loaded."""
