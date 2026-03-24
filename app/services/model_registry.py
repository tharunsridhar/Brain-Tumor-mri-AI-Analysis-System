from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.exceptions import ModelUnavailableError

LOGGER = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.classifier_paths = {
            "EfficientNetV2-S": settings.efficientnet_model_path,
            "MobileNetV3": settings.mobilenet_model_path,
            "ConvNeXt Tiny": settings.convnext_model_path,
        }
        self.segmentation_path = settings.segmentation_model_path
        self.classifiers_loaded = [name for name, path in self.classifier_paths.items() if Path(path).exists()]
        self.segmentation_available = Path(self.segmentation_path).exists()
        if not self.classifiers_loaded:
            LOGGER.warning("No classifier model files were found. /predict will return 503 until model paths are configured.")
        if not self.segmentation_available:
            LOGGER.warning("No segmentation model file was found. Automatic segmentation is unavailable.")

    def predict(self, image_np: np.ndarray) -> dict:
        if not self.classifiers_loaded:
            raise ModelUnavailableError("Classifier models are not configured. Set model paths in the environment before using /predict.")
        raise ModelUnavailableError("Classifier inference wiring is pending actual model export integration. The backend will not fabricate predictions.")

    def segment(self, image_np: np.ndarray) -> np.ndarray:
        if not self.segmentation_available:
            raise ModelUnavailableError("Segmentation model is not configured. Provide a segmentation model path or upload a mask to /analyze.")
        raise ModelUnavailableError("Segmentation inference wiring is pending actual model export integration. The backend will not fabricate masks.")
