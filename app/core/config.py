from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "NeuroScan AI API")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_dir: Path = Path(os.getenv("MODEL_DIR", "./models")).resolve()
    efficientnet_model_path: Path = Path(os.getenv("EFFICIENTNET_MODEL_PATH", "./models/class_Tumor_v2s_clean.keras")).resolve()
    mobilenet_model_path: Path = Path(os.getenv("MOBILENET_MODEL_PATH", "./models/class_Tumor_mobilenet_v3.keras")).resolve()
    convnext_model_path: Path = Path(os.getenv("CONVNEXT_MODEL_PATH", "./models/class_Tumor_convnext_tiny_tumor.keras")).resolve()
    segmentation_model_path: Path = Path(os.getenv("SEGMENTATION_MODEL_PATH", "./models/Segmentation_brisc_effunet.keras")).resolve()


def get_settings() -> Settings:
    return Settings()
