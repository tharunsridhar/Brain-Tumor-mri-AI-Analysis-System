from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
SAMPLE_DIR = DATA_DIR / 'sample'
MODELS_DIR = PROJECT_ROOT / 'models'
CLASSIFICATION_MODELS_DIR = MODELS_DIR / 'Classification'
SEGMENTATION_MODELS_DIR = MODELS_DIR / 'Segmentation'
REPORTS_DIR = PROJECT_ROOT / 'reports'
CASE_HISTORY_PATH = REPORTS_DIR / 'case_history.json'

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
DEFAULT_CONFIDENCE = 0.72

DEFAULT_CLASSIFICATION_MODEL = CLASSIFICATION_MODELS_DIR / 'Tumor_v2s_clean.keras'
DEFAULT_SEGMENTATION_MODEL = SEGMENTATION_MODELS_DIR / 'unet_lgg_segmentation_u-net.keras'
