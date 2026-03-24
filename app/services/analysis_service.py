from __future__ import annotations

import json
from typing import Any

from app.analytics.explainability_overlap import overlap_metrics
from app.analytics.prior_case_comparison import compare_with_prior
from app.analytics.risk_scoring import reliability_and_risk
from app.analytics.scan_quality import quality_metrics
from app.analytics.shape_irregularity import analyze_shape, mass_effect
from app.analytics.tumor_size import estimate_size
from app.core.exceptions import ModelUnavailableError
from app.services.model_registry import ModelRegistry


def parse_history(history_raw: str | None) -> list[dict[str, Any]]:
    if not history_raw:
        return []
    data = json.loads(history_raw)
    if not isinstance(data, list):
        raise ValueError("history must be a JSON array")
    return data


def build_analysis(image_np, filename: str, registry: ModelRegistry, mask_np=None, heatmap_np=None, history=None):
    history = history or []
    quality = quality_metrics(image_np)
    response: dict[str, Any] = {
        "filename": filename,
        "quality": quality,
        "prediction": None,
        "size_info": None,
        "shape_info": None,
        "mass_info": None,
        "overlap": None,
        "risk_info": None,
        "progression_info": None,
        "warnings": [],
    }

    try:
        response["prediction"] = registry.predict(image_np)
    except ModelUnavailableError as exc:
        response["warnings"].append(str(exc))

    if mask_np is None:
        if registry.segmentation_available:
            try:
                mask_np = registry.segment(image_np)
            except ModelUnavailableError as exc:
                response["warnings"].append(str(exc))
        else:
            response["warnings"].append("No mask was supplied and no segmentation model is configured.")

    if mask_np is not None:
        size_info = estimate_size(mask_np)
        shape_info = analyze_shape(mask_np)
        mass_info = mass_effect(mask_np)
        response["size_info"] = size_info
        response["shape_info"] = shape_info
        response["mass_info"] = mass_info

        if heatmap_np is not None:
            response["overlap"] = overlap_metrics(heatmap_np, mask_np)

        if response["prediction"] is not None:
            overlap_score = response["overlap"]["overlap_score"] if response["overlap"] else 0.0
            response["risk_info"] = reliability_and_risk(
                response["prediction"]["final_class"],
                response["prediction"]["fused_confidence"],
                response["prediction"]["agreement_score"],
                quality["quality_score"],
                size_info,
                shape_info,
                mass_info,
                overlap_score,
            )
            response["progression_info"] = compare_with_prior(history, filename, size_info["area_cm2"])
        else:
            response["warnings"].append("Risk scoring was skipped because no classifier prediction is available.")
    return response
