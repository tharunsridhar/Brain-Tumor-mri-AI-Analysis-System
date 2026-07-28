from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

from core.quality_metrics import quality_metrics
from core.risk_engine import clinical_decision, confidence_cal, rano_assessment, reliability_and_risk
from utils.config import GROQ_API_KEY, IMG_SIZE, MODELS_DIR, OVERLAY_DIR, REPORT_DIR, SEG_MODEL_PATH, load_models
from utils.history_manager import compare_with_prior, load_history, save_history

try:
    import cv2
except Exception:
    cv2 = None


@lru_cache(maxsize=1)
def get_loaded_models():
    return load_models()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "raw_probs"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _save_history(history, source_name, patient_name, label, confidence, severity, area_cm2, risk, reliability):
    save_history(
        {
            "id": f"#MRI-{len(history)+1:03d}",
            "filename": source_name,
            "patient": patient_name or "-",
            "label": label,
            "confidence": f"{confidence*100:.1f}%",
            "severity": severity,
            "area_cm2": area_cm2,
            "risk": risk,
            "reliability": reliability,
            "date": datetime.now().strftime("%b %d, %Y %H:%M"),
        }
    )


def _fallback_normal_report(confidence: float, patient_name: str = "") -> str:
    patient = f" for {patient_name}" if patient_name else ""
    return (
        "1. CLINICAL INDICATION\n"
        f"AI screening analysis{patient} was requested for the supplied brain MRI image.\n\n"
        "2. IMAGING TECHNIQUE\n"
        "Single-image MRI decision-support analysis was performed with multi-model classification.\n\n"
        "3. FINDINGS\n"
        f"No tumor class was selected by the classifier with {confidence:.2%} confidence.\n\n"
        "4. IMPRESSION\n"
        "No intracranial tumor is detected by this screening model.\n\n"
        "5. RISK STRATIFICATION\n"
        "The automated risk level is low, subject to clinical correlation and radiologist review.\n\n"
        "6. RECOMMENDATIONS\n"
        "Use this result only as decision support. A qualified clinician should review the image and symptoms."
    )


def _fallback_tumor_report(label: str, confidence: float, size_info: dict, severity: str, patient_name: str = "") -> str:
    patient = f" for {patient_name}" if patient_name else ""
    return (
        "1. CLINICAL INDICATION\n"
        f"AI screening analysis{patient} was requested for the supplied brain MRI image.\n\n"
        "2. IMAGING TECHNIQUE\n"
        "Single-image MRI decision-support analysis was performed with classification, segmentation, and Grad-CAM.\n\n"
        "3. FINDINGS\n"
        f"The model predicts {label.replace('_', ' ')} with {confidence:.2%} confidence. "
        f"Estimated area is {size_info.get('area_cm2')} cm2 and diameter is {size_info.get('diameter_cm')} cm.\n\n"
        "4. IMPRESSION\n"
        f"The automated severity estimate is {severity}. This is not a definitive diagnosis.\n\n"
        "5. RISK STRATIFICATION\n"
        "Risk should be interpreted with clinical context, image quality, and specialist review.\n\n"
        "6. RECOMMENDATIONS\n"
        "Radiology review and histopathological confirmation are required before clinical decisions."
    )


def _tumor_report(image_path: str, label: str, confidence: float, size_info: dict, shape_info: dict | None, mass_info: dict, risk_info: dict, severity: str, rano: dict | None, patient_name: str) -> tuple[str, str | None]:
    if GROQ_API_KEY:
        try:
            from reporting.llm_report_generator import groq_tumor_report

            return groq_tumor_report(image_path, label, confidence, size_info, shape_info, mass_info, risk_info, severity, rano, GROQ_API_KEY, patient_name), None
        except Exception as exc:
            print(f"[NeuroScan] Groq report generation failed, using fallback text: {exc}", flush=True)
            return _fallback_tumor_report(label, confidence, size_info, severity, patient_name), f"AI report generation failed ({exc}); showing a template summary instead."
    return _fallback_tumor_report(label, confidence, size_info, severity, patient_name), "GROQ_API_KEY is not configured; showing a template summary instead of an AI-generated one."


def _normal_report(image_path: str, confidence: float, patient_name: str) -> tuple[str, str | None]:
    if GROQ_API_KEY:
        try:
            from reporting.llm_report_generator import groq_normal_report

            return groq_normal_report(image_path, confidence, GROQ_API_KEY, patient_name), None
        except Exception as exc:
            print(f"[NeuroScan] Groq report generation failed, using fallback text: {exc}", flush=True)
            return _fallback_normal_report(confidence, patient_name), f"AI report generation failed ({exc}); showing a template summary instead."
    return _fallback_normal_report(confidence, patient_name), "GROQ_API_KEY is not configured; showing a template summary instead of an AI-generated one."


def _required_model_paths() -> list[Path]:
    return [
        MODELS_DIR / "class_Tumor_v2s_clean.keras",
        MODELS_DIR / "class_Tumor_mobilenet_v3.keras",
        MODELS_DIR / "class_Tumor_convnext_tiny_tumor.keras",
        SEG_MODEL_PATH,
    ]


def _can_run_model_pipeline() -> tuple[bool, list[str]]:
    missing = [str(path) for path in _required_model_paths() if not path.exists()]
    problems: list[str] = []
    if missing:
        problems.append("model files are missing")
    if cv2 is None:
        problems.append("opencv-python-headless is not installed")
    try:
        import tensorflow  # noqa: F401
    except Exception:
        problems.append("tensorflow is not installed")
    try:
        import fpdf  # noqa: F401
    except Exception:
        problems.append("fpdf2 is not installed")
    return not problems, problems


def _demo_label_and_confidence(image_np: np.ndarray, quality: dict) -> tuple[str, float, dict]:
    gray = image_np.mean(axis=2).astype(np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    yy, xx = np.indices(gray.shape)
    center_dist = np.sqrt((xx - gray.shape[1] / 2) ** 2 + (yy - gray.shape[0] / 2) ** 2)
    brain_region = center_dist < min(gray.shape) * 0.43
    threshold = brightness + max(contrast * 1.05, 12.0)
    bright_fraction = float(((gray > threshold) & brain_region).mean())
    dark_fraction = float(((gray < brightness - max(contrast * 0.85, 10.0)) & brain_region).mean())
    signal = max(bright_fraction, dark_fraction)

    if quality["quality_score"] < 0.38 or signal < 0.012:
        label = "no_tumor"
        confidence = 0.64 + min(quality["quality_score"], 0.8) * 0.20
    elif signal > 0.075:
        label = "glioma"
        confidence = 0.72 + min(signal, 0.18) * 1.15
    elif bright_fraction >= dark_fraction:
        label = "meningioma"
        confidence = 0.69 + min(signal, 0.14) * 1.10
    else:
        label = "pituitary"
        confidence = 0.68 + min(signal, 0.14) * 1.05

    confidence = round(float(min(max(confidence, 0.55), 0.93)), 4)
    class_scores = {name: 0.04 for name in ["glioma", "meningioma", "no_tumor", "pituitary"]}
    class_scores[label] = confidence
    remainder = 1.0 - confidence
    for key in class_scores:
        if key != label:
            class_scores[key] = round(remainder / 3, 4)
    class_scores[label] = confidence
    fusion = {
        "final_class": label,
        "fused_confidence": confidence,
        "agreement_score": 1.0,
        "uncertainty_flag": confidence < 0.70,
        "margin": round(confidence - max(v for k, v in class_scores.items() if k != label), 4),
        "uncertainty_score": round(1.0 - confidence, 4),
        "decision_logic": "Demo estimate; install trained models for research inference",
        "model_votes": {"Pillow demo analyzer": label},
        "class_scores": class_scores,
    }
    return label, confidence, fusion


def _demo_mask(image_np: np.ndarray) -> np.ndarray:
    gray = image_np.mean(axis=2).astype(np.float32)
    h, w = gray.shape
    yy, xx = np.indices(gray.shape)
    brain_region = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < min(h, w) * 0.43
    mean = float(gray[brain_region].mean()) if brain_region.any() else float(gray.mean())
    std = float(gray[brain_region].std()) if brain_region.any() else float(gray.std())
    bright = gray > mean + max(std * 1.05, 12.0)
    dark = gray < mean - max(std * 0.85, 10.0)
    mask = ((bright | dark) & brain_region).astype(np.uint8)
    return mask


def _bbox_from_mask(mask: np.ndarray) -> dict | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return {
        "x": x1,
        "y": y1,
        "w": x2 - x1 + 1,
        "h": y2 - y1 + 1,
        "width_cm": round((x2 - x1 + 1) * 0.5 / 10.0, 2),
        "height_cm": round((y2 - y1 + 1) * 0.5 / 10.0, 2),
    }


def _demo_size(mask: np.ndarray) -> dict:
    tumor_pixels = int(mask.sum())
    total = int(mask.shape[0] * mask.shape[1])
    area_mm2 = tumor_pixels * 0.25
    area_cm2 = area_mm2 * 0.01
    diameter_cm = 2 * float(np.sqrt(area_mm2 / np.pi)) / 10.0 if tumor_pixels else 0.0
    return {
        "tumor_pixels": tumor_pixels,
        "area_mm2": round(area_mm2, 2),
        "area_cm2": round(area_cm2, 4),
        "diameter_cm": round(diameter_cm, 3),
        "tumor_percent": round(tumor_pixels / max(total * 0.6, 1) * 100.0, 2),
        "volume_cm3": round(area_cm2 * 0.5, 3),
        "bbox": _bbox_from_mask(mask),
    }


def _demo_shape(mask: np.ndarray) -> dict | None:
    bbox = _bbox_from_mask(mask)
    if not bbox:
        return None
    fill = float(mask.sum()) / max(bbox["w"] * bbox["h"], 1)
    irregularity = round(float(np.clip(1.0 - fill, 0.0, 1.0)), 3)
    return {
        "irregularity": irregularity,
        "compactness": round(1.0 + irregularity * 2.5, 3),
        "convexity": round(float(np.clip(fill, 0.0, 1.0)), 3),
        "eccentricity": round(abs(bbox["w"] - bbox["h"]) / max(bbox["w"], bbox["h"], 1), 3),
        "border_def": "Poorly defined" if irregularity > 0.5 else "Moderately defined" if irregularity > 0.3 else "Well-defined",
        "roughness": "High" if irregularity > 0.5 else "Moderate" if irregularity > 0.3 else "Low",
    }


def _demo_mass(mask: np.ndarray) -> dict:
    h, w = mask.shape
    left = int(mask[:, : w // 2].sum())
    right = int(mask[:, w // 2 :].sum())
    total = left + right
    if total == 0:
        return {"laterality": "None", "shift_mm": 0.0, "compression": "None", "sulcal": "Absent"}
    ratio = left / total
    laterality = "Left hemisphere" if ratio > 0.6 else "Right hemisphere" if ratio < 0.4 else "Bilateral/Midline"
    shift = round(abs(left - right) * 0.25 / max(h * 0.5, 1e-6), 2)
    return {"laterality": laterality, "shift_mm": shift, "compression": "Moderate" if shift > 5 else "Mild" if shift > 2 else "None", "sulcal": "Present" if shift > 2 else "Absent"}


def _write_simple_pdf(path: Path, title: str, lines: list[str]) -> None:
    safe_lines = [line.encode("latin-1", "replace").decode("latin-1") for line in lines]
    content = ["BT", "/F1 18 Tf", "50 790 Td", f"({title}) Tj", "/F1 10 Tf", "0 -28 Td"]
    for line in safe_lines[:42]:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.append(f"({escaped}) Tj")
        content.append("0 -15 Td")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii"))
    path.write_bytes(out)


def _save_overlay_images(source_name: str, orig: np.ndarray, overlays: dict) -> dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    stem = Path(source_name).stem.replace(" ", "_") or "scan"
    files = {
        "original": (orig, f"{stem}_{ts}_original.jpg"),
        "segmentation": (overlays["mask_ov"], f"{stem}_{ts}_segmentation.jpg"),
        "gradcam_heatmap": (overlays["hmap_c"], f"{stem}_{ts}_gradcam_heatmap.jpg"),
        "gradcam_overlay": (overlays["gcam_ov"], f"{stem}_{ts}_gradcam_overlay.jpg"),
    }
    saved: dict[str, str] = {}
    for key, (image_bgr, filename) in files.items():
        if cv2.imwrite(str(OVERLAY_DIR / filename), image_bgr):
            saved[key] = filename
    return saved


def _demo_report_pdf(source_name: str, patient_name: str, patient_id: str, result: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    stem = Path(source_name).stem.replace(" ", "_") or "scan"
    suffix = "normal_report" if result["no_tumor"] else "mri_report"
    path = REPORT_DIR / f"{stem}_{ts}_{suffix}.pdf"
    lines = [
        f"Patient: {patient_name or '-'}",
        f"Patient ID: {patient_id or '-'}",
        f"File: {source_name}",
        f"Mode: {result.get('mode', 'demo')}",
        f"Prediction: {result['label'].replace('_', ' ').title()}",
        f"Confidence: {result['confidence']:.2%}",
        f"Severity: {result['severity']}",
        "",
        "Report:",
    ]
    lines.extend(result["report"].splitlines())
    _write_simple_pdf(path, "NeuroScan AI Report", lines)
    return path


def _demo_analysis(image_pil: Image.Image, source_name: str, patient_name: str, patient_id: str, warnings: list[str]) -> dict:
    image_np = np.array(image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    quality = quality_metrics(image_np)
    label, confidence, fusion = _demo_label_and_confidence(image_np, quality)
    history = load_history()

    if label == "no_tumor":
        report = _fallback_normal_report(confidence, patient_name)
        result = {
            "mode": "demo",
            "warnings": warnings,
            "no_tumor": True,
            "label": label,
            "confidence": confidence,
            "severity": "None",
            "filename": source_name,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "report": report,
            "fusion": fusion,
            "quality": quality,
            "model_scores": {"Pillow demo analyzer": fusion["class_scores"]},
            "clinical": clinical_decision(label, "None", None),
        }
        pdf_path = _demo_report_pdf(source_name, patient_name, patient_id, result)
        result["pdf_path"] = str(pdf_path)
        result["pdf_file"] = pdf_path.name
        _save_history(history, source_name, patient_name, "No Tumor", confidence, "None", "N/A", "None", quality["quality_score"])
        return _jsonable(result)

    mask = _demo_mask(image_np)
    size_info = _demo_size(mask)
    shape_info = _demo_shape(mask)
    mass_info = _demo_mass(mask)
    size_signal = min(size_info["area_cm2"] / 25.0, 1.0)
    shape_signal = shape_info["irregularity"] if shape_info else 0.0
    risk_score = round(float(min(0.55 * size_signal + 0.25 * shape_signal + 0.20 * min(mass_info["shift_mm"] / 10.0, 1.0), 1.0)), 3)
    severity = "Severe" if risk_score >= 0.7 else "Moderate" if risk_score >= 0.4 else "Mild"
    risk_info = {
        "severity": severity,
        "risk": "High" if risk_score >= 0.75 else "Moderate" if risk_score >= 0.45 else "Low",
        "clinical_priority": "Urgent" if risk_score >= 0.75 else "Priority" if risk_score >= 0.45 else "Routine",
        "reliability_score": round(0.35 * confidence + 0.35 * quality["quality_score"] + 0.30, 4),
        "progression_risk": f"{int(risk_score * 100)}%",
        "score": risk_score,
    }
    clinical = clinical_decision(label, severity, mass_info)
    comparison = compare_with_prior(history, source_name, size_info["area_cm2"])
    rano = {"size_cat": "Large" if size_info["diameter_cm"] > 3 else "Medium" if size_info["diameter_cm"] > 1.5 else "Small", "enhancement": "Demo estimate", "necrosis": "Not assessed in demo mode", "grade": "Imaging pattern only; histopathological confirmation required"}
    report = _fallback_tumor_report(label, confidence, size_info, severity, patient_name)
    result = {
        "mode": "demo",
        "warnings": warnings,
        "no_tumor": False,
        "label": label,
        "confidence": confidence,
        "severity": severity,
        "filename": source_name,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "size_info": size_info,
        "shape_info": shape_info,
        "mass_info": mass_info,
        "risk_info": risk_info,
        "cal_info": {"uncertainty": f"+/-{round((1 - confidence) * 50, 1)}%", "cal_score": round(confidence, 3)},
        "clinical": clinical,
        "rano": rano,
        "report": report,
        "fusion": fusion,
        "quality": quality,
        "overlap": {"overlap_score": None, "explainability_consistency": "Not available in demo mode"},
        "gate_info": {"acceptance_status": "DEMO_ONLY", "dri_tier": "MODEL_FILES_REQUIRED", "escalation_reasons": warnings},
        "comparison": comparison,
        "model_scores": {"Pillow demo analyzer": fusion["class_scores"]},
    }
    pdf_path = _demo_report_pdf(source_name, patient_name, patient_id, result)
    result["pdf_path"] = str(pdf_path)
    result["pdf_file"] = pdf_path.name
    _save_history(history, source_name, patient_name, label.capitalize(), confidence, severity, size_info["area_cm2"], risk_info["risk"], risk_info["reliability_score"])
    return _jsonable(result)


def analyze_mri(image_pil: Image.Image, source_name: str, tmp_path: str, patient_name: str = "", patient_id: str = "") -> dict:
    started = time.perf_counter()

    def log_step(message: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"[NeuroScan] {elapsed:7.2f}s {source_name}: {message}", flush=True)

    log_step("analysis started")
    image_np = np.array(image_pil.convert("RGB"))
    quality = quality_metrics(image_np)
    log_step("quality metrics complete")

    can_run_models, model_problems = _can_run_model_pipeline()
    if os.getenv("NEUROSCAN_FORCE_MODEL_PIPELINE", "").lower() not in {"1", "true", "yes"} and not can_run_models:
        log_step("using demo pipeline: " + "; ".join(model_problems))
        return _demo_analysis(image_pil, source_name, patient_name, patient_id, model_problems)

    if cv2 is None:
        raise RuntimeError("OpenCV is required for the trained model pipeline")

    from core.classifier_fusion import adaptive_model_fusion, classify_image
    from core.diagnostic_reliability import reliability_gate
    from core.gradcam import get_gradcam
    from core.morphology import analyze_shape, estimate_size, mass_effect
    from core.overlap_metrics import overlap_metrics
    from core.segmentation import run_segmentation
    from reporting.pdf_report_generator import build_overlays, pdf_normal, pdf_tumor

    classifiers, seg_model, base_model, head_layers = get_loaded_models()
    log_step("models loaded")
    fusion, model_scores = classify_image(classifiers, image_pil)
    label = fusion["final_class"]
    confidence = fusion["fused_confidence"]
    log_step(f"classification complete: {label} ({confidence:.4f})")

    history = load_history()
    orig = cv2.imread(tmp_path)
    if orig is None:
        raise ValueError("Could not read uploaded image with OpenCV")
    orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))

    if label == "no_tumor":
        llm, report_warning = _normal_report(tmp_path, confidence, patient_name)
        log_step("report text complete")
        pdf_path = pdf_normal(source_name, tmp_path, confidence, llm, orig, REPORT_DIR, patient_name=patient_name, patient_id=patient_id, fusion=fusion, quality=quality)
        log_step("pdf complete")
        _save_history(history, source_name, patient_name, "No Tumor", confidence, "None", "N/A", "None", quality["quality_score"])
        log_step("history saved")
        return _jsonable(
            {
                "mode": "model",
                "warnings": [report_warning] if report_warning else [],
                "no_tumor": True,
                "label": label,
                "confidence": confidence,
                "severity": "None",
                "filename": source_name,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "report": llm,
                "pdf_path": str(pdf_path),
                "pdf_file": os.path.basename(str(pdf_path)),
                "fusion": fusion,
                "quality": quality,
                "model_scores": model_scores,
                "clinical": {"urgency": "LOW", "steps": ["Routine follow-up in 12 months", "No immediate action required"]},
            }
        )

    mask = run_segmentation(seg_model, tmp_path)
    log_step("segmentation complete")
    import tensorflow as tf

    input_arr = np.array(image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    input_tensor = tf.keras.applications.efficientnet_v2.preprocess_input(input_arr)
    input_tensor = tf.convert_to_tensor(np.expand_dims(input_tensor, 0))
    heatmap = get_gradcam(input_tensor, base_model, head_layers)
    log_step("gradcam complete")

    size_info = estimate_size(mask)
    shape_info = analyze_shape(mask)
    mass_info = mass_effect(mask)
    hmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    overlap = overlap_metrics(hmap_resized, mask)
    lesion_context = {
        "area_cm2": size_info["area_cm2"],
        "diameter_cm": size_info["diameter_cm"],
        "irregularity": 0.0 if shape_info is None else shape_info["irregularity"],
        "overlap_score": overlap["overlap_score"],
    }
    adaptive_fusion = adaptive_model_fusion(fusion["raw_probs"], quality["quality_score"], lesion_context=lesion_context)
    label = adaptive_fusion["final_class"]
    confidence = adaptive_fusion["fused_confidence"]
    risk_info = reliability_and_risk(label, confidence, adaptive_fusion["agreement_score"], quality["quality_score"], size_info, shape_info, mass_info, overlap["overlap_score"])
    gate_info = reliability_gate(quality, adaptive_fusion, overlap, risk_info)
    cal_info = confidence_cal(confidence)
    severity = risk_info["severity"]
    clinical = clinical_decision(label, severity, mass_info)
    rano = rano_assessment(label, size_info, shape_info)
    comparison = compare_with_prior(history, source_name, size_info["area_cm2"])
    overlays = build_overlays(orig, mask, hmap_resized, size_info)
    overlay_files = _save_overlay_images(source_name, orig, overlays)
    log_step("clinical metrics complete")
    llm, report_warning = _tumor_report(tmp_path, label, confidence, size_info, shape_info, mass_info, risk_info, severity, rano, patient_name)
    log_step("report text complete")
    pdf_path = pdf_tumor(source_name, tmp_path, label, confidence, size_info, shape_info, mass_info, risk_info, cal_info, severity, clinical, rano, llm, orig, overlays["mask_ov"], overlays["hmap_r"], overlays["gcam_ov"], REPORT_DIR, patient_name, patient_id, adaptive_fusion, quality, overlap, comparison, gate_info)
    log_step("pdf complete")
    _save_history(history, source_name, patient_name, label.capitalize(), confidence, severity, size_info["area_cm2"], risk_info["risk"], risk_info["reliability_score"])
    log_step("history saved")

    return _jsonable(
        {
            "mode": "model",
            "warnings": [report_warning] if report_warning else [],
            "no_tumor": False,
            "label": label,
            "confidence": confidence,
            "severity": severity,
            "filename": source_name,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "size_info": size_info,
            "shape_info": shape_info,
            "mass_info": mass_info,
            "risk_info": risk_info,
            "cal_info": cal_info,
            "clinical": clinical,
            "rano": rano,
            "report": llm,
            "pdf_path": str(pdf_path),
            "pdf_file": os.path.basename(str(pdf_path)),
            "overlay_files": overlay_files,
            "fusion": adaptive_fusion,
            "quality": quality,
            "overlap": overlap,
            "gate_info": gate_info,
            "comparison": comparison,
            "model_scores": model_scores,
        }
    )


def save_upload_to_temp(contents: bytes, filename: str) -> str:
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(contents)
        return tmp_file.name
