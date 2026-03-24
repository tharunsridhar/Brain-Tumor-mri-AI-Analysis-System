from __future__ import annotations

from src.features.prior_case_comparison import compare_with_prior
from src.features.structured_report import build_report_payload, generate_structured_report


def build_case_report(
    *,
    filename: str,
    label: str,
    confidence: float,
    size_info: dict,
    shape_info: dict | None,
    mass_info: dict,
    risk_info: dict,
    history: list[dict],
    patient_name: str = '',
    groq_api_key: str = '',
) -> dict[str, object]:
    progression_info = compare_with_prior(history, filename, size_info['area_cm2'])
    payload = build_report_payload(
        label,
        confidence,
        size_info,
        shape_info,
        mass_info,
        risk_info,
        progression_info,
    )
    report_text = generate_structured_report(payload, patient_name=patient_name, groq_api_key=groq_api_key)
    return {
        'payload': payload,
        'progression_info': progression_info,
        'report_text': report_text,
    }
