from __future__ import annotations

from app.core.config import Settings
from app.inference.report_builder import build_report_payload, generate_structured_report


class ReportService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, analysis: dict, patient_name: str = "") -> str:
        required_keys = ["prediction", "size_info", "shape_info", "mass_info", "risk_info", "progression_info"]
        missing = [key for key in required_keys if not analysis.get(key)]
        if missing:
            raise ValueError(f"Missing analysis sections for report generation: {', '.join(missing)}")
        prediction = analysis["prediction"]
        payload = build_report_payload(
            prediction["final_class"],
            prediction["fused_confidence"],
            analysis["size_info"],
            analysis["shape_info"],
            analysis["mass_info"],
            analysis["risk_info"],
            analysis["progression_info"],
        )
        return generate_structured_report(payload, patient_name=patient_name, groq_api_key=self.settings.groq_api_key)
