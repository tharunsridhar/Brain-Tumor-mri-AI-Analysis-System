from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CASE_HISTORY_PATH, REPORTS_DIR, SAMPLE_DIR
from app.pipeline.classification import classify_scan
from app.pipeline.explainability import generate_heatmap
from app.pipeline.report import build_case_report
from app.pipeline.segmentation import segment_scan
from app.pipeline.utils import ensure_reports_dir, load_image
from src.features.explainability_overlap import overlap_metrics
from src.features.history_dashboard import latest_cases, summarize_history
from src.features.risk_scoring import reliability_and_risk
from src.features.scan_quality import quality_metrics
from src.features.shape_irregularity import analyze_shape, mass_effect
from src.features.tumor_size import estimate_size

st.set_page_config(page_title='NeuroScan AI', layout='wide')


def load_history() -> list[dict]:
    if not CASE_HISTORY_PATH.exists():
        return []
    try:
        return json.loads(CASE_HISTORY_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []


def save_history(history: list[dict]) -> None:
    ensure_reports_dir(REPORTS_DIR)
    CASE_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding='utf-8')


def run_pipeline(uploaded_file, patient_name: str = '') -> dict[str, object]:
    image_np = load_image(uploaded_file)
    classification = classify_scan(image_np)
    label = classification['final_class']
    confidence = float(classification['confidence'])

    segmentation = segment_scan(image_np, label)
    explainability = generate_heatmap(image_np, segmentation['mask'])

    quality_info = quality_metrics(image_np)
    size_info = estimate_size(segmentation['mask'])
    shape_info = analyze_shape(segmentation['mask'])
    mass_info = mass_effect(segmentation['mask'])
    overlap_info = overlap_metrics(explainability['heat_values'], segmentation['mask'])
    risk_info = reliability_and_risk(
        label,
        confidence,
        classification['agreement_score'],
        quality_info['quality_score'],
        size_info,
        shape_info,
        mass_info,
        overlap_info['overlap_score'],
    )

    history = load_history()
    report_info = build_case_report(
        filename=uploaded_file.name,
        label=label,
        confidence=confidence,
        size_info=size_info,
        shape_info=shape_info,
        mass_info=mass_info,
        risk_info=risk_info,
        history=history,
        patient_name=patient_name,
    )

    history.append(
        {
            'filename': uploaded_file.name,
            'predicted_label': label,
            'area_cm2': size_info['area_cm2'],
            'severity': risk_info['severity'],
        }
    )
    save_history(history)

    return {
        'image': image_np,
        'classification': classification,
        'segmentation': segmentation,
        'explainability': explainability,
        'quality_info': quality_info,
        'size_info': size_info,
        'shape_info': shape_info,
        'mass_info': mass_info,
        'overlap_info': overlap_info,
        'risk_info': risk_info,
        'report_info': report_info,
        'history': history,
    }


st.title('NeuroScan AI')
st.caption('Brain MRI analysis workspace organized for app, source modules, sample data, and reports.')

with st.sidebar:
    st.subheader('Demo Data')
    samples = sorted(SAMPLE_DIR.glob('*'))
    for sample in samples[:3]:
        st.write(sample.name)
    st.caption('Heavy trained models are intentionally excluded from the repository.')

patient_name = st.text_input('Patient name', placeholder='Optional')
uploaded = st.file_uploader('Upload MRI scan', type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded is not None:
    result = run_pipeline(uploaded, patient_name=patient_name)
    class_info = result['classification']
    risk_info = result['risk_info']
    size_info = result['size_info']

    st.subheader('Analysis Summary')
    col1, col2, col3 = st.columns(3)
    col1.metric('Predicted Class', class_info['final_class'].replace('_', ' ').title())
    col2.metric('Confidence', f"{class_info['confidence'] * 100:.1f}%")
    col3.metric('Severity', risk_info['severity'])

    col4, col5, col6 = st.columns(3)
    col4.metric('Tumor Area', f"{size_info['area_cm2']} cm2")
    col5.metric('Diameter', f"{size_info['diameter_cm']} cm")
    col6.metric('Risk Score', f"{risk_info['score']:.3f}")

    st.subheader('Visual Outputs')
    v1, v2, v3 = st.columns(3)
    v1.image(result['image'], caption='Uploaded MRI', use_container_width=True)
    v2.image(result['segmentation']['mask_overlay'], caption='Segmentation Overlay', use_container_width=True)
    v3.image(result['explainability']['overlay'], caption='Explainability Overlay', use_container_width=True)

    st.subheader('Structured Report')
    st.text(result['report_info']['report_text'])

history = load_history()
if history:
    st.subheader('History Snapshot')
    st.json(
        {
            'summary': summarize_history(history),
            'latest': latest_cases(history, limit=3),
        }
    )
