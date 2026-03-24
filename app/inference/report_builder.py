from __future__ import annotations

from textwrap import dedent


def build_report_payload(label, confidence, size_info, shape_info, mass_info, risk_info, progression_info):
    return {
        'label': label,
        'confidence': round(float(confidence), 4),
        'size_info': size_info,
        'shape_info': shape_info,
        'mass_info': mass_info,
        'risk_info': risk_info,
        'progression_info': progression_info,
    }


def build_report_prompt(payload, patient_name=''):
    pt_line = f'Patient Name: {patient_name}\n' if patient_name else ''
    return dedent(
        f'''
        {pt_line}AI ANALYSIS FINDINGS:
        - Tumor Type: {payload['label'].upper()} | Confidence: {payload['confidence']:.2%}
        - Tumor Area: {payload['size_info']['area_cm2']} cm2 | Diameter: {payload['size_info']['diameter_cm']} cm
        - Laterality: {payload['mass_info']['laterality']} | Midline Shift: {payload['mass_info']['shift_mm']} mm
        - Risk Score: {payload['risk_info']['score']}/1.0 | Growth Risk: {payload['risk_info']['risk']}
        - Progression Flag: {payload['progression_info']['progression_flag']}

        Generate a structured report with these 6 sections:
        1. Clinical Indication
        2. Imaging Technique
        3. Findings
        4. Impression
        5. Severity Assessment
        6. Recommendations
        '''
    ).strip()


def fallback_report(payload, patient_name=''):
    label = payload['label'].replace('_', ' ').title()
    confidence = payload['confidence'] * 100.0
    size_info = payload['size_info']
    mass_info = payload['mass_info']
    risk_info = payload['risk_info']
    progression = payload['progression_info']
    return dedent(
        f'''
        1. CLINICAL INDICATION
        Brain MRI reviewed for automated tumor analysis{' for ' + patient_name if patient_name else ''}. The current AI pipeline was used for classification, lesion measurement, and clinical risk support.

        2. IMAGING TECHNIQUE
        Single uploaded MRI image processed through quality validation, fusion-based classification, lesion quantification, and history comparison. Structured output generated from the combined imaging features.

        3. FINDINGS
        The fused prediction is {label} with confidence {confidence:.1f}%. Estimated lesion area is {size_info['area_cm2']} cm2 with equivalent diameter {size_info['diameter_cm']} cm. Laterality is {mass_info['laterality']} with estimated shift {mass_info['shift_mm']} mm.

        4. IMPRESSION
        Imaging features suggest {label}. This output is intended as decision support and should be correlated clinically and confirmed by expert review.

        5. SEVERITY ASSESSMENT
        Severity is assessed as {risk_info['severity']} and overall risk as {risk_info['risk']}. Reliability score is {risk_info['reliability_score']} and progression status is {progression['progression_flag']}.

        6. RECOMMENDATIONS
        Recommend neuroradiology or neurosurgical review as appropriate. Histopathological confirmation remains required for definitive diagnosis.
        '''
    ).strip()


def generate_structured_report(payload, patient_name='', groq_api_key=''):
    if not groq_api_key:
        return fallback_report(payload, patient_name=patient_name)
    try:
        from groq import Groq
    except Exception:
        return fallback_report(payload, patient_name=patient_name)
    client = Groq(api_key=groq_api_key)
    prompt = build_report_prompt(payload, patient_name=patient_name)
    response = client.chat.completions.create(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=900,
    )
    return response.choices[0].message.content.strip()


def run_demo():
    from app.analytics.tumor_size import estimate_size
    from app.analytics.shape_irregularity import analyze_shape, mass_effect
    from app.analytics.explainability_overlap import overlap_metrics
    from app.analytics.risk_scoring import reliability_and_risk
    from app.analytics.prior_case_comparison import compare_with_prior
    from app.analytics.common import sample_heatmap, sample_history, sample_mask

    mask = sample_mask(center=(62, 74))
    heat = sample_heatmap(mask)
    size_info = estimate_size(mask)
    shape_info = analyze_shape(mask)
    mass_info = mass_effect(mask)
    overlap = overlap_metrics(heat, mask)
    risk_info = reliability_and_risk('glioma', 0.84, 0.67, 0.72, size_info, shape_info, mass_info, overlap['overlap_score'])
    progression = compare_with_prior(sample_history(), 'sample_case.jpg', size_info['area_cm2'])
    payload = build_report_payload('glioma', 0.84, size_info, shape_info, mass_info, risk_info, progression)
    return fallback_report(payload, patient_name='Demo Patient')


if __name__ == '__main__':
    print(run_demo())

