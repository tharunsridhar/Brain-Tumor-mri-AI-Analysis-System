<div align="center">

# NeuroScan AI

### Brain MRI Screening, Explainability, Reliability Gating, and Automated Reporting

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-App-009688?style=flat-square&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-F57C00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-0C7BDC?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Ready-1E88E5?style=flat-square)

Research-focused MRI analysis system that combines multi-model tumor classification, lesion-aware fusion, segmentation, Grad-CAM explainability, diagnostic reliability scoring, Groq-based report generation, and PDF export in one FastAPI workflow.

</div>

> Important: This repository is built for research, screening, demonstrations, and workflow support. It is not a standalone clinical diagnosis system.

## Table of Contents

- [Overview](#overview)
- [Why This Project Stands Out](#why-this-project-stands-out)
- [System Preview](#system-preview)
- [Pipeline](#pipeline)
- [Model Stack](#model-stack)
- [Results Gallery](#results-gallery)
- [Repository Structure](#repository-structure)
- [Core Modules](#core-modules)
- [Setup](#setup)
- [Testing](#testing)
- [Strengths and Limitations](#strengths-and-limitations)
- [Clinical Safety Note](#clinical-safety-note)

## Overview

NeuroScan AI is designed as a full MRI analysis workflow rather than a basic classifier demo. The app validates scan quality, classifies the scan with three models, refines confidence with lesion-aware fusion, segments the lesion, compares Grad-CAM evidence with the segmentation mask, estimates morphology and risk, applies a Diagnostic Reliability Index (DRI) gate, generates an imaging summary with Groq, and exports a PDF report.

The current FastAPI interface includes:

- `/` for MRI upload and full analysis
- `/api/history` for previously processed cases
- `/api/model-info` for model and imaging metadata

## Why This Project Stands Out

- `Closed-loop design`: classification is not treated as the final step; segmentation and overlap signals feed back into confidence and reliability.
- `Lesion-aware fusion`: the ensemble is adjusted using scan quality, certainty, morphology, and explainability-derived trust.
- `Reliability gating`: the repo includes a DRI-based acceptance layer instead of only returning a prediction.
- `Reporting workflow`: results are turned into structured imaging text and downloadable PDF reports.
- `Presentation-ready assets`: training curves, confusion matrices, classification reports, and architecture visuals are already included.

## System Preview

![System Architecture](docs/system_architecture.png)

## Pipeline

```text
Input MRI
  ->
Scan quality validation
  ->
3-model classification
  ->
Adaptive lesion-aware fusion
  ->
Tumor segmentation
  ->
Grad-CAM heatmap generation
  ->
Tumor size + morphology analysis
  ->
Overlap consistency + DRI gate
  ->
Risk and urgency support
  ->
Groq imaging summary
  ->
PDF report export
```

### Step-by-step

1. **Input MRI** — the user uploads a PNG/JPG/JPEG/BMP scan through `/api/analyze` ([app/main.py](app/main.py)). File type and size (default 25 MB max) are validated before anything else runs, so bad input fails fast instead of wasting a model load.

2. **Scan quality validation** — [`core/quality_metrics.py`](core/quality_metrics.py) scores blur, brightness, and contrast into a single `quality_score`. This runs *before* classification because a low-quality scan makes every downstream prediction unreliable; the score is carried forward and factored into the reliability gate later instead of silently trusting every image as diagnostic-grade.

3. **3-model classification** — [`core/classifier_fusion.py`](core/classifier_fusion.py) runs the scan through three independently trained CNNs — EfficientNetV2-S, MobileNetV3, and ConvNeXt Tiny — each predicting `glioma`, `meningioma`, `pituitary`, or `no_tumor`. Three architecturally different backbones are used so that one model's blind spot on a given scan doesn't become the final answer unchallenged.

4. **Adaptive lesion-aware fusion** — also in `core/classifier_fusion.py`, the three predictions are combined with weights based on each model's confidence, entropy-based certainty, the scan's quality score, and a lesion-trust multiplier (derived from tumor size, shape irregularity, and Grad-CAM/segmentation overlap — see step 8). This replaces a naive equal-weight average, since how much to trust each model should depend on the specific scan, not be fixed in advance.

5. **Tumor segmentation** — [`core/segmentation.py`](core/segmentation.py) runs an EfficientNetB4-backed Attention U-Net to produce a binary tumor mask. Classification alone answers *what*; segmentation is required to answer *where* and *how large*, which drives every measurement in the steps that follow.

6. **Grad-CAM heatmap generation** — [`core/gradcam.py`](core/gradcam.py) computes a gradient-based class activation map showing which pixels actually drove the classifier's decision. This is the explainability layer, and it also feeds the consistency check in step 8.

7. **Tumor size + morphology analysis** — [`core/morphology.py`](core/morphology.py) measures area, diameter, volume, irregularity, convexity, eccentricity, laterality, and midline shift directly from the segmentation mask. These are the same quantitative measurements a radiologist would use, so the report gives numbers, not just a class label.

8. **Overlap consistency + DRI gate** — [`core/overlap_metrics.py`](core/overlap_metrics.py) computes IoU between the Grad-CAM attention region and the segmented lesion; [`core/diagnostic_reliability.py`](core/diagnostic_reliability.py) combines that overlap score with scan quality, model agreement, confidence margin, and lesion trust into a Diagnostic Reliability Index (DRI). A high-confidence prediction whose attention isn't actually on the tumor is a real failure mode for CNNs — the DRI turns that risk into an explicit `Accepted` / `Caution` / `Specialist Review Required` decision instead of hiding it behind a single confidence percentage.

9. **Risk and urgency support** — [`core/risk_engine.py`](core/risk_engine.py) turns the morphology, confidence, and DRI data into severity, growth-risk, clinical priority, and concrete next-step recommendations (e.g. "Neurosurgical consultation", "MR Spectroscopy"). This is the step that converts raw numbers into decision-support language.

10. **Groq imaging summary** — [`reporting/llm_report_generator.py`](reporting/llm_report_generator.py) sends the scan and structured findings to a vision-capable LLM (via the Groq API) to draft a radiology-style narrative report. If `GROQ_API_KEY` is missing or the API call fails, the pipeline falls back to a deterministic template so the app still returns a complete report rather than erroring out.

11. **PDF report export** — [`reporting/pdf_report_generator.py`](reporting/pdf_report_generator.py) assembles the original scan, segmentation overlay, Grad-CAM heatmap and overlay, a risk-analytics chart, and the written report into a multi-page PDF saved under `reports/` and downloadable via `/api/reports/{filename}`. The same overlay images are also saved individually under `reports/overlays/` and served via `/api/overlays/{filename}`, so the segmentation and Grad-CAM visuals show up directly in the web UI's result panel, not only inside the PDF.

## Feature Summary

| Area | Capability |
|---|---|
| Classification | Predicts `glioma`, `meningioma`, `pituitary`, or `no_tumor` |
| Fusion | Uses adaptive weighting based on confidence, entropy, quality, and lesion trust |
| Segmentation | Produces a binary tumor mask for lesion localization |
| Explainability | Generates Grad-CAM heatmaps for visual reasoning support |
| Morphology | Estimates area, diameter, volume, irregularity, convexity, and mass effect |
| Reliability | Computes DRI score, tier, escalation reasons, and gate decision |
| Risk Support | Produces severity, progression risk, urgency, and clinical steps |
| Reporting | Builds structured AI summaries and PDF reports |
| History | Saves prior cases and compares progression where available |

## Model Stack

| Model | Role | Input Size |
|---|---|---:|
| EfficientNetV2-S | primary classifier | 384 x 384 |
| MobileNetV3 | ensemble classifier | 384 x 384 |
| ConvNeXt Tiny | ensemble classifier | 384 x 384 |
| EfficientNet-based U-Net | segmentation model | 256 x 256 |

Classification targets:

- `glioma`
- `meningioma`
- `no_tumor`
- `pituitary`

## Results Gallery

The repository includes visual results in [`docs/`](docs). For a cleaner project presentation, this README highlights the primary classifier results from EfficientNetV2-S only.

### Architecture

![Architecture](docs/system_architecture.png)

### EfficientNetV2-S Highlights

![EfficientNetV2-S Graph](docs/v2s%20graph.png)
![EfficientNetV2-S Confusion Matrix](docs/v2s%20confustion%20matrix.png)
![EfficientNetV2-S Classification Report](docs/V2S%20CR.png)
![EfficientNetV2-S Test Output](docs/v2s%20testing.png)

## Repository Structure

```text
NeuroScan-AI/
|-- app/           FastAPI interface and workflow orchestration
|-- core/          fusion, segmentation, explainability, morphology, risk, reliability
|-- reporting/     LLM report generation and PDF export
|-- utils/         config, history, and I/O helpers
|-- training/      model training scripts
|-- experiments/   notebooks for validation and benchmarking
|-- tests/         pytest checks
|-- deploy/        deployment utilities
|-- docs/          architecture diagram and result images
|-- model/         trained .keras weights (downloaded, not committed)
|-- reports/       generated PDF reports and overlay images (not committed)
|-- history/       local case history (not committed)
|-- requirements.txt
`-- README.md
```

## Core Modules

- [`app/main.py`](app/main.py): FastAPI routes and upload page
- [`app/pipeline.py`](app/pipeline.py): end-to-end MRI analysis workflow
- [`core/classifier_fusion.py`](core/classifier_fusion.py): multi-model prediction, adaptive fusion, lesion trust
- [`core/segmentation.py`](core/segmentation.py): segmentation inference and mask generation
- [`core/diagnostic_reliability.py`](core/diagnostic_reliability.py): DRI scoring, tiering, and escalation logic
- [`core/risk_engine.py`](core/risk_engine.py): severity, urgency, progression, and decision support
- [`reporting/llm_report_generator.py`](reporting/llm_report_generator.py): Groq-based imaging summary generation
- [`reporting/pdf_report_generator.py`](reporting/pdf_report_generator.py): report visualization and PDF export
- [`utils/config.py`](utils/config.py): paths, constants, and model loading
- [`utils/history_manager.py`](utils/history_manager.py): local history persistence and prior-case comparison

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

Use Python 3.10, 3.11, or 3.12 for TensorFlow 2.16. Python 3.14 is not supported by the pinned TensorFlow build.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Main packages include TensorFlow, FastAPI, Uvicorn, OpenCV, NumPy, pandas, Pillow, fpdf2, Groq, matplotlib, requests, and pytest.

### 3. Configure secrets

The project uses `.env` in the repository root for local secrets.

Create `.env` with:

```env
GROQ_API_KEY=your_key_here
```

Use [`.env.example`](.env.example) as the safe template. The config also keeps legacy support for `env.txt`, but `.env` is now the standard format.

### 4. Prepare project directories

The app creates `reports/` and `history/` automatically on first run. You only need to prepare `model/` yourself:

```text
NeuroScan-AI/
|-- model/
|-- reports/     (auto-created)
`-- history/     (auto-created)
```

Folder purposes:

- `model/` stores trained `.keras` weights. Download them from [Hugging Face](https://huggingface.co/tharunsridhar/brain_tumor_net-ensemble/tree/main/models), then place the files in `model/`.
- `reports/` stores generated PDF reports and the segmentation/Grad-CAM overlay images shown in the UI
- `history/` stores saved case history

### 5. Run the app

Always launch with the project's `.venv` interpreter, not a system-wide `python`/`uvicorn` — a system interpreter without `opencv-python-headless`, `tensorflow`, and `fpdf2` installed will silently fall back to a heuristic "demo mode" instead of running the real trained models. Check `GET /ready` after startup; `dependencies_available` must be `true`.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open <http://127.0.0.1:8000> in your browser. The first analysis request loads TensorFlow and the three classifier models, so it can take up to a minute; later requests are fast.

## FastAPI Endpoints

- `GET /health`: service health
- `GET /ready`: model-file and dependency readiness check
- `POST /api/analyze`: upload MRI image and receive JSON analysis plus PDF and overlay-image URLs
- `GET /api/history`: local analyzed-case history
- `GET /api/reports`: generated PDF report list
- `GET /api/reports/{filename}`: download one PDF report
- `GET /api/overlays/{filename}`: fetch one segmentation/Grad-CAM overlay image
- `GET /api/model-info`: model classes, input sizes, MRI metadata, and model file paths
- `GET /api/config`: runtime config (upload limits, allowed extensions, storage paths)
- `GET /docs`: interactive Swagger documentation

## Deployment

Windows local service:

```powershell
.\deploy\start.ps1 -HostName 127.0.0.1 -Port 8000
```

Docker:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Deployment details are in [`deploy/README.md`](deploy/README.md).

## Testing

The repo currently includes lightweight tests for:

- quality metrics
- adaptive fusion
- diagnostic reliability gating
- segmentation mask output validation

Run them with:

```bash
pytest
```

## Strengths and Limitations

### Strengths

- modular codebase with clear separation between app, core logic, utilities, reporting, and tests
- code-aware reliability layer that goes beyond plain classification confidence
- built-in visual assets that help with demos, portfolio posts, and presentations
- report generation pipeline that makes the project feel end-to-end
- prior-case comparison support through local history tracking

### Limitations

- trained model files are external and not bundled in this repository. Download them from [Hugging Face](https://huggingface.co/tharunsridhar/brain_tumor_net-ensemble/tree/main/models).
- Groq-based reporting requires a valid `GROQ_API_KEY`
- test coverage is useful but still lightweight
- the project is suited to research and demonstration rather than direct clinical deployment

## Clinical Safety Note

All outputs should be treated as decision support only. Final diagnosis, tumor grading, treatment planning, and case interpretation must be confirmed by qualified clinicians and, when appropriate, histopathology.
