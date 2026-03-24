# Project 1

Project 1 is a streamlined brain MRI analysis application built with Streamlit. It is designed as a lightweight, runnable project that combines modular image analysis, tumor-focused feature extraction, explainability support, risk-oriented assessment, and structured reporting in a clean folder layout.

The repository has been cleaned to keep the runtime code focused and easy to understand. It is ready for local execution in its current lightweight form, and it can also be extended later with real trained classification and segmentation models.

## Overview

This project is built to:

- upload and analyze brain MRI images through a Streamlit interface
- run classification-style prediction and segmentation-style localization
- generate explainability-oriented visual outputs
- estimate tumor-related measurements such as area, diameter, and volume
- compute risk-oriented summary information
- produce a structured report for review

## Project Structure

```text
Project 1/
|
|-- app/                         # Production app (ONLY runnable code)
|   |-- main.py                  # Streamlit application entrypoint
|   |-- pipeline/
|   |   |-- classification.py    # Classification pipeline logic
|   |   |-- segmentation.py      # Segmentation-style processing logic
|   |   |-- explainability.py    # Explainability and heatmap generation
|   |   |-- report.py            # Structured report builder
|   |   `-- utils.py             # Shared pipeline helpers
|   `-- config.py                # Centralized runtime paths and settings
|
|-- models/                      # Model guidance only; heavy weights are not stored here
|   `-- README.md
|
|-- data/
|   |-- sample/                  # Demo MRI images for quick local testing
|   `-- README.md
|
|-- src/                         # Reusable source logic used by the app
|   |-- features/
|   |   |-- scan_quality.py
|   |   |-- tumor_size.py
|   |   |-- risk_scoring.py
|   |   |-- shape_irregularity.py
|   |   |-- explainability_overlap.py
|   |   |-- prior_case_comparison.py
|   |   |-- structured_report.py
|   |   |-- history_dashboard.py
|   |   |-- decision_logic.py
|   |   `-- _common.py
|   `-- fusion/
|       `-- multi_model.py
|
|-- reports/                     # Generated outputs such as case history and sample reports
|   `-- case_history.json
|
|-- docs/                        # Supporting PDFs, graphs, images, and reference material
|
|-- requirements.txt            # Runtime dependencies
`-- README.md                   # Project documentation
```

## How The Pipeline Works

The application follows a simple analysis flow:

1. A brain MRI image is uploaded through the Streamlit interface.
2. The image is processed by the classification module.
3. A segmentation-style mask is generated for localization.
4. Explainability output is created for visual interpretation.
5. Tumor metrics and risk-related summaries are computed.
6. A structured text report is generated for review.

## Requirements

Install the dependencies from the project root:

```bash
pip install -r requirements.txt
```

Current runtime dependencies:

```txt
numpy
Pillow
streamlit
```

## How To Run

Start the Streamlit app from the project root with:

```bash
streamlit run app/main.py
```

After running the command, open the local Streamlit URL shown in the terminal, upload an MRI image, and run the analysis from the web interface.

## Models

Heavy trained model files are intentionally not included in this repository.

If you want to connect real model weights later, create and use:

- `models/Classification/`
- `models/Segmentation/`

The default model-related paths are managed in `app/config.py`.

## Notes

- the current version is fully runnable in a lightweight form
- the app uses fallback pipeline logic, so it can run even without heavy trained models
- supporting reports, graphs, and reference materials are available in `reports/` and `docs/`
- the repository has been cleaned to remove old duplicate code and keep only the relevant project files

## Author

**Tharun Sridhar Natarajan**
