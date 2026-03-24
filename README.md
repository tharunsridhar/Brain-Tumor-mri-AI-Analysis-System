# Project 1

Project 1 is a lightweight brain MRI analysis application built with Streamlit. It is organized as a clean runtime package with app code, reusable source modules, demo images, model guidance, generated reports, and supporting documents.

## What It Does

- uploads a brain MRI image through a Streamlit interface
- runs classification-style analysis and segmentation-style localization
- generates explainability output and tumor-related measurements
- computes risk-oriented summary information
- produces a structured report for review

## Project Structure

```text
Project 1/
│
├── app/                         # Production app (ONLY runnable code)
│   ├── main.py                  # Streamlit application entrypoint
│   ├── pipeline/
│   │   ├── classification.py
│   │   ├── segmentation.py
│   │   ├── explainability.py
│   │   ├── report.py
│   │   └── utils.py
│   └── config.py
│
├── models/                      # Model notes and future model location guidance
│   └── README.md
│
├── data/
│   ├── sample/                  # Demo MRI images for quick local testing
│   └── README.md
│
├── src/                         # Feature engineering and reusable logic
│   ├── features/
│   │   ├── scan_quality.py
│   │   ├── tumor_size.py
│   │   ├── risk_scoring.py
│   │   └── ...
│   └── fusion/
│       └── multi_model.py
│
├── reports/                     # Generated outputs such as case history and report PDFs
│   └── case_history.json
│
├── docs/                        # Graphs, PDFs, images, and supporting reference material
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Requirements

Install the project dependencies with:

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

From the project root, start the app with:

```bash
streamlit run app/main.py
```

Then open the local Streamlit URL shown in the terminal and upload an MRI image to run the pipeline.

## Models

Heavy trained model files are intentionally not included in this repository. When you want to connect real models, place them under:

- `models/Classification/`
- `models/Segmentation/`

The default runtime paths are managed in `app/config.py`.

## Notes

- the current project is runnable without heavy models because it uses lightweight fallback logic
- supporting PDFs, graphs, and report files have been transferred into `docs/` and `reports/`
- the repository is intentionally cleaned to keep only the files needed to run or extend the app

## Author

**Tharun Sridhar Natarajan**
