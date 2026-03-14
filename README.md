# Brain Tumor MRI AI Analysis System

An end-to-end deep learning project for brain tumor MRI analysis that combines image classification, tumor segmentation, explainability, risk estimation, and AI-assisted radiology report generation.

This repository brings together the full workflow:

- multi-class brain tumor classification
- tumor region segmentation
- Grad-CAM based visual explainability
- quantitative tumor analytics
- structured PDF report generation
- an interactive Streamlit-style deployment prototype

## Why This Project Stands Out

This is not just a single-model notebook. The project is organized as a complete MRI analysis pipeline designed to move from raw scan input to clinically readable output.

Key highlights:

- Classifies MRI scans into `glioma`, `meningioma`, `pituitary`, and `no_tumor`
- Uses an EfficientNetV2-S based classifier for high-resolution image understanding
- Uses a U-Net style segmentation model to localize tumor regions
- Generates Grad-CAM heatmaps for model interpretability
- Estimates tumor area, diameter, volume, bounding box, mass effect, and risk score
- Produces AI-assisted radiology summaries and downloadable PDF reports
- Includes example outputs, evaluation plots, and sample test images inside the repo

## Project Preview

### Classification Performance

![Classification report](results/Classification%20Report.png)

### Confusion Matrix

![Confusion matrix](results/v2s%20confustion%20matrix.png)

### Training Curves

![Training curves](results/v2s%20graph.png)

### Sample Testing Output

![Testing output](results/v2s%20testing.png)

## System Pipeline

```text
MRI Input
   |
   v
Classification Model -> Tumor Type Prediction
   |
   +-> If no_tumor: generate normal MRI report
   |
   v
Segmentation Model -> Tumor Mask
   |
   v
Explainability -> Grad-CAM + overlay maps
   |
   v
Quantitative Analysis -> size, shape, volume, shift, risk
   |
   v
LLM-Assisted Reporting -> structured radiology summary + PDF export
```

## Model Stack

### Classification

The repository includes multiple classification experiments:

- EfficientNetV2-S
- ConvNeXt-Tiny
- MobileNetV3
- TinyViT Transformer

The deployed analysis flow is centered around `EfficientNetV2-S`, trained on MRI images resized to `384 x 384`.

Classification labels:

- `glioma`
- `meningioma`
- `no_tumor`
- `pituitary`

### Segmentation

The segmentation pipeline is built around a U-Net style architecture operating on `256 x 256` MRI slices, with preprocessing and filtering focused on tumor-positive masks.

### Reporting Layer

The reporting pipeline combines imaging-derived metrics with an LLM-assisted report generator to produce:

- structured findings
- severity assessment
- recommendation text
- PDF reports for detected tumor and no-tumor cases

## Repository Structure

```text
Brain-Tumor-mri-AI-Analysis-System/
|
+-- deployment/
|   +-- app.py
|
+-- src/
|   +-- classification/
|   |   +-- classification_model_efficientnet_v2_s.py
|   |   +-- classification_model_convnext_tiny.py
|   |   +-- classification_model_mobilenet_v3.py
|   |   +-- classification_model_tinyvit_transformer.py
|   |   +-- classification_testing.py
|   |
|   +-- segmentation/
|   |   +-- segmentation_brain_seg_final.py
|   |   +-- segmentaion_dataset.py
|   |
|   +-- reporting/
|   |   +-- reporting.py
|   |
|   +-- utils/
|       +-- image_loader.py
|       +-- preprocessing.py
|
+-- results/
|   +-- evaluation plots and visual outputs
|
+-- sample_images/
|   +-- example MRI scans
|
+-- example_reports/
|   +-- generated PDF report samples
|
+-- requirements.txt
+-- .env.example
```

## Features

### 1. MRI Tumor Classification

Predicts one of four diagnostic categories from brain MRI images.

### 2. Tumor Segmentation

Extracts the suspected tumor region for downstream measurement and visualization.

### 3. Explainable AI

Generates Grad-CAM heatmaps and overlays to show where the classifier is focusing.

### 4. Quantitative Tumor Analytics

Computes imaging-derived statistics such as:

- tumor area
- estimated diameter
- estimated volume
- bounding box dimensions
- irregularity and compactness
- hemisphere laterality
- approximate midline shift
- derived progression and severity risk

### 5. AI-Assisted Radiology Reporting

Creates structured report text with sections such as:

- clinical indication
- imaging technique
- findings
- impression
- severity assessment
- recommendations

### 6. Deployment Prototype

The `deployment/app.py` script contains an interactive Streamlit-based interface for running the end-to-end analysis workflow and exporting reports.

## Sample Assets Included

The repository includes ready-to-browse assets:

- sample MRI scans in [`sample_images/`](sample_images/)
- generated example reports in [`example_reports/`](example_reports/)
- evaluation visuals in [`results/`](results/)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/tharunsridhar/Brain-Tumor-mri-AI-Analysis-System.git
cd Brain-Tumor-mri-AI-Analysis-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For the reporting and deployment flow, you may also need:

```bash
pip install groq fpdf2 pyngrok streamlit python-dotenv pillow
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Project

This repository currently contains exported notebook-style Python scripts from the original experimentation workflow. Depending on your setup, you can:

- run the training and testing scripts inside Google Colab
- adapt the paths to local datasets and model files
- use `deployment/app.py` as the starting point for the interactive demo flow

## Technical Notes

- Several scripts were exported from Google Colab and still contain Colab-specific path assumptions.
- Model files referenced by the scripts are expected to exist in external storage paths and are not included in this repo.
- The deployment prototype is best treated as a showcase and integration layer built on top of the trained models.
- This project is for educational and research purposes and is not a medical device.

## Example Outputs

Generated PDF examples are available in the repo:

- `example_reports/test_no tumor_20260314084022_mri_report.pdf`
- `example_reports/test_pituitary_20260314084348_mri_report.pdf`

## Future Improvements

- package the app as a clean local deployment instead of a Colab-exported script
- add model weights loading instructions for reproducibility
- add automated evaluation scripts and benchmark tables
- containerize the deployment workflow
- add unit tests for preprocessing and reporting utilities

## Disclaimer

This project is intended for academic, portfolio, and research demonstration purposes only. Any generated analysis or report must be reviewed by a qualified medical professional before clinical use.

## Author

Built as an applied AI project focused on medical image analysis, interpretability, and report generation for brain MRI workflows.
