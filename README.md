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
- Uses a residual U-Net style segmentation model to localize tumor regions
- Generates Grad-CAM heatmaps for model interpretability
- Estimates tumor area, diameter, volume, bounding box, mass effect, and risk score
- Produces AI-assisted radiology summaries and downloadable PDF reports
- Includes example outputs, evaluation plots, and sample test images inside the repo

## Project Preview

### Classification Performance

![Classification report](results/Classification%20Report.png)

### Confusion Matrix

![Confusion matrix](results/v2s%20confusion%20matrix.png)

### Training Curves

![Training curves](results/v2s%20graph.png)

### Sample Testing Output

![Testing output](results/v2s%20testing.png)

### Segmentation Training Curve

![Segmentation dice curve](results/segmentation%20dice%20curve.png)

## Performance Snapshot

### Classification Results

Based on the saved classification report in the repository, the EfficientNetV2-S classifier reaches:

- Overall accuracy: `0.98`
- Macro average precision: `0.98`
- Macro average recall: `0.98`
- Macro average F1-score: `0.98`
- Evaluation support: `2180` MRI images

Per-class performance:

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| Glioma | 0.98 | 0.96 | 0.97 | 750 |
| Meningioma | 0.96 | 0.96 | 0.96 | 518 |
| No Tumor | 0.98 | 1.00 | 0.99 | 455 |
| Pituitary | 0.98 | 1.00 | 0.99 | 457 |

### Segmentation Results

The segmentation pipeline is trained and evaluated with the Dice coefficient as the primary metric.

- Input resolution: `256 x 256`
- Loss: `binary cross-entropy + Dice loss`
- Evaluation metric: `Dice coefficient`
- Saved training curve shows validation Dice stabilizing around the `0.60+` range, with training Dice reaching the high `0.80+` range

This gives the project both a classification component for tumor type prediction and a segmentation component for tumor localization and downstream measurement.

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

The segmentation pipeline is built around a residual U-Net style architecture operating on `256 x 256` grayscale MRI slices.

Segmentation workflow details:

- tumor-positive masks are filtered before training
- MRI slices are resized and normalized
- CLAHE is applied to enhance grayscale contrast
- masks are binarized for pixel-wise prediction
- augmentation includes flips and 90-degree rotations
- optimization uses a combined BCE + Dice objective
- model quality is tracked with Dice score on validation and test data

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
|   |   +-- segmentation_dataset.py
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

The segmentation stage supports:

- binary tumor mask generation
- localization for heatmap and overlay creation
- shape and boundary analysis
- tumor size and approximate volume estimation
- downstream clinical analytics such as laterality and mass-effect style features

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

### Suggested Entry Points

Use these files depending on what you want to demonstrate:

- `src/classification/classification_model_efficientnet_v2_s.py` for the main classification training pipeline
- `src/classification/classification_testing.py` for manual classification checks across saved models
- `src/segmentation/segmentation_brain_seg_final.py` for tumor segmentation training and Dice-based evaluation
- `src/reporting/reporting.py` for report generation logic
- `deployment/app.py` for the full showcase pipeline with classification, segmentation, explainability, analytics, and reporting

### Model Weights Note

The trained model files are not stored in this GitHub repository.

The code expects external model artifacts such as:

- classification model weights in `.keras` or `.h5` format
- segmentation model weights in `.keras` format
- local or Drive-based dataset folders that match the paths used in the scripts

If you want to run the full pipeline locally, update the dataset paths and model paths in the scripts to match your machine.

## Technical Notes

- Several scripts were exported from Google Colab and still contain Colab-specific path assumptions.
- Colab-specific parts include `drive.mount(...)`, `!pip install ...`, and `/content/drive/...` file paths.
- Model files referenced by the scripts are expected to exist in external storage paths and are not included in this repo.
- The deployment prototype is best treated as a showcase and integration layer built on top of the trained models.
- The repository preserves the full project workflow, including classification research, segmentation research, reporting logic, and deployment code.
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
THARUN SRIDHAR NATARAJAN
Built as an applied AI project focused on medical image analysis, interpretability, and report generation for brain MRI workflows.
