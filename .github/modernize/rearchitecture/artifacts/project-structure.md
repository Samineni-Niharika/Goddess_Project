# Project Structure

## Project type

Python image-classification research/application project. It contains offline dataset preparation and model-training scripts plus two prediction interfaces: a Streamlit web app and a command-line predictor.

## Functional domains

- Dataset preparation: validate, resize, normalize, split, and prepare new images.
- Model training: train ResNet18 multi-task classifiers for six goddess classes and six material classes.
- Model evaluation: test accuracy, confusion matrices, error analysis, and final result reporting.
- Calibration and OOD handling: temperature scaling, confidence threshold search, and feature-centroid cosine similarity.
- Inference: single-image CLI, batch new-image prediction, and Streamlit multi-upload UI.

## Runtime layers

1. Presentation: `app.py` (Streamlit).
2. Inference orchestration: model loading, preprocessing, two classifier heads, confidence/OOD gating.
3. ML model: torchvision ResNet18 backbone with shared feature representation and two linear heads.
4. Data layer: filesystem image folders under `Goddess_Resized`, `Goddess_Split`, and new-image folders.
5. Offline analysis: PyTorch training/evaluation and CSV/PNG/text debug outputs.

There is no database, HTTP API, message broker, persistent service layer, or multi-process application protocol in the repository.
