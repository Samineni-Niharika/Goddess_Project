# Technology Stack

- Language/runtime: Python; no `requirements.txt`, `pyproject.toml`, or lockfile was found, so versions are not pinned in source.
- UI: Streamlit (`st.set_page_config`, file uploader, cached model resource).
- Deep learning: PyTorch (`torch`, `torch.nn`, `DataLoader`).
- Computer vision: torchvision (`models.resnet18`, image transforms, `ImageFolder`).
- Image I/O: Pillow (`PIL.Image`).
- Numerical processing: NumPy.
- Scientific utilities: SciPy softmax.
- Metrics: scikit-learn classification reports, confusion matrices, and cosine similarity.
- Persistence: PyTorch `state_dict` checkpoint files (`.pth`); CSV, PNG, and TXT diagnostics.
- Hardware selection: CUDA when available, otherwise CPU.

## Model versions

- Model 3: `goddess_material_resnet18_v3_best.pth`.
- Model 4 training variant: `goddess_material_resnet18_v4_best.pth`, with material loss weight 1.5.
- The Streamlit app and primary CLI use the Model 3 checkpoint.

## Main blockers/constraints

- Relative filesystem paths make the application dependent on being launched from the project root.
- Class labels are duplicated across scripts rather than centralized.
- Centroid computation in the UI uses `ImageFolder`, while training labels are derived from nested goddess/material path components; this mapping should be verified before production use.
