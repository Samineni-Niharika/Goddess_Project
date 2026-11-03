import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "Goddess_Split/goddess_material_resnet18_v3_best.pth"

IMAGE_SIZE = 224

# Minimum softmax confidence required to accept a prediction (0-1). Below this the image is treated as unrelated
RECOGNITION_THRESHOLD = 0.70  # chosen by validation-constrained grid search
BEST_GODDESS_TEMP = 1.0
BEST_MATERIAL_TEMP = 1.2
CENTROID_THRESHOLD = 0.60  # max cosine similarity required to accept a material prediction

GODDESSES = [
    "Annapurna",
    "Durga",
    "Kali",
    "Lakshmi",
    "Lalitha",
    "Saraswati"
]

MATERIALS = [
    "Brass",
    "Bronze",
    "Copper",
    "Silver",
    "Stone",
    "Wood"
]

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("GODDESS IDOL SINGLE IMAGE PREDICTION")
print("=" * 60)

print("\nDevice:", device)

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# CREATE RESNET18 MODEL
# ============================================================

print("\nCreating ResNet18 model...")

model = models.resnet18(weights=None)

number_of_features = model.fc.in_features

model.fc = nn.Identity()

model.goddess_classifier = nn.Linear(
    number_of_features,
    6
)

model.material_classifier = nn.Linear(
    number_of_features,
    6
)

model = model.to(device)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained Model 3...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")

# Compute material centroids from validation set for OOD checks
import os
import numpy as np
from torchvision.datasets import ImageFolder

BASE_PATH = os.path.dirname(MODEL_PATH)
VAL_DIR = os.path.join('Goddess_Split', 'validation')
centroid_matrix_normed = None
try:
    print('\nComputing material centroids from validation images...')
    val_dataset = ImageFolder(VAL_DIR, transform=transform)
    if len(val_dataset) > 0:
        features_list = {i: [] for i in range(len(MATERIALS))}
        for img, label in val_dataset:
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(img)
            feat_np = feat.cpu().numpy().reshape(-1)
            # ImageFolder for validation: class folders correspond to goddess/material? Use mapping by folder name
            # Expect val_dataset.samples to contain (path, class_index) where class_index is index by folder; want material label
            # Fallback: distribute features evenly if mapping unknown
            features_list[label].append(feat_np)
        centroid_matrix = np.stack([np.mean(np.stack(features_list[i], axis=0), axis=0) if len(features_list[i])>0 else np.zeros(feat_np.shape) for i in range(len(MATERIALS))], axis=0)
        centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        centroid_matrix_normed = centroid_matrix / np.clip(centroid_norms, 1e-8, None)
        print('Computed centroids')
except Exception as e:
    print('Could not compute centroids:', e)

# ============================================================
# GET IMAGE PATH
# ============================================================

image_path = input(
    "\nEnter the complete path of the image:\n"
)

# Remove accidental quotation marks
image_path = image_path.strip().strip('"')

# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(image_path):

    print("\nERROR: Image not found.")
    raise SystemExit

# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(image_path).convert("RGB")

except Exception as error:

    print("\nERROR: Could not open image.")
    print(error)

    raise SystemExit

# ============================================================
# PREPARE IMAGE
# ============================================================

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)

# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    features = model(image_tensor)

    goddess_output = model.goddess_classifier(
        features
    )

    material_output = model.material_classifier(
        features
    )

    # Apply temperature scaling to logits before softmax
    goddess_probability = torch.softmax(
        goddess_output / BEST_GODDESS_TEMP,
        dim=1
    )

    material_probability = torch.softmax(
        material_output / BEST_MATERIAL_TEMP,
        dim=1
    )

    goddess_confidence, goddess_prediction = torch.max(
        goddess_probability,
        dim=1
    )

    material_confidence, material_prediction = torch.max(
        material_probability,
        dim=1
    )

    # compute centroid similarity if available
    max_sim = 0.0
    try:
        feat_np = features.cpu().numpy().reshape(-1)
        feat_norm = feat_np / (np.linalg.norm(feat_np) + 1e-8)
        sims = centroid_matrix_normed.dot(feat_norm)
        max_sim = float(np.max(sims))
    except Exception:
        max_sim = 0.0

# ============================================================
# RESULTS
# ============================================================

predicted_goddess = GODDESSES[
    goddess_prediction.item()
]

predicted_material = MATERIALS[
    material_prediction.item()
]

# Convert to percentages
goddess_confidence = (
    goddess_confidence.item() * 100
)

material_confidence = (
    material_confidence.item() * 100
)

# Reject images where either classifier is not confident enough OR centroid similarity is low
if ((goddess_confidence / 100.0) < RECOGNITION_THRESHOLD or (material_confidence / 100.0) < RECOGNITION_THRESHOLD) or (max_sim < CENTROID_THRESHOLD):
    predicted_goddess = "Image Not Recognized"
    predicted_material = "Image Not Recognized"
    goddess_confidence = 0.0
    material_confidence = 0.0

# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("\nImage:")
print(image_path)

print("\nPredicted Goddess:")
print(predicted_goddess)

print(
    "Goddess Confidence:",
    round(goddess_confidence, 2),
    "%"
)

print("\nPredicted Material:")
print(predicted_material)

print(
    "Material Confidence:",
    round(material_confidence, 2),
    "%"
)

print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)