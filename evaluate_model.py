import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
from evaluation_metrics import print_evaluation_summary


# ============================================================
# SETTINGS
# ============================================================

BASE_PATH = "Goddess_Split"

TEST_PATH = os.path.join(
    BASE_PATH,
    "test"
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "goddess_material_resnet18.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

# Minimum softmax confidence required to accept a prediction (0-1). Below this the image is treated as unrelated
RECOGNITION_THRESHOLD = 0.70  # chosen by validation-constrained grid search
# Temperature scaling (found on validation)
BEST_GODDESS_TEMP = 1.0
BEST_MATERIAL_TEMP = 1.2
# Centroid similarity threshold (max cosine similarity to any material centroid)
CENTROID_THRESHOLD = 0.60


# ============================================================
# CLASS NAMES
# ============================================================

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


GODDESS_TO_INDEX = {
    "Annapurna": 0,
    "Durga": 1,
    "Kali": 2,
    "Lakshmi": 3,
    "Lalitha": 4,
    "Saraswati": 5
}


MATERIAL_TO_INDEX = {
    "Brass": 0,
    "Bronze": 1,
    "Copper": 2,
    "Silver": 3,
    "Stone": 4,
    "Wood": 5
}


# ============================================================
# DATASET
# ============================================================

class GoddessMaterialDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for current_root, folders, files in os.walk(root_dir):

            for file in files:

                if not file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                image_path = os.path.join(
                    current_root,
                    file
                )

                relative_path = os.path.relpath(
                    image_path,
                    root_dir
                )

                path_parts = relative_path.split(
                    os.sep
                )

                goddess = None
                material = None

                for name in GODDESSES:

                    if name in path_parts:

                        goddess = name
                        break

                for name in MATERIALS:

                    if name in path_parts:

                        material = name
                        break

                if goddess is not None and material is not None:

                    goddess_label = GODDESS_TO_INDEX[
                        goddess
                    ]

                    material_label = MATERIAL_TO_INDEX[
                        material
                    ]

                    self.samples.append(
                        (
                            image_path,
                            goddess_label,
                            material_label
                        )
                    )

        print(
            "Loaded",
            len(self.samples),
            "test images"
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_path, goddess_label, material_label = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return (
            image,
            goddess_label,
            material_label
        )


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

test_dataset = GoddessMaterialDataset(
    TEST_PATH,
    test_transform
)


if len(test_dataset) == 0:

    print("ERROR: No test images found.")

    raise SystemExit


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)


model = models.resnet18(
    weights=None
)


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


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model = model.to(device)

model.eval()


print("Model loaded successfully.")

# ============================================================
# Build material centroids from validation set for OOD/centroid check
# ============================================================
import numpy as np
VALIDATION_PATH = os.path.join(BASE_PATH, 'validation')

print('\nComputing material centroids from validation set...')
val_dataset = GoddessMaterialDataset(VALIDATION_PATH, test_transform)
if len(val_dataset) == 0:
    print('Warning: no validation images found; centroid checks will be disabled')
    centroid_matrix_normed = None
else:
    material_features = {i: [] for i in range(len(MATERIALS))}
    for img, g_lbl, m_lbl in val_dataset:
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img)
        feat_np = feat.cpu().numpy().reshape(-1)
        material_features[int(m_lbl)].append(feat_np)
    centroid_matrix = np.stack([np.mean(np.stack(material_features[i], axis=0), axis=0) if len(material_features[i])>0 else np.zeros(feat_np.shape) for i in range(len(MATERIALS))], axis=0)
    centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    centroid_matrix_normed = centroid_matrix / np.clip(centroid_norms, 1e-8, None)
    print('Computed centroids for', len(MATERIALS), 'materials')


# ============================================================
# PREDICTIONS
# ============================================================

all_goddess_labels = []
all_goddess_predictions = []

all_material_labels = []
all_material_predictions = []

# For reporting how many images were rejected because of low confidence
accepted_goddess_labels = []
accepted_goddess_predictions = []

accepted_material_labels = []
accepted_material_predictions = []

rejected_count = 0

total_images = 0


with torch.no_grad():

    for images, goddess_labels, material_labels in test_loader:

        images = images.to(device)


        features = model(images)


        goddess_outputs = model.goddess_classifier(
            features
        )


        material_outputs = model.material_classifier(
            features
        )


        # Apply temperature scaling when computing softmax probabilities
        goddess_probabilities = torch.softmax(
            goddess_outputs / BEST_GODDESS_TEMP,
            dim=1
        )

        material_probabilities = torch.softmax(
            material_outputs / BEST_MATERIAL_TEMP,
            dim=1
        )

        goddess_confidences, goddess_predictions = torch.max(
            goddess_probabilities,
            dim=1
        )

        material_confidences, material_predictions = torch.max(
            material_probabilities,
            dim=1
        )

        # Append labels and predictions; also maintain accepted-only lists
        for i in range(goddess_labels.size(0)):

            true_g = goddess_labels[i].item()
            pred_g = goddess_predictions[i].item()
            conf_g = goddess_confidences[i].item()

            true_m = material_labels[i].item()
            pred_m = material_predictions[i].item()
            conf_m = material_confidences[i].item()

            # compute centroid similarity for this sample (if centroids exist)
            max_sim = 0.0
            try:
                feat_np = features[i].cpu().numpy().reshape(-1)
                feat_norm = feat_np / (np.linalg.norm(feat_np) + 1e-8)
                sims = np.dot(centroid_matrix_normed, feat_norm)
                max_sim = float(np.max(sims))
            except Exception:
                max_sim = 0.0

            # Record overall
            all_goddess_labels.append(true_g)
            all_goddess_predictions.append(pred_g)

            all_material_labels.append(true_m)
            all_material_predictions.append(pred_m)

            # Check acceptance: both confidences and centroid similarity
            if (conf_g >= RECOGNITION_THRESHOLD) and (conf_m >= RECOGNITION_THRESHOLD) and (max_sim >= CENTROID_THRESHOLD):
                accepted_goddess_labels.append(true_g)
                accepted_goddess_predictions.append(pred_g)

                accepted_material_labels.append(true_m)
                accepted_material_predictions.append(pred_m)
            else:
                rejected_count += 1

        total_images += goddess_labels.size(0)

print_evaluation_summary(
    all_goddess_labels,
    all_goddess_predictions,
    GODDESSES,
    "Goddess classification",
)
print_evaluation_summary(
    all_material_labels,
    all_material_predictions,
    MATERIALS,
    "Material classification",
)

# ============================================================
# GODDESS RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GODDESS CLASSIFICATION RESULTS")
print("=" * 60)


# Overall goddess accuracy (includes rejected images as normal predictions)
goddess_accuracy = accuracy_score(
    all_goddess_labels,
    all_goddess_predictions
)

print(
    "\nGoddess Accuracy (overall):",
    round(goddess_accuracy * 100, 2),
    "%"
)

# Accepted-only goddess accuracy (only images not rejected)
if len(accepted_goddess_labels) > 0:
    accepted_goddess_accuracy = accuracy_score(
        accepted_goddess_labels,
        accepted_goddess_predictions
    )

    print(
        "Goddess Accuracy (accepted images only):",
        round(accepted_goddess_accuracy * 100, 2),
        "%"
    )
else:
    print("Goddess Accuracy (accepted images only): No accepted images to evaluate")

print("\nClassification Report (overall):\n")
print(
    classification_report(
        all_goddess_labels,
        all_goddess_predictions,
        labels=list(range(6)),
        target_names=GODDESSES,
        zero_division=0
    )
)

# ============================================================
# GODDESS CONFUSION MATRIX
# ============================================================

goddess_matrix = confusion_matrix(
    all_goddess_labels,
    all_goddess_predictions,
    labels=list(range(6))
)

print("Goddess Confusion Matrix:")
print(goddess_matrix)

plt.figure(figsize=(8, 7))
plt.imshow(goddess_matrix)
plt.title("Goddess Confusion Matrix")
plt.xlabel("Predicted Goddess")
plt.ylabel("Actual Goddess")
plt.xticks(range(6), GODDESSES, rotation=45, ha="right")
plt.yticks(range(6), GODDESSES)

for i in range(6):
    for j in range(6):
        plt.text(j, i, goddess_matrix[i, j], ha="center", va="center")

plt.tight_layout()

goddess_matrix_path = os.path.join(BASE_PATH, "goddess_confusion_matrix.png")
plt.savefig(goddess_matrix_path, dpi=300)
plt.show()


# ============================================================
# MATERIAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MATERIAL CLASSIFICATION RESULTS")
print("=" * 60)


# Overall material accuracy (includes rejected images as normal predictions)
material_accuracy = accuracy_score(
    all_material_labels,
    all_material_predictions
)

print(
    "\nMaterial Accuracy (overall):",
    round(material_accuracy * 100, 2),
    "%"
)

# Accepted-only material accuracy
if len(accepted_material_labels) > 0:
    accepted_material_accuracy = accuracy_score(
        accepted_material_labels,
        accepted_material_predictions
    )

    print(
        "Material Accuracy (accepted images only):",
        round(accepted_material_accuracy * 100, 2),
        "%"
    )
else:
    print("Material Accuracy (accepted images only): No accepted images to evaluate")

print("\nClassification Report (overall):\n")
print(
    classification_report(
        all_material_labels,
        all_material_predictions,
        labels=list(range(6)),
        target_names=MATERIALS,
        zero_division=0
    )
)


# ============================================================
# MATERIAL CONFUSION MATRIX
# ============================================================

material_matrix = confusion_matrix(
    all_material_labels,
    all_material_predictions,
    labels=list(range(6))
)


print("Material Confusion Matrix:")

print(material_matrix)


plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    material_matrix
)

plt.title(
    "Material Confusion Matrix"
)

plt.xlabel(
    "Predicted Material"
)

plt.ylabel(
    "Actual Material"
)

plt.xticks(
    range(6),
    MATERIALS,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(6),
    MATERIALS
)


for i in range(6):

    for j in range(6):

        plt.text(
            j,
            i,
            material_matrix[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


material_matrix_path = os.path.join(
    BASE_PATH,
    "material_confusion_matrix.png"
)


plt.savefig(
    material_matrix_path,
    dpi=300
)


plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL BASELINE SUMMARY")
print("=" * 60)


print(
    "Test Images:",
    len(test_dataset)
)

print(
    "Rejected images:",
    rejected_count,
    "(",
    round((rejected_count / max(1, len(test_dataset))) * 100, 2),
    "%)"
)


print(
    "Goddess Accuracy (overall):",
    round(goddess_accuracy * 100, 2),
    "%"
)

if 'accepted_goddess_accuracy' in globals():
    print(
        "Goddess Accuracy (accepted images):",
        round(accepted_goddess_accuracy * 100, 2),
        "%"
    )

print(
    "Material Accuracy (overall):",
    round(material_accuracy * 100, 2),
    "%"
)

if 'accepted_material_accuracy' in globals():
    print(
        "Material Accuracy (accepted images):",
        round(accepted_material_accuracy * 100, 2),
        "%"
    )

print(
    "\nGoddess confusion matrix saved at:"
)

print(
    goddess_matrix_path
)

print(
    "\nMaterial confusion matrix saved at:"
)

print(
    material_matrix_path
)

print("=" * 60)