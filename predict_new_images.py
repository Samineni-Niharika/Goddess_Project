import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

IMAGE_FOLDER = "New_Test_Images_Processed"

MODEL_PATH = "Goddess_Split/goddess_material_resnet18.pth"

IMAGE_SIZE = 224

# Minimum softmax confidence required to accept a prediction (0-1). Below this the image is treated as unrelated
RECOGNITION_THRESHOLD = 0.70  # chosen by validation-constrained grid search
BEST_GODDESS_TEMP = 1.0
BEST_MATERIAL_TEMP = 1.2
CENTROID_THRESHOLD = 0.60  # max cosine similarity required to accept a material prediction


# ============================================================
# CLASSES
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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("FINAL PREDICTION ON NEW IMAGES")
print("=" * 60)

print("Device:", device)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
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
# CREATE RESNET18 MODEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING RESNET18 MODEL")
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

model = model.to(device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")

# Compute material centroids from validation set for OOD checks
import numpy as np
VAL_DIR = os.path.join('Goddess_Split', 'validation')
centroid_matrix_normed = None
try:
    print('\nComputing material centroids from validation images...')
    from torchvision.datasets import ImageFolder
    val_dataset = ImageFolder(VAL_DIR, transform=transform)
    if len(val_dataset) > 0:
        features_list = {i: [] for i in range(len(MATERIALS))}
        for img, label in val_dataset:
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(img)
            feat_np = feat.cpu().numpy().reshape(-1)
            features_list[label].append(feat_np)
        centroid_matrix = np.stack([np.mean(np.stack(features_list[i], axis=0), axis=0) if len(features_list[i])>0 else np.zeros(feat_np.shape) for i in range(len(MATERIALS))], axis=0)
        centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        centroid_matrix_normed = centroid_matrix / np.clip(centroid_norms, 1e-8, None)
        print('Computed centroids')
except Exception as e:
    print('Could not compute centroids:', e)


# ============================================================
# FIND IMAGES
# ============================================================

images = []

for file in os.listdir(IMAGE_FOLDER):

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        images.append(file)


images.sort()


print("\nNew images found:", len(images))


if len(images) == 0:

    print("\nERROR: No images found.")

    raise SystemExit


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)


with torch.no_grad():

    for number, file in enumerate(images, start=1):

        image_path = os.path.join(
            IMAGE_FOLDER,
            file
        )

        try:

            # Open image
            image = Image.open(
                image_path
            ).convert("RGB")

            # Transform
            image_tensor = transform(
                image
            )

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(
                0
            )

            # Move to device
            image_tensor = image_tensor.to(
                device
            )

            # Extract features
            features = model(
                image_tensor
            )

            # Predictions
            goddess_output = model.goddess_classifier(
                features
            )

            material_output = model.material_classifier(
                features
            )

            # Apply temperature scaling to logits before softmax
            goddess_probabilities = torch.softmax(
                goddess_output / BEST_GODDESS_TEMP,
                dim=1
            )

            material_probabilities = torch.softmax(
                material_output / BEST_MATERIAL_TEMP,
                dim=1
            )

            # Best prediction
            goddess_confidence, goddess_index = torch.max(
                goddess_probabilities,
                dim=1
            )

            material_confidence, material_index = torch.max(
                material_probabilities,
                dim=1
            )

            # centroid similarity
            max_sim = 0.0
            try:
                feat_np = features.cpu().numpy().reshape(-1)
                feat_norm = feat_np / (np.linalg.norm(feat_np) + 1e-8)
                sims = centroid_matrix_normed.dot(feat_norm)
                max_sim = float(np.max(sims))
            except Exception:
                max_sim = 0.0

            goddess_name = GODDESSES[
                goddess_index.item()
            ]

            material_name = MATERIALS[
                material_index.item()
            ]

            goddess_confidence = (
                goddess_confidence.item()
                * 100
            )

            material_confidence = (
                material_confidence.item()
                * 100
            )

            # If either classifier is not confident enough or centroid sim is low, mark image as not recognized
            if ((goddess_confidence / 100.0) < RECOGNITION_THRESHOLD or (material_confidence / 100.0) < RECOGNITION_THRESHOLD) or (max_sim < CENTROID_THRESHOLD):
                goddess_name = "Image Not Recognized"
                material_name = "Image Not Recognized"
                goddess_confidence = 0.0
                material_confidence = 0.0

            # Print result
            print("\n" + "-" * 60)

            print(
                "Image", number,
                ":",
                file
            )

            print(
                "Predicted Goddess :",
                goddess_name
            )

            print(
                "Goddess Confidence:",
                round(
                    goddess_confidence,
                    2
                ),
                "%"
            )

            print(
                "Predicted Material:",
                material_name
            )

            print(
                "Material Confidence:",
                round(
                    material_confidence,
                    2
                ),
                "%"
            )

        except Exception as e:

            print("\nERROR:", file)
            print("Reason:", e)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("FINAL PREDICTION COMPLETED")
print("=" * 60)