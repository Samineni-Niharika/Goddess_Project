import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
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
    "goddess_material_resnet18_v3_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16


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
# LABEL DICTIONARIES
# ============================================================

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


INDEX_TO_GODDESS = {
    0: "Annapurna",
    1: "Durga",
    2: "Kali",
    3: "Lakshmi",
    4: "Lalitha",
    5: "Saraswati"
}

INDEX_TO_MATERIAL = {
    0: "Brass",
    1: "Bronze",
    2: "Copper",
    3: "Silver",
    4: "Stone",
    5: "Wood"
}


# ============================================================
# DATASET
# ============================================================

class GoddessDataset(Dataset):

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

                parts = relative_path.split(os.sep)

                goddess = None
                material = None

                for name in GODDESSES:

                    if name in parts:

                        goddess = name
                        break

                for name in MATERIALS:

                    if name in parts:

                        material = name
                        break

                if goddess is not None and material is not None:

                    self.samples.append(
                        (
                            image_path,
                            GODDESS_TO_INDEX[goddess],
                            MATERIAL_TO_INDEX[material]
                        )
                    )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_path = self.samples[index][0]

        goddess_label = self.samples[index][1]

        material_label = self.samples[index][2]

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
# TEST TRANSFORM
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
print("LOADING TEST DATA - MODEL 3")
print("=" * 60)

test_dataset = GoddessDataset(
    TEST_PATH,
    test_transform
)

print(
    "Test images:",
    len(test_dataset)
)

if len(test_dataset) == 0:

    print("\nERROR: No test images found.")

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

print(
    "Device:",
    device
)


# ============================================================
# CREATE SAME RESNET18 ARCHITECTURE
# ============================================================

print("\n" + "=" * 60)
print("CREATING RESNET18 MODEL - VERSION 3")
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

if not os.path.exists(MODEL_PATH):

    print(
        "\nERROR: Model file not found:"
    )

    print(MODEL_PATH)

    raise SystemExit


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# PREDICTIONS
# ============================================================

actual_goddess = []
predicted_goddess = []

actual_material = []
predicted_material = []


print("\n" + "=" * 60)
print("ANALYZING TEST DATA")
print("=" * 60)


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

        goddess_predictions = torch.argmax(
            goddess_outputs,
            dim=1
        )

        material_predictions = torch.argmax(
            material_outputs,
            dim=1
        )

        actual_goddess.extend(
            goddess_labels.numpy()
        )

        predicted_goddess.extend(
            goddess_predictions.cpu().numpy()
        )

        actual_material.extend(
            material_labels.numpy()
        )

        predicted_material.extend(
            material_predictions.cpu().numpy()
        )

print_evaluation_summary(
    actual_goddess,
    predicted_goddess,
    GODDESSES,
    "Goddess classification",
)
print_evaluation_summary(
    actual_material,
    predicted_material,
    MATERIALS,
    "Material classification",
)

# ============================================================
# GODDESS ACCURACY
# ============================================================

goddess_correct = 0

for i in range(len(actual_goddess)):

    if actual_goddess[i] == predicted_goddess[i]:

        goddess_correct += 1


goddess_accuracy = (
    goddess_correct
    /
    len(actual_goddess)
) * 100


# ============================================================
# MATERIAL ACCURACY
# ============================================================

material_correct = 0

for i in range(len(actual_material)):

    if actual_material[i] == predicted_material[i]:

        material_correct += 1


material_accuracy = (
    material_correct
    /
    len(actual_material)
) * 100


# ============================================================
# GODDESS RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GODDESS CLASSIFICATION RESULTS - MODEL 3")
print("=" * 60)

print(
    "\nGoddess Accuracy:",
    round(goddess_accuracy, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        actual_goddess,
        predicted_goddess,
        labels=list(range(len(GODDESSES))),
        target_names=GODDESSES,
        zero_division=0,
        digits=2
    )
)


# ============================================================
# GODDESS CONFUSION MATRIX
# ============================================================

goddess_cm = confusion_matrix(
    actual_goddess,
    predicted_goddess,
    labels=list(range(len(GODDESSES)))
)

print("Goddess Confusion Matrix:")

print(goddess_cm)


# ============================================================
# MATERIAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MATERIAL CLASSIFICATION RESULTS - MODEL 3")
print("=" * 60)

print(
    "\nMaterial Accuracy:",
    round(material_accuracy, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        actual_material,
        predicted_material,
        labels=list(range(len(MATERIALS))),
        target_names=MATERIALS,
        zero_division=0,
        digits=2
    )
)


# ============================================================
# MATERIAL CONFUSION MATRIX
# ============================================================

material_cm = confusion_matrix(
    actual_material,
    predicted_material,
    labels=list(range(len(MATERIALS)))
)

print("Material Confusion Matrix:")

print(material_cm)


# ============================================================
# SAVE GODDESS CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    goddess_cm,
    interpolation="nearest"
)

plt.title(
    "Model 3 - Goddess Confusion Matrix"
)

plt.xlabel("Predicted Goddess")

plt.ylabel("Actual Goddess")

plt.xticks(
    range(6),
    GODDESSES,
    rotation=45
)

plt.yticks(
    range(6),
    GODDESSES
)

for i in range(6):

    for j in range(6):

        plt.text(
            j,
            i,
            goddess_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

goddess_matrix_path = os.path.join(
    BASE_PATH,
    "model3_goddess_confusion_matrix.png"
)

plt.savefig(
    goddess_matrix_path
)

plt.show()


# ============================================================
# SAVE MATERIAL CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    material_cm,
    interpolation="nearest"
)

plt.title(
    "Model 3 - Material Confusion Matrix"
)

plt.xlabel("Predicted Material")

plt.ylabel("Actual Material")

plt.xticks(
    range(6),
    MATERIALS,
    rotation=45
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
            material_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

material_matrix_path = os.path.join(
    BASE_PATH,
    "model3_material_confusion_matrix.png"
)

plt.savefig(
    material_matrix_path
)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL 3 FINAL EVALUATION SUMMARY")
print("=" * 60)

print(
    "Test Images:",
    len(actual_goddess)
)

print(
    "Goddess Accuracy:",
    round(goddess_accuracy, 2),
    "%"
)

print(
    "Material Accuracy:",
    round(material_accuracy, 2),
    "%"
)

print("\nGoddess confusion matrix saved at:")

print(
    goddess_matrix_path
)

print("\nMaterial confusion matrix saved at:")

print(
    material_matrix_path
)

print("\n" + "=" * 60)
print("MODEL 3 EVALUATION COMPLETED")
print("=" * 60)