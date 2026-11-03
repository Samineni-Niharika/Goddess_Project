import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from evaluation_metrics import print_evaluation_summary

# ============================================================
# SETTINGS
# ============================================================

BASE_PATH = "Goddess_Split"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
VALIDATION_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "test")

MODEL_PATH = os.path.join(
    BASE_PATH,
    "goddess_material_resnet18_v4_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
PATIENCE = 5

# Material gets more importance
GODDESS_WEIGHT = 1.0
MATERIAL_WEIGHT = 1.5

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

                if (
                    goddess is not None
                    and
                    material is not None
                ):

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
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.90, 1.0)
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

validation_transform = transforms.Compose([

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
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING DATASETS - MODEL 4")
print("=" * 60)

train_dataset = GoddessDataset(
    TRAIN_PATH,
    train_transform
)

validation_dataset = GoddessDataset(
    VALIDATION_PATH,
    validation_transform
)

test_dataset = GoddessDataset(
    TEST_PATH,
    validation_transform
)

print(
    "Training images   :",
    len(train_dataset)
)

print(
    "Validation images :",
    len(validation_dataset)
)

print(
    "Testing images    :",
    len(test_dataset)
)

if len(train_dataset) == 0:
    print("\nERROR: No training images found.")
    raise SystemExit

if len(validation_dataset) == 0:
    print("\nERROR: No validation images found.")
    raise SystemExit

if len(test_dataset) == 0:
    print("\nERROR: No test images found.")
    raise SystemExit


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

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

print("\nDevice:", device)


# ============================================================
# CREATE RESNET18
# ============================================================

print("\n" + "=" * 60)
print("CREATING RESNET18 MODEL - VERSION 4")
print("=" * 60)

model = models.resnet18(
    weights="DEFAULT"
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
# LOSS FUNCTIONS
# ============================================================

goddess_loss_function = nn.CrossEntropyLoss()

material_loss_function = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING MODEL 4 TRAINING")
print("=" * 60)

print(
    "\nLoss = Goddess Loss + 1.5 × Material Loss"
)

best_validation_score = 0.0
epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    total_loss = 0.0

    goddess_correct = 0
    material_correct = 0

    total_train = 0

    for images, goddess_labels, material_labels in train_loader:

        images = images.to(device)
        goddess_labels = goddess_labels.to(device)
        material_labels = material_labels.to(device)

        optimizer.zero_grad()

        features = model(images)

        goddess_outputs = model.goddess_classifier(
            features
        )

        material_outputs = model.material_classifier(
            features
        )

        goddess_loss = goddess_loss_function(
            goddess_outputs,
            goddess_labels
        )

        material_loss = material_loss_function(
            material_outputs,
            material_labels
        )

        # ====================================================
        # MODEL 4 WEIGHTED LOSS
        # ====================================================

        loss = (
            GODDESS_WEIGHT * goddess_loss
            +
            MATERIAL_WEIGHT * material_loss
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        goddess_predictions = torch.argmax(
            goddess_outputs,
            dim=1
        )

        material_predictions = torch.argmax(
            material_outputs,
            dim=1
        )

        goddess_correct += (
            goddess_predictions == goddess_labels
        ).sum().item()

        material_correct += (
            material_predictions == material_labels
        ).sum().item()

        total_train += images.size(0)


    train_goddess_accuracy = (
        goddess_correct /
        total_train
    ) * 100

    train_material_accuracy = (
        material_correct /
        total_train
    ) * 100


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_goddess_correct = 0
    validation_material_correct = 0

    total_validation = 0

    with torch.no_grad():

        for images, goddess_labels, material_labels in validation_loader:

            images = images.to(device)
            goddess_labels = goddess_labels.to(device)
            material_labels = material_labels.to(device)

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

            validation_goddess_correct += (
                goddess_predictions == goddess_labels
            ).sum().item()

            validation_material_correct += (
                material_predictions == material_labels
            ).sum().item()

            total_validation += images.size(0)


    validation_goddess_accuracy = (
        validation_goddess_correct /
        total_validation
    ) * 100

    validation_material_accuracy = (
        validation_material_correct /
        total_validation
    ) * 100


    # ========================================================
    # COMBINED VALIDATION SCORE
    # ========================================================

    combined_validation_score = (
        validation_goddess_accuracy
        +
        validation_material_accuracy
    ) / 2


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step(
        combined_validation_score
    )

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "-" * 60)

    print(
        "Epoch",
        epoch + 1,
        "/",
        EPOCHS
    )

    print(
        "Loss:",
        round(
            total_loss / len(train_loader),
            4
        )
    )

    print(
        "Learning Rate:",
        current_learning_rate
    )

    print(
        "Train Goddess Accuracy:",
        round(
            train_goddess_accuracy,
            2
        ),
        "%"
    )

    print(
        "Train Material Accuracy:",
        round(
            train_material_accuracy,
            2
        ),
        "%"
    )

    print(
        "Validation Goddess Accuracy:",
        round(
            validation_goddess_accuracy,
            2
        ),
        "%"
    )

    print(
        "Validation Material Accuracy:",
        round(
            validation_material_accuracy,
            2
        ),
        "%"
    )

    print(
        "Combined Validation Score:",
        round(
            combined_validation_score,
            2
        ),
        "%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if combined_validation_score > best_validation_score:

        best_validation_score = (
            combined_validation_score
        )

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ BEST MODEL SAVED"
        )

    else:

        epochs_without_improvement += 1

        print(
            "No improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print("\n" + "=" * 60)
        print("EARLY STOPPING")
        print("=" * 60)

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    "Best model loaded."
)

print(
    "Best validation score:",
    round(
        best_validation_score,
        2
    ),
    "%"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION - MODEL 4")
print("=" * 60)

test_goddess_correct = 0
test_material_correct = 0

total_test = 0

test_goddess_labels = []
test_goddess_predictions = []
test_material_labels = []
test_material_predictions = []

with torch.no_grad():

    for images, goddess_labels, material_labels in test_loader:

        images = images.to(device)
        goddess_labels = goddess_labels.to(device)
        material_labels = material_labels.to(device)

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

        test_goddess_labels.extend(goddess_labels.cpu().tolist())
        test_goddess_predictions.extend(goddess_predictions.cpu().tolist())
        test_material_labels.extend(material_labels.cpu().tolist())
        test_material_predictions.extend(material_predictions.cpu().tolist())

        material_predictions = torch.argmax(
            material_outputs,
            dim=1
        )

        test_goddess_correct += (
            goddess_predictions == goddess_labels
        ).sum().item()

        test_material_correct += (
            material_predictions == material_labels
        ).sum().item()

        total_test += images.size(0)

print_evaluation_summary(
    test_goddess_labels,
    test_goddess_predictions,
    GODDESSES,
    "Goddess classification",
)
print_evaluation_summary(
    test_material_labels,
    test_material_predictions,
    MATERIALS,
    "Material classification",
)

test_goddess_accuracy = (
    test_goddess_correct /
    total_test
) * 100

test_material_accuracy = (
    test_material_correct /
    total_test
) * 100


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL 4 FINAL RESULTS")
print("=" * 60)

print(
    "Test Images:",
    total_test
)

print(
    "Test Goddess Accuracy:",
    round(
        test_goddess_accuracy,
        2
    ),
    "%"
)

print(
    "Test Material Accuracy:",
    round(
        test_material_accuracy,
        2
    ),
    "%"
)

print(
    "\nBest validation score:",
    round(
        best_validation_score,
        2
    ),
    "%"
)

print(
    "\nModel saved at:"
)

print(
    MODEL_PATH
)

print("\n" + "=" * 60)
print("MODEL 4 TRAINING COMPLETED")
print("=" * 60)