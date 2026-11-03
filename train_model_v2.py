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

IMAGE_SIZE = 224
BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 0.0001

PATIENCE = 5

MODEL_PATH = os.path.join(
    BASE_PATH,
    "goddess_material_resnet18_v2_best.pth"
)


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
            "images from",
            root_dir
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
            torch.tensor(
                goddess_label,
                dtype=torch.long
            ),
            torch.tensor(
                material_label,
                dtype=torch.long
            )
        )


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.90, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
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

normal_transform = transforms.Compose([

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
print("LOADING DATASETS")
print("=" * 60)


train_dataset = GoddessMaterialDataset(
    TRAIN_PATH,
    train_transform
)


validation_dataset = GoddessMaterialDataset(
    VALIDATION_PATH,
    normal_transform
)


test_dataset = GoddessMaterialDataset(
    TEST_PATH,
    normal_transform
)


if len(train_dataset) == 0:

    print("ERROR: No training images found.")

    raise SystemExit


if len(validation_dataset) == 0:

    print("ERROR: No validation images found.")

    raise SystemExit


if len(test_dataset) == 0:

    print("ERROR: No test images found.")

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
# MODEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING RESNET18 MODEL - VERSION 2")
print("=" * 60)


model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
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
# LOSS
# ============================================================

goddess_loss_function = nn.CrossEntropyLoss()

material_loss_function = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
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
# BEST MODEL VARIABLES
# ============================================================

best_validation_score = 0.0

epochs_without_improvement = 0


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []

train_goddess_accuracies = []

train_material_accuracies = []

validation_goddess_accuracies = []

validation_material_accuracies = []


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING MODEL 2 TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    model.train()


    total_loss = 0

    goddess_correct = 0

    material_correct = 0

    total_samples = 0


    # ========================================================
    # TRAIN
    # ========================================================

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


        loss = goddess_loss + material_loss


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


        total_samples += images.size(0)


    # ========================================================
    # TRAINING METRICS
    # ========================================================

    train_loss = (
        total_loss /
        len(train_loader)
    )


    train_goddess_accuracy = (
        goddess_correct /
        total_samples
    ) * 100


    train_material_accuracy = (
        material_correct /
        total_samples
    ) * 100


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    validation_goddess_correct = 0

    validation_material_correct = 0

    validation_total = 0


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


            validation_total += images.size(0)


    validation_goddess_accuracy = (
        validation_goddess_correct /
        validation_total
    ) * 100


    validation_material_accuracy = (
        validation_material_correct /
        validation_total
    ) * 100


    # ========================================================
    # COMBINED VALIDATION SCORE
    # ========================================================

    validation_score = (
        validation_goddess_accuracy +
        validation_material_accuracy
    ) / 2


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    train_losses.append(
        train_loss
    )

    train_goddess_accuracies.append(
        train_goddess_accuracy
    )

    train_material_accuracies.append(
        train_material_accuracy
    )

    validation_goddess_accuracies.append(
        validation_goddess_accuracy
    )

    validation_material_accuracies.append(
        validation_material_accuracy
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        validation_score
    )


    # ========================================================
    # PRINT
    # ========================================================

    current_lr = optimizer.param_groups[0]["lr"]


    print("\n" + "-" * 60)

    print(
        "Epoch",
        epoch + 1,
        "/",
        EPOCHS
    )

    print(
        "Loss:",
        round(train_loss, 4)
    )

    print(
        "Learning Rate:",
        current_lr
    )

    print(
        "Train Goddess Accuracy:",
        round(train_goddess_accuracy, 2),
        "%"
    )

    print(
        "Train Material Accuracy:",
        round(train_material_accuracy, 2),
        "%"
    )

    print(
        "Validation Goddess Accuracy:",
        round(validation_goddess_accuracy, 2),
        "%"
    )

    print(
        "Validation Material Accuracy:",
        round(validation_material_accuracy, 2),
        "%"
    )

    print(
        "Combined Validation Score:",
        round(validation_score, 2),
        "%"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if validation_score > best_validation_score:

        best_validation_score = validation_score

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

        print(
            "EARLY STOPPING"
        )

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


model = model.to(device)

model.eval()


print(
    "Best model loaded."
)


print(
    "Best validation score:",
    round(best_validation_score, 2),
    "%"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION - MODEL 2")
print("=" * 60)


test_goddess_correct = 0

test_material_correct = 0

test_total = 0

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


        test_total += images.size(0)

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

# ============================================================
# FINAL ACCURACY
# ============================================================

test_goddess_accuracy = (
    test_goddess_correct /
    test_total
) * 100


test_material_accuracy = (
    test_material_correct /
    test_total
) * 100


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2 FINAL RESULTS")
print("=" * 60)


print(
    "Test Images:",
    test_total
)


print(
    "Test Goddess Accuracy:",
    round(test_goddess_accuracy, 2),
    "%"
)


print(
    "Test Material Accuracy:",
    round(test_material_accuracy, 2),
    "%"
)


print(
    "\nBest validation score:",
    round(best_validation_score, 2),
    "%"
)


print(
    "\nModel saved at:"
)


print(
    MODEL_PATH
)


print("=" * 60)
print("MODEL 2 TRAINING COMPLETED")
print("=" * 60)