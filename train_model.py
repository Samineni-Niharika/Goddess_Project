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
EPOCHS = 20
LEARNING_RATE = 0.001


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
# CUSTOM DATASET
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

                # Find goddess from folder path
                for name in GODDESSES:

                    if name in path_parts:

                        goddess = name
                        break

                # Find material from folder path
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
# VALIDATION / TEST TRANSFORMATION
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


# ============================================================
# CHECK DATASET
# ============================================================

if len(train_dataset) == 0:

    print("\nERROR: No training images found.")

    print(
        "Training path:",
        TRAIN_PATH
    )

    raise SystemExit


if len(validation_dataset) == 0:

    print("\nERROR: No validation images found.")

    print(
        "Validation path:",
        VALIDATION_PATH
    )

    raise SystemExit


if len(test_dataset) == 0:

    print("\nERROR: No test images found.")

    print(
        "Test path:",
        TEST_PATH
    )

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
print("CREATING RESNET18 MODEL")
print("=" * 60)


model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


number_of_features = model.fc.in_features


# Remove original ImageNet classifier
model.fc = nn.Identity()


# Goddess classifier
model.goddess_classifier = nn.Linear(
    number_of_features,
    6
)


# Material classifier
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
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    correct_goddess = 0

    correct_material = 0

    total_samples = 0


    # ========================================================
    # TRAINING BATCHES
    # ========================================================

    for images, goddess_labels, material_labels in train_loader:

        images = images.to(device)

        goddess_labels = goddess_labels.to(device)

        material_labels = material_labels.to(device)


        optimizer.zero_grad()


        # Feature extraction
        features = model(images)


        # Goddess prediction
        goddess_outputs = model.goddess_classifier(
            features
        )


        # Material prediction
        material_outputs = model.material_classifier(
            features
        )


        # Calculate losses
        goddess_loss = goddess_loss_function(
            goddess_outputs,
            goddess_labels
        )


        material_loss = material_loss_function(
            material_outputs,
            material_labels
        )


        # Combined loss
        loss = goddess_loss + material_loss


        # Backpropagation
        loss.backward()


        optimizer.step()


        total_loss += loss.item()


        # Predictions
        goddess_predictions = torch.argmax(
            goddess_outputs,
            dim=1
        )


        material_predictions = torch.argmax(
            material_outputs,
            dim=1
        )


        # Correct predictions
        correct_goddess += (
            goddess_predictions == goddess_labels
        ).sum().item()


        correct_material += (
            material_predictions == material_labels
        ).sum().item()


        total_samples += images.size(0)


    # ========================================================
    # TRAINING ACCURACY
    # ========================================================

    train_goddess_accuracy = (
        correct_goddess /
        total_samples
    ) * 100


    train_material_accuracy = (
        correct_material /
        total_samples
    ) * 100


    average_loss = (
        total_loss /
        len(train_loader)
    )


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
        round(average_loss, 4)
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


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_PATH,
    "goddess_material_resnet18.pth"
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)


print(
    "Model saved at:"
)


print(MODEL_PATH)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


model.eval()


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
# TEST ACCURACY
# ============================================================

test_goddess_accuracy = (
    test_goddess_correct /
    test_total
) * 100


test_material_accuracy = (
    test_material_correct /
    test_total
) * 100


print(
    "\nTest images:",
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


print("=" * 60)