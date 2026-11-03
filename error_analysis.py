import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


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

ERROR_PATH = os.path.join(
    BASE_PATH,
    "error_analysis"
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
# TEST DATASET
# ============================================================

class TestDataset(Dataset):

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

                parts = relative_path.split(
                    os.sep
                )

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
                            goddess,
                            material
                        )
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, goddess, material = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return (
            image,
            GODDESS_TO_INDEX[goddess],
            MATERIAL_TO_INDEX[material],
            image_path
        )


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
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)


dataset = TestDataset(
    TEST_PATH,
    transform
)


print(
    "Loaded",
    len(dataset),
    "test images"
)


if len(dataset) == 0:

    print(
        "ERROR: No test images found."
    )

    raise SystemExit


loader = DataLoader(
    dataset,
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
# MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MODEL 1")
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


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model = model.to(device)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# CREATE ERROR FOLDERS
# ============================================================

if os.path.exists(ERROR_PATH):

    shutil.rmtree(
        ERROR_PATH
    )


os.makedirs(
    ERROR_PATH,
    exist_ok=True
)


goddess_error_path = os.path.join(
    ERROR_PATH,
    "goddess_errors"
)


material_error_path = os.path.join(
    ERROR_PATH,
    "material_errors"
)


both_error_path = os.path.join(
    ERROR_PATH,
    "both_errors"
)


os.makedirs(
    goddess_error_path,
    exist_ok=True
)

os.makedirs(
    material_error_path,
    exist_ok=True
)

os.makedirs(
    both_error_path,
    exist_ok=True
)


# ============================================================
# COUNTERS
# ============================================================

total = 0

goddess_correct = 0

material_correct = 0

goddess_errors = 0

material_errors = 0

both_errors = 0


# ============================================================
# ERROR LISTS
# ============================================================

goddess_error_list = []

material_error_list = []

both_error_list = []


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("ANALYZING TEST IMAGES")
print("=" * 60)


with torch.no_grad():

    for images, goddess_labels, material_labels, paths in loader:

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


        goddess_probabilities = torch.softmax(
            goddess_outputs,
            dim=1
        )


        material_probabilities = torch.softmax(
            material_outputs,
            dim=1
        )


        goddess_predictions = torch.argmax(
            goddess_outputs,
            dim=1
        )


        material_predictions = torch.argmax(
            material_outputs,
            dim=1
        )


        for i in range(len(paths)):

            total += 1


            actual_goddess = INDEX_TO_GODDESS[
                goddess_labels[i].item()
            ]


            predicted_goddess = INDEX_TO_GODDESS[
                goddess_predictions[i].item()
            ]


            actual_material = INDEX_TO_MATERIAL[
                material_labels[i].item()
            ]


            predicted_material = INDEX_TO_MATERIAL[
                material_predictions[i].item()
            ]


            goddess_confidence = (
                goddess_probabilities[i][
                    goddess_predictions[i]
                ].item()
                * 100
            )


            material_confidence = (
                material_probabilities[i][
                    material_predictions[i]
                ].item()
                * 100
            )


            goddess_is_wrong = (
                actual_goddess !=
                predicted_goddess
            )


            material_is_wrong = (
                actual_material !=
                predicted_material
            )


            # ==================================================
            # CORRECT / WRONG COUNTS
            # ==================================================

            if not goddess_is_wrong:

                goddess_correct += 1

            else:

                goddess_errors += 1


            if not material_is_wrong:

                material_correct += 1

            else:

                material_errors += 1


            # ==================================================
            # GODDESS ERROR
            # ==================================================

            if goddess_is_wrong:

                filename = os.path.basename(
                    paths[i]
                )

                destination = os.path.join(
                    goddess_error_path,
                    "actual_"
                    + actual_goddess
                    + "__predicted_"
                    + predicted_goddess
                    + "__"
                    + filename
                )


                shutil.copy2(
                    paths[i],
                    destination
                )


                goddess_error_list.append(
                    (
                        actual_goddess,
                        predicted_goddess,
                        round(
                            goddess_confidence,
                            2
                        ),
                        paths[i]
                    )
                )


            # ==================================================
            # MATERIAL ERROR
            # ==================================================

            if material_is_wrong:

                filename = os.path.basename(
                    paths[i]
                )

                destination = os.path.join(
                    material_error_path,
                    "actual_"
                    + actual_material
                    + "__predicted_"
                    + predicted_material
                    + "__"
                    + filename
                )


                shutil.copy2(
                    paths[i],
                    destination
                )


                material_error_list.append(
                    (
                        actual_material,
                        predicted_material,
                        round(
                            material_confidence,
                            2
                        ),
                        paths[i]
                    )
                )


            # ==================================================
            # BOTH WRONG
            # ==================================================

            if goddess_is_wrong and material_is_wrong:

                both_errors += 1


                filename = os.path.basename(
                    paths[i]
                )


                destination = os.path.join(
                    both_error_path,
                    "actual_"
                    + actual_goddess
                    + "_"
                    + actual_material
                    + "__predicted_"
                    + predicted_goddess
                    + "_"
                    + predicted_material
                    + "__"
                    + filename
                )


                shutil.copy2(
                    paths[i],
                    destination
                )


                both_error_list.append(
                    (
                        actual_goddess,
                        predicted_goddess,
                        actual_material,
                        predicted_material,
                        paths[i]
                    )
                )


# ============================================================
# FINAL RESULTS
# ============================================================

goddess_accuracy = (
    goddess_correct /
    total
) * 100


material_accuracy = (
    material_correct /
    total
) * 100


print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)


print(
    "Total test images:",
    total
)


print(
    "Goddess correct:",
    goddess_correct
)


print(
    "Goddess errors:",
    goddess_errors
)


print(
    "Goddess accuracy:",
    round(
        goddess_accuracy,
        2
    ),
    "%"
)


print()


print(
    "Material correct:",
    material_correct
)


print(
    "Material errors:",
    material_errors
)


print(
    "Material accuracy:",
    round(
        material_accuracy,
        2
    ),
    "%"
)


print()


print(
    "Both Goddess + Material wrong:",
    both_errors
)


# ============================================================
# GODDESS ERROR DETAILS
# ============================================================

print("\n" + "=" * 60)
print("GODDESS MISCLASSIFICATIONS")
print("=" * 60)


for error in goddess_error_list:

    print(
        "Actual:",
        error[0],
        "| Predicted:",
        error[1],
        "| Confidence:",
        error[2],
        "%"
    )

    print(
        "Image:",
        error[3]
    )

    print()


# ============================================================
# MATERIAL ERROR DETAILS
# ============================================================

print("\n" + "=" * 60)
print("MATERIAL MISCLASSIFICATIONS")
print("=" * 60)


for error in material_error_list:

    print(
        "Actual:",
        error[0],
        "| Predicted:",
        error[1],
        "| Confidence:",
        error[2],
        "%"
    )

    print(
        "Image:",
        error[3]
    )

    print()


# ============================================================
# IMPORTANT MATERIAL CONFUSIONS
# ============================================================

print("\n" + "=" * 60)
print("BRASS / BRONZE / COPPER CONFUSIONS")
print("=" * 60)


bbc_count = 0


for error in material_error_list:

    actual = error[0]

    predicted = error[1]


    if (
        actual in ["Brass", "Bronze", "Copper"]
        and
        predicted in ["Brass", "Bronze", "Copper"]
    ):

        print(
            "Actual:",
            actual,
            "| Predicted:",
            predicted,
            "| Confidence:",
            error[2],
            "%"
        )

        print(
            "Image:",
            error[3]
        )

        print()

        bbc_count += 1


if bbc_count == 0:

    print(
        "No Brass/Bronze/Copper confusion found."
    )


# ============================================================
# SAVED FOLDERS
# ============================================================

print("\n" + "=" * 60)
print("ERROR IMAGES SAVED")
print("=" * 60)


print(
    "Goddess errors:"
)


print(
    goddess_error_path
)


print(
    "\nMaterial errors:"
)


print(
    material_error_path
)


print(
    "\nBoth errors:"
)


print(
    both_error_path
)


print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETED")
print("=" * 60)