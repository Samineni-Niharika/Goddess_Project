import os
import random
import shutil

SOURCE = "Goddess_Resized"
DESTINATION = "Goddess_Split"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

random.seed(42)

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

# Remove the old split
if os.path.exists(DESTINATION):
    shutil.rmtree(DESTINATION)

total_train = 0
total_validation = 0
total_test = 0

print("=" * 60)
print("CREATING FINAL TRAIN / VALIDATION / TEST DATASET")
print("=" * 60)

for goddess in GODDESSES:

    for material in MATERIALS:

        if material in ["Brass", "Bronze", "Copper", "Silver"]:
            source_folder = os.path.join(
                SOURCE,
                goddess,
                "Metals",
                material
            )
        else:
            source_folder = os.path.join(
                SOURCE,
                goddess,
                material
            )

        if not os.path.exists(source_folder):
            print("MISSING:", source_folder)
            continue

        images = []

        for file in os.listdir(source_folder):
            if file.lower().endswith(".jpg"):
                images.append(file)

        random.shuffle(images)

        total = len(images)

        train_count = round(total * TRAIN_RATIO)
        validation_count = round(total * VALIDATION_RATIO)

        # Make sure at least one image remains for testing
        if train_count + validation_count >= total:
            validation_count = max(1, validation_count - 1)

        train_images = images[:train_count]

        validation_images = images[
            train_count:train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]

        splits = {
            "train": train_images,
            "validation": validation_images,
            "test": test_images
        }

        for split_name, split_images in splits.items():

            if material in ["Brass", "Bronze", "Copper", "Silver"]:
                destination_folder = os.path.join(
                    DESTINATION,
                    split_name,
                    goddess,
                    "Metals",
                    material
                )
            else:
                destination_folder = os.path.join(
                    DESTINATION,
                    split_name,
                    goddess,
                    material
                )

            os.makedirs(destination_folder, exist_ok=True)

            for image in split_images:

                source_path = os.path.join(
                    source_folder,
                    image
                )

                destination_path = os.path.join(
                    destination_folder,
                    image
                )

                shutil.copy2(
                    source_path,
                    destination_path
                )

        total_train += len(train_images)
        total_validation += len(validation_images)
        total_test += len(test_images)

        print(
            goddess,
            "-",
            material,
            ":",
            total,
            "→ Train:",
            len(train_images),
            "Validation:",
            len(validation_images),
            "Test:",
            len(test_images)
        )

print("\n" + "=" * 60)
print("FINAL SPLIT COMPLETED")
print("=" * 60)

print("Training images   :", total_train)
print("Validation images :", total_validation)
print("Testing images    :", total_test)
print(
    "Total images      :",
    total_train + total_validation + total_test
)

print("=" * 60)