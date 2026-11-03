import os
from PIL import Image

DATASET_PATH = "Goddess_Resized"

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

total_images = 0

print("=" * 60)
print("MODEL DATASET CHECK")
print("=" * 60)

for goddess in GODDESSES:

    goddess_path = os.path.join(DATASET_PATH, goddess)

    print("\n" + goddess)
    print("-" * 40)

    for material in MATERIALS:

        if material in ["Brass", "Bronze", "Copper", "Silver"]:
            folder_path = os.path.join(
                goddess_path,
                "Metals",
                material
            )
        else:
            folder_path = os.path.join(
                goddess_path,
                material
            )

        count = 0

        if os.path.exists(folder_path):

            for file in os.listdir(folder_path):

                if file.lower().endswith(".jpg"):

                    image_path = os.path.join(folder_path, file)

                    try:
                        with Image.open(image_path) as img:
                            img.verify()

                        count += 1
                        total_images += 1

                    except Exception:
                        print("ERROR:", image_path)

        print(material, ":", count)

print("\n" + "=" * 60)
print("TOTAL IMAGES :", total_images)
print("=" * 60)