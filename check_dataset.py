import os
from PIL import Image

DATASET_PATH = "Goddess_Resized"



VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

total_images = 0
total_corrupted = 0

print("=" * 60)
print("GODDESS DATASET CHECK")
print("=" * 60)

for goddess in os.listdir(DATASET_PATH):

    goddess_path = os.path.join(DATASET_PATH, goddess)

    if not os.path.isdir(goddess_path):
        continue

    print("\n" + goddess)
    print("-" * 40)

    goddess_total = 0

    for root, dirs, files in os.walk(goddess_path):

        folder_count = 0

        for file in files:

            if file.lower().endswith(VALID_EXTENSIONS):

                image_path = os.path.join(root, file)

                try:
                    with Image.open(image_path) as img:
                        img.verify()

                    folder_count += 1
                    goddess_total += 1
                    total_images += 1

                except Exception:
                    total_corrupted += 1
                    print("CORRUPTED:", image_path)

        if folder_count > 0:
            relative_folder = os.path.relpath(root, goddess_path)
            print(relative_folder, ":", folder_count)

    print("Total for", goddess, ":", goddess_total)

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Total valid images :", total_images)
print("Corrupted images   :", total_corrupted)

print("=" * 60)
