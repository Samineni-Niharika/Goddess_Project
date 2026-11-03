import os
from PIL import Image

DATASET_PATH = "Goddess_Resized"

total_images = 0
total_failed = 0
non_jpg = 0

print("=" * 60)
print("STANDARD DATASET CHECK")
print("=" * 60)

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        file_path = os.path.join(root, file)

        if not file.lower().endswith(".jpg"):
            non_jpg += 1
            print("NOT JPG:", file_path)
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()

            total_images += 1

        except Exception:
            total_failed += 1
            print("CORRUPTED:", file_path)

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Total JPG images :", total_images)
print("Non-JPG files    :", non_jpg)
print("Corrupted images :", total_failed)

print("=" * 60)