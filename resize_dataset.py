import os
from PIL import Image

SOURCE = "Goddess_Resized"
DESTINATION = "Goddess_Split"

IMAGE_SIZE = (224, 224)

total_images = 0
failed_images = 0

print("=" * 60)
print("CREATING RESIZED DATASET")
print("=" * 60)

for root, dirs, files in os.walk(SOURCE):

    for file in files:

        if not file.lower().endswith(".jpg"):
            continue

        source_path = os.path.join(root, file)

        relative_path = os.path.relpath(root, SOURCE)

        destination_folder = os.path.join(
            DESTINATION,
            relative_path
        )

        os.makedirs(destination_folder, exist_ok=True)

        destination_path = os.path.join(
            destination_folder,
            file
        )

        try:
            with Image.open(source_path) as img:

                img = img.convert("RGB")

                resized_img = img.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                resized_img.save(
                    destination_path,
                    "JPEG",
                    quality=95
                )

            total_images += 1

        except Exception as e:

            failed_images += 1

            print("FAILED:", source_path)
            print("Reason:", e)

print("\n" + "=" * 60)
print("RESIZED DATASET CREATED")
print("=" * 60)

print("Images resized :", total_images)
print("Images failed  :", failed_images)
print("Image size     :", IMAGE_SIZE)
print("Output folder  :", DESTINATION)

print("=" * 60)