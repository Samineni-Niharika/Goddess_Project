import os
from PIL import Image


SOURCE = "Goddess_Resized"

DESTINATION = "Goddess_Split"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

converted = 0
failed = 0

print("=" * 60)
print("CREATING STANDARD DATASET")
print("=" * 60)

for root, dirs, files in os.walk(SOURCE):

    for file in files:

        if not file.lower().endswith(VALID_EXTENSIONS):
            continue

        source_path = os.path.join(root, file)

        # Get the folder structure after C:\Goddess
        relative_path = os.path.relpath(root, SOURCE)

        destination_folder = os.path.join(
            DESTINATION,
            relative_path
        )

        os.makedirs(destination_folder, exist_ok=True)

        # Change extension to JPG
        file_name = os.path.splitext(file)[0] + ".jpg"

        destination_path = os.path.join(
            destination_folder,
            file_name
        )

        try:
            with Image.open(source_path) as img:

                # Convert to RGB because JPG doesn't support transparency
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(
                    destination_path,
                    "JPEG",
                    quality=95
                )

            converted += 1

        except Exception as e:

            failed += 1
            print("FAILED:", source_path)
            print("Reason:", e)

print("\n" + "=" * 60)
print("STANDARD DATASET CREATED")
print("=" * 60)

print("Images converted :", converted)
print("Images failed    :", failed)
print("Output folder    :", DESTINATION)

print("=" * 60)