import os
from PIL import Image

# ============================================================
# PATHS
# ============================================================

SOURCE = "New_Test_Images"

DESTINATION = "New_Test_Images_Processed"

IMAGE_SIZE = (224, 224)


# ============================================================
# CREATE DESTINATION FOLDER
# ============================================================

os.makedirs(DESTINATION, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("PREPARING NEW TEST IMAGES")
print("=" * 60)

count = 0
failed = 0


# ============================================================
# PROCESS IMAGES
# ============================================================

for file in os.listdir(SOURCE):

    if not file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    source_path = os.path.join(
        SOURCE,
        file
    )

    try:

        # Open image
        image = Image.open(
            source_path
        ).convert("RGB")

        # Resize to 224 x 224
        image = image.resize(
            IMAGE_SIZE
        )

        # Remove old extension
        name = os.path.splitext(file)[0]

        # Save everything as JPG
        destination_path = os.path.join(
            DESTINATION,
            name + ".jpg"
        )

        image.save(
            destination_path,
            "JPEG",
            quality=95
        )

        count += 1

        print(
            "Processed:",
            file
        )

    except Exception as e:

        failed += 1

        print(
            "ERROR:",
            file
        )

        print(
            "Reason:",
            e
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREPARATION COMPLETED")
print("=" * 60)

print(
    "Images processed :",
    count
)

print(
    "Images failed    :",
    failed
)

print(
    "\nProcessed images saved at:"
)

print(
    DESTINATION
)

print("=" * 60)