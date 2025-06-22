import os
import shutil

# Define paths
source_folder = "blueberries"
destination_folder = "Data/blueberries"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Move image files
for filename in os.listdir(source_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)

        shutil.move(src_path, dst_path)
        print(f"✅ Moved: {filename}")

print("✅ All images moved successfully.")

folder_path = "Data/blueberries"

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".webp")

# Count
image_count = sum(
    1 for file in os.listdir(folder_path)
    if file.lower().endswith(image_extensions)
)

print(f"🫐 Total images in '{folder_path}': {image_count}")