import os
import hashlib
from PIL import Image
from tqdm import tqdm

def dhash(image, hash_size=8):
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            diff.append(pixel_left > pixel_right)
    decimal_value = 0
    hex_string = []
    for i, v in enumerate(diff):
        if v:
            decimal_value |= 1 << (i % 8)
        if (i % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0
    return ''.join(hex_string)

def remove_duplicate_images(folder_path):
    hashes = {}
    duplicates = []

    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                hash_val = dhash(img)
                if hash_val in hashes:
                    duplicates.append(file_path)
                else:
                    hashes[hash_val] = file_path
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Found {len(duplicates)} duplicates.")
    for dup in duplicates:
        os.remove(dup)
        print(f"Deleted: {dup}")

# === USAGE ===
# Change this to your image folder (e.g., "blueberries")
remove_duplicate_images("Data/blueberries")


folder_path = "Output/blueberries"

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".webp")

# Count
image_count = sum(
    1 for file in os.listdir(folder_path)
    if file.lower().endswith(image_extensions)
)

print(f"🫐 Total images in '{folder_path}': {image_count}")