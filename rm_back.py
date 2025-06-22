import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image

# Paths
DATA_DIR = ""
OUTPUT_DIR = "Output"
CLASSES = [ "pomegranates"]

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_bg_and_save(image_path, output_path):
    # Load BGR image, convert to RGB
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Failed to load {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb).convert("RGBA")

    # Remove background using rembg
    result = remove(pil_image)

    # Save output
    result.save(output_path)
    print(f"✅ Saved: {output_path}")

def process_class_folder(class_name):
    input_folder = os.path.join(DATA_DIR, class_name)
    output_folder = os.path.join(OUTPUT_DIR, class_name)

    ensure_folder(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

            if os.path.exists(output_path):
                print(f"⏩ Skipped (already processed): {output_path}")
                continue

            remove_bg_and_save(input_path, output_path)

# Process all classes
for cls in CLASSES:
    class_folder = os.path.join(DATA_DIR, cls)
    if os.path.isdir(class_folder):
        print(f"\n🔍 Processing: {cls}")
        process_class_folder(cls)
    else:
        print(f"⚠️ Skipping missing folder: {class_folder}")


print("\n✅ All classes processed successfully!")