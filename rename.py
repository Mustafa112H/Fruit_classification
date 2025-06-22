
import os

folder = "Data/blueberries"
image_extensions = (".jpg", ".jpeg", ".png", ".webp")

# Get image files
image_files = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith(image_extensions)
])

# 1️⃣ TEMPORARY rename to avoid collisions
for i, filename in enumerate(image_files):
    ext = os.path.splitext(filename)[1].lower()
    temp_name = f"temp_{i}{ext}"
    os.rename(os.path.join(folder, filename), os.path.join(folder, temp_name))

# 2️⃣ FINAL rename to ordered format
temp_files = sorted([
    f for f in os.listdir(folder)
    if f.startswith("temp_")
])

for i, filename in enumerate(temp_files):
    ext = os.path.splitext(filename)[1].lower()
    final_name = f"{i}{ext}"
    os.rename(os.path.join(folder, filename), os.path.join(folder, final_name))
    print(f"✅ Renamed: {filename} ➜ {final_name}")

print("🎉 All files renamed safely in order.")
