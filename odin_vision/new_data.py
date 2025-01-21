import os

for _, _, files in os.walk("train/images"):
    old_images = files
for _, _, files in os.walk("train/labels"):
    old_labels = files

for _, _, files in os.walk("staging/images"):
    for file in files:
        if file in old_images:
            os.remove(f"staging/images/{file}")
    break

for _, _, files in os.walk("staging/labels"):
    for file in files:
        if file in old_labels:
            os.remove(f"staging/labels/{file}")
    break
