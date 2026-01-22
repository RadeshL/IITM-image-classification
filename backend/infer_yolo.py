from ultralytics import YOLO
import os
from collections import Counter

model = YOLO("runs/detect/parts_detector/weights/best.pt")

IMAGE_DIR = "dataset/images/val"

for img in os.listdir(IMAGE_DIR):
    if not img.endswith(".jpg"):
        continue

    result = model(os.path.join(IMAGE_DIR, img), conf=0.3)[0]

    counts = Counter()
    for cls_id in result.boxes.cls.tolist():
        counts[int(cls_id)] += 1

    print(f"\nImage: {img}")
    print({
        "bolt": counts.get(0, 0),
        "locatingpin": counts.get(1, 0),
        "nut": counts.get(2, 0),
        "washer": counts.get(3, 0)
    })
