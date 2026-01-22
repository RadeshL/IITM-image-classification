from ultralytics import YOLO

# Load pretrained YOLOv8
model = YOLO("yolov8n.pt")  # s = safe balance for hackathon

model.train(
    data="src/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,   # use CPU if no GPU
    project="runs",
    workers=0,
    name="parts_detector"
)
