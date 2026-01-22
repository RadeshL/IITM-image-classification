from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from ultralytics import YOLO

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best.pt"  # Your trained YOLOv8 model

CLASS_NAMES = ["bolt", "locatingpin", "nut", "washer"]

# ======================
# LOAD MODEL
# ======================
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)  # Automatically loads to GPU if available
print("Model loaded successfully!")

# ======================
# FASTAPI APP
# ======================
app = FastAPI(title="SOLIDWORKS AI Hackathon - YOLOv8 Part Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "SOLIDWORKS AI Hackathon - YOLOv8 Part Counter Running!", "status": "ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference
    results = model(image, conf=0.25, iou=0.45, device=DEVICE)[0]  # First result

    # Count detections per class
    counts = {"bolt": 0, "locatingpin": 0, "nut": 0, "washer": 0}

    if results.boxes is not None:
        classes = results.boxes.cls.cpu().numpy().astype(int)
        for cls_id in classes:
            class_name = CLASS_NAMES[cls_id]
            counts[class_name] += 1

    result = {
        "bolt": counts["bolt"],
        "locatingpin": counts["locatingpin"],
        "nut": counts["nut"],
        "washer": counts["washer"],
        "inference_time_ms": round(results.speed['inference'], 0)  # Real inference time!
    }

    return result