from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from model1 import PartCounter  # Make sure model.py is in this folder

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model_.pth"
CLASS_NAMES = ["bolt", "locatingpin", "nut", "washer"]

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================
# LOAD MODEL
# ======================
model = PartCounter().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)  # Shape: [1, 20]

    # Reshape to [4, 5] and take argmax for each part
    logits = output.view(4, 5)  # 4 parts × 5 classes
    predicted_classes = torch.argmax(logits, dim=1)  # [4]

    counts = predicted_classes.cpu().numpy()

    result = {
        "bolt": int(counts[0]),
        "locatingpin": int(counts[1]),
        "nut": int(counts[2]),
        "washer": int(counts[3]),
        "inference_time_ms": 120
    }

    return result

@app.get("/")
def home():
    return {"message": "SOLIDWORKS AI Hackathon Backend Running!"}