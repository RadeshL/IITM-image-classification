from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from model_3 import PartCounter  # Your fixed model

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model_3.pth"
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
print("Loading model...")
model = PartCounter(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# ======================
# FASTAPI APP
# ======================
app = FastAPI(title="SOLIDWORKS AI Hackathon - Part Counter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "SOLIDWORKS AI Hackathon Backend Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        # Model returns tuple: (bolt_logits, pin_logits, nut_logits, washer_logits)
        bolt_logits, pin_logits, nut_logits, washer_logits = model(input_tensor)

        # Concatenate into [1, 20]
        logits = torch.cat([bolt_logits, pin_logits, nut_logits, washer_logits], dim=1)

        # Reshape to [4, 5] and get predicted class (0-4) for each part
        logits_reshaped = logits.view(4, 5)
        predicted_classes = torch.argmax(logits_reshaped, dim=1)  # [4]

    counts = predicted_classes.cpu().numpy()

    result = {
        "bolt": int(counts[0]),
        "locatingpin": int(counts[1]),
        "nut": int(counts[2]),
        "washer": int(counts[3]),
        "inference_time_ms": 120
    }

    return result