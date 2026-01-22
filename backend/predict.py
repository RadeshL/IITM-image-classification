# src/predict.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import PartsDataset
from model_classifier import ClassifierModel
from model_regression import RegressionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLS_PATH = os.path.join(BASE_DIR, "outputs", "best_model.pth")
REG_PATH = os.path.join(BASE_DIR, "outputs", "finetunemodel.pth")

# ======================
# LOAD DATA
# ======================
image_dir = os.path.join(BASE_DIR, "dataset", "test_images")

image_names = sorted(
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

test_df = pd.DataFrame({"image_name": image_names})
dataset = PartsDataset(test_df, image_dir=image_dir, train=False)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# ======================
# LOAD MODELS
# ======================
cls_model = ClassifierModel(num_classes=5).to(DEVICE)
cls_model.load_state_dict(torch.load(CLS_PATH, map_location=DEVICE))
cls_model.eval()

reg_model = RegressionModel().to(DEVICE)
reg_model.load_state_dict(torch.load(REG_PATH, map_location=DEVICE))
reg_model.eval()

print("✅ Hybrid models loaded")

# ======================
# CONFIDENCE GATE
# ======================
def hybrid_predict(logits, reg, thresh=0.7):
    probs = torch.softmax(logits, dim=1)
    conf, cls_pred = probs.max(dim=1)

    reg_pred = torch.clamp(torch.round(reg), 0, 4).long()
    final = cls_pred.clone()
    final[conf < thresh] = reg_pred[conf < thresh]
    return final

# ======================
# INFERENCE
# ======================
rows = []

with torch.no_grad():
    for imgs, names in loader:   # ✅ FIXED
        imgs = imgs.to(DEVICE)

        b, p, n, w = cls_model(imgs)
        reg = reg_model(imgs)

        preds = torch.stack([
            hybrid_predict(b, reg[:, 0]),
            hybrid_predict(p, reg[:, 1]),
            hybrid_predict(n, reg[:, 2]),
            hybrid_predict(w, reg[:, 3]),
        ], dim=1).cpu().numpy()

        for name, r in zip(names, preds):
            rows.append([name, *r])

# ======================
# SAVE
# ======================
df = pd.DataFrame(rows, columns=["image_name", "bolt", "locatingpin", "nut", "washer"])
df.to_csv("submission.csv", index=False)

print(f"🏁 Hybrid submission.csv generated with {len(rows)} rows")
