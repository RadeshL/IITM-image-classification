# Mechanical Component Multi-Object Counter  
### IIT Madras – SOLIDWORKS AI Hackathon Project

A system that **predicts exact counts of mechanical components** (bolts, locating pins, nuts, washers) from a single image using a multi-task classification architecture.
---

## 📌 Problem Statement

Given an input image, precisely count the number of:

- Bolt
- Locating Pin
- Nut
- Washer

### Output Format

Y = [bolt_count, locating_pin_count, nut_count, washer_count]


Each image may contain zero or more instances of each component.

---

## 💡 Key Idea

Instead of:
- Object detection (bounding boxes)
- Instance segmentation
- Localization-heavy pipelines

We treat the task as:

> **Multi-Object Discrete Counting via Classification**

This allows:
- Single forward pass
- No bounding boxes
- Faster inference
- Integer predictions by design
- Lightweight deployment

---

## 🧠 Architecture

### Backbone
**EfficientNet-B0 (ImageNet pretrained)**

Used as a shared feature extractor to learn:
- shapes
- contours
- geometric structure
- contextual relationships

---

### Multi-Head Counting Design

Input Image
    ↓
EfficientNet Backbone
    ↓
Global Feature Vector
    ↓
4 Independent Classification Heads


| Head | Predicts |
|--------|-----------|
| Head 1 | Bolt count |
| Head 2 | Locating pin count |
| Head 3 | Nut count |
| Head 4 | Washer count |

Each head predicts a **count class (0–4)**.

# 📁 Project Structure
'''
main/
|
├── backend/ # FastAPI + PyTorch inference backend
│ ├── main.py # API server (loads model and serves predictions)
│ ├── model_3.py # Final EfficientNet multi-head model (production)
│ ├── predict.py # Inference utilities
│ ├── train_yolo.py # YOLO experiment training script
│ ├── infer_yolo.py # YOLO inference script
│ ├── model_classifier.py # Classification-based experiments
│ ├── model_regression.py # Regression-based experiments
│ └── init.py
│
├── src/ # React frontend (Vite)
│ ├── App.jsx 
│ ├── UploadCard.jsx # Image upload component
│ ├── ResultsCard.jsx # Prediction display
│ ├── ClassifyButton.jsx # API trigger button
│ ├── LightPillar.jsx # UI animation components
│ ├── ShinyText.jsx
│ ├── TextType.jsx
│ ├── assets/
│ └── styles (.css files)
├── package.json # Frontend dependencies
├── vite.config.js
├── README.md
└── .gitignore
'''
---

# Model Files & Experiments

Due to GitHub's file size limits, trained model weights (`.pth`, `.pt`) are **not stored in this repository**.

All trained checkpoints and experimental models can be accessed here:

### Google Drive (Models & Weights)
**[Download from Google Drive](https://drive.google.com/drive/u/0/folders/1O8PJYVl0c-KNazUjE-ZKy_uJY1M_WU7L)**

Contents include:

- Final multi-head EfficientNet model (best_model_3.pth)
- Earlier EfficientNet fine-tuning checkpoints
- Classification vs regression experiments
- YOLO detection-based baselines
---
To run inference locally:

1. Download weights from Drive
2. Place inside `backend/`
3. Update `MODEL_PATH` if required


