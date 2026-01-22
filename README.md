Mechanical Component Multi-Object Counter
IIT Madras – SOLIDWORKS AI Hackathon Project

An end-to-end deep learning system that directly predicts exact counts of mechanical parts (bolt, nut, washer, locating pin) from a single image using multi-task learning and EfficientNet.

Problem Statement

Given an image containing mechanical assemblies, precisely count the number of:

Bolt

Locating Pin

Nut

Washer

Instead of object detection or segmentation, the problem is formulated as multi-object discrete counting, where the system predicts a structured output:

Y = [bolt_count, locating_pin_count, nut_count, washer_count]

Key Idea

We avoid bounding boxes and detection pipelines.

Instead:

Single forward pass

Direct count prediction

Faster, simpler, more robust

This makes the system:

Lightweight

Edge-device friendly

CPU compatible

Easy to deploy

Architecture
Backbone

EfficientNet-B0 (ImageNet pretrained)

Extracts global semantic features

Captures shape + structural cues

Multi-Head Counting Design

Shared CNN backbone → 4 independent heads

Each head:

Predicts discrete count class (0–4)

Uses classification instead of regression

Heads:

Bolt head

Locating pin head

Nut head

Washer head

Why Classification Instead of Regression?

Counting is discrete and small-range.

Regression problems:

produce decimals

require rounding

unstable training

Classification advantages:

guaranteed integers

stable cross-entropy training

better convergence

aligns with evaluation metric

Loss Function

Total Loss =

CE(bolt)

CE(pin)

CE(nut)

CE(washers)

Multi-task learning improves generalization by sharing features across part types.

Data Processing

Resize → 224×224

ImageNet normalization

RGB conversion

No augmentation (synthetic dataset)

Integrity checks for labels

We intentionally avoided heavy augmentation to preserve mechanical geometry.

Evaluation Metric

Exact-Match Accuracy

A prediction is correct only if all 4 counts are correct simultaneously.

Example:

GT: [2,1,0,3]
Pred: [2,1,0,3] → Correct
Pred: [2,1,1,3] → Incorrect

This strict metric ensures industrial reliability.

Backend API (FastAPI)

The model is deployed using FastAPI.

Endpoint

POST /predict

Input:

image file

Output:

{
  "bolt": 2,
  "locatingpin": 1,
  "nut": 0,
  "washer": 3,
  "inference_time_ms": 120
}

Tech Stack

PyTorch

EfficientNet

FastAPI

Torchvision

PIL

Uvicorn

Run Locally
Install dependencies
pip install -r requirements.txt

Start server
uvicorn main:app --reload

Test API

Open:

http://127.0.0.1:8000/docs


Swagger UI will appear.

Project Structure
├── main.py              # FastAPI server
├── model_3.py          # PartCounter model
├── best_model_3.pth    # trained weights
├── requirements.txt
├── README.md
└── frontend/           # optional UI

Performance Highlights

CPU-only inference supported

<150ms per image

Lightweight (~20MB)

No detection overhead

Easy scaling to new parts

Scalability

To add a new component:

Add new classification head

Add labels

Retrain

No architectural redesign needed.

Applications

Industrial inspection

Assembly verification

Automated quality control

Robotics part counting

Manufacturing analytics
