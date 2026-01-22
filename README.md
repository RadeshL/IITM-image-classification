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



