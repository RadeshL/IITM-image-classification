import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# -------------------------------------------------
# MODEL
# -------------------------------------------------

class PartCounter(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 1280  # EfficientNet-B0 output channels

        self.bolt_head   = nn.Linear(feat_dim, num_classes)
        self.pin_head    = nn.Linear(feat_dim, num_classes)
        self.nut_head    = nn.Linear(feat_dim, num_classes)
        self.washer_head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return (
            self.bolt_head(x),
            self.pin_head(x),
            self.nut_head(x),
            self.washer_head(x)
        )

# -------------------------------------------------
# TTA TRANSFORMS
# -------------------------------------------------

def get_tta_transforms():
    """
    Safe test-time augmentations
    """
    return [
        lambda x: x,                               # original
        lambda x: torch.flip(x, dims=[3]),         # horizontal flip
    ]

# -------------------------------------------------
# CONFIDENCE GUARD
# -------------------------------------------------

def predict_with_confidence(logits, conf_threshold=0.6):
    """
    logits: [B, num_classes]
    returns:
        pred: class index or -1 (uncertain)
        conf: confidence score
    """
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)

    pred = torch.where(
        conf >= conf_threshold,
        pred,
        torch.full_like(pred, -1)
    )

    return pred, conf

# -------------------------------------------------
# TTA INFERENCE
# -------------------------------------------------

@torch.no_grad()
def infer_with_tta(model, image, conf_threshold=0.6, device="cuda"):
    """
    image: Tensor [1, 3, H, W]
    """

    model.eval()
    image = image.to(device)

    tta_transforms = get_tta_transforms()

    bolt_logits = []
    pin_logits = []
    nut_logits = []
    washer_logits = []

    for tta in tta_transforms:
        img = tta(image)

        b, p, n, w = model(img)

        bolt_logits.append(b)
        pin_logits.append(p)
        nut_logits.append(n)
        washer_logits.append(w)

    # Average logits across TTA runs
    bolt_logits   = torch.mean(torch.stack(bolt_logits), dim=0)
    pin_logits    = torch.mean(torch.stack(pin_logits), dim=0)
    nut_logits    = torch.mean(torch.stack(nut_logits), dim=0)
    washer_logits = torch.mean(torch.stack(washer_logits), dim=0)

    # Confidence guard
    bolt_pred, bolt_conf     = predict_with_confidence(bolt_logits, conf_threshold)
    pin_pred, pin_conf       = predict_with_confidence(pin_logits, conf_threshold)
    nut_pred, nut_conf       = predict_with_confidence(nut_logits, conf_threshold)
    washer_pred, washer_conf = predict_with_confidence(washer_logits, conf_threshold)

    return {
        "bolt":   {"count": bolt_pred.item(),   "confidence": bolt_conf.item()},
        "pin":    {"count": pin_pred.item(),    "confidence": pin_conf.item()},
        "nut":    {"count": nut_pred.item(),    "confidence": nut_conf.item()},
        "washer": {"count": washer_pred.item(), "confidence": washer_conf.item()},
    }




