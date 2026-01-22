# model_3.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PartCounter(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()  # ← Fixed: __init__

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 1280  # EfficientNet-B0 feature dimension

        self.bolt_head = nn.Linear(feat_dim, num_classes)
        self.pin_head = nn.Linear(feat_dim, num_classes)
        self.nut_head = nn.Linear(feat_dim, num_classes)
        self.washer_head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Same as .flatten(1)

        return (
            self.bolt_head(x),
            self.pin_head(x),
            self.nut_head(x),
            self.washer_head(x)
        )