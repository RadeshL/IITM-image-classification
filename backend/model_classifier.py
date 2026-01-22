# src/model_classifier.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ClassifierModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # IMPORTANT: use `features`, NOT `backbone.features`
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = 1280

        self.bolt_head   = nn.Linear(in_features, num_classes)
        self.pin_head    = nn.Linear(in_features, num_classes)
        self.nut_head    = nn.Linear(in_features, num_classes)
        self.washer_head = nn.Linear(in_features, num_classes)

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
