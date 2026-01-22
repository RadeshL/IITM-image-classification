# model1.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PartCounter(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained EfficientNet-B0
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # EfficientNet-B0 has 1280 features before classifier
        in_features = 1280  # Hardcoded — matches standard EfficientNet-B0

        # Each head outputs 5 classes (for counts 0,1,2,3,4)
        self.bolt_head   = nn.Linear(in_features, 5)
        self.pin_head    = nn.Linear(in_features, 5)
        self.nut_head    = nn.Linear(in_features, 5)
        self.washer_head = nn.Linear(in_features, 5)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # Return concatenated logits: [batch, 20]
        return torch.cat([
            self.bolt_head(x),
            self.pin_head(x),
            self.nut_head(x),
            self.washer_head(x)
        ], dim=1)