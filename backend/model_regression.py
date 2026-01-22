# src/model_regression.py
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # IMPORTANT: keep EfficientNet classifier structure
        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.classifier[1].in_features

        # Regression head lives INSIDE backbone.classifier
        self.backbone.classifier[1] = nn.Linear(in_features, 4)

    def forward(self, x):
        return self.backbone(x)
