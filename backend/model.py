import torch.nn as nn
from torchvision import models

class PartCounter(nn.Module):
    def __init__(self):
        super().__init__()

        # Load EfficientNet-B0 pretrained on ImageNet
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Replace ImageNet classifier with regression head (4 outputs)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 4)

    def forward(self, x):
        """
        Output:
        Tensor of shape [batch_size, 4]
        [bolt, locatingpin, nut, washer]
        """
        return self.backbone(x)
