import torch.nn as nn

from .gaze_head import GazeHead
from .mobileone import MobileOneS1


class GazeModel(nn.Module):
    """
    Complete gaze estimation model combining MobileOne S1 backbone with gaze head.
    """

    def __init__(self):
        super().__init__()
        self.backbone = MobileOneS1()
        self.head = GazeHead(
            in_channels=self.backbone.out_channels, num_bins=90
        )  # 90 matches training (Gaze360)

    def forward(self, x):
        features = self.backbone(x)
        pitch_logits, yaw_logits = self.head(features)
        return pitch_logits, yaw_logits
