import torch.nn as nn


class GazeHead(nn.Module):
    """
    GazeHead for binned gaze estimation.
    Takes backbone features and predicts binned pitch and yaw angles.

    For Gaze360: 90 bins covering -180 to +180 degrees range.
    """

    def __init__(self, in_channels: int, num_bins: int = 90, dropout_rate: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)

        # Separate linear layers for pitch and yaw classification
        self.fc_pitch = nn.Linear(in_channels, num_bins)
        self.fc_yaw = nn.Linear(in_channels, num_bins)

    def forward(self, x):
        x = self.pool(x)  # [B, C, 1, 1]
        x = self.flatten(x)  # [B, C]
        x = self.dropout(x)

        # Get binned predictions
        pitch_logits = self.fc_pitch(x)  # [B, num_bins]
        yaw_logits = self.fc_yaw(x)  # [B, num_bins]

        return pitch_logits, yaw_logits
