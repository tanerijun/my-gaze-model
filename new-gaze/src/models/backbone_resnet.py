import torch.nn as nn
import torchvision.models as models

# ResNetBackbone: wraps a torchvision ResNet (18/34/50) for feature extraction.
class ResNetBackbone(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True, use_se=False):
        """
        Args:
            arch (str): Which ResNet variant to use ('resnet18', 'resnet34', 'resnet50').
            pretrained (bool): If True, loads ImageNet weights.
            use_se (bool): If True, adds a SE block after the backbone.
        """
        super().__init__()
        assert arch in ['resnet18', 'resnet34', 'resnet50'], 'Unsupported ResNet arch'

        weights = None
        if pretrained:
            if arch == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT
            elif arch == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT
            else: # resnet50
                weights = models.ResNet50_Weights.DEFAULT

        # Load the chosen ResNet model with the specified weights
        if arch == 'resnet18':
            self.backbone = models.resnet18(weights=weights)
        elif arch == 'resnet34':
            self.backbone = models.resnet34(weights=weights)
        else:
            self.backbone = models.resnet50(weights=weights)

        # Remove the final pooling and fc layers to use as a feature extractor
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Output channels depend on ResNet depth
        self.out_channels = 512 if arch in ['resnet18', 'resnet34'] else 2048

        self.use_se = use_se
        if use_se:
            self.se = SEModule(self.out_channels)
        else:
            self.se = None

    def forward(self, x):
        x = self.backbone(x)  # [B, C, H, W]
        if self.se:
            x = self.se(x)
        return x

# SEModule: Squeeze-and-Excitation block for channel-wise feature recalibration.
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for bottleneck.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
