import timm
import torch.nn as nn

from .backbone_lowformer import LowFormerBackbone
from .backbone_mobileone import MobileOneBackbone
from .backbone_resnet import ResNetBackbone
from .gaze_head import GazeHead
from .gaze_model import GazeModel


def build_model(config, **backbone_kwargs):
    """
    Builds the complete gaze model from backbone and head based on the config.
    Any additional keyword arguments are passed directly to the backbone constructor.
    """
    backbone_name = config["backbone"]

    if backbone_name.startswith("resnet"):
        backbone = ResNetBackbone(
            arch=backbone_name,
            pretrained=config.get("pretrained", True),
            **backbone_kwargs,
        )
    elif backbone_name.startswith("mobileone"):
        backbone = MobileOneBackbone(
            arch=backbone_name,
            pretrained=config.get("pretrained", True),
            **backbone_kwargs,
        )
    elif backbone_name.startswith("lowformer"):
        backbone = LowFormerBackbone(
            arch=backbone_name,
            pretrained=config.get("pretrained", True),
            **backbone_kwargs,
        )
    else:
        try:
            timm_model = timm.create_model(
                backbone_name,
                pretrained=config.get("pretrained", True),
                features_only=True,
                **backbone_kwargs,
            )

            class TimmBackbone(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.out_channels = model.feature_info.channels()[-1]

                def forward(self, x):
                    # Return only the last feature map
                    return self.model(x)[-1]

            backbone = TimmBackbone(timm_model)
        except Exception as e:
            raise ValueError(f"Unknown backbone or TIMM error: {backbone_name}") from e

    head = GazeHead(in_channels=backbone.out_channels, num_bins=config["num_bins"])

    model = GazeModel(backbone, head)

    return model
