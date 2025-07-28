from .backbone_resnet import ResNetBackbone
from .backbone_mobileone import MobileOneBackbone
from .gaze_head import GazeHead
from .gaze_model import GazeModel

def build_model(config, **backbone_kwargs):
    """
    Builds the complete gaze model from backbone and head based on the config.
    Any additional keyword arguments are passed directly to the backbone constructor.
    """
    backbone_name = config['backbone']

    if backbone_name.startswith('resnet'):
        backbone = ResNetBackbone(
            arch=backbone_name,
            pretrained=config.get('pretrained', True),
            **backbone_kwargs
        )
    elif backbone_name.startswith('mobileone'):
        backbone = MobileOneBackbone(
            arch=backbone_name,
            pretrained=config.get('pretrained', True),
            **backbone_kwargs
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    head = GazeHead(
        in_channels=backbone.out_channels,
        num_bins=config['num_bins']
    )

    model = GazeModel(backbone, head)

    return model
