from .backbone_resnet import ResNetBackbone
from .backbone_mobileone import mobileone_s0, mobileone_s1, mobileone_s2, mobileone_s3, mobileone_s4
from .gaze_head import GazeHead
from .gaze_model import GazeModel

def build_model(config):
    """
    Builds the complete gaze model from backbone and head based on the config.
    """
    backbone_name = config['backbone']

    if backbone_name.startswith('resnet'):
        backbone = ResNetBackbone(
            arch=backbone_name,
            pretrained=config.get('pretrained', True),
            use_se=config.get('use_se', False)
        )
    elif backbone_name == 'mobileone_s0':
        backbone = mobileone_s0(pretrained=config.get('pretrained', True))
    elif backbone_name == 'mobileone_s1':
        backbone = mobileone_s1(pretrained=config.get('pretrained', True))
    elif backbone_name == 'mobileone_s2':
        backbone = mobileone_s2(pretrained=config.get('pretrained', True))
    elif backbone_name == 'mobileone_s3':
        backbone = mobileone_s3(pretrained=config.get('pretrained', True))
    elif backbone_name == 'mobileone_s4':
        backbone = mobileone_s4(pretrained=config.get('pretrained', True))
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    head = GazeHead(
        in_channels=backbone.out_channels,
        num_bins=config['num_bins']
    )

    model = GazeModel(backbone, head)

    return model
