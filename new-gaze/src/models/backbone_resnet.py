import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True, use_se=False):
        """
        Args:
            arch (str): Which ResNet variant to use ('resnet18', 'resnet34', 'resnet50').
            pretrained (bool): If True, loads ImageNet weights.
            use_se (bool): If True, adds a SE block after the backbone. (TO BE IMPLEMENTED)
        """
        super().__init__()
        assert arch in ['resnet18', 'resnet34', 'resnet50'], 'Unsupported ResNet arch'

        weights = None
        if pretrained:
            if arch == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT
            elif arch == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT
            else:
                weights = models.ResNet50_Weights.DEFAULT

        if arch == 'resnet18':
            self.net = models.resnet18(weights=weights)
        elif arch == 'resnet34':
            self.net = models.resnet34(weights=weights)
        else:
            self.net = models.resnet50(weights=weights)

        self.out_channels = 512 if arch in ['resnet18', 'resnet34'] else 2048

    def forward(self, x):
        # Custom forward pass that uses the parts of the original model
        # This stops before the avgpool and fc layers, returning the feature map.
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        return x
