import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Union, cast
import logging
from src.utils.model_utils import load_filtered_state_dict

class SqueezeExcitationBlock(nn.Module):
    """ Squeeze and Excitation Block. """
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * rd_ratio), kernel_size=1, stride=1, bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio), out_channels=in_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x

class MobileOneBlock(nn.Module):
    """
    MobileOne building block.
    This block has a multi-branched architecture at training time,
    and can be re-parameterized to a single vanilla convolution at inference time.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        super().__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.dilation = dilation

        self.se = SqueezeExcitationBlock(out_channels) if use_se else nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv: Union[nn.Conv2d, None] = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.reparam_conv = None
            self.rbr_skip: Union[nn.BatchNorm2d, None] = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)
            self.rbr_scale: Union[nn.Sequential, None] = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            assert self.reparam_conv is not None
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = self.rbr_skip(x) if self.rbr_skip is not None else 0
        scale_out = self.rbr_scale(x) if self.rbr_scale is not None else 0

        conv_out: Union[int, torch.Tensor] = 0
        for branch in self.rbr_conv:
            conv_out = conv_out + branch(x)

        return self.activation(self.se(cast(torch.Tensor, conv_out) + scale_out + identity_out))


    def reparameterize(self):
        if self.inference_mode: return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size//2, dilation=self.dilation, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = kernel
        assert self.reparam_conv.bias is not None
        self.reparam_conv.bias.data = bias
        for para in self.parameters(): para.detach_()
        self.__delattr__('rbr_conv'); self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'): self.__delattr__('rbr_skip')
        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)

        if kernel_scale.dim() > 0:
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        first_branch_seq = cast(nn.Sequential, self.rbr_conv[0])
        first_conv = cast(nn.Conv2d, first_branch_seq[0])
        first_bn = cast(nn.BatchNorm2d, first_branch_seq[1])

        kernel_conv = torch.zeros_like(first_conv.weight)
        bias_conv = torch.zeros_like(first_bn.bias)

        for branch_module in self.rbr_conv:
            branch_seq_loop = cast(nn.Sequential, branch_module)
            _kernel, _bias = self._fuse_bn_tensor(branch_seq_loop)
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch: Union[nn.Sequential, nn.BatchNorm2d, None]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not branch: return torch.tensor(0.0), torch.tensor(0.0)

        if isinstance(branch, nn.Sequential):
            conv = cast(nn.Conv2d, branch[0])
            bn = cast(nn.BatchNorm2d, branch[1])
            kernel = conv.weight
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            bn = branch
        else:
             # Should be unreachable due to the initial check, but good for safety
             return torch.tensor(0.0), torch.tensor(0.0)

        assert bn.running_mean is not None and bn.running_var is not None
        running_mean, running_var = bn.running_mean, bn.running_var
        gamma, beta, eps = bn.weight, bn.bias, bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta - (running_mean * gamma / std)
        return fused_kernel, fused_bias

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

class MobileOneBackbone(nn.Module):
    PARAMS: Dict[str, Dict[str, Any]] = {
        "mobileone_s0": {"num_blocks": [2, 8, 10, 1], "width": [0.75, 1.0, 1.0, 2.0], "conv_branches": 4, "use_se": False},
        "mobileone_s1": {"num_blocks": [2, 8, 10, 1], "width": [1.5, 1.5, 2.0, 2.5], "conv_branches": 1, "use_se": False},
        "mobileone_s2": {"num_blocks": [2, 8, 10, 1], "width": [1.5, 2.0, 2.5, 4.0], "conv_branches": 1, "use_se": False},
        "mobileone_s3": {"num_blocks": [2, 8, 10, 1], "width": [2.0, 2.5, 3.0, 4.0], "conv_branches": 1, "use_se": False},
        "mobileone_s4": {"num_blocks": [2, 8, 10, 1], "width": [3.0, 3.5, 3.5, 4.0], "conv_branches": 1, "use_se": True},
    }
    PRETRAINED_URLS: Dict[str, str] = {
        "mobileone_s0": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0_unfused.pth.tar",
        "mobileone_s1": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s1_unfused.pth.tar",
        "mobileone_s2": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2_unfused.pth.tar",
        "mobileone_s3": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s3_unfused.pth.tar",
        "mobileone_s4": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s4_unfused.pth.tar",
    }

    def __init__(self, arch: str, pretrained: bool = False, inference_mode: bool = False):
        super().__init__()
        if arch not in self.PARAMS:
            raise ValueError(f"Unsupported MobileOne architecture: {arch}. Available: {list(self.PARAMS.keys())}")

        params = self.PARAMS[arch]
        self.in_planes = 0
        self._build_network(params, inference_mode)
        self.out_channels = self.in_planes

        if pretrained:
            self._load_pretrained(arch)

    def _build_network(self, params: Dict[str, Any], inference_mode: bool):
        num_blocks_per_stage: List[int] = params["num_blocks"]
        width_multipliers: List[float] = params["width"]
        num_conv_branches: int = params["conv_branches"]
        use_se: bool = params["use_se"]

        assert len(width_multipliers) == 4
        self.in_planes = min(64, int(64 * width_multipliers[0]))

        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, inference_mode=inference_mode, num_conv_branches=num_conv_branches)
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0, inference_mode=inference_mode, num_conv_branches=num_conv_branches)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0, inference_mode=inference_mode, num_conv_branches=num_conv_branches)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2], num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0, inference_mode=inference_mode, num_conv_branches=num_conv_branches)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3], num_se_blocks=num_blocks_per_stage[3] if use_se else 0, inference_mode=inference_mode, num_conv_branches=num_conv_branches)

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int, inference_mode: bool, num_conv_branches: int) -> nn.Sequential:
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = (ix >= (num_blocks - num_se_blocks))
            blocks.append(MobileOneBlock(in_channels=self.in_planes, out_channels=self.in_planes, kernel_size=3, stride=stride, padding=1, groups=self.in_planes, use_se=use_se, num_conv_branches=num_conv_branches, inference_mode=inference_mode))
            blocks.append(MobileOneBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=1, stride=1, padding=0, groups=1, use_se=use_se, num_conv_branches=num_conv_branches, inference_mode=inference_mode))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def _load_pretrained(self, arch: str):
        logger = logging.getLogger('new_gaze_logger')
        url = self.PRETRAINED_URLS.get(arch)
        if not url:
            logger.warning(f"No pretrained weights available for {arch}")
            return
        try:
            state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
            load_filtered_state_dict(self, state_dict)
        except Exception as e:
            logger.error(f"Failed to load pretrained weights for {arch} from {url}. Error: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
