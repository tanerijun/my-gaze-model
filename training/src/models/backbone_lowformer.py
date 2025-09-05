import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from functools import partial
from typing import List, Tuple, Optional, Union, Dict, Any

from src.utils.model_utils import load_filtered_state_dict

def get_same_padding(kernel_size: Union[int, Tuple[int, int]]) -> Union[int, Tuple[int, int]]:
    """
    Calculates the padding value for 'same' padding for nn.Conv2d.
    """
    if isinstance(kernel_size, int):
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size {kernel_size} must be odd for 'same' padding.")
        return kernel_size // 2

    # The type hint Union[int, Tuple[int, int]] makes the length check unnecessary.
    k_h, k_w = kernel_size
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError(f"All kernel sizes in tuple {kernel_size} must be odd for 'same' padding.")

    return (k_h // 2, k_w // 2)

def val2tuple(x: Any, min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    x = val2list(x)
    if len(x) > 0 and len(x) < min_len:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)

def val2list(x: Any, repeat_time: int = 1) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def build_norm(name: str, num_features: int, **kwargs: Any) -> Optional[nn.Module]:
    if name == "bn2d":
        return nn.BatchNorm2d(num_features=num_features, **kwargs)
    elif name == "ln":
        return nn.LayerNorm(normalized_shape=num_features, **kwargs)
    return None

def build_act(name: str, **kwargs: Any) -> Optional[nn.Module]:
    if name == "relu": return nn.ReLU(**kwargs)
    if name == "relu6": return nn.ReLU6(**kwargs)
    if name == "hswish": return nn.Hardswish(**kwargs)
    if name == "silu": return nn.SiLU(**kwargs)
    if name == "gelu": return partial(nn.GELU, approximate="tanh")(**kwargs)
    return None

# --- Core Modules ---

class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x

class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class ResidualBlock(nn.Module):
    def __init__(self, main: Optional[nn.Module], shortcut: Optional[nn.Module], post_act: Optional[str] = None, pre_norm: Optional[nn.Module] = None):
        super(ResidualBlock, self).__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act) if post_act else None

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm and self.main: return self.main(self.pre_norm(x))
        if self.main: return self.main(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act: res = self.post_act(res)
        return res

class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1, groups: int = 1, use_bias: bool = False, dropout: float = 0, norm: str = "bn2d", act_func: Optional[str] = "relu"):
        super(ConvLayer, self).__init__()

        padding_val = get_same_padding(kernel_size)

        # The type hint `dilation: int` makes this check unreachable.
        padding: Union[int, Tuple[int, int]]
        if isinstance(padding_val, int):
            padding = padding_val * dilation
        else:
            padding = (padding_val[0] * dilation, padding_val[1] * dilation)

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, use_bias)
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func) if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout: x = self.dropout(x)
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act: x = self.act(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, expand_ratio: float = 6, grouping: int = 1, use_bias: Union[bool, Tuple[bool, ...]] = False, norm: Union[str, Tuple[Optional[str], ...]] = ("bn2d", "bn2d", "bn2d"), act_func: Union[str, Tuple[Optional[str], ...]] = ("relu6", "relu6", None)):
        super(MBConv, self).__init__()
        mid_channels = round(in_channels * expand_ratio)
        use_bias_tuple, norm_tuple, act_func_tuple = val2tuple(use_bias, 3), val2tuple(norm, 3), val2tuple(act_func, 3)

        self.inverted_conv = ConvLayer(in_channels, mid_channels, 1, 1, groups=grouping, norm=norm_tuple[0], act_func=act_func_tuple[0], use_bias=use_bias_tuple[0])
        self.depth_conv = ConvLayer(mid_channels, mid_channels, kernel_size, stride, groups=mid_channels, norm=norm_tuple[1], act_func=act_func_tuple[1], use_bias=use_bias_tuple[1])
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, groups=grouping, norm=norm_tuple[2], act_func=act_func_tuple[2], use_bias=use_bias_tuple[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(self.inverted_conv(x)))

class FusedMBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, expand_ratio: float = 6, groups: int = 1, use_bias: Union[bool, Tuple[bool, ...]] = False, norm: Union[str, Tuple[Optional[str], ...]] = ("bn2d", "bn2d"), act_func: Union[str, Tuple[Optional[str], ...]] = ("relu6", None), fusedgroup: bool = False):
        super().__init__()
        mid_channels = round(in_channels * expand_ratio)
        use_bias_tuple, norm_tuple, act_func_tuple = val2tuple(use_bias, 2), val2tuple(norm, 2), val2tuple(act_func, 2)
        self.spatial_conv = ConvLayer(in_channels, mid_channels, kernel_size, stride, groups=2 if fusedgroup and groups == 1 else groups, use_bias=use_bias_tuple[0], norm=norm_tuple[0], act_func=act_func_tuple[0])
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, use_bias=use_bias_tuple[1], norm=norm_tuple[1], act_func=act_func_tuple[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.spatial_conv(x))

class SDALayer(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d_k = q.size(-1)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        return torch.matmul(attention, v)

class ConvAttention(nn.Module):
    def __init__(self, input_dim: int, head_dim_mul: float = 1.0, att_stride: int = 4, att_kernel: int = 7, fuseconv: bool = False):
        super().__init__()
        self.num_heads = max(1, int((input_dim * head_dim_mul) // 30))
        self.head_dim = int((input_dim // self.num_heads) * head_dim_mul)
        self.o_proj_inpdim = self.head_dim * self.num_heads
        total_dim = self.o_proj_inpdim * 3

        self.conv_proj = nn.Sequential(nn.Conv2d(input_dim, input_dim, att_kernel, att_stride, att_kernel // 2, groups=input_dim, bias=False), nn.BatchNorm2d(input_dim))
        self.pwise = nn.Conv2d(input_dim, total_dim, 1, 1, 0, bias=False)
        self.sda = SDALayer()

        if fuseconv:
            self.o_proj = nn.Identity()
            self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2, groups=1)
        else:
            self.o_proj = nn.Conv2d(self.o_proj_inpdim, input_dim, 1, 1, 0)
            self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=att_stride * 2, stride=att_stride, padding=att_stride // 2, groups=input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        xout = self.pwise(self.conv_proj(x))
        _, _, h, w = xout.size()
        qkv = xout.reshape(N, self.num_heads, 3 * self.head_dim, h * w).permute(0, 1, 3, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        values = self.sda(q, k, v)
        values_reshaped = values.permute(0, 1, 3, 2).reshape(N, self.o_proj_inpdim, h, w)
        o = self.o_proj(values_reshaped)
        o = self.upsampling(o)
        return o[:, :, :H, :W]

class LowFormerBlock(nn.Module):
    def __init__(self, in_channels: int, expand_ratio: float = 4, norm: str = "bn2d", act_func: str = "hswish", fuseconv: bool = False, just_unfused: bool = False, mlpremoved: bool = False, noattention: bool = False, att_stride: int = 1, mlpexpans: int = 4, stage_num: int = -1, **kwargs: Any):
        super().__init__()
        att_kernel = 5 if att_stride > 1 else 3
        block = ConvAttention(input_dim=in_channels, att_stride=att_stride, att_kernel=att_kernel, head_dim_mul=0.5, fuseconv=fuseconv and not just_unfused)

        if noattention:
            context_module = nn.Identity()
        elif not mlpremoved:
            context_module = ResidualBlock(nn.Sequential(nn.GroupNorm(1, in_channels), block), IdentityLayer())
            mlp = nn.Sequential(nn.GroupNorm(1, in_channels), nn.Conv2d(in_channels, in_channels * mlpexpans, 1), nn.GELU(), nn.Conv2d(in_channels * mlpexpans, in_channels, 1))
            context_module = nn.Sequential(context_module, ResidualBlock(mlp, IdentityLayer()))
        else:
            context_module = ResidualBlock(block, IdentityLayer())

        local_module = FusedMBConv(in_channels, in_channels, expand_ratio=expand_ratio, use_bias=(True, False), norm=norm, act_func=(act_func, None)) if (fuseconv and in_channels < 256) and not just_unfused else MBConv(in_channels, in_channels, expand_ratio=expand_ratio, use_bias=(True, True, False), norm=(None, None, norm), act_func=(act_func, act_func, None))
        self.total = nn.Sequential(context_module, ResidualBlock(local_module, IdentityLayer()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.total(x)

# --- Main LowFormerBackbone Class for Integration ---

class LowFormerBackbone(nn.Module):
    PARAMS: Dict[str, Dict[str, List[int]]] = {
        "lowformer_b0": {"width": [16, 32, 64, 128, 256], "depth": [0, 1, 1, 3, 4]},
        "lowformer_b1": {"width": [16, 32, 64, 128, 256], "depth": [0, 1, 1, 5, 5]},
        "lowformer_b15": {"width": [20, 40, 80, 160, 320], "depth": [0, 1, 1, 6, 6]},
        "lowformer_b2": {"width": [24, 48, 96, 192, 384], "depth": [0, 0, 0, 6, 6]},
        "lowformer_b3": {"width": [32, 64, 128, 256, 512], "depth": [1, 2, 3, 6, 6]},
    }
    PRETRAINED_PATHS: Dict[str, str] = {f"lowformer_b{v}": f"src/models/pretrained/lowformer/b{v}.pt" for v in ['0', '1', '15', '2', '3']}

    def __init__(self, arch: str, pretrained: bool = True, **kwargs: Any):
        super().__init__()
        if arch not in self.PARAMS:
            raise ValueError(f"Unsupported LowFormer architecture: {arch}. Available: {list(self.PARAMS.keys())}")

        params = self.PARAMS[arch]
        self.out_channels = params["width"][-1]
        self._build_network(width_list=params["width"], depth_list=params["depth"], **kwargs)

        if pretrained:
            self._load_pretrained(arch)

    def _build_network(self, width_list: List[int], depth_list: List[int], in_channels: int = 3, norm: str = "bn2d", act_func: str = "hswish", expand_ratio: float = 4, **kwargs: Any):
        stem_ops: List[Optional[nn.Module]] = [ConvLayer(in_channels, width_list[0], stride=2, norm=norm, act_func=act_func)]
        for _ in range(depth_list[0]):
            block = MBConv(width_list[0], width_list[0], 1, expand_ratio=2, use_bias=(True, True, False), norm=(None, None, norm), act_func=(act_func, act_func, None))
            stem_ops.append(ResidualBlock(block, IdentityLayer()))
        self.input_stem = OpSequential(stem_ops)

        current_channels = width_list[0]
        self.stages = nn.ModuleList()
        stage_num = 1

        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage_blocks: List[Optional[nn.Module]] = []
            for i in range(d):
                in_c = current_channels if i == 0 else w
                block = MBConv(in_c, w, stride=(2 if i == 0 else 1), expand_ratio=expand_ratio, use_bias=(True, True, False), norm=(None, None, norm), act_func=(act_func, act_func, None))
                stage_blocks.append(ResidualBlock(block, IdentityLayer() if i > 0 else None))
            self.stages.append(OpSequential(stage_blocks))
            current_channels = w
            stage_num += 1

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage_blocks = []
            block = MBConv(current_channels, w, stride=2, expand_ratio=expand_ratio, use_bias=(True, True, False), norm=(None, None, norm), act_func=(act_func, act_func, None))
            stage_blocks.append(ResidualBlock(block, None))
            for _ in range(d):
                stage_blocks.append(LowFormerBlock(in_channels=w, expand_ratio=expand_ratio, norm=norm, act_func=act_func, att_stride=(2 if stage_num == 3 else 1), stage_num=stage_num))
            self.stages.append(OpSequential(stage_blocks))
            current_channels = w
            stage_num += 1

    def _load_pretrained(self, arch: str):
        logger = logging.getLogger('new_gaze_logger')
        path = self.PRETRAINED_PATHS.get(arch)
        if not path:
            logger.warning(f"No pretrained weights path found for {arch}")
            return
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            processed_state_dict = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
            load_filtered_state_dict(self, processed_state_dict)
        except Exception as e:
            logger.error(f"Failed to load pretrained weights for {arch} from {path}. Error: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
