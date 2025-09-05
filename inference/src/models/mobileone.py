import torch
import torch.nn as nn


class MobileOneBlock(nn.Module):
    """
    MobileOne building block - inference mode only.
    This is a simplified version that only supports the fused/reparameterized form.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        # MobileOne S1 doesn't use SE blocks, so this is always Identity
        self.se = nn.Identity()
        self.activation = nn.ReLU()

        # Only the fused convolution for inference
        self.reparam_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.se(self.reparam_conv(x)))


class MobileOneS1(nn.Module):
    """
    MobileOne S1 backbone for inference only.
    Matches the exact architecture from the original training code.
    """

    def __init__(self):
        super().__init__()

        # MobileOne S1 parameters (matches training code)
        num_blocks = [2, 8, 10, 1]
        width_multipliers = [1.5, 1.5, 2.0, 2.5]

        # Matches training
        self.in_planes = min(64, int(64 * width_multipliers[0]))  # 64

        # Build network stages (matches training)
        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,  # 64
            kernel_size=3,
            stride=2,
            padding=1,
            use_se=False,
        )

        # Stage 1: 64 channels, 2 blocks
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]),  # 96
            num_blocks[0],  # 2
            use_se=False,
        )

        # Stage 2: 128 * 1.5 = 192 channels, 8 blocks
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]),  # 192
            num_blocks[1],  # 8
            use_se=False,
        )

        # Stage 3: 256 * 2.0 = 512 channels, 10 blocks
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),  # 512
            num_blocks[2],  # 10
            use_se=False,
        )

        # Stage 4: 512 * 2.5 = 1280 channels, 1 block
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),  # 1280
            num_blocks[3],  # 1
            use_se=False,
        )

        self.out_channels = self.in_planes

    def _make_stage(
        self, planes: int, num_blocks: int, use_se: bool = False
    ) -> nn.Sequential:
        """Create a stage with depthwise + pointwise blocks - matches training"""
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []

        for ix, stride in enumerate(strides):
            # Depthwise convolution (groups = in_channels)
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,  # This makes it depthwise
                    use_se=use_se,
                )
            )

            # Pointwise convolution (1x1, groups=1)
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    use_se=use_se,
                )
            )

            # Update in_planes for next iteration
            self.in_planes = planes

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
