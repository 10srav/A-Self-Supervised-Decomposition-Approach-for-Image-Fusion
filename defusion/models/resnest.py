"""
ResNeSt Split-Attention Blocks
==============================
Implementation of Split-Attention mechanism from ResNeSt paper
Used in DeFusion projection heads (Section 3.3)

Reference: "ResNeSt: Split-Attention Networks" (Zhang et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SplitAttention(nn.Module):
    """
    Split-Attention module for ResNeSt blocks.

    Splits channels into 'radix' groups and applies attention across groups.
    This enables the network to learn channel-wise attention patterns.

    Paper correspondence: Section 3.3 mentions ResNeSt layers in projections
    """

    def __init__(
        self,
        channels: int,
        radix: int = 2,
        cardinality: int = 1,
        reduction_factor: int = 4,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            channels: Number of input/output channels
            radix: Number of splits within a cardinal group
            cardinality: Number of cardinal groups
            reduction_factor: Channel reduction for attention FC layers
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels

        inter_channels = max(channels * radix // reduction_factor, 32)

        # Global average pooling + FC layers for attention
        # No normalization after GAP (1x1 spatial) - use ReLU directly
        self.fc1 = nn.Conv2d(channels, inter_channels, kernel_size=1, groups=cardinality)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, kernel_size=1, groups=cardinality)

        self.rsoftmax = RadixSoftmax(radix, cardinality)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C*radix, H, W]

        Returns:
            Attention-weighted output [B, C, H, W]
        """
        batch, rchannel = x.shape[:2]

        # Split and sum across radix groups
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x

        # Global average pooling
        gap = F.adaptive_avg_pool2d(gap, 1)

        # Attention computation (no norm after GAP)
        gap = self.fc1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        # Apply attention
        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x

        return out.contiguous()


class RadixSoftmax(nn.Module):
    """Radix-wise softmax for split-attention."""

    def __init__(self, radix: int, cardinality: int):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class ResNeStBlock(nn.Module):
    """
    ResNeSt Block with Split-Attention.

    This is the building block used in DeFusion's encoder and projection heads.
    Implements: Conv -> Norm -> ReLU -> SplitAttention -> Conv -> Norm + Skip

    Paper correspondence: "ResNeSt layers" mentioned in Section 3.3
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        radix: int = 2,
        cardinality: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        dropout_rate: float = 0.0
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            downsample: Downsampling layer for residual connection
            radix: Split-attention radix
            cardinality: Number of cardinal groups
            base_width: Base width for computing group width
            dilation: Convolution dilation
            norm_layer: Normalization layer type
            dropout_rate: Dropout probability
        """
        super().__init__()

        group_width = int(out_channels * (base_width / 64.0)) * cardinality

        # First conv block
        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)

        # Split-attention conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width * radix,
                kernel_size=3, stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality * radix, bias=False
            ),
            norm_layer(group_width * radix),
            nn.ReLU(inplace=True)
        )
        self.attention = SplitAttention(group_width, radix, cardinality)

        # Final conv block
        self.conv3 = nn.Conv2d(group_width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.attention(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicResBlock(nn.Module):
    """
    Basic Residual Block (without split-attention).

    Used in encoder where we need simpler residual connections.
    Implements: Conv -> Norm -> ReLU -> Conv -> Norm + Skip
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_layer(
    block: nn.Module,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
    norm_layer: nn.Module = nn.BatchNorm2d,
    **kwargs
) -> nn.Sequential:
    """
    Create a sequential layer of residual blocks.

    Args:
        block: Block class to use
        in_channels: Input channels for first block
        out_channels: Output channels for all blocks
        num_blocks: Number of blocks in the layer
        stride: Stride for first block (others use stride=1)
        norm_layer: Normalization layer type
        **kwargs: Additional arguments for block

    Returns:
        Sequential container of blocks
    """
    downsample = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            norm_layer(out_channels * block.expansion)
        )

    layers = [block(in_channels, out_channels, stride, downsample, norm_layer=norm_layer, **kwargs)]

    for _ in range(1, num_blocks):
        layers.append(block(out_channels * block.expansion, out_channels, norm_layer=norm_layer, **kwargs))

    return nn.Sequential(*layers)
