b"""
Projection Heads
================
Implementation of projection heads from Section 3.3 of
"Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion"

Projection Heads upsample feature maps back to image space (256x256):
- Pc: fc -> xc (common projection)
- Pu: f1u/f2u -> x1u/x2u (unique projections)
- Pr: concat(fc, f1u, f2u) -> x (reconstruction)

Paper: "PROJECTION HEADS (upsample to 256x256):
        Pc: fc -> xc (common projection, ResNest layers)
        Pu: f1u/f2u -> x1u/x2u (unique projections)
        Pr: concat(fc, f1u, f2u) -> x (reconstruction)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .resnest import ResNeStBlock, BasicResBlock


class UpsampleBlock(nn.Module):
    """
    Upsampling block with optional residual processing.

    Performs 2x bilinear upsampling followed by conv layers.
    Used in projection heads to upsample from 32x32 to 256x256.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_resnest: bool = True,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: Input channel count
            out_channels: Output channel count
            use_resnest: Whether to use ResNeSt blocks (vs basic ResBlocks)
            norm_layer: Normalization layer type
        """
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Conv after upsampling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        # Residual block (ResNeSt for projection heads as per paper)
        if use_resnest:
            self.res_block = ResNeStBlock(
                out_channels, out_channels,
                stride=1,
                downsample=None,
                radix=2,
                cardinality=1,
                norm_layer=norm_layer
            )
        else:
            self.res_block = BasicResBlock(
                out_channels, out_channels,
                stride=1,
                downsample=None,
                norm_layer=norm_layer
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample and process.

        Args:
            x: Input features [B, C_in, H, W]

        Returns:
            Upsampled features [B, C_out, 2*H, 2*W]
        """
        x = self.upsample(x)    # [B, C_in, 2H, 2W]
        x = self.conv(x)        # [B, C_out, 2H, 2W]
        x = self.res_block(x)   # [B, C_out, 2H, 2W]
        return x


class ProjectionHead(nn.Module):
    """
    Base Projection Head

    Upsamples feature maps from H/8 x W/8 back to H x W (256x256).
    Uses progressive 2x upsampling with ResNeSt blocks.

    3 upsample stages: 32 -> 64 -> 128 -> 256

    Paper Section 3.3: "ResNest layers" used in projections
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 3,
        use_resnest: bool = True,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: Input feature channels (k)
            hidden_channels: Intermediate channel count
            out_channels: Output channels (3 for RGB)
            use_resnest: Whether to use ResNeSt blocks
            norm_layer: Normalization layer type
        """
        super().__init__()

        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=3, padding=1, bias=False),
            norm_layer(hidden_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Upsample 32 -> 64 (channels: 256 -> 256)
        self.up1 = UpsampleBlock(
            hidden_channels * 2, hidden_channels * 2,
            use_resnest=use_resnest, norm_layer=norm_layer
        )

        # Upsample 64 -> 128 (channels: 256 -> 128)
        self.up2 = UpsampleBlock(
            hidden_channels * 2, hidden_channels,
            use_resnest=use_resnest, norm_layer=norm_layer
        )

        # Upsample 128 -> 256 (channels: 128 -> 64)
        self.up3 = UpsampleBlock(
            hidden_channels, hidden_channels // 2,
            use_resnest=use_resnest, norm_layer=norm_layer
        )

        # Final output conv: 64 -> 3
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to image space.

        Args:
            x: Feature map [B, C_in, 32, 32]

        Returns:
            Projected image [B, 3, 256, 256]
        """
        x = self.input_conv(x)  # [B, 256, 32, 32]
        x = self.up1(x)         # [B, 256, 64, 64]
        x = self.up2(x)         # [B, 128, 128, 128]
        x = self.up3(x)         # [B, 64, 256, 256]
        x = self.output_conv(x) # [B, 3, 256, 256]
        return x


class ProjectionHeadC(ProjectionHead):
    """
    Common Feature Projection Head (Pc)

    Projects common features fc to image space.
    Pc: fc -> xc

    Output represents content common to both source images.
    During training, supervised by (M1 ∩ M2) ⊙ x (overlapping masked regions).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            use_resnest=True,
            norm_layer=norm_layer
        )


class ProjectionHeadU(ProjectionHead):
    """
    Unique Feature Projection Head (Pu)

    Projects unique features f1u/f2u to image space.
    Pu: f1u -> x1u, f2u -> x2u

    Same network used for both unique features (shared weights).
    During training:
    - x1u supervised by (M1 - M2) ⊙ x
    - x2u supervised by (M2 - M1) ⊙ x
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            use_resnest=True,
            norm_layer=norm_layer
        )


class ProjectionHeadR(nn.Module):
    """
    Reconstruction Projection Head (Pr)

    Projects concatenated features to reconstructed image.
    Pr: concat(fc, f1u, f2u) -> x

    Input is 3x the feature dimension (fc + f1u + f2u concatenated).
    During training, supervised by the original scene x.
    During inference, this produces the fused image.

    Paper: "Pr: concat(fc, f1u, f2u) -> x (reconstruction)"
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            feature_dim: Single feature dimension k
            hidden_channels: Intermediate channel count
            out_channels: Output channels (3 for RGB)
            norm_layer: Normalization layer type
        """
        super().__init__()

        # Input: concat(fc, f1u, f2u) = 3 * feature_dim channels
        in_channels = feature_dim * 3

        # Reduce from 3*k to 2*hidden
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 4, kernel_size=1, bias=False),
            norm_layer(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1, bias=False),
            norm_layer(hidden_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Upsample stages (same as other projection heads)
        self.up1 = UpsampleBlock(
            hidden_channels * 2, hidden_channels * 2,
            use_resnest=True, norm_layer=norm_layer
        )

        self.up2 = UpsampleBlock(
            hidden_channels * 2, hidden_channels,
            use_resnest=True, norm_layer=norm_layer
        )

        self.up3 = UpsampleBlock(
            hidden_channels, hidden_channels // 2,
            use_resnest=True, norm_layer=norm_layer
        )

        # Final output
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(
        self,
        fc: torch.Tensor,
        f1u: torch.Tensor,
        f2u: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct/fuse from decomposed features.

        Args:
            fc: Common features [B, k, H/8, W/8]
            f1u: Unique features from image 1 [B, k, H/8, W/8]
            f2u: Unique features from image 2 [B, k, H/8, W/8]

        Returns:
            Reconstructed/fused image [B, 3, H, W]
        """
        # Concatenate all features
        x = torch.cat([fc, f1u, f2u], dim=1)  # [B, 3k, H/8, W/8]

        # Process and upsample
        x = self.input_conv(x)  # [B, 256, 32, 32]
        x = self.up1(x)         # [B, 256, 64, 64]
        x = self.up2(x)         # [B, 128, 128, 128]
        x = self.up3(x)         # [B, 64, 256, 256]
        x = self.output_conv(x) # [B, 3, 256, 256]

        return x


class ProjectionHeadRSingle(ProjectionHead):
    """
    Alternative Reconstruction Projection Head

    Version that takes pre-concatenated features as a single tensor.
    Useful when features are already concatenated elsewhere.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__(
            in_channels=feature_dim * 3,  # 3x for concatenated features
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            use_resnest=True,
            norm_layer=norm_layer
        )

        # Override input conv to handle larger input
        self.input_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 3, hidden_channels * 4, kernel_size=1, bias=False),
            norm_layer(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1, bias=False),
            norm_layer(hidden_channels * 2),
            nn.ReLU(inplace=True)
        )
