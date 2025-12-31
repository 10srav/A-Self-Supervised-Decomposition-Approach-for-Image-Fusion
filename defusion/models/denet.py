"""
DeNet: Decomposition Network
============================
Implementation of the core DeNet architecture from Section 3.3 of
"Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion"

DeNet = Encoder(E) + Ensembler(Ec) + Decoder(D = Du + Dc)

Architecture:
- Encoder (E): 3 MaxPool layers with residual blocks, outputs Ex1, Ex2 at H/8 x W/8 x k
- Ensembler (Ec): 1 Residual layer to combine features
- DecoderU (Du): Generates unique features f1u, f2u from concat(Ex, fc)
- DecoderC (Dc): Generates common features fc from ensembled features

Paper Section 3.3:
"The encoder is composed by 3 residual layers with maxpool function,
each layer contains several residual blocks (ResNest [23] layers),
the number of layers is set to {2,2,2}."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .resnest import BasicResBlock, ResNeStBlock, make_layer


class Encoder(nn.Module):
    """
    Encoder Network (E)

    Extracts features from input images with 3 stages of downsampling.
    Each stage: ResBlocks -> MaxPool(stride=2)

    Input: [B, 3, 256, 256]
    Output: [B, k, 32, 32] where k=256 (feature dimension)

    Paper: "The encoder is composed by 3 residual layers with maxpool function"
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        feature_dim: int = 256,
        num_blocks: Tuple[int, int, int] = (2, 2, 2),
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base channel dimension (doubles each stage)
            feature_dim: Output feature dimension k (default 256)
            num_blocks: Number of residual blocks per stage
            norm_layer: Normalization layer type
        """
        super().__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Initial convolution: 3 -> base_channels (64)
        # Paper doesn't specify, using 7x7 conv similar to ResNet
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(base_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 64 -> 64 channels, 256x256 -> 128x128
        self.layer1 = make_layer(
            BasicResBlock, base_channels, base_channels,
            num_blocks[0], stride=1, norm_layer=norm_layer
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: 64 -> 128 channels, 128x128 -> 64x64
        self.layer2 = make_layer(
            BasicResBlock, base_channels, base_channels * 2,
            num_blocks[1], stride=1, norm_layer=norm_layer
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: 128 -> 256 channels, 64x64 -> 32x32
        self.layer3 = make_layer(
            BasicResBlock, base_channels * 2, feature_dim,
            num_blocks[2], stride=1, norm_layer=norm_layer
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input image [B, 3, H, W] (H=W=256)

        Returns:
            features: Encoded features [B, k, H/8, W/8] (k=256, size 32x32)
        """
        # Initial conv
        x = self.conv1(x)  # [B, 64, 256, 256]

        # Stage 1
        x = self.layer1(x)  # [B, 64, 256, 256]
        x = self.pool1(x)   # [B, 64, 128, 128]

        # Stage 2
        x = self.layer2(x)  # [B, 128, 128, 128]
        x = self.pool2(x)   # [B, 128, 64, 64]

        # Stage 3
        x = self.layer3(x)  # [B, 256, 64, 64]
        x = self.pool3(x)   # [B, 256, 32, 32]

        return x


class Ensembler(nn.Module):
    """
    Ensembler Network (Ec)

    Combines features from two encoded images into common representation.
    Uses a single residual layer as per paper specification.

    Paper Section 3.3: "Ensembler (Ec) [Ex1+Ex2 -> fc_compressed]: 1 Residual layer only"

    Input: Concatenated Ex1, Ex2 [B, 2*k, H/8, W/8]
    Output: Common features [B, k, H/8, W/8]
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: Input channels (2 * feature_dim for concatenated features)
            out_channels: Output channels (feature_dim)
            norm_layer: Normalization layer type
        """
        super().__init__()

        # Channel reduction conv
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        # Single residual layer for ensembling
        self.res_block = BasicResBlock(
            out_channels, out_channels,
            stride=1, downsample=None, norm_layer=norm_layer
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - combine two feature maps.

        Args:
            x1: Features from image 1 [B, k, H/8, W/8]
            x2: Features from image 2 [B, k, H/8, W/8]

        Returns:
            fc: Common/ensembled features [B, k, H/8, W/8]
        """
        # Element-wise addition first (as per paper: Ex1 + Ex2)
        combined = x1 + x2  # [B, k, H/8, W/8]

        # Process through residual block
        fc = self.res_block(combined)

        return fc


class DecoderC(nn.Module):
    """
    Common Feature Decoder (Dc)

    Decodes the ensembled features to produce common features fc.
    This represents content shared between both source images.

    Paper: "Dc: Ec(Ex1+Ex2) -> fc"

    Input: Ensembled features [B, k, H/8, W/8]
    Output: Common features fc [B, k, H/8, W/8]
    """

    def __init__(
        self,
        channels: int = 256,
        num_blocks: int = 2,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            channels: Feature channel dimension
            num_blocks: Number of residual blocks
            norm_layer: Normalization layer type
        """
        super().__init__()

        # Stack of residual blocks for decoding common features
        self.decoder = make_layer(
            BasicResBlock, channels, channels,
            num_blocks, stride=1, norm_layer=norm_layer
        )

    def forward(self, fc_ensemble: torch.Tensor) -> torch.Tensor:
        """
        Decode ensembled features to common features.

        Args:
            fc_ensemble: Ensembled features from Ec [B, k, H/8, W/8]

        Returns:
            fc: Common features [B, k, H/8, W/8]
        """
        fc = self.decoder(fc_ensemble)
        return fc


class DecoderU(nn.Module):
    """
    Unique Feature Decoder (Du)

    Decodes concatenation of encoder features and common features
    to produce unique features for each source image.

    Paper:
    - Du1: concat(Ex1, Ec(Ex1+Ex2)) -> f1u
    - Du2: concat(Ex2, Ec(Ex1+Ex2)) -> f2u

    Input: concat(Ex, fc) [B, 2*k, H/8, W/8]
    Output: Unique features fu [B, k, H/8, W/8]
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        num_blocks: int = 2,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: Input channels (2 * feature_dim for concatenated features)
            out_channels: Output channels (feature_dim)
            num_blocks: Number of residual blocks
            norm_layer: Normalization layer type
        """
        super().__init__()

        # Channel reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder residual blocks
        self.decoder = make_layer(
            BasicResBlock, out_channels, out_channels,
            num_blocks, stride=1, norm_layer=norm_layer
        )

    def forward(self, ex: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        Decode unique features from encoder features and common features.

        Args:
            ex: Encoder features for one image [B, k, H/8, W/8]
            fc: Common features [B, k, H/8, W/8]

        Returns:
            fu: Unique features [B, k, H/8, W/8]
        """
        # Concatenate encoder features and common features
        combined = torch.cat([ex, fc], dim=1)  # [B, 2k, H/8, W/8]

        # Reduce channels
        x = self.reduce(combined)  # [B, k, H/8, W/8]

        # Decode through residual blocks
        fu = self.decoder(x)  # [B, k, H/8, W/8]

        return fu


class DeNet(nn.Module):
    """
    Complete Decomposition Network

    Combines Encoder, Ensembler, and Decoders into a single module.
    Decomposes two source images into common (fc) and unique (f1u, f2u) features.

    Paper Section 3.3: "DeNet = Encoder(E) + Ensembler(Ec) + Decoder(D = Du + Dc)"

    Input: Two images x1, x2 [B, 3, H, W]
    Output: fc, f1u, f2u [B, k, H/8, W/8] each
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        base_channels: int = 64,
        encoder_blocks: Tuple[int, int, int] = (2, 2, 2),
        decoder_blocks: int = 2,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: Input image channels
            feature_dim: Feature dimension k
            base_channels: Base channels for encoder
            encoder_blocks: Number of blocks per encoder stage
            decoder_blocks: Number of blocks in decoders
            norm_layer: Normalization layer type
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Shared encoder for both images
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
            num_blocks=encoder_blocks,
            norm_layer=norm_layer
        )

        # Ensembler to combine features
        self.ensembler = Ensembler(
            in_channels=feature_dim,  # Takes summed features
            out_channels=feature_dim,
            norm_layer=norm_layer
        )

        # Common decoder
        self.decoder_c = DecoderC(
            channels=feature_dim,
            num_blocks=decoder_blocks,
            norm_layer=norm_layer
        )

        # Unique decoders (shared weights for both sources)
        self.decoder_u = DecoderU(
            in_channels=feature_dim * 2,
            out_channels=feature_dim,
            num_blocks=decoder_blocks,
            norm_layer=norm_layer
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - decompose two images into common and unique features.

        Args:
            x1: First source image [B, 3, H, W]
            x2: Second source image [B, 3, H, W]

        Returns:
            fc: Common features [B, k, H/8, W/8]
            f1u: Unique features for x1 [B, k, H/8, W/8]
            f2u: Unique features for x2 [B, k, H/8, W/8]
        """
        # Encode both images (shared encoder)
        ex1 = self.encoder(x1)  # [B, k, H/8, W/8]
        ex2 = self.encoder(x2)  # [B, k, H/8, W/8]

        # Ensemble features (Ex1 + Ex2 -> fc)
        fc_ensemble = self.ensembler(ex1, ex2)  # [B, k, H/8, W/8]

        # Decode common features
        fc = self.decoder_c(fc_ensemble)  # [B, k, H/8, W/8]

        # Decode unique features
        # Du1: concat(Ex1, fc) -> f1u
        f1u = self.decoder_u(ex1, fc)  # [B, k, H/8, W/8]
        # Du2: concat(Ex2, fc) -> f2u
        f2u = self.decoder_u(ex2, fc)  # [B, k, H/8, W/8]

        return fc, f1u, f2u

    def get_encoder_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get raw encoder features (useful for analysis).

        Args:
            x1: First source image [B, 3, H, W]
            x2: Second source image [B, 3, H, W]

        Returns:
            ex1: Encoder features for x1 [B, k, H/8, W/8]
            ex2: Encoder features for x2 [B, k, H/8, W/8]
        """
        ex1 = self.encoder(x1)
        ex2 = self.encoder(x2)
        return ex1, ex2
