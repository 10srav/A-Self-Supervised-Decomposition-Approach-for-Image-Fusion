"""
DeFusion: Complete Model
========================
Implementation of the complete DeFusion model from
"Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion"

This module integrates all components:
- DeNet for feature decomposition
- Projection heads for reconstruction
- Training and inference forward passes

Paper Section 3:
"Given scene x ∈ R^{H×W×3}, generate 2 masks M1, M2 s.t. M1 ∩ M2 = ∅
DeNet(x1,x2) → fc, f1u, f2u
Fused = Pr(concat(fc, f1u, f2u))"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .denet import DeNet, Encoder, Ensembler, DecoderC, DecoderU
from .projection_heads import ProjectionHeadC, ProjectionHeadU, ProjectionHeadR


class DeFusion(nn.Module):
    """
    DeFusion: Self-Supervised Decomposition for Image Fusion

    Complete model implementing the CUD (Common and Unique Decomposition)
    pretext task for self-supervised image fusion.

    Architecture:
        DeNet: Encoder(E) + Ensembler(Ec) + Decoder(Du, Dc)
        Projections: Pc (common), Pu (unique), Pr (reconstruction)

    Training:
        Input: Two masked/noisy versions of same scene
        Output: Decomposed features + reconstructed projections
        Loss: MAE on projections vs masked ground truth

    Inference:
        Input: Two source images (IR+Vis, multi-focus, multi-exposure)
        Output: Single fused image

    Paper: Section 3.2-3.3
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        base_channels: int = 64,
        encoder_blocks: Tuple[int, int, int] = (2, 2, 2),
        decoder_blocks: int = 2,
        proj_hidden_channels: int = 128,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB)
            feature_dim: Feature dimension k (default 256 per paper)
            base_channels: Base channels for encoder (64)
            encoder_blocks: Number of blocks per encoder stage (2,2,2)
            decoder_blocks: Number of blocks in decoders (2)
            proj_hidden_channels: Hidden channels in projection heads
            norm_layer: Normalization layer type
        """
        super().__init__()

        self.feature_dim = feature_dim

        # DeNet: Decomposition Network
        self.denet = DeNet(
            in_channels=in_channels,
            feature_dim=feature_dim,
            base_channels=base_channels,
            encoder_blocks=encoder_blocks,
            decoder_blocks=decoder_blocks,
            norm_layer=norm_layer
        )

        # Projection Heads
        # Pc: fc -> xc (common features to image)
        self.proj_c = ProjectionHeadC(
            feature_dim=feature_dim,
            hidden_channels=proj_hidden_channels,
            out_channels=in_channels,
            norm_layer=norm_layer
        )

        # Pu: fu -> xu (unique features to image, shared for f1u and f2u)
        self.proj_u = ProjectionHeadU(
            feature_dim=feature_dim,
            hidden_channels=proj_hidden_channels,
            out_channels=in_channels,
            norm_layer=norm_layer
        )

        # Pr: concat(fc, f1u, f2u) -> x (reconstruction/fusion)
        self.proj_r = ProjectionHeadR(
            feature_dim=feature_dim,
            hidden_channels=proj_hidden_channels,
            out_channels=in_channels,
            norm_layer=norm_layer
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard forward pass for inference/fusion.

        Args:
            x1: First source image [B, 3, H, W]
            x2: Second source image [B, 3, H, W]

        Returns:
            fused: Fused output image [B, 3, H, W]
            fc: Common features [B, k, H/8, W/8]
            f1u: Unique features for x1 [B, k, H/8, W/8]
            f2u: Unique features for x2 [B, k, H/8, W/8]
        """
        # Decompose into common and unique features
        fc, f1u, f2u = self.denet(x1, x2)

        # Reconstruct/fuse using projection head
        fused = self.proj_r(fc, f1u, f2u)

        return fused, fc, f1u, f2u

    def forward_train(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with all projections.

        Used during CUD pretext training. Returns all intermediate
        projections for computing the full loss.

        Args:
            x1: First augmented image (masked + noise) [B, 3, H, W]
            x2: Second augmented image (masked + noise) [B, 3, H, W]

        Returns:
            Dictionary containing:
                - fc: Common features [B, k, H/8, W/8]
                - f1u: Unique features for x1 [B, k, H/8, W/8]
                - f2u: Unique features for x2 [B, k, H/8, W/8]
                - xc: Common projection [B, 3, H, W]
                - x1u: Unique projection for x1 [B, 3, H, W]
                - x2u: Unique projection for x2 [B, 3, H, W]
                - xr: Reconstruction [B, 3, H, W]
        """
        # Decompose
        fc, f1u, f2u = self.denet(x1, x2)

        # All projections
        xc = self.proj_c(fc)        # Common projection
        x1u = self.proj_u(f1u)      # Unique projection for image 1
        x2u = self.proj_u(f2u)      # Unique projection for image 2
        xr = self.proj_r(fc, f1u, f2u)  # Full reconstruction

        return {
            'fc': fc,
            'f1u': f1u,
            'f2u': f2u,
            'xc': xc,
            'x1u': x1u,
            'x2u': x2u,
            'xr': xr
        }

    def forward_fusion(
        self,
        i1: torch.Tensor,
        i2: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference fusion forward pass.

        Takes two source images and returns the fused result.
        This is the primary method for applying the trained model
        to actual fusion tasks.

        Paper Section 3.2 (Inference Pipeline, Fig 3):
        "Given source images I1, I2:
         DeNet(I1,I2) → fc, f1u, f2u
         fused = Pr(concat(fc, f1u, f2u))"

        Args:
            i1: First source image [B, 3, H, W]
            i2: Second source image [B, 3, H, W]

        Returns:
            fused: Fused output image [B, 3, H, W]
        """
        # Decompose source images
        fc, f1u, f2u = self.denet(i1, i2)

        # Fuse via reconstruction projection
        fused = self.proj_r(fc, f1u, f2u)

        return fused

    def get_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get decomposed features without projection.

        Useful for feature visualization and analysis.

        Args:
            x1: First image [B, 3, H, W]
            x2: Second image [B, 3, H, W]

        Returns:
            fc: Common features [B, k, H/8, W/8]
            f1u: Unique features for x1 [B, k, H/8, W/8]
            f2u: Unique features for x2 [B, k, H/8, W/8]
        """
        return self.denet(x1, x2)

    def get_encoder_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get raw encoder features before decomposition.

        Args:
            x1: First image [B, 3, H, W]
            x2: Second image [B, 3, H, W]

        Returns:
            ex1: Encoder features for x1 [B, k, H/8, W/8]
            ex2: Encoder features for x2 [B, k, H/8, W/8]
        """
        return self.denet.get_encoder_features(x1, x2)

    @torch.no_grad()
    def fuse_images(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Convenience method for fusing images with optional normalization.

        Args:
            image1: First source image [B, 3, H, W] or [3, H, W]
            image2: Second source image [B, 3, H, W] or [3, H, W]
            normalize: Whether to normalize output to [0, 1]

        Returns:
            Fused image [B, 3, H, W] or [3, H, W]
        """
        # Handle unbatched inputs
        squeeze_output = False
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)
            squeeze_output = True

        # Fuse
        fused = self.forward_fusion(image1, image2)

        # Normalize from [-1, 1] (Tanh output) to [0, 1]
        if normalize:
            fused = (fused + 1) / 2
            fused = fused.clamp(0, 1)

        # Remove batch dimension if input was unbatched
        if squeeze_output:
            fused = fused.squeeze(0)

        return fused


def build_defusion(
    config: Optional[Dict] = None,
    pretrained: Optional[str] = None
) -> DeFusion:
    """
    Build DeFusion model from config or with defaults.

    Args:
        config: Optional configuration dictionary with model parameters
        pretrained: Optional path to pretrained weights

    Returns:
        Initialized DeFusion model
    """
    # Default configuration
    default_config = {
        'in_channels': 3,
        'feature_dim': 256,
        'base_channels': 64,
        'encoder_blocks': (2, 2, 2),
        'decoder_blocks': 2,
        'proj_hidden_channels': 128,
    }

    # Merge with provided config
    if config is not None:
        default_config.update(config)

    # Build model
    model = DeFusion(**default_config)

    # Load pretrained weights if provided
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained}")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    print("Testing DeFusion model...")

    # Create model
    model = DeFusion()
    print(f"Total parameters: {count_parameters(model):,}")

    # Test input
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)

    # Test training forward
    outputs = model.forward_train(x1, x2)
    print("\nTraining forward pass:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test fusion forward
    fused = model.forward_fusion(x1, x2)
    print(f"\nFusion output: {fused.shape}")

    # Test convenience method
    fused_norm = model.fuse_images(x1, x2, normalize=True)
    print(f"Normalized fusion output: {fused_norm.shape}")
    print(f"  min: {fused_norm.min():.4f}, max: {fused_norm.max():.4f}")

    print("\nAll tests passed!")
