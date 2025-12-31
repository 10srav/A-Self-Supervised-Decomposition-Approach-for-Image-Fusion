"""
CUD Loss Functions
==================
Loss functions for the Common and Unique Decomposition pretext task.

Paper Section 3.2 (Loss Function):
L = ||Pc(fc) - (M1∩M2)⊙x||           # Common loss
  + ||Pu(f1u) - (M1-M2)⊙x||          # Unique1 loss
  + ||Pu(f2u) - (M2-M1)⊙x||          # Unique2 loss
  + ||Pr(fc+f1u+f2u) - x||            # Reconstruction loss

"All losses use MAE (L1 loss)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CUDLoss(nn.Module):
    """
    Complete CUD Loss for DeFusion training.

    Computes all four loss components:
    1. Common loss: Penalizes projection of common features
    2. Unique1 loss: Penalizes projection of unique features for image 1
    3. Unique2 loss: Penalizes projection of unique features for image 2
    4. Reconstruction loss: Penalizes full reconstruction

    Paper Section 3.2:
    "LOSS (MAE on projections)"
    """

    def __init__(
        self,
        weight_common: float = 1.0,
        weight_unique: float = 1.0,
        weight_recon: float = 1.0,
        masked_loss: bool = True
    ):
        """
        Args:
            weight_common: Weight for common projection loss
            weight_unique: Weight for unique projection losses
            weight_recon: Weight for reconstruction loss
            masked_loss: Whether to mask losses to relevant regions
        """
        super().__init__()

        self.weight_common = weight_common
        self.weight_unique = weight_unique
        self.weight_recon = weight_recon
        self.masked_loss = masked_loss

        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CUD loss.

        Args:
            predictions: Model outputs containing:
                - xc: Common projection [B, 3, H, W]
                - x1u: Unique projection 1 [B, 3, H, W]
                - x2u: Unique projection 2 [B, 3, H, W]
                - xr: Reconstruction [B, 3, H, W]
            targets: Ground truth containing:
                - x: Original image [B, 3, H, W]
                - m1: Mask 1 [B, H, W]
                - m2: Mask 2 [B, H, W]
                - m_common: Common mask (M1 ∩ M2) [B, H, W]
                - m1_unique: Unique mask 1 (M1 - M2) [B, H, W]
                - m2_unique: Unique mask 2 (M2 - M1) [B, H, W]

        Returns:
            Dictionary containing:
                - loss: Total loss
                - loss_common: Common projection loss
                - loss_u1: Unique1 projection loss
                - loss_u2: Unique2 projection loss
                - loss_recon: Reconstruction loss
        """
        x = targets['x']
        m_common = targets['m_common']
        m1_unique = targets['m1_unique']
        m2_unique = targets['m2_unique']

        xc = predictions['xc']
        x1u = predictions['x1u']
        x2u = predictions['x2u']
        xr = predictions['xr']

        B, C, H, W = x.shape

        # Expand masks to match image channels [B, 1, H, W] -> [B, C, H, W]
        m_common_exp = m_common.unsqueeze(1).expand(B, C, H, W)
        m1_unique_exp = m1_unique.unsqueeze(1).expand(B, C, H, W)
        m2_unique_exp = m2_unique.unsqueeze(1).expand(B, C, H, W)

        # Common loss: ||Pc(fc) - (M1∩M2)⊙x||
        # Target is the common region (overlap), which should be ~0 for non-overlapping masks
        target_common = m_common_exp * x
        loss_common_raw = self.l1_loss(xc, target_common)

        if self.masked_loss:
            # Only compute loss where mask is active
            loss_common = (loss_common_raw * m_common_exp).sum() / (m_common_exp.sum() + 1e-8)
        else:
            loss_common = loss_common_raw.mean()

        # Unique1 loss: ||Pu(f1u) - (M1-M2)⊙x||
        target_u1 = m1_unique_exp * x
        loss_u1_raw = self.l1_loss(x1u, target_u1)

        if self.masked_loss:
            loss_u1 = (loss_u1_raw * m1_unique_exp).sum() / (m1_unique_exp.sum() + 1e-8)
        else:
            loss_u1 = loss_u1_raw.mean()

        # Unique2 loss: ||Pu(f2u) - (M2-M1)⊙x||
        target_u2 = m2_unique_exp * x
        loss_u2_raw = self.l1_loss(x2u, target_u2)

        if self.masked_loss:
            loss_u2 = (loss_u2_raw * m2_unique_exp).sum() / (m2_unique_exp.sum() + 1e-8)
        else:
            loss_u2 = loss_u2_raw.mean()

        # Reconstruction loss: ||Pr(fc+f1u+f2u) - x||
        loss_recon = self.l1_loss(xr, x).mean()

        # Total weighted loss
        loss_total = (
            self.weight_common * loss_common +
            self.weight_unique * (loss_u1 + loss_u2) +
            self.weight_recon * loss_recon
        )

        return {
            'loss': loss_total,
            'loss_common': loss_common,
            'loss_u1': loss_u1,
            'loss_u2': loss_u2,
            'loss_recon': loss_recon
        }


def compute_cud_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weight_common: float = 1.0,
    weight_unique: float = 1.0,
    weight_recon: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute CUD loss from model output and batch.

    Args:
        model_output: Output from DeFusion.forward_train()
        batch: Batch from COCODataset
        weight_common: Weight for common loss
        weight_unique: Weight for unique losses
        weight_recon: Weight for reconstruction loss

    Returns:
        Loss dictionary
    """
    criterion = CUDLoss(
        weight_common=weight_common,
        weight_unique=weight_unique,
        weight_recon=weight_recon
    )

    predictions = {
        'xc': model_output['xc'],
        'x1u': model_output['x1u'],
        'x2u': model_output['x2u'],
        'xr': model_output['xr']
    }

    targets = {
        'x': batch['x'],
        'm_common': batch['m_common'],
        'm1_unique': batch['m1_unique'],
        'm2_unique': batch['m2_unique']
    }

    return criterion(predictions, targets)


class PerceptualLoss(nn.Module):
    """
    Optional perceptual loss using VGG features.

    Not mentioned in paper but can improve visual quality.
    """

    def __init__(self, layers: list = ['relu2_2', 'relu3_3']):
        super().__init__()

        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except ImportError:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features

        self.slices = nn.ModuleList()
        self.layer_names = []

        # VGG layer indices for common perceptual layers
        layer_indices = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
        }

        prev_idx = 0
        for name in layers:
            idx = layer_indices.get(name, 16)
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:idx]))
            self.layer_names.append(name)
            prev_idx = idx

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # Denormalize from [-1, 1] to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        # Apply ImageNet normalization
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0
        x_pred = pred
        x_target = target

        for slice_module in self.slices:
            x_pred = slice_module(x_pred)
            x_target = slice_module(x_target)
            loss += F.l1_loss(x_pred, x_target)

        return loss / len(self.slices)


class GradientLoss(nn.Module):
    """
    Gradient loss for preserving edges.

    Computes L1 loss on Sobel gradients.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude."""
        B, C, H, W = x.shape

        # Expand Sobel kernels for all channels
        sobel_x = self.sobel_x.expand(C, 1, 3, 3)
        sobel_y = self.sobel_y.expand(C, 1, 3, 3)

        # Compute gradients
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return grad_mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss."""
        grad_pred = self._compute_gradient(pred)
        grad_target = self._compute_gradient(target)

        return F.l1_loss(grad_pred, grad_target)


if __name__ == '__main__':
    # Test loss functions
    print("Testing CUD Loss...")

    B, C, H, W = 2, 3, 256, 256

    # Create dummy predictions
    predictions = {
        'xc': torch.randn(B, C, H, W),
        'x1u': torch.randn(B, C, H, W),
        'x2u': torch.randn(B, C, H, W),
        'xr': torch.randn(B, C, H, W)
    }

    # Create dummy targets
    targets = {
        'x': torch.randn(B, C, H, W),
        'm_common': torch.zeros(B, H, W),  # Non-overlapping
        'm1_unique': torch.rand(B, H, W) > 0.5,
        'm2_unique': torch.rand(B, H, W) > 0.5
    }

    # Ensure non-overlapping
    targets['m2_unique'] = targets['m2_unique'].float() * (1 - targets['m1_unique'].float())

    # Compute loss
    criterion = CUDLoss()
    losses = criterion(predictions, targets)

    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"Common loss: {losses['loss_common'].item():.4f}")
    print(f"Unique1 loss: {losses['loss_u1'].item():.4f}")
    print(f"Unique2 loss: {losses['loss_u2'].item():.4f}")
    print(f"Reconstruction loss: {losses['loss_recon'].item():.4f}")

    # Test gradient loss
    grad_loss = GradientLoss()
    g_loss = grad_loss(predictions['xr'], targets['x'])
    print(f"Gradient loss: {g_loss.item():.4f}")

    print("\nAll tests passed!")
