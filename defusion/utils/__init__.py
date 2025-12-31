"""
DeFusion Utilities
==================
Utility functions for training, evaluation, and visualization.

Components:
- losses: CUD loss functions
- metrics: Fusion quality metrics (SSIM, MEF-SSIM, etc.)
- visualization: Feature and result visualization
"""

from .losses import CUDLoss, compute_cud_loss
from .metrics import (
    compute_ssim,
    compute_psnr,
    compute_mef_ssim,
    compute_mutual_information,
    compute_entropy,
    compute_qcv,
    FusionMetrics
)
from .visualization import (
    visualize_features,
    visualize_fusion_result,
    plot_training_curves
)

__all__ = [
    'CUDLoss',
    'compute_cud_loss',
    'compute_ssim',
    'compute_psnr',
    'compute_mef_ssim',
    'compute_mutual_information',
    'compute_entropy',
    'compute_qcv',
    'FusionMetrics',
    'visualize_features',
    'visualize_fusion_result',
    'plot_training_curves'
]
