"""
CUD Augmentation: Common and Unique Decomposition
=================================================
Implementation of the CUD pretext task augmentation from Section 3.2

Given scene x ∈ R^{H×W×3}:
    Generate 2 masks M1, M2 s.t. M1 ∩ M2 = ∅ (non-overlapping)
    x1 = M1 ⊙ x + (1-M1) ⊙ n1  # n1 = Gaussian noise
    x2 = M2 ⊙ x + (1-M2) ⊙ n2  # n2 = Gaussian noise

Paper Section 3.2:
"We propose the Common and Unique Decomposition (CUD) task as a self-supervised
pretext for image fusion. Given an image x, we generate two non-overlapping masks
M1 and M2, and use them to create two augmented views x1 and x2."
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import random


def generate_random_rectangles(
    height: int,
    width: int,
    num_rects: int = 3,
    min_size: float = 0.1,
    max_size: float = 0.5
) -> torch.Tensor:
    """
    Generate a binary mask with random rectangles.

    Args:
        height: Image height
        width: Image width
        num_rects: Number of random rectangles to generate
        min_size: Minimum rectangle size as fraction of image
        max_size: Maximum rectangle size as fraction of image

    Returns:
        Binary mask [H, W] with rectangles set to 1
    """
    mask = torch.zeros(height, width)

    for _ in range(num_rects):
        # Random rectangle size
        rect_h = int(height * random.uniform(min_size, max_size))
        rect_w = int(width * random.uniform(min_size, max_size))

        # Random position
        y = random.randint(0, height - rect_h)
        x = random.randint(0, width - rect_w)

        # Set rectangle to 1
        mask[y:y+rect_h, x:x+rect_w] = 1

    return mask


def generate_multi_resolution_mask(
    height: int,
    width: int,
    scales: Tuple[int, ...] = (8, 16, 32, 64),
    coverage_target: float = 0.5
) -> torch.Tensor:
    """
    Generate mask at multiple resolutions (block sizes).

    Paper mentions "2 random masks at different resolutions".

    Args:
        height: Image height
        width: Image width
        scales: Block sizes to use
        coverage_target: Target coverage fraction

    Returns:
        Binary mask [H, W]
    """
    mask = torch.zeros(height, width)

    for scale in scales:
        # Generate random blocks at this scale
        blocks_h = height // scale
        blocks_w = width // scale

        if blocks_h == 0 or blocks_w == 0:
            continue

        # Random block selection
        block_mask = torch.rand(blocks_h, blocks_w) < (coverage_target / len(scales))

        # Upsample to full resolution
        upsampled = F.interpolate(
            block_mask.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest'
        ).squeeze()

        mask = torch.clamp(mask + upsampled, 0, 1)

    return mask


def generate_non_overlapping_masks(
    height: int,
    width: int,
    method: str = 'random_rects',
    min_coverage: float = 0.3,
    max_coverage: float = 0.7,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two non-overlapping binary masks.

    Paper Section 3.2:
    "Generate 2 masks M1, M2 s.t. M1 ∩ M2 = ∅"

    Args:
        height: Image height
        width: Image width
        method: Mask generation method ('random_rects', 'multi_scale', 'stripes')
        min_coverage: Minimum coverage for each mask
        max_coverage: Maximum coverage for each mask
        **kwargs: Additional arguments for mask generation

    Returns:
        m1: First binary mask [H, W]
        m2: Second binary mask [H, W]
    """
    if method == 'random_rects':
        # Generate first mask with random rectangles
        m1 = generate_random_rectangles(
            height, width,
            num_rects=kwargs.get('num_rects', 4),
            min_size=kwargs.get('min_size', 0.1),
            max_size=kwargs.get('max_size', 0.4)
        )

        # Generate second mask from complement
        complement = 1 - m1

        # Add random rectangles to second mask (within complement)
        m2_rects = generate_random_rectangles(
            height, width,
            num_rects=kwargs.get('num_rects', 4),
            min_size=kwargs.get('min_size', 0.1),
            max_size=kwargs.get('max_size', 0.4)
        )

        # Ensure non-overlapping: m2 only where m1 is 0
        m2 = m2_rects * complement

    elif method == 'multi_scale':
        # Multi-scale block masks
        target_coverage = random.uniform(min_coverage, max_coverage)

        m1 = generate_multi_resolution_mask(
            height, width,
            coverage_target=target_coverage
        )

        # Generate m2 from complement
        complement = 1 - m1

        m2_raw = generate_multi_resolution_mask(
            height, width,
            coverage_target=target_coverage
        )

        m2 = m2_raw * complement

    elif method == 'stripes':
        # Alternating stripes (simpler but effective)
        stripe_width = kwargs.get('stripe_width', 32)

        y_coords = torch.arange(height).unsqueeze(1).expand(height, width)
        x_coords = torch.arange(width).unsqueeze(0).expand(height, width)

        # Horizontal and vertical stripes
        h_stripes = ((y_coords // stripe_width) % 2).float()
        v_stripes = ((x_coords // stripe_width) % 2).float()

        # Checkerboard pattern
        checker = ((h_stripes + v_stripes) % 2).float()

        # Random rotation/flip for variety
        if random.random() > 0.5:
            m1 = checker
            m2 = 1 - checker
        else:
            m1 = 1 - checker
            m2 = checker

        # Add some randomness
        noise = (torch.rand(height, width) > 0.9).float()
        m1 = torch.clamp(m1 + noise * (1 - m1) * 0.1, 0, 1)

    else:
        raise ValueError(f"Unknown mask method: {method}")

    # Ensure masks are binary
    m1 = (m1 > 0.5).float()
    m2 = (m2 > 0.5).float()

    # Ensure non-overlapping (safety check)
    m2 = m2 * (1 - m1)

    return m1, m2


class CUDAugmentation:
    """
    Common and Unique Decomposition Augmentation.

    Applies the CUD pretext task augmentation to an image:
    1. Generate non-overlapping masks M1, M2
    2. Create augmented views: x1 = M1*x + (1-M1)*noise
    3. Return augmented images and masks for loss computation

    Paper Section 3.2:
    "x1 = M1 ⊙ x + (1-M1) ⊙ n1, x2 = M2 ⊙ x + (1-M2) ⊙ n2"
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        mask_method: str = 'random_rects',
        min_coverage: float = 0.3,
        max_coverage: float = 0.7,
        **mask_kwargs
    ):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise (paper uses 0.1)
            mask_method: Method for mask generation
            min_coverage: Minimum mask coverage
            max_coverage: Maximum mask coverage
            **mask_kwargs: Additional arguments for mask generation
        """
        self.noise_std = noise_std
        self.mask_method = mask_method
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.mask_kwargs = mask_kwargs

    def __call__(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply CUD augmentation to an image.

        Args:
            x: Input image [C, H, W] or [B, C, H, W]

        Returns:
            Dictionary containing:
                - x: Original image [C, H, W]
                - x1: First augmented view [C, H, W]
                - x2: Second augmented view [C, H, W]
                - m1: First mask [H, W]
                - m2: Second mask [H, W]
                - m_common: M1 ∩ M2 intersection (should be ~0) [H, W]
                - m1_unique: M1 - M2 (unique to first) [H, W]
                - m2_unique: M2 - M1 (unique to second) [H, W]
        """
        # Handle batch dimension
        squeeze_batch = False
        if x.dim() == 4:
            # Batch mode - process first image only for simplicity
            x = x[0]
            squeeze_batch = True

        C, H, W = x.shape

        # Generate non-overlapping masks
        m1, m2 = generate_non_overlapping_masks(
            H, W,
            method=self.mask_method,
            min_coverage=self.min_coverage,
            max_coverage=self.max_coverage,
            **self.mask_kwargs
        )

        # Generate Gaussian noise
        n1 = torch.randn_like(x) * self.noise_std
        n2 = torch.randn_like(x) * self.noise_std

        # Expand masks to channel dimension
        m1_expanded = m1.unsqueeze(0).expand(C, -1, -1)
        m2_expanded = m2.unsqueeze(0).expand(C, -1, -1)

        # Apply CUD augmentation
        # x1 = M1 ⊙ x + (1-M1) ⊙ n1
        x1 = m1_expanded * x + (1 - m1_expanded) * n1

        # x2 = M2 ⊙ x + (1-M2) ⊙ n2
        x2 = m2_expanded * x + (1 - m2_expanded) * n2

        # Compute derived masks for loss computation
        m_common = m1 * m2  # Should be ~0 (non-overlapping)
        m1_unique = m1 * (1 - m2)  # M1 - M2: unique to first
        m2_unique = m2 * (1 - m1)  # M2 - M1: unique to second

        return {
            'x': x,
            'x1': x1,
            'x2': x2,
            'm1': m1,
            'm2': m2,
            'm_common': m_common,
            'm1_unique': m1_unique,
            'm2_unique': m2_unique
        }

    def compute_targets(
        self,
        x: torch.Tensor,
        m1: torch.Tensor,
        m2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute target images for each projection loss.

        Paper Loss Function:
            L_c = ||Pc(fc) - (M1∩M2)⊙x||    -> target_common
            L_u1 = ||Pu(f1u) - (M1-M2)⊙x||  -> target_u1
            L_u2 = ||Pu(f2u) - (M2-M1)⊙x||  -> target_u2
            L_r = ||Pr(features) - x||       -> x (full reconstruction)

        Args:
            x: Original image [C, H, W]
            m1: First mask [H, W]
            m2: Second mask [H, W]

        Returns:
            Dictionary with targets for each loss term
        """
        C = x.shape[0]

        # Derived masks
        m_common = m1 * m2
        m1_unique = m1 * (1 - m2)
        m2_unique = m2 * (1 - m1)

        # Expand to channel dimension
        m_common_exp = m_common.unsqueeze(0).expand(C, -1, -1)
        m1_unique_exp = m1_unique.unsqueeze(0).expand(C, -1, -1)
        m2_unique_exp = m2_unique.unsqueeze(0).expand(C, -1, -1)

        return {
            'target_common': m_common_exp * x,
            'target_u1': m1_unique_exp * x,
            'target_u2': m2_unique_exp * x,
            'target_recon': x,
            # Masks for weighted loss
            'mask_common': m_common,
            'mask_u1': m1_unique,
            'mask_u2': m2_unique
        }


def visualize_cud_augmentation(
    x: torch.Tensor,
    cud_output: Dict[str, torch.Tensor]
) -> None:
    """
    Visualize CUD augmentation results.

    Args:
        x: Original image
        cud_output: Output from CUDAugmentation
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original
    axes[0, 0].imshow(x.permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title('Original')

    # Mask 1
    axes[0, 1].imshow(cud_output['m1'], cmap='gray')
    axes[0, 1].set_title('Mask M1')

    # Mask 2
    axes[0, 2].imshow(cud_output['m2'], cmap='gray')
    axes[0, 2].set_title('Mask M2')

    # Common (should be ~empty)
    axes[0, 3].imshow(cud_output['m_common'], cmap='gray')
    axes[0, 3].set_title('M1 ∩ M2 (common)')

    # Augmented x1
    axes[1, 0].imshow(cud_output['x1'].permute(1, 2, 0).clamp(0, 1))
    axes[1, 0].set_title('x1 = M1⊙x + (1-M1)⊙n1')

    # Augmented x2
    axes[1, 1].imshow(cud_output['x2'].permute(1, 2, 0).clamp(0, 1))
    axes[1, 1].set_title('x2 = M2⊙x + (1-M2)⊙n2')

    # M1 unique
    axes[1, 2].imshow(cud_output['m1_unique'], cmap='gray')
    axes[1, 2].set_title('M1 - M2 (unique to x1)')

    # M2 unique
    axes[1, 3].imshow(cud_output['m2_unique'], cmap='gray')
    axes[1, 3].set_title('M2 - M1 (unique to x2)')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Test CUD augmentation
    print("Testing CUD Augmentation...")

    # Create test image (gradient for visibility)
    x = torch.zeros(3, 256, 256)
    for c in range(3):
        x[c] = torch.linspace(0, 1, 256).unsqueeze(1).expand(256, 256)
        if c == 1:
            x[c] = x[c].T  # Different gradient direction

    # Apply CUD
    cud = CUDAugmentation(noise_std=0.1, mask_method='random_rects')
    output = cud(x)

    print(f"Original shape: {output['x'].shape}")
    print(f"Augmented x1 shape: {output['x1'].shape}")
    print(f"Augmented x2 shape: {output['x2'].shape}")
    print(f"Mask M1 coverage: {output['m1'].mean():.2%}")
    print(f"Mask M2 coverage: {output['m2'].mean():.2%}")
    print(f"Overlap (should be ~0): {output['m_common'].mean():.4f}")

    # Visualize if matplotlib available
    try:
        visualize_cud_augmentation(x, output)
    except ImportError:
        print("matplotlib not available for visualization")
