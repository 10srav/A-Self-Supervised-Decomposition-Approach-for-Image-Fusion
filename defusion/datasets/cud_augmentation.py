"""
CUD Augmentation: Common and Unique Decomposition
=================================================
Implementation of the CUD pretext task augmentation from Section 3.2

CRITICAL: Masks must OVERLAP to have common visible region!
    M1 ∩ M2 > 0 (common region - visible in both)
    M1 - M2 > 0 (unique to x1)
    M2 - M1 > 0 (unique to x2)
    M1 ∪ M2 = 1 (full coverage between both masks)

Given scene x:
    x1 = M1 * x + (1-M1) * noise1  (M1 regions show x, rest is noise)
    x2 = M2 * x + (1-M2) * noise2  (M2 regions show x, rest is noise)

Loss targets:
    L_c = ||Pc(fc) - (M1∩M2)*x||    # Common visible in both
    L_u1 = ||Pu(f1u) - (M1-M2)*x||  # Only visible in x1
    L_u2 = ||Pu(f2u) - (M2-M1)*x||  # Only visible in x2
    L_r = ||Pr(fc,f1u,f2u) - x||    # Full reconstruction
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import random


def generate_overlapping_masks(
    height: int,
    width: int,
    common_ratio: float = 0.3,
    unique_ratio: float = 0.35
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two OVERLAPPING masks with common and unique regions.

    Paper requirement: M1 and M2 must have:
    - Non-empty intersection (common region)
    - Non-empty unique regions for each
    - Together cover the full image

    Args:
        height: Image height
        width: Image width
        common_ratio: Target ratio for common region (M1 ∩ M2)
        unique_ratio: Target ratio for each unique region

    Returns:
        m1: First mask [H, W] - regions visible in x1
        m2: Second mask [H, W] - regions visible in x2
    """
    # Total pixels
    total = height * width

    # Block-based mask generation at multiple scales
    mask_base = torch.zeros(height, width)

    # Generate random assignment: 0=common, 1=unique_to_m1, 2=unique_to_m2, 3=both_noise (small)
    # Use multi-scale blocks for realistic mask patterns

    scales = [8, 16, 32, 64]
    assignment = torch.zeros(height, width)

    for scale in scales:
        bh, bw = height // scale, width // scale
        if bh == 0 or bw == 0:
            continue

        # Random block assignments
        block_assign = torch.rand(bh, bw)

        # Upsample to full resolution
        upsampled = F.interpolate(
            block_assign.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest'
        ).squeeze()

        # Blend with existing assignment
        assignment = 0.5 * assignment + 0.5 * upsampled

    # Threshold to create regions
    # Common region: middle values -> both M1 and M2 are 1
    # Unique to M1: low values -> only M1 is 1
    # Unique to M2: high values -> only M2 is 1

    # Compute thresholds based on target ratios
    t1 = unique_ratio  # Below this -> unique to M1
    t2 = unique_ratio + common_ratio  # Below this but above t1 -> common
    # Above t2 -> unique to M2

    m1 = (assignment < t2).float()  # M1 covers unique_m1 + common
    m2 = (assignment > t1).float()  # M2 covers common + unique_m2

    # Verify overlap exists
    intersection = m1 * m2
    if intersection.sum() < total * 0.1:  # Less than 10% overlap
        # Force some overlap in center region
        ch, cw = height // 4, width // 4
        m1[ch:height-ch, cw:width-cw] = 1
        m2[ch:height-ch, cw:width-cw] = 1

    return m1, m2


def generate_complementary_masks(
    height: int,
    width: int,
    overlap_ratio: float = 0.3,
    method: str = 'blocks'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate masks that together cover the image with some overlap.

    M1 ∪ M2 = 1 (full coverage)
    M1 ∩ M2 > 0 (overlap exists)

    Args:
        height, width: Image dimensions
        overlap_ratio: Fraction of image that should be common
        method: 'blocks', 'stripes', or 'random'

    Returns:
        m1, m2: Overlapping masks
    """
    if method == 'blocks':
        # Generate base random field at multiple scales
        base = torch.zeros(height, width)
        for scale in [64, 32, 16]:
            bh, bw = height // scale, width // scale
            if bh == 0 or bw == 0:
                continue
            blocks = torch.rand(bh, bw)
            up = F.interpolate(blocks.unsqueeze(0).unsqueeze(0),
                              size=(height, width), mode='nearest').squeeze()
            base = 0.5 * base + 0.5 * up

        # Create three regions:
        # - Common: both M1=1 and M2=1 (overlap_ratio of image)
        # - Unique to M1: M1=1, M2=0
        # - Unique to M2: M1=0, M2=1
        # Target: ~30% common, ~35% unique each

        # Thresholds for region assignment
        t_unique1 = 0.35  # [0, t_unique1) -> unique to M1
        t_common = 0.35 + overlap_ratio  # [t_unique1, t_common) -> common
        # [t_common, 1.0] -> unique to M2

        unique1_region = (base < t_unique1).float()
        common_region = ((base >= t_unique1) & (base < t_common)).float()
        unique2_region = (base >= t_common).float()

        # M1 = unique1 + common
        m1 = torch.clamp(unique1_region + common_region, 0, 1)
        # M2 = common + unique2
        m2 = torch.clamp(common_region + unique2_region, 0, 1)

    elif method == 'stripes':
        # Overlapping stripe patterns
        stripe_w = random.choice([16, 32, 64])

        y = torch.arange(height).unsqueeze(1).expand(height, width).float()
        x = torch.arange(width).unsqueeze(0).expand(height, width).float()

        # Diagonal stripes with overlap
        diag1 = ((x + y) / stripe_w).floor() % 2
        diag2 = ((x - y + width) / stripe_w).floor() % 2

        m1 = ((diag1 + torch.rand(1) * 0.3) > 0.5).float()
        m2 = ((diag2 + torch.rand(1) * 0.3) > 0.5).float()

        # Ensure overlap
        m1 = torch.clamp(m1 + (torch.rand(height, width) < overlap_ratio).float(), 0, 1)
        m2 = torch.clamp(m2 + (torch.rand(height, width) < overlap_ratio).float(), 0, 1)

    else:  # random
        # Random pixel-wise masks with controlled overlap
        base = torch.rand(height, width)

        # M1: below threshold1 OR in overlap region
        # M2: above threshold2 OR in overlap region
        overlap_region = torch.rand(height, width) < overlap_ratio

        m1 = ((base < 0.6) | overlap_region).float()
        m2 = ((base > 0.4) | overlap_region).float()

    return m1, m2


class CUDAugmentation:
    """
    Common and Unique Decomposition Augmentation.

    IMPORTANT: This implementation ensures masks OVERLAP so that:
    - M1 ∩ M2 > 0 (common visible region exists)
    - M1 - M2 > 0 (unique to x1 exists)
    - M2 - M1 > 0 (unique to x2 exists)
    - M1 ∪ M2 ≈ 1 (most of image covered)
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        overlap_ratio: float = 0.3,
        mask_method: str = 'blocks'
    ):
        """
        Args:
            noise_std: Gaussian noise std (paper uses 0.1)
            overlap_ratio: Target overlap between masks (0.2-0.4 recommended)
            mask_method: 'blocks', 'stripes', or 'random'
        """
        self.noise_std = noise_std
        self.overlap_ratio = overlap_ratio
        self.mask_method = mask_method

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply CUD augmentation.

        Args:
            x: Input image [C, H, W] or [B, C, H, W]

        Returns:
            Dictionary with x, x1, x2, masks, and derived mask regions
        """
        if x.dim() == 4:
            x = x[0]  # Take first if batch

        C, H, W = x.shape

        # Generate OVERLAPPING masks
        m1, m2 = generate_complementary_masks(
            H, W,
            overlap_ratio=self.overlap_ratio,
            method=self.mask_method
        )

        # Generate Gaussian noise
        n1 = torch.randn_like(x) * self.noise_std
        n2 = torch.randn_like(x) * self.noise_std

        # Expand masks for broadcasting [H,W] -> [C,H,W]
        m1_exp = m1.unsqueeze(0).expand(C, -1, -1)
        m2_exp = m2.unsqueeze(0).expand(C, -1, -1)

        # Create augmented views
        # x1 shows x where m1=1, noise where m1=0
        x1 = m1_exp * x + (1 - m1_exp) * n1
        # x2 shows x where m2=1, noise where m2=0
        x2 = m2_exp * x + (1 - m2_exp) * n2

        # Compute derived mask regions for loss
        m_common = m1 * m2           # Visible in BOTH x1 and x2
        m1_unique = m1 * (1 - m2)    # Visible ONLY in x1
        m2_unique = m2 * (1 - m1)    # Visible ONLY in x2

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


def create_batch_cud(
    images: torch.Tensor,
    noise_std: float = 0.1,
    overlap_ratio: float = 0.3
) -> Dict[str, torch.Tensor]:
    """
    Apply CUD augmentation to a batch of images.

    Args:
        images: Batch of images [B, C, H, W]
        noise_std: Gaussian noise std
        overlap_ratio: Target mask overlap

    Returns:
        Batch dictionary with all augmented data
    """
    B, C, H, W = images.shape
    cud = CUDAugmentation(noise_std=noise_std, overlap_ratio=overlap_ratio)

    # Initialize batch tensors
    batch = {
        'x': images,
        'x1': torch.zeros_like(images),
        'x2': torch.zeros_like(images),
        'm1': torch.zeros(B, H, W),
        'm2': torch.zeros(B, H, W),
        'm_common': torch.zeros(B, H, W),
        'm1_unique': torch.zeros(B, H, W),
        'm2_unique': torch.zeros(B, H, W)
    }

    # Process each image
    for i in range(B):
        result = cud(images[i])
        batch['x1'][i] = result['x1']
        batch['x2'][i] = result['x2']
        batch['m1'][i] = result['m1']
        batch['m2'][i] = result['m2']
        batch['m_common'][i] = result['m_common']
        batch['m1_unique'][i] = result['m1_unique']
        batch['m2_unique'][i] = result['m2_unique']

    return batch


if __name__ == '__main__':
    print("Testing CUD Augmentation (with OVERLAPPING masks)...")
    print("=" * 60)

    # Create test image
    x = torch.zeros(3, 256, 256)
    for c in range(3):
        grad = torch.linspace(0, 1, 256).unsqueeze(1).expand(256, 256).clone()
        if c == 1:
            grad = grad.T.clone()
        x[c] = grad

    # Apply CUD
    cud = CUDAugmentation(noise_std=0.1, overlap_ratio=0.3)
    output = cud(x)

    print(f"Original shape: {output['x'].shape}")
    print(f"x1 shape: {output['x1'].shape}")
    print(f"x2 shape: {output['x2'].shape}")
    print()
    print("Mask Statistics:")
    print(f"  M1 coverage: {output['m1'].mean():.1%}")
    print(f"  M2 coverage: {output['m2'].mean():.1%}")
    print(f"  Common (M1 AND M2): {output['m_common'].mean():.1%}")
    print(f"  Unique to M1: {output['m1_unique'].mean():.1%}")
    print(f"  Unique to M2: {output['m2_unique'].mean():.1%}")
    print(f"  Total coverage: {(output['m1'] + output['m2'] - output['m_common']).clamp(0,1).mean():.1%}")
    print()

    # Verify constraints
    assert output['m_common'].mean() > 0.05, "ERROR: No common region!"
    assert output['m1_unique'].mean() > 0.05, "ERROR: No unique region for M1!"
    assert output['m2_unique'].mean() > 0.05, "ERROR: No unique region for M2!"

    print("All constraints satisfied!")
    print("  - Common region exists (M1 AND M2 > 0)")
    print("  - Unique regions exist")
    print("  - Ready for CUD training!")
