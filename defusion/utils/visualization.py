"""
Visualization Utilities
=======================
Functions for visualizing DeFusion features, fusion results, and training progress.

Paper Section 4.3 (Fig 7-8):
"Feature visualization of fc, f1u, f2u"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def tensor_to_image(
    tensor: torch.Tensor,
    denormalize: bool = True
) -> np.ndarray:
    """
    Convert tensor to numpy image for visualization.

    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
        denormalize: Whether to denormalize from [-1, 1] to [0, 1]

    Returns:
        Numpy array [H, W, C] in range [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch item

    # Move to CPU
    img = tensor.detach().cpu()

    # Denormalize from [-1, 1] to [0, 1]
    if denormalize:
        img = (img + 1) / 2

    # Clamp and convert
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()

    return img


def visualize_features(
    fc: torch.Tensor,
    f1u: torch.Tensor,
    f2u: torch.Tensor,
    num_channels: int = 16,
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Visualize decomposed features (fc, f1u, f2u).

    Creates a grid showing selected channels from each feature map.
    Corresponds to Fig 7-8 in the paper.

    Args:
        fc: Common features [B, C, H, W]
        f1u: Unique features 1 [B, C, H, W]
        f2u: Unique features 2 [B, C, H, W]
        num_channels: Number of channels to visualize
        save_path: Optional path to save figure

    Returns:
        Visualization as numpy array if matplotlib available
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        print("matplotlib not available for visualization")
        return None

    # Take first batch item
    if fc.dim() == 4:
        fc = fc[0]
        f1u = f1u[0]
        f2u = f2u[0]

    # Move to CPU
    fc = fc.detach().cpu()
    f1u = f1u.detach().cpu()
    f2u = f2u.detach().cpu()

    # Select channels to visualize
    C = fc.shape[0]
    channel_indices = np.linspace(0, C - 1, num_channels, dtype=int)

    # Create figure
    fig, axes = plt.subplots(3, num_channels, figsize=(2 * num_channels, 6))

    titles = ['Common (fc)', 'Unique 1 (f1u)', 'Unique 2 (f2u)']
    features = [fc, f1u, f2u]

    for row, (title, feat) in enumerate(zip(titles, features)):
        for col, ch_idx in enumerate(channel_indices):
            ax = axes[row, col]
            im = ax.imshow(feat[ch_idx].numpy(), cmap='viridis')
            ax.axis('off')

            if col == 0:
                ax.set_ylabel(title, fontsize=10)
            if row == 0:
                ax.set_title(f'Ch {ch_idx}', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature visualization to {save_path}")

    # Convert to array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return img_array


def visualize_fusion_result(
    source1: torch.Tensor,
    source2: torch.Tensor,
    fused: torch.Tensor,
    title: str = "Fusion Result",
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Visualize fusion result with source images.

    Args:
        source1: First source image [C, H, W]
        source2: Second source image [C, H, W]
        fused: Fused image [C, H, W]
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        Visualization as numpy array
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return None

    # Convert to images
    img1 = tensor_to_image(source1)
    img2 = tensor_to_image(source2)
    img_fused = tensor_to_image(fused)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img1)
    axes[0].set_title('Source 1')
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title('Source 2')
    axes[1].axis('off')

    axes[2].imshow(img_fused)
    axes[2].set_title('Fused')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved fusion visualization to {save_path}")

    # Convert to array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return img_array


def visualize_cud_training(
    original: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    xc: torch.Tensor,
    x1u: torch.Tensor,
    x2u: torch.Tensor,
    xr: torch.Tensor,
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Visualize CUD training step.

    Shows original, augmented views, masks, and reconstructions.

    Args:
        original: Original image
        x1, x2: CUD augmented views
        m1, m2: Masks
        xc: Common projection
        x1u, x2u: Unique projections
        xr: Full reconstruction
        save_path: Optional save path

    Returns:
        Visualization array
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Row 1: Original, x1, x2, m1, m2
    axes[0, 0].imshow(tensor_to_image(original))
    axes[0, 0].set_title('Original')

    axes[0, 1].imshow(tensor_to_image(x1))
    axes[0, 1].set_title('x1 (M1*x + noise)')

    axes[0, 2].imshow(tensor_to_image(x2))
    axes[0, 2].set_title('x2 (M2*x + noise)')

    if m1.dim() == 3:
        m1 = m1[0]
    if m2.dim() == 3:
        m2 = m2[0]

    axes[0, 3].imshow(m1.cpu().numpy(), cmap='gray')
    axes[0, 3].set_title('Mask M1')

    axes[0, 4].imshow(m2.cpu().numpy(), cmap='gray')
    axes[0, 4].set_title('Mask M2')

    # Row 2: xc, x1u, x2u, xr, difference
    axes[1, 0].imshow(tensor_to_image(xc))
    axes[1, 0].set_title('Common Proj (xc)')

    axes[1, 1].imshow(tensor_to_image(x1u))
    axes[1, 1].set_title('Unique1 Proj (x1u)')

    axes[1, 2].imshow(tensor_to_image(x2u))
    axes[1, 2].set_title('Unique2 Proj (x2u)')

    axes[1, 3].imshow(tensor_to_image(xr))
    axes[1, 3].set_title('Reconstruction (xr)')

    # Difference map
    diff = (tensor_to_image(xr) - tensor_to_image(original)) ** 2
    diff = diff.sum(axis=-1)  # Sum over channels
    axes[1, 4].imshow(diff, cmap='hot')
    axes[1, 4].set_title('Recon Error')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return img_array


def plot_training_curves(
    losses: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Plot training loss curves.

    Args:
        losses: Dictionary mapping loss names to lists of values
        save_path: Optional save path

    Returns:
        Plot as numpy array
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    if 'loss' in losses:
        axes[0].plot(losses['loss'], label='Total Loss', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Component losses
    components = ['loss_common', 'loss_u1', 'loss_u2', 'loss_recon']
    colors = ['blue', 'green', 'red', 'purple']

    for comp, color in zip(components, colors):
        if comp in losses:
            axes[1].plot(losses[comp], label=comp, color=color, alpha=0.7)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Component Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return img_array


def create_comparison_grid(
    images: List[torch.Tensor],
    labels: List[str],
    nrow: int = 4,
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Create a grid comparing multiple images.

    Args:
        images: List of image tensors
        labels: Labels for each image
        nrow: Number of images per row
        save_path: Optional save path

    Returns:
        Grid as numpy array
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    n = len(images)
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(4 * ncol, 4 * nrow_actual))

    if nrow_actual == 1:
        axes = [axes]
    if ncol == 1:
        axes = [[ax] for ax in axes]

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // ncol
        col = idx % ncol
        axes[row][col].imshow(tensor_to_image(img))
        axes[row][col].set_title(label)
        axes[row][col].axis('off')

    # Hide empty subplots
    for idx in range(len(images), nrow_actual * ncol):
        row = idx // ncol
        col = idx % ncol
        axes[row][col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return img_array


def save_feature_maps(
    features: torch.Tensor,
    output_dir: str,
    prefix: str = 'feature'
):
    """
    Save individual feature map channels as images.

    Args:
        features: Feature tensor [C, H, W]
        output_dir: Output directory
        prefix: Filename prefix
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if features.dim() == 4:
        features = features[0]

    features = features.detach().cpu()
    C = features.shape[0]

    for c in range(C):
        # Normalize channel to [0, 255]
        feat = features[c].numpy()
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        feat = (feat * 255).astype(np.uint8)

        # Save
        img = Image.fromarray(feat)
        img.save(output_dir / f'{prefix}_ch{c:03d}.png')

    print(f"Saved {C} feature maps to {output_dir}")


if __name__ == '__main__':
    # Test visualization functions
    print("Testing Visualization Utilities...")

    # Create test data
    H, W = 256, 256
    C = 256  # Feature channels

    # Test images
    source1 = torch.rand(3, H, W) * 2 - 1  # [-1, 1]
    source2 = torch.rand(3, H, W) * 2 - 1
    fused = (source1 + source2) / 2

    # Test features
    fc = torch.rand(C, 32, 32)
    f1u = torch.rand(C, 32, 32)
    f2u = torch.rand(C, 32, 32)

    print("Creating visualizations...")

    # Test feature visualization
    feat_vis = visualize_features(fc, f1u, f2u, num_channels=8)
    if feat_vis is not None:
        print(f"Feature visualization shape: {feat_vis.shape}")

    # Test fusion visualization
    fusion_vis = visualize_fusion_result(source1, source2, fused)
    if fusion_vis is not None:
        print(f"Fusion visualization shape: {fusion_vis.shape}")

    # Test training curves
    losses = {
        'loss': list(np.random.rand(100) * 0.5 + 0.5 * np.exp(-np.arange(100) / 30)),
        'loss_common': list(np.random.rand(100) * 0.2),
        'loss_u1': list(np.random.rand(100) * 0.2),
        'loss_u2': list(np.random.rand(100) * 0.2),
        'loss_recon': list(np.random.rand(100) * 0.3)
    }

    curves_vis = plot_training_curves(losses)
    if curves_vis is not None:
        print(f"Training curves shape: {curves_vis.shape}")

    print("\nAll tests passed!")
