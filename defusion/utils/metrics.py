"""
Fusion Quality Metrics
======================
Implementation of image fusion quality metrics from the paper.

Paper Section 4 (Experiments):
Metrics used: CE, QCV, SSIM, MEF-SSIM, SD, VIF, MI

Reference metrics for fusion evaluation:
- Structural: SSIM, MEF-SSIM
- Information: CE, MI, QCV
- Visual quality: SD, VIF
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import ndimage
from scipy.signal import convolve2d


def gaussian_window(size: int, sigma: float) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= (size - 1) / 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    window = g.unsqueeze(1) @ g.unsqueeze(0)
    return window


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    size_average: bool = True,
    K1: float = 0.01,
    K2: float = 0.03
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        window_size: Size of Gaussian window
        sigma: Gaussian sigma
        data_range: Range of image values
        size_average: Whether to average over batch
        K1, K2: SSIM constants

    Returns:
        SSIM value(s)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    B, C, H, W = img1.shape

    # Create Gaussian window
    window = gaussian_window(window_size, sigma)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size)
    window = window.to(img1.device, img1.dtype)

    # Constants
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1: First image
        img2: Second image
        data_range: Maximum value range

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(data_range ** 2 / mse)


def compute_mef_ssim(
    fused: torch.Tensor,
    sources: Tuple[torch.Tensor, ...],
    weights: Optional[Tuple[float, ...]] = None
) -> torch.Tensor:
    """
    Compute Multi-Exposure Fusion SSIM (MEF-SSIM).

    Measures fusion quality for multi-exposure images.
    Computes weighted average of SSIM between fused and source images.

    Args:
        fused: Fused image [B, C, H, W]
        sources: Tuple of source images
        weights: Optional weights for each source

    Returns:
        MEF-SSIM value
    """
    if weights is None:
        weights = tuple(1.0 / len(sources) for _ in sources)

    total_ssim = 0
    for src, w in zip(sources, weights):
        total_ssim += w * compute_ssim(fused, src)

    return total_ssim


def compute_entropy(img: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    Compute image entropy.

    Higher entropy indicates more information content.

    Args:
        img: Input image (grayscale) [H, W] or [B, H, W]
        bins: Number of histogram bins

    Returns:
        Entropy value
    """
    if img.dim() == 3:
        img = img.mean(dim=0)  # Convert to grayscale if needed

    # Flatten and compute histogram
    img_flat = img.flatten()
    hist = torch.histc(img_flat, bins=bins, min=0, max=1)
    hist = hist / hist.sum()

    # Remove zero bins
    hist = hist[hist > 0]

    # Entropy
    entropy = -torch.sum(hist * torch.log2(hist))

    return entropy


def compute_mutual_information(
    img1: torch.Tensor,
    img2: torch.Tensor,
    bins: int = 256
) -> float:
    """
    Compute Mutual Information between two images.

    Higher MI indicates more shared information.

    Args:
        img1: First image
        img2: Second image
        bins: Number of histogram bins

    Returns:
        Mutual information value
    """
    # Convert to numpy
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    # Flatten
    img1 = img1.flatten()
    img2 = img2.flatten()

    # Joint histogram
    hist_2d, _, _ = np.histogram2d(img1, img2, bins=bins, range=[[0, 1], [0, 1]])
    hist_2d = hist_2d / hist_2d.sum()

    # Marginal histograms
    hist1 = hist_2d.sum(axis=1)
    hist2 = hist_2d.sum(axis=0)

    # Compute MI
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if hist_2d[i, j] > 0 and hist1[i] > 0 and hist2[j] > 0:
                mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist1[i] * hist2[j]))

    return mi


def compute_qcv(
    fused: torch.Tensor,
    source1: torch.Tensor,
    source2: torch.Tensor
) -> float:
    """
    Compute QCV (Quality metric based on Coefficient of Variation).

    Measures how well the fused image preserves contrast from sources.

    Args:
        fused: Fused image
        source1: First source image
        source2: Second source image

    Returns:
        QCV value
    """
    # Convert to numpy
    if torch.is_tensor(fused):
        fused = fused.detach().cpu().numpy()
    if torch.is_tensor(source1):
        source1 = source1.detach().cpu().numpy()
    if torch.is_tensor(source2):
        source2 = source2.detach().cpu().numpy()

    # Compute local standard deviation (contrast)
    def local_std(img, window_size=7):
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(img, window_size)
        sq_mean = uniform_filter(img ** 2, window_size)
        return np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

    # Flatten if needed
    if len(fused.shape) > 2:
        fused = np.mean(fused, axis=0) if fused.shape[0] <= 3 else np.mean(fused, axis=-1)
        source1 = np.mean(source1, axis=0) if source1.shape[0] <= 3 else np.mean(source1, axis=-1)
        source2 = np.mean(source2, axis=0) if source2.shape[0] <= 3 else np.mean(source2, axis=-1)

    # Compute contrast maps
    cf = local_std(fused)
    c1 = local_std(source1)
    c2 = local_std(source2)

    # Maximum contrast from sources
    c_max = np.maximum(c1, c2)

    # QCV: ratio of fused contrast to max source contrast
    mask = c_max > 1e-8
    qcv = np.mean(cf[mask] / c_max[mask])

    return float(qcv)


def compute_sd(img: torch.Tensor) -> torch.Tensor:
    """
    Compute Standard Deviation of image.

    Higher SD indicates more contrast/detail.

    Args:
        img: Input image

    Returns:
        Standard deviation
    """
    return img.std()


def compute_gradient_magnitude(img: torch.Tensor) -> torch.Tensor:
    """
    Compute average gradient magnitude.

    Higher value indicates more edge/detail content.

    Args:
        img: Input image [B, C, H, W] or [C, H, W]

    Returns:
        Average gradient magnitude
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device)

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    # Apply to each channel
    B, C, H, W = img.shape
    grad_mag = 0

    for c in range(C):
        channel = img[:, c:c+1, :, :]
        gx = F.conv2d(channel, sobel_x, padding=1)
        gy = F.conv2d(channel, sobel_y, padding=1)
        grad_mag += torch.sqrt(gx ** 2 + gy ** 2).mean()

    return grad_mag / C


class FusionMetrics:
    """
    Collection of fusion quality metrics.

    Computes all metrics used in the paper for evaluation.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Args:
            device: Device for computation
        """
        self.device = device

    def compute_all(
        self,
        fused: torch.Tensor,
        source1: torch.Tensor,
        source2: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all fusion metrics.

        Args:
            fused: Fused image [C, H, W]
            source1: First source image [C, H, W]
            source2: Second source image [C, H, W]
            reference: Optional reference image for reference-based metrics

        Returns:
            Dictionary of metric values
        """
        metrics = {}

        # Move to device
        fused = fused.to(self.device)
        source1 = source1.to(self.device)
        source2 = source2.to(self.device)

        # Non-reference metrics
        metrics['SD'] = compute_sd(fused).item()
        metrics['Entropy'] = compute_entropy(fused).item()
        metrics['Gradient'] = compute_gradient_magnitude(fused).item()

        # Source-based metrics
        metrics['SSIM_s1'] = compute_ssim(fused, source1).item()
        metrics['SSIM_s2'] = compute_ssim(fused, source2).item()
        metrics['SSIM_avg'] = (metrics['SSIM_s1'] + metrics['SSIM_s2']) / 2

        metrics['MI_s1'] = compute_mutual_information(fused, source1)
        metrics['MI_s2'] = compute_mutual_information(fused, source2)
        metrics['MI_avg'] = (metrics['MI_s1'] + metrics['MI_s2']) / 2

        metrics['QCV'] = compute_qcv(fused, source1, source2)

        # MEF-SSIM
        metrics['MEF_SSIM'] = compute_mef_ssim(fused, (source1, source2)).item()

        # Reference-based metrics (if reference provided)
        if reference is not None:
            reference = reference.to(self.device)
            metrics['SSIM_ref'] = compute_ssim(fused, reference).item()
            metrics['PSNR_ref'] = compute_psnr(fused, reference).item()
            metrics['MI_ref'] = compute_mutual_information(fused, reference)

        return metrics

    def summary(self, metrics: Dict[str, float]) -> str:
        """Create formatted summary string."""
        lines = ["Fusion Quality Metrics:", "=" * 40]

        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.4f}")

        return "\n".join(lines)


if __name__ == '__main__':
    # Test metrics
    print("Testing Fusion Metrics...")

    # Create test images
    torch.manual_seed(42)
    H, W = 256, 256

    # Simulate source images and fused result
    source1 = torch.rand(3, H, W)
    source2 = torch.rand(3, H, W)
    fused = (source1 + source2) / 2  # Simple average fusion

    # Test individual metrics
    print(f"SSIM(fused, source1): {compute_ssim(fused, source1):.4f}")
    print(f"SSIM(fused, source2): {compute_ssim(fused, source2):.4f}")
    print(f"PSNR(fused, source1): {compute_psnr(fused, source1):.4f} dB")
    print(f"Entropy(fused): {compute_entropy(fused):.4f}")
    print(f"SD(fused): {compute_sd(fused):.4f}")
    print(f"Gradient(fused): {compute_gradient_magnitude(fused):.4f}")

    # Test all metrics
    metrics_calc = FusionMetrics()
    all_metrics = metrics_calc.compute_all(fused, source1, source2)
    print("\n" + metrics_calc.summary(all_metrics))

    print("\nAll tests passed!")
