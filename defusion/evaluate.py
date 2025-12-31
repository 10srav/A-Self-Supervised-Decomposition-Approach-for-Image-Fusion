"""
DeFusion Evaluation Script
==========================
Evaluate fusion quality using standard metrics.

Paper Section 4 (Experiments):
Evaluation on:
- Multi-exposure: MEFB, SICE datasets
- Multi-focus: Real-MFF dataset
- IR-Visible: TNO, RoadScene datasets

Metrics: CE, QCV, SSIM, MEF-SSIM, SD, VIF, MI

Usage:
    python evaluate.py --checkpoint model.pth --dataset_dir ./test_data --task ir_vis
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import DeFusion, build_defusion
from utils.metrics import (
    FusionMetrics,
    compute_ssim,
    compute_psnr,
    compute_mef_ssim,
    compute_entropy,
    compute_sd,
    compute_mutual_information,
    compute_qcv,
    compute_gradient_magnitude
)


def load_image(path: str, device: str = 'cpu') -> torch.Tensor:
    """Load image as tensor normalized to [-1, 1]."""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).to(device)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


class FusionEvaluator:
    """
    Evaluator for fusion quality assessment.

    Computes all metrics used in the paper across multiple test pairs.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.metrics = FusionMetrics(device)

    def evaluate_pair(
        self,
        fused: torch.Tensor,
        source1: torch.Tensor,
        source2: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single fusion result.

        Args:
            fused: Fused image [C, H, W] or [1, C, H, W]
            source1: First source [C, H, W]
            source2: Second source [C, H, W]
            reference: Optional ground truth reference

        Returns:
            Dictionary of metric values
        """
        # Remove batch dimension if present
        if fused.dim() == 4:
            fused = fused[0]
        if source1.dim() == 4:
            source1 = source1[0]
        if source2.dim() == 4:
            source2 = source2[0]

        # Denormalize to [0, 1]
        fused = denormalize(fused)
        source1 = denormalize(source1)
        source2 = denormalize(source2)

        return self.metrics.compute_all(fused, source1, source2, reference)

    def evaluate_dataset(
        self,
        model: DeFusion,
        dataset_dir: str,
        task: str = 'ir_vis',
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            model: DeFusion model
            dataset_dir: Path to test dataset
            task: Fusion task type
            output_dir: Optional directory to save fused results

        Returns:
            Average metrics across all test pairs
        """
        dataset_dir = Path(dataset_dir)
        model.eval()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Collect metrics for all pairs
        all_metrics = defaultdict(list)
        results = []

        # Get image pairs based on task
        pairs = self._get_pairs(dataset_dir, task)
        print(f"Found {len(pairs)} image pairs for evaluation")

        for pair_name, (i1_path, i2_path) in pairs:
            print(f"Evaluating: {pair_name}")

            # Load images
            i1 = load_image(str(i1_path), self.device).unsqueeze(0)
            i2 = load_image(str(i2_path), self.device).unsqueeze(0)

            # Ensure same size
            if i1.shape != i2.shape:
                import torch.nn.functional as F
                i2 = F.interpolate(i2, size=i1.shape[2:], mode='bilinear', align_corners=False)

            # Fuse
            with torch.no_grad():
                fused = model.forward_fusion(i1, i2)

            # Evaluate
            metrics = self.evaluate_pair(fused, i1, i2)

            # Store results
            for key, value in metrics.items():
                all_metrics[key].append(value)

            results.append({
                'name': pair_name,
                **metrics
            })

            # Save fused image if requested
            if output_dir:
                fused_np = denormalize(fused[0]).cpu().permute(1, 2, 0).numpy()
                fused_np = (fused_np * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(fused_np).save(output_dir / f"{pair_name}.png")

        # Compute averages
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        std_metrics = {f"{key}_std": np.std(values) for key, values in all_metrics.items()}

        return avg_metrics, std_metrics, results

    def _get_pairs(self, dataset_dir: Path, task: str) -> List:
        """Get list of image pairs from dataset directory."""
        pairs = []

        if task == 'ir_vis':
            ir_dir = dataset_dir / 'ir'
            vis_dir = dataset_dir / 'vis'

            if ir_dir.exists() and vis_dir.exists():
                for ir_path in sorted(ir_dir.glob('*')):
                    if ir_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
                        vis_path = vis_dir / ir_path.name
                        if not vis_path.exists():
                            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                                alt = vis_dir / (ir_path.stem + ext)
                                if alt.exists():
                                    vis_path = alt
                                    break

                        if vis_path.exists():
                            pairs.append((ir_path.stem, (ir_path, vis_path)))

        elif task in ['multi_focus', 'multi_exposure']:
            for scene_dir in sorted(dataset_dir.iterdir()):
                if scene_dir.is_dir():
                    images = sorted([
                        p for p in scene_dir.glob('*')
                        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
                    ])
                    if len(images) >= 2:
                        pairs.append((scene_dir.name, (images[0], images[-1])))

        else:
            # Generic pairing
            images = sorted([
                p for p in dataset_dir.glob('*')
                if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
            ])
            for i in range(0, len(images) - 1, 2):
                name = f"{images[i].stem}_{images[i+1].stem}"
                pairs.append((name, (images[i], images[i+1])))

        return pairs


def print_results(avg_metrics: Dict, std_metrics: Dict, title: str = "Evaluation Results"):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

    # Group metrics
    structural = ['SSIM_s1', 'SSIM_s2', 'SSIM_avg', 'MEF_SSIM']
    information = ['Entropy', 'MI_s1', 'MI_s2', 'MI_avg', 'QCV']
    quality = ['SD', 'Gradient']

    print("\nStructural Metrics:")
    print("-" * 40)
    for key in structural:
        if key in avg_metrics:
            std_key = f"{key}_std"
            std = std_metrics.get(std_key, 0)
            print(f"  {key:15s}: {avg_metrics[key]:.4f} ± {std:.4f}")

    print("\nInformation Metrics:")
    print("-" * 40)
    for key in information:
        if key in avg_metrics:
            std_key = f"{key}_std"
            std = std_metrics.get(std_key, 0)
            print(f"  {key:15s}: {avg_metrics[key]:.4f} ± {std:.4f}")

    print("\nQuality Metrics:")
    print("-" * 40)
    for key in quality:
        if key in avg_metrics:
            std_key = f"{key}_std"
            std = std_metrics.get(std_key, 0)
            print(f"  {key:15s}: {avg_metrics[key]:.4f} ± {std:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeFusion Model')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Model feature dimension')

    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--task', type=str, default='ir_vis',
                        choices=['ir_vis', 'multi_focus', 'multi_exposure', 'generic'],
                        help='Fusion task type')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save fused images')
    parser.add_argument('--results_file', type=str, default='evaluation_results.json',
                        help='Path to save detailed results')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = build_defusion(
        config={'feature_dim': args.feature_dim},
        pretrained=args.checkpoint
    ).to(device)
    model.eval()

    # Create evaluator
    evaluator = FusionEvaluator(device=device)

    # Evaluate
    print(f"\nEvaluating on {args.dataset_dir}")
    print(f"Task: {args.task}")

    avg_metrics, std_metrics, results = evaluator.evaluate_dataset(
        model,
        args.dataset_dir,
        task=args.task,
        output_dir=args.output_dir
    )

    # Print results
    print_results(avg_metrics, std_metrics, f"DeFusion - {args.task}")

    # Save detailed results
    output = {
        'task': args.task,
        'dataset': args.dataset_dir,
        'checkpoint': args.checkpoint,
        'average_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'per_image_results': results
    }

    with open(args.results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to {args.results_file}")

    if args.output_dir:
        print(f"Fused images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
