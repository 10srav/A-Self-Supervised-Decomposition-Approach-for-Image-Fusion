"""
DeFusion Inference Script
=========================
Apply trained DeFusion model to fuse images.

Paper Section 3.2 (Inference Pipeline, Fig 3):
"Given source images I1, I2:
 DeNet(I1,I2) → fc, f1u, f2u
 fused = Pr(concat(fc, f1u, f2u))"

Usage:
    python test_fusion.py --checkpoint model.pth --i1 image1.png --i2 image2.png

For batch processing:
    python test_fusion.py --checkpoint model.pth --input_dir ./test_images --output_dir ./results
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import DeFusion, build_defusion


def load_image(
    path: str,
    size: Optional[Tuple[int, int]] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load and preprocess image.

    Args:
        path: Image path
        size: Optional resize dimensions (H, W)
        device: Target device

    Returns:
        Preprocessed image tensor [1, 3, H, W] in [-1, 1]
    """
    img = Image.open(path).convert('RGB')

    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform = transforms.Compose(transform_list)
    img_tensor = transform(img).unsqueeze(0).to(device)

    return img_tensor


def save_image(
    tensor: torch.Tensor,
    path: str,
    denormalize: bool = True
):
    """
    Save tensor as image.

    Args:
        tensor: Image tensor [1, 3, H, W] or [3, H, W]
        path: Output path
        denormalize: Whether to denormalize from [-1, 1] to [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu()

    if denormalize:
        tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)

    # Convert to PIL
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)

    print(f"Saved fused image to {path}")


def fuse_single_pair(
    model: DeFusion,
    i1_path: str,
    i2_path: str,
    output_path: str,
    size: Optional[Tuple[int, int]] = None,
    device: str = 'cpu',
    save_features: bool = False
):
    """
    Fuse a single pair of images.

    Args:
        model: DeFusion model
        i1_path: Path to first source image
        i2_path: Path to second source image
        output_path: Path for fused output
        size: Optional resize dimensions
        device: Device to use
        save_features: Whether to save feature visualizations
    """
    # Load images
    i1 = load_image(i1_path, size, device)
    i2 = load_image(i2_path, size, device)

    # Ensure same size
    if i1.shape != i2.shape:
        # Resize i2 to match i1
        i2 = F.interpolate(i2, size=i1.shape[2:], mode='bilinear', align_corners=False)

    # Fuse
    with torch.no_grad():
        fused = model.forward_fusion(i1, i2)

    # Save result
    save_image(fused, output_path)

    # Optionally save feature visualizations
    if save_features:
        from utils.visualization import visualize_features

        fc, f1u, f2u = model.get_features(i1, i2)

        output_dir = Path(output_path).parent
        feat_path = output_dir / f"{Path(output_path).stem}_features.png"

        visualize_features(fc, f1u, f2u, save_path=str(feat_path))


def fuse_directory(
    model: DeFusion,
    input_dir: str,
    output_dir: str,
    task: str = 'ir_vis',
    size: Optional[Tuple[int, int]] = None,
    device: str = 'cpu'
):
    """
    Fuse all image pairs in a directory.

    Supports different fusion task directory structures:
    - ir_vis: input_dir/ir/ and input_dir/vis/
    - multi_focus: input_dir/{scene}/img1.png, img2.png
    - multi_exposure: input_dir/{scene}/exp1.png, exp2.png, ...

    Args:
        model: DeFusion model
        input_dir: Input directory
        output_dir: Output directory
        task: Fusion task type
        size: Optional resize dimensions
        device: Device to use
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if task == 'ir_vis':
        # IR-Visible fusion
        ir_dir = input_dir / 'ir'
        vis_dir = input_dir / 'vis'

        if not ir_dir.exists() or not vis_dir.exists():
            print(f"Expected 'ir' and 'vis' subdirectories in {input_dir}")
            return

        ir_images = sorted(ir_dir.glob('*'))

        for ir_path in ir_images:
            vis_path = vis_dir / ir_path.name

            if not vis_path.exists():
                # Try different extensions
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    alt_path = vis_dir / (ir_path.stem + ext)
                    if alt_path.exists():
                        vis_path = alt_path
                        break

            if vis_path.exists():
                output_path = output_dir / ir_path.name
                print(f"Fusing: {ir_path.name}")

                fuse_single_pair(model, str(ir_path), str(vis_path), str(output_path), size, device)
            else:
                print(f"Warning: No matching visible image for {ir_path.name}")

    elif task in ['multi_focus', 'multi_exposure']:
        # Multi-focus or multi-exposure: each subdirectory is a scene
        for scene_dir in input_dir.iterdir():
            if scene_dir.is_dir():
                images = sorted(scene_dir.glob('*'))
                images = [p for p in images if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]

                if len(images) >= 2:
                    # Fuse first and last (or first two)
                    i1_path = images[0]
                    i2_path = images[-1] if len(images) > 2 else images[1]

                    output_path = output_dir / f"{scene_dir.name}.png"
                    print(f"Fusing scene: {scene_dir.name}")

                    fuse_single_pair(model, str(i1_path), str(i2_path), str(output_path), size, device)

    else:
        # Generic: look for pairs of images
        images = sorted(input_dir.glob('*'))
        images = [p for p in images if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]

        for i in range(0, len(images) - 1, 2):
            i1_path = images[i]
            i2_path = images[i + 1]

            output_path = output_dir / f"fused_{i1_path.stem}_{i2_path.stem}.png"
            print(f"Fusing: {i1_path.name} + {i2_path.name}")

            fuse_single_pair(model, str(i1_path), str(i2_path), str(output_path), size, device)


def main():
    parser = argparse.ArgumentParser(description='DeFusion Image Fusion')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Model feature dimension')

    # Single pair mode
    parser.add_argument('--i1', '--ir', type=str, default=None,
                        help='Path to first source image (or IR image)')
    parser.add_argument('--i2', '--vis', type=str, default=None,
                        help='Path to second source image (or visible image)')
    parser.add_argument('--output', '-o', type=str, default='fused.png',
                        help='Output path for fused image')

    # Batch mode
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str, default='./fused_results',
                        help='Output directory for batch processing')
    parser.add_argument('--task', type=str, default='ir_vis',
                        choices=['ir_vis', 'multi_focus', 'multi_exposure', 'generic'],
                        help='Fusion task type')

    # Processing options
    parser.add_argument('--size', type=int, nargs=2, default=None,
                        help='Resize images to this size (H W)')
    parser.add_argument('--save_features', action='store_true',
                        help='Save feature visualizations')

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

    # Process size argument
    size = tuple(args.size) if args.size else None

    # Single pair or batch mode
    if args.i1 and args.i2:
        # Single pair mode
        print(f"Fusing: {args.i1} + {args.i2}")
        fuse_single_pair(
            model, args.i1, args.i2, args.output,
            size=size, device=device, save_features=args.save_features
        )

    elif args.input_dir:
        # Batch mode
        print(f"Processing directory: {args.input_dir}")
        fuse_directory(
            model, args.input_dir, args.output_dir,
            task=args.task, size=size, device=device
        )

    else:
        parser.print_help()
        print("\nPlease specify either:")
        print("  --i1 image1.png --i2 image2.png  (single pair mode)")
        print("  --input_dir ./images             (batch mode)")


if __name__ == '__main__':
    main()
