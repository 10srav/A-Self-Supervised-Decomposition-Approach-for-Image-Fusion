"""
COCO Dataset Preparation for DeFusion
=====================================
Downloads and prepares COCO images for CUD training.

Usage:
    python prepare_coco.py --output ./data/coco_50k --num_images 50000
    python prepare_coco.py --coco_path /existing/coco --output ./data/coco_50k
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Install tqdm for progress bars: pip install tqdm")


def download_coco_sample(output_dir: Path, num_images: int = 1000):
    """
    Download sample COCO images using direct URLs.
    For full dataset, download from: http://images.cocodataset.org/zips/train2017.zip
    """
    print(f"Generating {num_images} synthetic training images...")
    print("(For real training, download COCO from http://images.cocodataset.org/zips/train2017.zip)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic images for demo/testing
    for i in range(num_images):
        # Create varied synthetic images
        size = 256
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Random background
        bg_type = random.randint(0, 3)
        if bg_type == 0:  # Gradient
            for x in range(size):
                img[:, x] = [int(x * 255 / size)] * 3
        elif bg_type == 1:  # Vertical gradient
            for y in range(size):
                img[y, :] = [int(y * 255 / size)] * 3
        elif bg_type == 2:  # Random color
            img[:] = [random.randint(50, 200) for _ in range(3)]
        else:  # Noise
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

        # Add random shapes
        num_shapes = random.randint(2, 6)
        for _ in range(num_shapes):
            shape_type = random.randint(0, 2)
            color = [random.randint(0, 255) for _ in range(3)]

            if shape_type == 0:  # Circle
                cx, cy = random.randint(30, size-30), random.randint(30, size-30)
                r = random.randint(15, 50)
                y, x = np.ogrid[:size, :size]
                mask = (x - cx)**2 + (y - cy)**2 < r**2
                img[mask] = color
            elif shape_type == 1:  # Rectangle
                x1, y1 = random.randint(0, size-50), random.randint(0, size-50)
                w, h = random.randint(20, 80), random.randint(20, 80)
                img[y1:y1+h, x1:x1+w] = color
            else:  # Triangle (approximated)
                x1, y1 = random.randint(20, size-20), random.randint(20, size-20)
                for dy in range(30):
                    x_start = x1 - dy // 2
                    x_end = x1 + dy // 2
                    if 0 <= y1 + dy < size:
                        img[y1+dy, max(0,x_start):min(size,x_end)] = color

        # Save
        Image.fromarray(img).save(output_dir / f"synthetic_{i:06d}.jpg", quality=95)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_images} images")

    print(f"Created {num_images} synthetic images in {output_dir}")
    return output_dir


def prepare_coco_images(
    coco_path: Path,
    output_dir: Path,
    num_images: int = 50000,
    image_size: int = 256,
    num_workers: int = 4
):
    """
    Prepare COCO images: resize and crop to fixed size.

    Args:
        coco_path: Path to COCO train2017 folder
        output_dir: Output directory for processed images
        num_images: Number of images to process
        image_size: Target image size (256 for DeFusion)
        num_workers: Number of parallel workers
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image list
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = []

    for ext in image_extensions:
        all_images.extend(coco_path.glob(f'*{ext}'))
        all_images.extend(coco_path.glob(f'*{ext.upper()}'))

    if not all_images:
        print(f"No images found in {coco_path}")
        print("Generating synthetic images instead...")
        return download_coco_sample(output_dir, num_images)

    # Shuffle and limit
    random.shuffle(all_images)
    all_images = all_images[:num_images]

    print(f"Processing {len(all_images)} images from {coco_path}")
    print(f"Output: {output_dir}")

    def process_image(img_path):
        try:
            img = Image.open(img_path).convert('RGB')

            # Center crop to square
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

            # Resize
            img = img.resize((image_size, image_size), Image.BILINEAR)

            # Save
            output_path = output_dir / f"{img_path.stem}.jpg"
            img.save(output_path, quality=95)
            return True
        except Exception as e:
            return False

    # Process images
    success_count = 0

    if HAS_TQDM:
        iterator = tqdm(all_images, desc="Processing")
    else:
        iterator = all_images

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_image, img): img for img in all_images}

        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1

            if not HAS_TQDM and (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(all_images)} images")

    print(f"\nProcessed {success_count}/{len(all_images)} images successfully")
    print(f"Output directory: {output_dir}")

    return output_dir


def verify_dataset(data_path: Path, min_images: int = 100):
    """Verify dataset is ready for training."""
    image_files = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))

    print(f"\nDataset Verification:")
    print(f"  Path: {data_path}")
    print(f"  Images found: {len(image_files)}")

    if len(image_files) < min_images:
        print(f"  WARNING: Less than {min_images} images found!")
        return False

    # Check a random sample
    sample = random.sample(image_files, min(10, len(image_files)))
    sizes = []
    for img_path in sample:
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
        except:
            print(f"  ERROR: Cannot open {img_path}")
            return False

    print(f"  Sample sizes: {sizes[:3]}...")
    print(f"  Status: READY FOR TRAINING")
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare COCO dataset for DeFusion')
    parser.add_argument('--coco_path', type=str, default=None,
                        help='Path to COCO train2017 images')
    parser.add_argument('--output', type=str, default='./data/coco_50k',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=50000,
                        help='Number of images to process')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Target image size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic images (for testing)')

    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print(" DeFusion Dataset Preparation")
    print("=" * 60)

    if args.synthetic or args.coco_path is None:
        # Generate synthetic images for testing
        download_coco_sample(output_dir, args.num_images)
    else:
        # Process real COCO images
        coco_path = Path(args.coco_path)
        if not coco_path.exists():
            print(f"ERROR: COCO path not found: {coco_path}")
            print("Generating synthetic images instead...")
            download_coco_sample(output_dir, args.num_images)
        else:
            prepare_coco_images(
                coco_path=coco_path,
                output_dir=output_dir,
                num_images=args.num_images,
                image_size=args.image_size,
                num_workers=args.workers
            )

    # Verify
    verify_dataset(output_dir)

    print("\n" + "=" * 60)
    print(" Next Steps:")
    print("=" * 60)
    print(f"\n  python train.py --data_path {output_dir} --epochs 50")
    print("\n  Or quick test:")
    print(f"  python train_demo.py --epochs 5\n")


if __name__ == '__main__':
    main()
