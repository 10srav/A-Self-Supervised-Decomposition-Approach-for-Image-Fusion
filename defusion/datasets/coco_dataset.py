"""
COCO Dataset for DeFusion Training
==================================
Dataset loader for COCO images with CUD augmentation.

Paper Section 3.3 (Training Details):
"Dataset: COCO (50k images, random crop/resize to 256×256)"
"Data aug: Random crop/resize 256×256"
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from .cud_augmentation import CUDAugmentation


class COCODataset(Dataset):
    """
    COCO Dataset with CUD Augmentation for DeFusion training.

    Loads images from COCO dataset directory and applies:
    1. Random crop/resize to 256x256
    2. CUD augmentation (non-overlapping masks + noise)

    Paper Section 3.3:
    "Dataset: COCO (50k images, random crop/resize to 256×256)"
    """

    def __init__(
        self,
        root: str,
        split: str = 'train2017',
        image_size: int = 256,
        max_images: Optional[int] = 50000,
        cud_augmentation: Optional[CUDAugmentation] = None,
        transform: Optional[Callable] = None,
        noise_std: float = 0.1
    ):
        """
        Args:
            root: Root directory containing COCO images
            split: COCO split to use ('train2017', 'val2017', 'unlabeled2017')
            image_size: Target image size (256 per paper)
            max_images: Maximum number of images to use (50k per paper)
            cud_augmentation: CUD augmentation instance (created if None)
            transform: Additional transforms to apply
            noise_std: Noise standard deviation for CUD (0.1 per paper)
        """
        self.root = Path(root)
        self.image_size = image_size

        # Find image directory
        self.image_dir = self.root / split
        if not self.image_dir.exists():
            # Try without split suffix
            self.image_dir = self.root
            if not self.image_dir.exists():
                raise ValueError(f"Image directory not found: {self.root}")

        # Collect image paths
        self.image_paths = self._collect_images(max_images)
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")

        # CUD augmentation
        self.cud = cud_augmentation or CUDAugmentation(
            noise_std=noise_std,
            mask_method='random_rects'
        )

        # Image transforms
        self.transform = transform or transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalize to [-1, 1] for consistency with Tanh output
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _collect_images(self, max_images: Optional[int]) -> List[Path]:
        """Collect image paths from directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = []

        for path in self.image_dir.iterdir():
            if path.suffix.lower() in extensions:
                image_paths.append(path)

        # Sort for reproducibility
        image_paths.sort()

        # Limit number of images
        if max_images is not None and len(image_paths) > max_images:
            # Random sample for variety
            random.seed(42)  # Reproducible
            image_paths = random.sample(image_paths, max_images)
            image_paths.sort()

        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
                - x: Original image [3, H, W] (normalized to [-1, 1])
                - x1: First CUD augmented view [3, H, W]
                - x2: Second CUD augmented view [3, H, W]
                - m1: First mask [H, W]
                - m2: Second mask [H, W]
                - m_common: Common mask (M1 ∩ M2) [H, W]
                - m1_unique: Unique to first (M1 - M2) [H, W]
                - m2_unique: Unique to second (M2 - M1) [H, W]
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        x = self.transform(image)

        # Apply CUD augmentation
        cud_output = self.cud(x)

        return cud_output


class FusionTestDataset(Dataset):
    """
    Dataset for fusion evaluation.

    Loads pairs of source images for fusion testing.
    Supports multi-exposure, multi-focus, and IR-visible pairs.
    """

    def __init__(
        self,
        root: str,
        task: str = 'ir_vis',
        image_size: Optional[int] = None,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            root: Root directory containing image pairs
            task: Fusion task type ('ir_vis', 'multi_exposure', 'multi_focus')
            image_size: Target image size (None for original size)
            transform: Optional transforms
        """
        self.root = Path(root)
        self.task = task
        self.image_size = image_size

        # Collect image pairs
        self.pairs = self._collect_pairs()
        print(f"Found {len(self.pairs)} image pairs for {task}")

        # Transform
        transforms_list = []
        if image_size is not None:
            transforms_list.append(transforms.Resize((image_size, image_size)))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transform = transform or transforms.Compose(transforms_list)

    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        """Collect source image pairs."""
        pairs = []

        if self.task == 'ir_vis':
            # IR-Visible: expect folders like 'ir/' and 'vis/'
            ir_dir = self.root / 'ir'
            vis_dir = self.root / 'vis'

            if ir_dir.exists() and vis_dir.exists():
                ir_images = sorted(ir_dir.glob('*'))
                vis_images = sorted(vis_dir.glob('*'))

                # Match by filename
                for ir_path in ir_images:
                    vis_path = vis_dir / ir_path.name
                    if vis_path.exists():
                        pairs.append((ir_path, vis_path))

        elif self.task in ['multi_exposure', 'multi_focus']:
            # Multi-exposure/focus: expect folders with numbered images
            for subdir in self.root.iterdir():
                if subdir.is_dir():
                    images = sorted(subdir.glob('*'))
                    if len(images) >= 2:
                        # Pair first and last (or two extremes)
                        pairs.append((images[0], images[-1]))

        else:
            # Generic: look for paired naming convention
            extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [p for p in self.root.glob('*') if p.suffix.lower() in extensions]
            images.sort()

            # Pair consecutive images
            for i in range(0, len(images) - 1, 2):
                pairs.append((images[i], images[i + 1]))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a test sample.

        Returns:
            Dictionary with:
                - i1: First source image [3, H, W]
                - i2: Second source image [3, H, W]
                - name: Image pair identifier
        """
        path1, path2 = self.pairs[idx]

        i1 = Image.open(path1).convert('RGB')
        i2 = Image.open(path2).convert('RGB')

        # Ensure same size
        if i1.size != i2.size:
            min_w = min(i1.width, i2.width)
            min_h = min(i1.height, i2.height)
            i1 = i1.resize((min_w, min_h), Image.BILINEAR)
            i2 = i2.resize((min_w, min_h), Image.BILINEAR)

        i1 = self.transform(i1)
        i2 = self.transform(i2)

        return {
            'i1': i1,
            'i2': i2,
            'name': path1.stem
        }


def get_coco_dataloader(
    root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = 'train2017',
    max_images: int = 50000,
    image_size: int = 256,
    noise_std: float = 0.1,
    shuffle: bool = True
) -> DataLoader:
    """
    Create COCO dataloader for DeFusion training.

    Paper Section 3.3:
    "Batch size: 8"
    "Dataset: COCO (50k images)"

    Args:
        root: Path to COCO images directory
        batch_size: Batch size (8 per paper)
        num_workers: Number of data loading workers
        split: COCO split name
        max_images: Maximum images to use (50k per paper)
        image_size: Target image size (256 per paper)
        noise_std: Noise std for CUD (0.1 per paper)
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = COCODataset(
        root=root,
        split=split,
        image_size=image_size,
        max_images=max_images,
        noise_std=noise_std
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def get_fusion_dataloader(
    root: str,
    task: str = 'ir_vis',
    batch_size: int = 1,
    image_size: Optional[int] = None,
    num_workers: int = 2
) -> DataLoader:
    """
    Create dataloader for fusion evaluation.

    Args:
        root: Path to test images
        task: Fusion task type
        batch_size: Batch size (typically 1 for evaluation)
        image_size: Target size (None for original)
        num_workers: Number of workers

    Returns:
        DataLoader instance
    """
    dataset = FusionTestDataset(
        root=root,
        task=task,
        image_size=image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == '__main__':
    # Test dataset
    print("Testing COCO Dataset...")

    # Create dummy dataset for testing
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        for i in range(10):
            img = Image.new('RGB', (512, 512), color=(i * 25, i * 20, i * 15))
            img.save(os.path.join(tmpdir, f'image_{i:04d}.jpg'))

        # Test dataset
        dataset = COCODataset(
            root=tmpdir,
            split='',
            image_size=256,
            max_images=None
        )

        print(f"Dataset length: {len(dataset)}")

        # Get sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"x shape: {sample['x'].shape}")
        print(f"x1 shape: {sample['x1'].shape}")
        print(f"m1 shape: {sample['m1'].shape}")

        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        print(f"\nBatch x shape: {batch['x'].shape}")
        print(f"Batch x1 shape: {batch['x1'].shape}")
        print(f"Batch m1 shape: {batch['m1'].shape}")

    print("\nAll tests passed!")
