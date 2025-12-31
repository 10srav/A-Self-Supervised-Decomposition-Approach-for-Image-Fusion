"""
DeFusion Demo Training Script
=============================
Quick training demo using synthetic images (no COCO download required).
For full training, use train.py with actual COCO dataset.

This script demonstrates:
1. CUD augmentation
2. Loss computation
3. Training loop
4. Checkpoint saving
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from models.defusion import DeFusion
from utils.losses import CUDLoss


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for demo training.
    Generates random images with patterns for CUD training.
    """

    def __init__(self, num_samples=1000, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
        self.noise_std = 0.1

    def __len__(self):
        return self.num_samples

    def _generate_image(self):
        """Generate a random synthetic image with patterns."""
        size = self.image_size
        img = np.zeros((size, size, 3), dtype=np.float32)

        # Random background gradient
        direction = np.random.randint(4)
        for i in range(size):
            if direction == 0:  # Horizontal
                img[:, i, :] = i / size
            elif direction == 1:  # Vertical
                img[i, :, :] = i / size
            elif direction == 2:  # Diagonal
                img[:, :, 0] = np.linspace(0, 1, size).reshape(1, -1)
                img[:, :, 1] = np.linspace(0, 1, size).reshape(-1, 1)
            else:  # Random noise base
                img = np.random.rand(size, size, 3).astype(np.float32) * 0.5

        # Add random shapes
        num_shapes = np.random.randint(1, 5)
        for _ in range(num_shapes):
            shape_type = np.random.randint(3)
            color = np.random.rand(3)

            if shape_type == 0:  # Circle
                cx, cy = np.random.randint(20, size-20, 2)
                r = np.random.randint(10, 50)
                y, x = np.ogrid[:size, :size]
                mask = (x - cx)**2 + (y - cy)**2 < r**2
                img[mask] = color

            elif shape_type == 1:  # Rectangle
                x1, y1 = np.random.randint(0, size-40, 2)
                w, h = np.random.randint(20, 60, 2)
                img[y1:y1+h, x1:x1+w] = color

            else:  # Gradient overlay
                overlay = np.random.rand(size, size, 3).astype(np.float32) * 0.3
                img = img * 0.7 + overlay * 0.3

        return np.clip(img, 0, 1)

    def _generate_masks(self, size):
        """Generate non-overlapping masks."""
        m1 = np.zeros((size, size), dtype=np.float32)
        m2 = np.zeros((size, size), dtype=np.float32)

        # Random rectangles for m1
        num_rects = np.random.randint(2, 5)
        for _ in range(num_rects):
            x1, y1 = np.random.randint(0, size-30, 2)
            w, h = np.random.randint(20, 80, 2)
            x2, y2 = min(x1+w, size), min(y1+h, size)
            m1[y1:y2, x1:x2] = 1

        # m2 from complement of m1
        complement = 1 - m1
        num_rects = np.random.randint(2, 5)
        for _ in range(num_rects):
            x1, y1 = np.random.randint(0, size-30, 2)
            w, h = np.random.randint(20, 80, 2)
            x2, y2 = min(x1+w, size), min(y1+h, size)
            m2[y1:y2, x1:x2] = 1

        # Ensure non-overlapping
        m2 = m2 * complement

        return m1, m2

    def __getitem__(self, idx):
        # Generate random image
        img = self._generate_image()

        # Generate masks
        m1, m2 = self._generate_masks(self.image_size)

        # Apply CUD augmentation
        noise1 = np.random.randn(*img.shape).astype(np.float32) * self.noise_std
        noise2 = np.random.randn(*img.shape).astype(np.float32) * self.noise_std

        m1_exp = np.expand_dims(m1, -1)
        m2_exp = np.expand_dims(m2, -1)

        x1 = m1_exp * img + (1 - m1_exp) * noise1
        x2 = m2_exp * img + (1 - m2_exp) * noise2

        # Convert to tensors [C, H, W] and normalize to [-1, 1]
        x = torch.from_numpy(img.transpose(2, 0, 1)) * 2 - 1
        x1 = torch.from_numpy(x1.transpose(2, 0, 1).astype(np.float32)) * 2 - 1
        x2 = torch.from_numpy(x2.transpose(2, 0, 1).astype(np.float32)) * 2 - 1
        m1 = torch.from_numpy(m1)
        m2 = torch.from_numpy(m2)

        # Compute derived masks
        m_common = m1 * m2
        m1_unique = m1 * (1 - m2)
        m2_unique = m2 * (1 - m1)

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


def train(args):
    print("=" * 60)
    print(" DeFusion Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print("\n[1/5] Creating model...")
    model = DeFusion().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Parameters: {num_params:,}")

    # Create dataset
    print("\n[2/5] Creating dataset...")
    dataset = SyntheticDataset(num_samples=args.num_samples, image_size=256)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    print(f"       Samples: {len(dataset)}")
    print(f"       Batches: {len(dataloader)}")

    # Loss and optimizer
    print("\n[3/5] Setting up training...")
    criterion = CUDLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    print(f"       Optimizer: Adam (lr={args.lr})")
    print(f"       LR Schedule: Halve every {args.lr_step} epochs")

    # Training loop
    print("\n[4/5] Training...")
    print("-" * 60)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_losses = {'common': 0, 'u1': 0, 'u2': 0, 'recon': 0}

        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            x = batch['x'].to(device)
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            m_common = batch['m_common'].to(device)
            m1_unique = batch['m1_unique'].to(device)
            m2_unique = batch['m2_unique'].to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model.forward_train(x1, x2)

            # Compute loss
            predictions = {
                'xc': outputs['xc'],
                'x1u': outputs['x1u'],
                'x2u': outputs['x2u'],
                'xr': outputs['xr']
            }
            targets = {
                'x': x,
                'm_common': m_common,
                'm1_unique': m1_unique,
                'm2_unique': m2_unique
            }

            losses = criterion(predictions, targets)

            # Backward
            losses['loss'].backward()
            optimizer.step()

            # Accumulate
            epoch_loss += losses['loss'].item()
            epoch_losses['common'] += losses['loss_common'].item()
            epoch_losses['u1'] += losses['loss_u1'].item()
            epoch_losses['u2'] += losses['loss_u2'].item()
            epoch_losses['recon'] += losses['loss_recon'].item()

        # Average losses
        num_batches = len(dataloader)
        epoch_loss /= num_batches
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"C: {epoch_losses['common']:.4f} | "
              f"U1: {epoch_losses['u1']:.4f} | "
              f"U2: {epoch_losses['u2']:.4f} | "
              f"R: {epoch_losses['recon']:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, output_dir / 'best_model.pth')

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, output_dir / f'checkpoint_epoch{epoch+1:03d}.pth')

        scheduler.step()

    # Save final model
    print("-" * 60)
    print("\n[5/5] Saving final model...")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss
    }, output_dir / 'final_model.pth')

    print(f"       Saved to: {output_dir}")
    print(f"       Best loss: {best_loss:.4f}")

    # Test fusion with trained model
    print("\n" + "=" * 60)
    print(" Testing Trained Model")
    print("=" * 60)

    model.eval()

    # Create test images
    img1 = np.zeros((256, 256, 3), dtype=np.float32)
    img2 = np.zeros((256, 256, 3), dtype=np.float32)

    for i in range(256):
        img1[:, i, 0] = i / 255
        img1[:, i, 2] = 1 - i / 255
        img2[i, :, 1] = i / 255

    y, x = np.ogrid[:256, :256]
    mask = (x - 128)**2 + (y - 128)**2 < 50**2
    img2[mask] = [1, 1, 0]

    # Fuse
    t1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).to(device) * 2 - 1
    t2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).to(device) * 2 - 1

    with torch.no_grad():
        fused, _, _, _ = model(t1, t2)

    # Save results
    fused_np = ((fused.squeeze(0).permute(1, 2, 0).cpu() + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)

    demo_dir = Path('demo_output')
    demo_dir.mkdir(exist_ok=True)

    Image.fromarray(img1_uint8).save(demo_dir / 'trained_source1.png')
    Image.fromarray(img2_uint8).save(demo_dir / 'trained_source2.png')
    Image.fromarray(fused_np).save(demo_dir / 'trained_fused.png')

    print(f"\nResults saved to {demo_dir}:")
    print("  - trained_source1.png")
    print("  - trained_source2.png")
    print("  - trained_fused.png")

    print("\n" + "=" * 60)
    print(" Training Complete!")
    print("=" * 60)
    print(f"\nTo use the trained model:")
    print(f"  python test_fusion.py --checkpoint {output_dir}/best_model.pth --i1 img1.png --i2 img2.png")

    return model


def main():
    parser = argparse.ArgumentParser(description='DeFusion Demo Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=5, help='LR decay step')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of synthetic samples')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
