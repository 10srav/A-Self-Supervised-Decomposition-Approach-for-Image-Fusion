"""
DeFusion Training Script
========================
Training script for the DeFusion model using CUD pretext task.

Paper Section 3.3 (Training Details):
- Dataset: COCO (50k images, random crop/resize to 256x256)
- Optimizer: Adam (lr=1e-3, halve every 10 epochs)
- Epochs: 50
- Batch size: 8

Usage:
    python train.py --coco_path /path/to/coco/train2017

Example:
    python train.py --coco_path ./data/coco --epochs 50 --batch_size 8
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import DeFusion
from datasets import COCODataset, get_coco_dataloader
from utils.losses import CUDLoss
from utils.visualization import visualize_cud_training, plot_training_curves


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    save_path: Path
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    checkpoint_path: Path
) -> int:
    """Load training checkpoint. Returns starting epoch."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'] + 1


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary of average losses
    """
    model.train()

    total_loss = 0
    loss_components = {'loss_common': 0, 'loss_u1': 0, 'loss_u2': 0, 'loss_recon': 0}
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        x = batch['x'].to(device)
        x1 = batch['x1'].to(device)
        x2 = batch['x2'].to(device)
        m_common = batch['m_common'].to(device)
        m1_unique = batch['m1_unique'].to(device)
        m2_unique = batch['m2_unique'].to(device)

        # Forward pass
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

        # Backward pass
        losses['loss'].backward()
        optimizer.step()

        # Accumulate losses
        total_loss += losses['loss'].item()
        for key in loss_components:
            loss_components[key] += losses[key].item()

        # Logging
        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                f"Loss: {losses['loss'].item():.4f} "
                f"(C:{losses['loss_common'].item():.4f} "
                f"U1:{losses['loss_u1'].item():.4f} "
                f"U2:{losses['loss_u2'].item():.4f} "
                f"R:{losses['loss_recon'].item():.4f})"
            )

        # TensorBoard logging
        if writer is not None:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/train', losses['loss'].item(), global_step)
            writer.add_scalar('Loss/common', losses['loss_common'].item(), global_step)
            writer.add_scalar('Loss/unique1', losses['loss_u1'].item(), global_step)
            writer.add_scalar('Loss/unique2', losses['loss_u2'].item(), global_step)
            writer.add_scalar('Loss/recon', losses['loss_recon'].item(), global_step)

    # Average losses
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {'loss': avg_loss, **loss_components}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model.

    Returns:
        Dictionary of average losses
    """
    model.eval()

    total_loss = 0
    loss_components = {'loss_common': 0, 'loss_u1': 0, 'loss_u2': 0, 'loss_recon': 0}
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            m_common = batch['m_common'].to(device)
            m1_unique = batch['m1_unique'].to(device)
            m2_unique = batch['m2_unique'].to(device)

            outputs = model.forward_train(x1, x2)

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

            total_loss += losses['loss'].item()
            for key in loss_components:
                loss_components[key] += losses[key].item()

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {'loss': avg_loss, **loss_components}


def main():
    parser = argparse.ArgumentParser(description='Train DeFusion model')

    # Data arguments
    parser.add_argument('--coco_path', type=str, required=True,
                        help='Path to COCO images directory')
    parser.add_argument('--split', type=str, default='train2017',
                        help='COCO split to use')
    parser.add_argument('--max_images', type=int, default=50000,
                        help='Maximum number of images (50k per paper)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (50 per paper)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (8 per paper)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (1e-3 per paper)')
    parser.add_argument('--lr_decay_epochs', type=int, default=10,
                        help='Epochs between LR decay (10 per paper)')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='LR decay factor (0.5 per paper)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension k')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size (256 per paper)')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='CUD noise standard deviation (0.1 per paper)')

    # Loss weights
    parser.add_argument('--weight_common', type=float, default=1.0,
                        help='Weight for common loss')
    parser.add_argument('--weight_unique', type=float, default=1.0,
                        help='Weight for unique losses')
    parser.add_argument('--weight_recon', type=float, default=1.0,
                        help='Weight for reconstruction loss')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Checkpoint save frequency (epochs)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Setup output directory
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting DeFusion training")
    logger.info(f"Arguments: {args}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating DeFusion model...")
    model = DeFusion(
        in_channels=3,
        feature_dim=args.feature_dim
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create dataset and dataloader
    logger.info("Creating dataloader...")
    train_loader = get_coco_dataloader(
        root=args.coco_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        max_images=args.max_images,
        image_size=args.image_size,
        noise_std=args.noise_std
    )
    logger.info(f"Dataset size: {len(train_loader.dataset)}")

    # Create loss function
    criterion = CUDLoss(
        weight_common=args.weight_common,
        weight_unique=args.weight_unique,
        weight_recon=args.weight_recon
    )

    # Create optimizer (Adam per paper)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create scheduler (halve LR every 10 epochs per paper)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_epochs,
        gamma=args.lr_decay_factor
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, Path(args.resume))
        logger.info(f"Resumed from epoch {start_epoch}")

    # TensorBoard writer
    writer = SummaryWriter(output_dir / 'tensorboard')

    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    loss_history = {'loss': [], 'loss_common': [], 'loss_u1': [], 'loss_u2': [], 'loss_recon': []}

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.6f})")

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, logger, writer
        )

        # Record losses
        for key in loss_history:
            loss_history[key].append(train_losses[key])

        # Step scheduler
        scheduler.step()

        # Log epoch summary
        logger.info(
            f"Epoch {epoch + 1} complete - "
            f"Loss: {train_losses['loss']:.4f} "
            f"(C:{train_losses['loss_common']:.4f} "
            f"U1:{train_losses['loss_u1']:.4f} "
            f"U2:{train_losses['loss_u2']:.4f} "
            f"R:{train_losses['loss_recon']:.4f})"
        )

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_losses['loss'],
                checkpoint_dir / f'checkpoint_epoch{epoch + 1:03d}.pth'
            )

        # Save best model
        if train_losses['loss'] < best_loss:
            best_loss = train_losses['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_losses['loss'],
                checkpoint_dir / 'best_model.pth'
            )
            logger.info(f"New best model saved (loss: {best_loss:.4f})")

        # TensorBoard epoch summary
        writer.add_scalar('Epoch/loss', train_losses['loss'], epoch)
        writer.add_scalar('Epoch/lr', scheduler.get_last_lr()[0], epoch)

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, train_losses['loss'],
        checkpoint_dir / 'final_model.pth'
    )

    # Save loss history
    plot_training_curves(loss_history, save_path=str(output_dir / 'training_curves.png'))

    # Close writer
    writer.close()

    logger.info(f"\nTraining complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
