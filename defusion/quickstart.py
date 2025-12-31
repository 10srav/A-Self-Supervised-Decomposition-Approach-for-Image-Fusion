#!/usr/bin/env python
"""
DeFusion Quickstart
===================
One-command setup: Dataset -> Train -> Test -> Visualize

Usage:
    python quickstart.py              # Full quickstart (5 epochs)
    python quickstart.py --full       # Full training (50 epochs)
    python quickstart.py --test-only  # Just test existing model
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run command with nice output."""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if check and result.returncode != 0:
        print(f"\nERROR: {description} failed!")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='DeFusion Quickstart')
    parser.add_argument('--full', action='store_true', help='Full 50-epoch training')
    parser.add_argument('--test-only', action='store_true', help='Test existing model')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--samples', type=int, default=200, help='Training samples')

    args = parser.parse_args()

    print("""
    ============================================================
           DeFusion Quickstart
       Self-Supervised Image Fusion
    ============================================================
    """)

    # Get script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check dependencies
    print("\n[1/5] Checking dependencies...")
    try:
        import torch
        import numpy
        from PIL import Image
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  Missing dependency: {e}")
        print("  Run: pip install -r requirements.txt")
        return 1

    # Test model architecture
    print("\n[2/5] Testing model architecture...")
    try:
        from models.defusion import DeFusion, count_parameters
        model = DeFusion()
        params = count_parameters(model)
        print(f"  Model created successfully!")
        print(f"  Parameters: {params:,}")

        # Test forward pass
        import torch
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            fused, fc, f1u, f2u = model(x1, x2)
        print(f"  Forward pass: OK")
        print(f"  Output shape: {list(fused.shape)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    if args.test_only:
        # Just test existing model
        checkpoint = script_dir / 'checkpoints' / 'best_model.pth'
        if checkpoint.exists():
            print(f"\n[3/5] Testing with existing checkpoint...")
            run_command(
                f'python test_fusion.py --checkpoint {checkpoint} '
                f'--i1 sample_images/exposure_bright.png '
                f'--i2 sample_images/exposure_dark.png '
                f'--output demo_output/quickstart_fused.png',
                "Image Fusion Test"
            )
        else:
            print(f"\nNo checkpoint found at {checkpoint}")
            print("Run training first: python quickstart.py")
        return 0

    # Training
    epochs = 50 if args.full else args.epochs
    samples = 1000 if args.full else args.samples

    print(f"\n[3/5] Training model ({epochs} epochs, {samples} samples)...")

    train_cmd = f'python train_demo.py --epochs {epochs} --batch_size 4 --num_samples {samples}'
    if not run_command(train_cmd, f"Training ({epochs} epochs)"):
        return 1

    # Test fusion
    print("\n[4/5] Testing fusion...")
    checkpoint = script_dir / 'checkpoints' / 'best_model.pth'

    if checkpoint.exists():
        # Create test images if needed
        sample_dir = script_dir / 'sample_images'
        if not sample_dir.exists():
            run_command('python realtime_inference.py --demo', "Creating test images")

        # Run fusion
        run_command(
            f'python test_fusion.py --checkpoint {checkpoint} '
            f'--i1 demo_output/source1.png '
            f'--i2 demo_output/source2.png '
            f'--output demo_output/quickstart_fused.png',
            "Image Fusion"
        )

    # Summary
    print(f"""
    ============================================================
           Quickstart Complete!
    ============================================================

    Results:
    - Checkpoint: checkpoints/best_model.pth
    - Fused image: demo_output/quickstart_fused.png

    Next steps:

    1. Web interface:
       streamlit run app.py

    2. Fuse your own images:
       python test_fusion.py --checkpoint checkpoints/best_model.pth \\
           --i1 your_image1.png --i2 your_image2.png

    3. Full training (50 epochs):
       python quickstart.py --full

    4. Real-time benchmark:
       python realtime_inference.py --benchmark
    """)

    return 0


if __name__ == '__main__':
    sys.exit(main())
