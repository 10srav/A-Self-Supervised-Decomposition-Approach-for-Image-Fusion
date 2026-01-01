"""
DeFusion Unit Tests
===================
Basic tests to verify model functionality.

Run: python -m pytest tests/ -v
Or:  python tests/test_model.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'defusion'))

import torch
import unittest


class TestDeFusionModel(unittest.TestCase):
    """Test DeFusion model architecture and forward pass."""

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        from models.defusion import DeFusion
        cls.model = DeFusion()
        cls.model.eval()

    def test_model_creation(self):
        """Test model instantiation."""
        from models.defusion import DeFusion
        model = DeFusion()
        self.assertIsNotNone(model)

    def test_parameter_count(self):
        """Test model has expected number of parameters (~17.7M)."""
        params = sum(p.numel() for p in self.model.parameters())
        # Should be approximately 17.7M (allow some variance)
        self.assertGreater(params, 15_000_000)
        self.assertLess(params, 20_000_000)
        print(f"Parameters: {params:,}")

    def test_forward_inference(self):
        """Test inference forward pass."""
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            fused, fc, f1u, f2u = self.model(x1, x2)

        # Check output shapes
        self.assertEqual(fused.shape, torch.Size([1, 3, 256, 256]))
        self.assertEqual(fc.shape, torch.Size([1, 256, 32, 32]))
        self.assertEqual(f1u.shape, torch.Size([1, 256, 32, 32]))
        self.assertEqual(f2u.shape, torch.Size([1, 256, 32, 32]))

    def test_forward_train(self):
        """Test training forward pass."""
        x1 = torch.randn(2, 3, 256, 256)
        x2 = torch.randn(2, 3, 256, 256)

        outputs = self.model.forward_train(x1, x2)

        # Check all outputs present
        expected_keys = ['fc', 'f1u', 'f2u', 'xc', 'x1u', 'x2u', 'xr']
        for key in expected_keys:
            self.assertIn(key, outputs)

        # Check shapes
        self.assertEqual(outputs['xc'].shape, torch.Size([2, 3, 256, 256]))
        self.assertEqual(outputs['xr'].shape, torch.Size([2, 3, 256, 256]))

    def test_batch_size_1(self):
        """Test model works with batch_size=1."""
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)

        # Should not raise error
        with torch.no_grad():
            fused = self.model.forward_fusion(x1, x2)

        self.assertEqual(fused.shape, torch.Size([1, 3, 256, 256]))

    def test_different_input_sizes(self):
        """Test model with different input sizes (must be divisible by 8)."""
        for size in [128, 256, 512]:
            x1 = torch.randn(1, 3, size, size)
            x2 = torch.randn(1, 3, size, size)

            with torch.no_grad():
                fused = self.model.forward_fusion(x1, x2)

            self.assertEqual(fused.shape, torch.Size([1, 3, size, size]))


class TestCUDAugmentation(unittest.TestCase):
    """Test CUD mask generation."""

    def test_mask_overlap(self):
        """Test that masks properly overlap."""
        from datasets.cud_augmentation import CUDAugmentation

        cud = CUDAugmentation(noise_std=0.1, overlap_ratio=0.3)
        x = torch.rand(3, 256, 256)

        output = cud(x)

        # Check common region exists (overlap)
        self.assertGreater(output['m_common'].mean(), 0.05,
                          "Common region too small!")

        # Check unique regions exist
        self.assertGreater(output['m1_unique'].mean(), 0.05,
                          "Unique region for M1 too small!")
        self.assertGreater(output['m2_unique'].mean(), 0.05,
                          "Unique region for M2 too small!")

    def test_mask_coverage(self):
        """Test that masks cover the full image."""
        from datasets.cud_augmentation import CUDAugmentation

        cud = CUDAugmentation(noise_std=0.1, overlap_ratio=0.3)
        x = torch.rand(3, 256, 256)

        output = cud(x)

        # Union should cover most of image
        union = (output['m1'] + output['m2'] - output['m_common']).clamp(0, 1)
        self.assertGreater(union.mean(), 0.9,
                          "Masks don't cover enough of the image!")


class TestCUDLoss(unittest.TestCase):
    """Test CUD loss computation."""

    def test_loss_computation(self):
        """Test loss computes without error."""
        from utils.losses import CUDLoss

        criterion = CUDLoss()

        B, C, H, W = 2, 3, 256, 256

        predictions = {
            'xc': torch.randn(B, C, H, W),
            'x1u': torch.randn(B, C, H, W),
            'x2u': torch.randn(B, C, H, W),
            'xr': torch.randn(B, C, H, W)
        }

        targets = {
            'x': torch.randn(B, C, H, W),
            'm_common': torch.rand(B, H, W) > 0.5,
            'm1_unique': torch.rand(B, H, W) > 0.5,
            'm2_unique': torch.rand(B, H, W) > 0.5
        }

        losses = criterion(predictions, targets)

        # Check all loss components present
        self.assertIn('loss', losses)
        self.assertIn('loss_common', losses)
        self.assertIn('loss_recon', losses)

        # Check loss is positive
        self.assertGreater(losses['loss'].item(), 0)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def test_ssim_computation(self):
        """Test SSIM metric."""
        from utils.metrics import compute_ssim

        # Same image should have SSIM = 1
        img = torch.rand(3, 256, 256)
        ssim = compute_ssim(img, img)
        self.assertGreater(ssim, 0.99)

        # Different images should have lower SSIM
        img2 = torch.rand(3, 256, 256)
        ssim2 = compute_ssim(img, img2)
        self.assertLess(ssim2, 0.5)

    def test_entropy_computation(self):
        """Test entropy metric."""
        from utils.metrics import compute_entropy

        img = torch.rand(3, 256, 256)
        entropy = compute_entropy(img)

        # Entropy should be positive
        self.assertGreater(entropy, 0)


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print(" DeFusion Unit Tests")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeFusionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestCUDAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestCUDLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(" All tests passed!")
    else:
        print(f" {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
