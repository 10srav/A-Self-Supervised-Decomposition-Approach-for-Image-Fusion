# Testing Patterns

**Analysis Date:** 2026-02-26

## Test Framework

**Runner:**
- `unittest` (Python standard library)
- Config: Not explicitly configured, uses defaults
- No pytest, nose, or other frameworks

**Assertion Library:**
- `unittest.TestCase` assertion methods: `assertEqual()`, `assertIsNotNone()`, `assertGreater()`, `assertLess()`, `assertIn()`

**Run Commands:**
```bash
python -m pytest tests/ -v                    # Standard pytest (if installed)
python tests/test_model.py                    # Direct execution
python -m unittest discover tests/            # unittest discovery
```

## Test File Organization

**Location:**
- Co-located in separate `tests/` directory at project root
- File: `tests/test_model.py`
- Directory structure: `/tests/` separate from `/defusion/` source code

**Naming:**
- File: `test_model.py` (prefix with `test_`)
- Classes: `TestDeFusionModel`, `TestCUDAugmentation`, `TestCUDLoss`, `TestMetrics` (prefix with `Test`)
- Methods: `test_model_creation()`, `test_parameter_count()`, `test_forward_inference()` (prefix with `test_`)

**Structure:**
```
tests/
└── test_model.py
    ├── TestDeFusionModel (5 tests)
    ├── TestCUDAugmentation (2 tests)
    ├── TestCUDLoss (1 test)
    ├── TestMetrics (2 tests)
    └── run_tests() function
```

## Test Structure

**Suite Organization:**
File: `tests/test_model.py`

```python
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
```

**Patterns:**
- Setup: `@classmethod setUpClass()` for one-time expensive operations (model loading)
- Teardown: Not used (default cleanup sufficient)
- Assertion pattern: Use `self.assert*()` methods
- Descriptive test names and docstrings explaining what is tested
- Example test (test_forward_inference in TestDeFusionModel):
  - Creates random input tensors with shape [1, 3, 256, 256]
  - Calls model forward pass within torch.no_grad() context
  - Validates all output shapes match expected dimensions
  - Checks fused output [1, 3, 256, 256] and feature outputs [1, 256, 32, 32]

## Mocking

**Framework:** No mocking framework used (no unittest.mock imports)

**Patterns:**
- No mocking - uses real objects and real computations
- Direct instantiation: `model = DeFusion()`
- Real tensor creation: `x1 = torch.randn(1, 3, 256, 256)`
- All dependencies are real PyTorch tensors and model components

**What to Mock:**
- Not applicable - codebase doesn't use mocks

**What NOT to Mock:**
- Model components (used directly)
- PyTorch tensors (created fresh for each test)
- Loss functions (computed with real tensors)
- Metrics (computed with real data)

## Fixtures and Factories

**Test Data:**
File: `tests/test_model.py` contains inline fixture generation

Common patterns:
- Random image tensors: `torch.randn(B, C, H, W)` where B in range 1-2
- Binary masks: `torch.rand(B, H, W) > 0.5` for mask boolean tensors
- Prediction dictionaries: Keys include 'xc', 'x1u', 'x2u', 'xr' matching model outputs
- Target dictionaries: Keys include 'x' (image), 'm_common', 'm1_unique', 'm2_unique' (masks)
- Standard dimensions: 256x256 images, 3 channels, batch sizes 1-2

**Location:**
- No factory classes - fixtures created inline within test methods
- Test data generated with `torch.randn()`, `torch.rand()` for reproducibility testing
- Hard-coded dimensions: 256x256 images, batch size 1-2, 3 channels

## Coverage

**Requirements:** Not enforced (no coverage config found)

**View Coverage:**
```bash
python -m coverage run -m unittest discover tests/
python -m coverage report
python -m coverage html
```
(Commands documented but not configured)

## Test Types

**Unit Tests:**
- Scope: Individual model components and functions
- Approach: Test single functionality in isolation
- Examples in `tests/test_model.py`:
  - `test_model_creation()` - Can instantiate model
  - `test_parameter_count()` - Model has approximately 17.7M parameters
  - `test_forward_inference()` - Forward pass produces correct shapes
  - `test_forward_train()` - Training forward pass returns all outputs
  - `test_batch_size_1()` - Works with batch size 1
  - `test_different_input_sizes()` - Works with 128, 256, 512 resolutions

**Integration Tests:**
- Scope: Component interactions (e.g., loss computation with model outputs)
- Approach: Feed real model outputs to loss functions
- Examples in `tests/test_model.py`:
  - `TestCUDLoss.test_loss_computation()` - Loss receives real prediction dictionaries
  - `TestMetrics.test_ssim_computation()` - SSIM metric works with image tensors
  - `TestMetrics.test_entropy_computation()` - Entropy metric produces positive values
  - `TestCUDAugmentation.test_mask_overlap()` - Mask generation produces valid overlaps

**E2E Tests:**
- Framework: Not formally used
- Full end-to-end pipeline tested in `defusion/quickstart.py`:
  - Data preparation
  - Model training
  - Inference/fusion
  - Metric evaluation

## Common Patterns

**Async Testing:**
Not applicable - synchronous code only, no async/await

**Error Testing:**
No explicit error/exception tests in current suite. When adding error tests, use pattern:
```python
def test_error_condition(self):
    """Test that invalid input raises appropriate error."""
    invalid_input = torch.randn(1, 4, 256, 256)
    with self.assertRaises(RuntimeError):
        self.model(invalid_input, invalid_input)
```

**Shape Validation:**
Standard pattern used throughout - check tensor shapes after operations:
```python
self.assertEqual(outputs['xc'].shape, torch.Size([2, 3, 256, 256]))
self.assertEqual(outputs['xr'].shape, torch.Size([2, 3, 256, 256]))
```

**No-Grad Context:**
Used for inference tests to avoid building computation graph:
```python
with torch.no_grad():
    fused, fc, f1u, f2u = self.model(x1, x2)
```

## Test Execution

**Main test runner in `tests/test_model.py`:**

Function `run_tests()` implements custom test orchestration:
- Creates `unittest.TestLoader()` instance
- Builds `unittest.TestSuite()` with all test classes
- Loads tests: TestDeFusionModel, TestCUDAugmentation, TestCUDLoss, TestMetrics
- Runs with `unittest.TextTestRunner(verbosity=2)`
- Returns boolean indicating success/failure
- Prints formatted summary with pass/fail counts

Entry point:
```python
if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
```

## Test Coverage Summary

**Tested Components:**
- Model architecture: `DeFusion` class instantiation and forward passes
- Output shapes: Verified for inference and training modes
- Parameter counts: Ensure approximately 17.7M parameters as expected
- Feature generation: Common/unique decomposition works
- Loss computation: All components (common, unique, reconstruction)
- Metrics: SSIM and entropy calculations
- Data augmentation: Mask generation and overlap validation
- Batch processing: Single samples and multiple sizes

**Currently Untested:**
- Training loops (full epoch training)
- Checkpoint save/load functionality
- Dataset loading and preprocessing
- Image I/O operations
- Evaluation metrics beyond SSIM/entropy
- GPU/device handling edge cases
- Error conditions and invalid inputs

---

*Testing analysis: 2026-02-26*
