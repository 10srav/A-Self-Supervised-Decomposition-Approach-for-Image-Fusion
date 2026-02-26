# Coding Conventions

**Analysis Date:** 2026-02-26

## Naming Patterns

**Files:**
- Snake case: `cud_augmentation.py`, `coco_dataset.py`, `resnest.py`
- Descriptive compound names reflecting module purpose
- Short, focused modules (dataset, model, utility components)

**Classes:**
- PascalCase: `DeFusion`, `DeNet`, `Encoder`, `Ensembler`, `CUDAugmentation`, `ProjectionHeadC`
- Descriptive names matching architectural components
- Example: `class DeFusion(nn.Module):` in `defusion/models/defusion.py`

**Functions:**
- Snake case: `train_epoch()`, `validate()`, `load_checkpoint()`, `save_checkpoint()`, `compute_ssim()`, `generate_overlapping_masks()`
- Action verbs for operations: `build_defusion()`, `count_parameters()`, `load_image()`, `save_image()`
- Private functions prefixed with underscore: `_collect_images()`, `_get_pairs()` in `defusion/datasets/coco_dataset.py`
- Example: `def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, writer=None)` in `defusion/train.py`

**Variables:**
- Snake case for regular variables: `num_batches`, `total_loss`, `loss_components`, `feature_dim`
- Abbreviations preserved from paper: `fc` (common features), `f1u` (unique features image 1), `f2u` (unique features image 2), `xc`, `x1u`, `x2u`, `xr` (projections)
- Tensor variable naming follows mathematical notation: `x1`, `x2` (input images), `m1`, `m2` (masks)
- Uppercase for constants: `C1`, `C2` in metrics
- Example in `defusion/models/defusion.py`: `fc, f1u, f2u = self.denet(x1, x2)`

**Types:**
- Type hints used extensively with `typing` module: `from typing import Dict, Tuple, Optional`
- Example signatures in `defusion/models/defusion.py`:
  ```python
  def forward_train(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
  def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  ```

## Code Style

**Formatting:**
- Max line length: ~100 characters (implicit based on code)
- Uses f-strings: `f"Found {len(pairs)} image pairs for evaluation"` in `defusion/evaluate.py`
- No explicit formatter configured (no .prettierrc or black config found)

**Linting:**
- No configured linting (no .eslintrc, .flake8, or .pylintrc found)
- PEP 8 conventions appear to be followed manually

## Import Organization

**Order:**
1. Standard library imports: `import os, sys, argparse, logging, json, subprocess, random, numpy`
2. Third-party imports: `import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F`
3. Local/relative imports: `from models import DeFusion`, `from datasets import COCODataset`, `from utils.losses import CUDLoss`

**Path Aliases:**
- No aliases configured (no .pth module or PATH setup in imports)
- Relative imports handled by adding parent directory to sys.path:
  ```python
  sys.path.insert(0, str(Path(__file__).parent))
  ```
  (See `defusion/train.py`, `defusion/test_fusion.py`, `defusion/evaluate.py`)

## Error Handling

**Patterns:**
- Try-except for dependency checks and file operations
- Example in `defusion/quickstart.py`:
  ```python
  try:
      import torch
      import numpy
      from PIL import Image
  except ImportError as e:
      print(f"Missing dependency: {e}")
      return 1
  ```
- Graceful error reporting with informative messages
- Path validation with `exists()` checks: `if not self.image_dir.exists()`
- Error propagation via return codes: `sys.exit(0 if success else 1)` in `tests/test_model.py`
- No exception specifications beyond `ImportError` and `Exception` - uses broad catching

## Logging

**Framework:** `logging` module (Python standard library)

**Patterns:**
- Logging setup in `defusion/train.py`:
  ```python
  def setup_logging(log_dir: Path) -> logging.Logger:
      logging.basicConfig(
          level=logging.INFO,
          format='%(asctime)s [%(levelname)s] %(message)s',
          handlers=[
              logging.FileHandler(log_dir / 'training.log'),
              logging.StreamHandler()
          ]
      )
      return logging.getLogger(__name__)
  ```
- Info-level logging for training progress: `logger.info(f"Epoch {epoch} [{batch_idx}/{num_batches}] Loss: {losses['loss'].item():.4f}")`
- Console output via `print()` for user-facing messages (not using logging)
- TensorBoard integration for metrics: `writer.add_scalar('Loss/train', losses['loss'].item(), global_step)`

## Comments

**When to Comment:**
- Module-level docstrings required - every file starts with triple-quoted description
- Complex algorithm explanations with paper references
- Example in `defusion/datasets/cud_augmentation.py`:
  ```python
  """
  CUD Augmentation: Common and Unique Decomposition
  =================================================
  Implementation of the CUD pretext task augmentation from Section 3.2
  ...
  """
  ```

**JSDoc/TSDoc:**
- Uses Python docstrings (not JSDoc - this is Python code)
- Comprehensive docstrings for all public classes and functions
- Google-style docstrings with Args, Returns, Paper references
- Example in `defusion/models/defusion.py`:
  ```python
  def forward_fusion(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
      """
      Inference fusion forward pass.

      Paper Section 3.2 (Inference Pipeline, Fig 3):
      "Given source images I1, I2:
       DeNet(I1,I2) → fc, f1u, f2u
       fused = Pr(concat(fc, f1u, f2u))"

      Args:
          i1: First source image [B, 3, H, W]
          i2: Second source image [B, 3, H, W]

      Returns:
          fused: Fused output image [B, 3, H, W]
      """
  ```

## Function Design

**Size:**
- Typical functions are 20-60 lines
- Complex functions like `train_epoch()` in `defusion/train.py` reach ~70 lines with logging/monitoring
- Model forward passes are 5-20 lines (simple delegation to sub-modules)

**Parameters:**
- Device parameter pattern: `device: torch.device` or `device: str = 'cpu'`
- Config dictionary pattern: `config: Optional[Dict] = None`
- Optional parameters with defaults: `normalize: bool = True`, `size_average: bool = True`
- Large parameter lists delegated to classes (e.g., `DeFusion.__init__` has 8 params)

**Return Values:**
- Single values: `torch.Tensor`
- Multiple related values: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- Structured data: `Dict[str, torch.Tensor]` for training outputs
- Status codes: `bool` for validation functions, `int` for CLI return codes
- Example in `defusion/models/defusion.py`:
  ```python
  return {
      'fc': fc,
      'f1u': f1u,
      'f2u': f2u,
      'xc': xc,
      'x1u': x1u,
      'x2u': x2u,
      'xr': xr
  }
  ```

## Module Design

**Exports:**
- Barrel files use `__all__` to define public interface
- Example in `defusion/models/__init__.py`:
  ```python
  __all__ = [
      'SplitAttention',
      'ResNeStBlock',
      'Encoder',
      'Ensembler',
      'DeFusion',
      'build_defusion'
  ]
  ```

**Barrel Files:**
- Used in `defusion/models/__init__.py` and `defusion/datasets/__init__.py`
- Consolidate related imports for cleaner external imports
- Pattern: Import all public items, define `__all__` list

---

*Convention analysis: 2026-02-26*
