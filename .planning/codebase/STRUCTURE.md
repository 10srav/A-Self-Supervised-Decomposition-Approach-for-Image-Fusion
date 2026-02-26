# Codebase Structure

**Analysis Date:** 2026-02-26

## Directory Layout

```
defusion/
├── app.py                      # Streamlit web interface (interactive fusion)
├── train.py                    # Main training script (full COCO training)
├── train_demo.py               # Demo training script (quick, synthetic data)
├── test_fusion.py              # Inference/testing script (fuse two images)
├── quickstart.py               # Orchestration script (runs full pipeline)
├── evaluate.py                 # Evaluation metrics computation
├── realtime_inference.py       # Optimized inference (FP16, TorchScript, benchmarking)
├── generate_client_demos.py    # Demo generation utilities
├── requirements.txt            # Python dependencies
│
├── models/                     # Core neural network architectures
│   ├── __init__.py
│   ├── defusion.py            # Complete DeFusion model (DeNet + projections)
│   ├── denet.py               # DeNet components (Encoder, Ensembler, Decoders)
│   ├── projection_heads.py    # Projection heads (Pc, Pu, Pr)
│   └── resnest.py             # ResNeSt building blocks (residual layers)
│
├── datasets/                   # Data loading and augmentation
│   ├── __init__.py
│   ├── coco_dataset.py        # COCODataset class (loads COCO images)
│   ├── cud_augmentation.py    # CUDAugmentation (mask generation, noise)
│   └── prepare_coco.py        # COCO download/preparation utilities
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── losses.py              # CUDLoss and component loss functions
│   ├── metrics.py             # Fusion quality metrics (SSIM, MEF-SSIM, etc)
│   └── visualization.py       # Training visualization utilities
│
├── configs/                    # Configuration files
│   └── defusion.yaml          # YAML config (model params, training hyperparams)
│
└── checkpoints/               # Model checkpoints (generated at runtime)
    └── best_model.pth         # Best trained model snapshot
```

## Directory Purposes

**models/:**
- Purpose: All neural network modules for feature extraction, decomposition, and fusion
- Contains: PyTorch nn.Module subclasses
- Key files: `defusion.py` (orchestrates all), `denet.py` (core decomposition), `resnest.py` (building blocks)
- Imports: Only torch, no external ML frameworks
- Usage: Imported by training scripts and inference scripts

**datasets/:**
- Purpose: Data loading pipeline with self-supervised augmentation (CUD pretext task)
- Contains: PyTorch Dataset subclasses, augmentation generators
- Key files: `coco_dataset.py` (main dataset), `cud_augmentation.py` (pretext task logic)
- Imports: torch, torchvision, PIL, numpy
- Usage: DataLoader wrapping in train.py, train_demo.py

**utils/:**
- Purpose: Shared functionality (loss computation, metrics, visualization)
- Contains: Loss functions (nn.Module), metric functions, plotting utilities
- Key files: `losses.py` (training objective), `metrics.py` (evaluation metrics)
- Imports: torch, numpy, scikit-image
- Usage: Imported by training and evaluation scripts

**configs/:**
- Purpose: Centralized hyperparameter specification
- Contains: Single YAML file with model architecture, training params, dataset info
- Key files: `defusion.yaml` (authoritative config)
- Format: YAML with sections: model, training, loss, evaluation, hardware, paths
- Usage: Loaded by train.py, train_demo.py (used but not enforced)

## Key File Locations

**Entry Points:**

| Purpose | File | Command |
|---------|------|---------|
| Interactive web UI | `defusion/app.py` | `streamlit run app.py` |
| Full training (50ep) | `defusion/train.py` | `python train.py --coco_path /path` |
| Demo training (5ep) | `defusion/train_demo.py` | `python train_demo.py --epochs 5` |
| Inference/testing | `defusion/test_fusion.py` | `python test_fusion.py --checkpoint model.pth --i1 img1.png --i2 img2.png` |
| Quick full pipeline | `defusion/quickstart.py` | `python quickstart.py` |
| Benchmarking | `defusion/realtime_inference.py` | `python realtime_inference.py --benchmark` |
| Evaluation | `defusion/evaluate.py` | `python evaluate.py --checkpoint model.pth` |

**Configuration:**
- `defusion/configs/defusion.yaml` - YAML with model architecture (feature_dim=256), training hyperparams (lr=1e-3, epochs=50, batch=8), dataset paths

**Core Logic:**
- `defusion/models/defusion.py` - DeFusion class (17.7M parameters, orchestrates DeNet + projections)
- `defusion/models/denet.py` - DeNet decomposition (Encoder→Ensembler→Decoders)
- `defusion/datasets/cud_augmentation.py` - CUD pretext task (mask generation, noise, loss targets)
- `defusion/utils/losses.py` - CUDLoss (4-component training objective)

**Testing/Evaluation:**
- `defusion/test_fusion.py` - Single pair fusion script
- `defusion/evaluate.py` - Multi-pair evaluation with metrics
- Tests directory at root level (parent): `tests/test_model.py`

## Naming Conventions

**Files:**
- Lowercase with underscores: `cud_augmentation.py`, `projection_heads.py`, `realtime_inference.py`
- Main entry scripts: Descriptive action verbs: `train.py`, `test_fusion.py`, `evaluate.py`
- Utilities organized by function: `losses.py`, `metrics.py`, `visualization.py`

**Directories:**
- Plural for collections: `models/`, `datasets/`, `utils/`, `configs/`
- Single word with lowercase: `checkpoints/` (generated outputs)

**Classes:**
- PascalCase with full words: `DeFusion`, `DeNet`, `Encoder`, `COCODataset`, `CUDAugmentation`, `CUDLoss`
- Descriptive names reflecting architecture: `ProjectionHeadC`, `ProjectionHeadU`, `ProjectionHeadR`

**Functions:**
- Lowercase snake_case for functions: `load_model()`, `preprocess_image()`, `generate_overlapping_masks()`, `save_checkpoint()`
- Helper functions often in main script: `load_image()`, `denormalize()`, `save_image()`

**Variables:**
- Tensor dimensions in comments: `[B, C, H, W]` for batch dimension convention
- Abbreviated math notation matching paper: `fc` (common features), `f1u`/`f2u` (unique), `x1`/`x2` (masked views), `xc`/`x1u`/`x2u`/`xr` (projections)
- Mask variables: `m1`, `m2`, `m_common`, `m1_unique`, `m2_unique` (boolean masks)

**Module/Package exports:**
- `models/__init__.py` exports: `DeFusion`, `build_defusion`, `count_parameters`
- `datasets/__init__.py` exports: `COCODataset`, `get_coco_dataloader`, `CUDAugmentation`
- `utils/__init__.py` - minimal exports (individual functions imported as needed)

## Where to Add New Code

**New Training Feature/Loss Component:**
- Primary code: `defusion/utils/losses.py`
- Integration: Add to `CUDLoss.forward()` and return dict
- Config: Update `defusion/configs/defusion.yaml` under `loss:` section
- Training: Reference in `train.py` line ~38 where CUDLoss is instantiated

**New Model Component (architecture change):**
- If encoder/ensembler/decoder: `defusion/models/denet.py`
- If projection head: `defusion/models/projection_heads.py`
- Integration point: Update `DeFusion.__init__()` in `defusion/models/defusion.py`
- Tests: Add to `tests/test_model.py` (verify forward pass shape)

**New Dataset or Augmentation:**
- If dataset: `defusion/datasets/coco_dataset.py` or new `defusion/datasets/new_dataset.py`
- If augmentation: Add to `defusion/datasets/cud_augmentation.py` (modify `CUDAugmentation` class)
- Integration: Update `train.py` DataLoader construction
- Config: Add dataset params to `defusion/configs/defusion.yaml`

**New Metric for Evaluation:**
- Primary code: `defusion/utils/metrics.py`
- Pattern: Add function `compute_metric_name(tensor1, tensor2, device)` returning float
- Integration: Import in `evaluate.py`, add to metric computation loop
- Returns: Should be differentiable if needed for loss, numpy-compatible for logging

**New Inference Mode/Optimization:**
- Primary code: `defusion/realtime_inference.py` (extends `RealTimeDeFusion` class)
- Alternative: New script `defusion/inference_mode_name.py` if standalone
- Pattern: Wrap DeFusion.forward(), handle preprocessing/postprocessing
- Device handling: Check `torch.cuda.is_available()` with CPU fallback

**Visualization/Analysis Utilities:**
- Primary code: `defusion/utils/visualization.py`
- Functions: `visualize_*()` or `plot_*()`
- Integration: Called from training scripts for debugging

## Special Directories

**checkpoints/:**
- Purpose: Stores trained model snapshots (PyTorch `.pth` files)
- Generated: Yes (created by train.py, train_demo.py)
- Committed: No (.gitignore entry exists)
- Format: PyTorch checkpoint dict with keys: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `loss`
- Expected: `checkpoints/best_model.pth` (auto-loaded by app.py if exists)

**output/ or demo_output/:**
- Purpose: Generated results (fused images, logs, metrics reports)
- Generated: Yes (created by inference and evaluation scripts)
- Committed: No (temporary outputs)
- Structure: Mirrors task type (e.g., `demo_output/source1.png`, `demo_output/quickstart_fused.png`)

**logs/ or tensorboard_logs/:**
- Purpose: Training logs and TensorBoard event files
- Generated: Yes (created by train.py with logging module + TensorBoard)
- Committed: No (training artifacts)
- Location: Configurable via argparse (default `logs/` in train.py)
- Viewing: `tensorboard --logdir ./logs`

**sample_images/ or data/:**
- Purpose: Test/demo images for development
- Generated: By `generate_client_demos.py` or user-provided
- Committed: No (except as documentation references)
- Usage: Fallback source for `quickstart.py` if COCO unavailable

---

*Structure analysis: 2026-02-26*
