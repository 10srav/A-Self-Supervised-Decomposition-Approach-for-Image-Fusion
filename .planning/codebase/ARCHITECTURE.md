# Architecture

**Analysis Date:** 2026-02-26

## Pattern Overview

**Overall:** Self-Supervised Decomposition Network (DeNet) with projection heads for image fusion

**Key Characteristics:**
- **Self-supervised learning** using Common and Unique Decomposition (CUD) pretext task
- **Feature decomposition** into common (fc), unique-to-source-1 (f1u), and unique-to-source-2 (f2u) components
- **Multi-layer projection heads** that reconstruct spatial information during training
- **Modular component design** separating encoder, ensembler, decoders, and projection heads
- **Two-phase operation:** Self-supervised training on unlabeled COCO, then inference for fusion tasks

## Layers

**Encoder (E):**
- Purpose: Extract hierarchical features from input images at multiple scales
- Location: `defusion/models/denet.py` - `Encoder` class
- Contains: Initial 7x7 convolution followed by 3 stages (Stage 1/2/3) with residual blocks and max pooling
- Depends on: `BasicResBlock` from `resnest.py`, normalization layers
- Used by: Ensembler and DecoderU for feature decomposition
- Architecture: 3->64->128->256 channels, output at 32x32 spatial resolution

**Ensembler (Ec):**
- Purpose: Combine extracted features from both source images to enable cross-image feature sharing
- Location: `defusion/models/denet.py` - `Ensembler` class
- Contains: Residual layers for feature combination
- Depends on: ResNeStBlock layers
- Used by: DecoderC to extract common features visible in both masked views
- Output: Combined feature tensor enabling common feature extraction

**DecoderC (Dc):**
- Purpose: Extract common features (fc) that should be visible in both masked/noisy views
- Location: `defusion/models/denet.py` - `DecoderC` class
- Contains: Upsampling blocks and residual processing
- Depends on: Ensembler output, UpsampleBlock for progressive upsampling
- Used by: ProjectionHeadC for reconstruction
- Output: Common feature map [B, 256, 32, 32]

**DecoderU (Du):**
- Purpose: Extract unique features from each source that are specific to its masked regions
- Location: `defusion/models/denet.py` - `DecoderU` class
- Contains: Two parallel branches for unique features f1u and f2u
- Depends on: Encoder outputs from both sources
- Used by: ProjectionHeadU for both unique projections
- Output: f1u and f2u feature maps [B, 256, 32, 32]

**ProjectionHeadC:**
- Purpose: Upsample common features (fc) back to image space for reconstruction loss
- Location: `defusion/models/projection_heads.py` - `ProjectionHeadC` class
- Contains: 3 progressive upsampling stages (32->64->128->256) with ResNeSt blocks
- Depends on: UpsampleBlock, feature dimension passed from DeNet
- Used by: DeFusion.forward() for training loss computation
- Output: Common projection xc [B, 3, 256, 256]

**ProjectionHeadU:**
- Purpose: Upsample unique features (f1u, f2u) back to image space for unique region reconstruction
- Location: `defusion/models/projection_heads.py` - `ProjectionHeadU` class
- Contains: 3 progressive upsampling stages shared across both unique branches
- Depends on: UpsampleBlock, feature dimension
- Used by: DeFusion.forward() for training loss on unique regions
- Output: Unique projections x1u, x2u [B, 3, 256, 256]

**ProjectionHeadR (Reconstruction):**
- Purpose: Fuse all decomposed features (concat fc+f1u+f2u) back to complete image
- Location: `defusion/models/projection_heads.py` - `ProjectionHeadR` class
- Contains: 3 progressive upsampling stages with feature concatenation handling
- Depends on: UpsampleBlock, input feature dimension (3 * feature_dim)
- Used by: DeFusion.forward() for full reconstruction during both training and inference
- Output: Fused image xr [B, 3, 256, 256]

## Data Flow

**Training Flow (with CUD augmentation):**

1. Load image from COCO dataset (256x256)
2. Generate two overlapping masks M1 and M2 with:
   - Common region: M1 ∩ M2 (visible in both)
   - Unique regions: M1-M2 (only x1), M2-M1 (only x2)
3. Create masked views with Gaussian noise:
   - x1 = M1⊙x + (1-M1)⊙noise1
   - x2 = M2⊙x + (1-M2)⊙noise2
4. Forward through DeNet:
   - Encoder: x1, x2 → Ex1, Ex2 (features at 32x32)
   - Ensembler: Ex1, Ex2 → combined features
   - DecoderC: → fc (common)
   - DecoderU: → f1u, f2u (unique)
5. Project to image space:
   - xc = ProjectionHeadC(fc)
   - x1u, x2u = ProjectionHeadU(f1u), ProjectionHeadU(f2u)
   - xr = ProjectionHeadR(concat(fc, f1u, f2u))
6. Compute losses on masked regions:
   - L_c = ||xc - (M1∩M2)⊙x|| (MAE)
   - L_u1 = ||x1u - (M1-M2)⊙x|| (MAE)
   - L_u2 = ||x2u - (M2-M1)⊙x|| (MAE)
   - L_r = ||xr - x|| (MAE)
   - Total Loss = L_c + L_u1 + L_u2 + L_r

**Inference Flow (image fusion):**

1. Load two source images (any size, resize to 256x256)
2. Forward through DeNet (no masking, no noise):
   - Encoder: I1, I2 → features at 32x32
   - Ensembler, Decoders → fc, f1u, f2u
3. Fuse via reconstruction projection:
   - Fused = ProjectionHeadR(concat(fc, f1u, f2u))
4. Output fused image (3, 256, 256)

**State Management:**
- Model state: Weights only (no persistent state between calls)
- Feature tensors: Passed forward through layer chain with no branching
- Loss accumulation: Per-batch during training via CUDLoss computation

## Key Abstractions

**DeFusion (Complete Model):**
- Purpose: Unified interface combining DeNet + projection heads
- Examples: `defusion/models/defusion.py` - `DeFusion` class
- Pattern: nn.Module that integrates encoder→ensembler→decoders→projections
- Public interface:
  - `.forward(x1, x2)` → (fused, fc, f1u, f2u) for inference
  - `.forward_train(x1, x2, x, masks)` → projection outputs for training
- Parameters: 17.7M total (feature_dim=256, base_channels=64)

**DeNet (Decomposition Network):**
- Purpose: Core feature decomposition component (E + Ec + D)
- Examples: `defusion/models/denet.py` - `DeNet` class
- Pattern: Composition of Encoder, Ensembler, DecoderC, DecoderU
- Returns: (fc, f1u, f2u) - three feature tensors for projection

**CUDAugmentation (Pretext Task):**
- Purpose: Generate training-time masking and noise for self-supervision
- Examples: `defusion/datasets/cud_augmentation.py` - `CUDAugmentation` class
- Pattern: Callable augmentation that generates overlapping masks and noise
- Key method: `__call__(image)` → (x1, x2, masks_dict)
- Ensures: Non-empty intersection + unique regions + full coverage

**CUDLoss (Training Loss):**
- Purpose: Compute all four loss components of the pretext task
- Examples: `defusion/utils/losses.py` - `CUDLoss` class
- Pattern: nn.Module loss function with weighted components
- Input: predictions dict (xc, x1u, x2u, xr) and targets dict (x, masks)
- Output: Dict with loss breakdown {'loss_common', 'loss_unique1', 'loss_unique2', 'loss_recon', 'total'}

**COCODataset (Training Data):**
- Purpose: Load 50k COCO images with CUD augmentation applied
- Examples: `defusion/datasets/coco_dataset.py` - `COCODataset` class
- Pattern: torch.utils.data.Dataset subclass
- Returns: (x1_augmented, x2_augmented, original_image, masks) tuples

## Entry Points

**app.py (Streamlit Web Interface):**
- Location: `defusion/app.py`
- Triggers: `streamlit run app.py`
- Responsibilities:
  - Loads trained model via `load_model()` helper (cached with @st.cache_resource)
  - Provides three tabs: Image Fusion (upload), Demo (test images), About
  - Handles image preprocessing/postprocessing
  - Displays inference metrics (time, FPS)
  - Allows checkpoint selection via sidebar

**train.py (Full Training):**
- Location: `defusion/train.py`
- Triggers: `python train.py --coco_path /path/to/coco`
- Responsibilities:
  - Sets up DataLoader with COCODataset + CUDAugmentation
  - Creates optimizer (Adam, lr=1e-3, halves every 10 epochs)
  - Trains DeFusion model on CUD pretext task (50 epochs, batch=8)
  - Saves checkpoints and logs to TensorBoard
  - Implements learning rate scheduling

**train_demo.py (Quick Training):**
- Location: `defusion/train_demo.py`
- Triggers: `python train_demo.py --epochs 5 --num_samples 200`
- Responsibilities:
  - Simplified training for demonstration
  - Creates synthetic image pairs if COCO unavailable
  - Faster iteration for testing pipeline

**test_fusion.py (Inference/Testing):**
- Location: `defusion/test_fusion.py`
- Triggers: `python test_fusion.py --checkpoint model.pth --i1 img1.png --i2 img2.png`
- Responsibilities:
  - Loads checkpoint and applies DeFusion.forward()
  - Preprocesses images (resize to 256x256, normalize)
  - Runs fusion without gradients
  - Saves fused output image

**realtime_inference.py (Optimized Inference):**
- Location: `defusion/realtime_inference.py`
- Triggers: `python realtime_inference.py --benchmark` or `--video`
- Responsibilities:
  - Wraps DeFusion in RealTimeDeFusion class for optimization
  - Supports FP16 (half precision) for 2x speed
  - TorchScript compilation support
  - Benchmarking across CPU/GPU
  - Video processing capability

**evaluate.py (Evaluation Suite):**
- Location: `defusion/evaluate.py`
- Triggers: `python evaluate.py --checkpoint model.pth --dataset_dir ./test_data`
- Responsibilities:
  - Evaluates fusion quality across multiple test pairs
  - Computes metrics: SSIM, MEF-SSIM, entropy, MI, QCV, SD
  - Aggregates results per test dataset
  - Generates comparison reports

**quickstart.py (Orchestration):**
- Location: `defusion/quickstart.py`
- Triggers: `python quickstart.py` (5 epochs) or `--full` (50 epochs)
- Responsibilities:
  - Orchestrates full pipeline: train → test → visualize
  - Checks dependencies
  - Runs all entry points in sequence with friendly output

## Error Handling

**Strategy:** Defensive checks with informative messages, graceful degradation to CPU

**Patterns:**
- Device fallback: Check `torch.cuda.is_available()` and fall back to CPU
- Model loading: Check checkpoint exists before load, warn if untrained weights used
- Image loading: Convert to RGB, handle grayscale/RGBA edge cases
- Tensor shape validation: Assertions on expected dimensions in forward passes
- Missing files: Try default paths, warn user with alternative suggestions

**Example from app.py:**
```python
if checkpoint_path and Path(checkpoint_path).exists():
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    st.success(f"Loaded trained model")
else:
    st.warning("Using untrained model. Train with: python train_demo.py --epochs 20")
```

## Cross-Cutting Concerns

**Logging:**
- Uses standard Python logging module in `train.py`, `evaluate.py`
- TensorBoard integration for training curves (enabled in config)
- File output: logs written to `logs/training.log`

**Validation:**
- Input shape validation: [B, 3, 256, 256] expected
- Mask properties: Verified by CUDAugmentation (overlap check, coverage check)
- Loss computation: Defensive masking to prevent NaN propagation

**Normalization:**
- Images normalized to [-1, 1] (mean=0.5, std=0.5) before input
- Denormalized to [0, 1] for visualization and output
- Applied consistently in `preprocessing` and `postprocessing` functions

**Checkpoint Management:**
- Saved format: Dict with `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `loss`
- Loading: Checks for `model_state_dict` key and extracts if present
- Best model tracking: Saves checkpoint when validation loss improves

---

*Architecture analysis: 2026-02-26*
