# External Integrations

**Analysis Date:** 2026-02-26

## APIs & External Services

**Dataset Sources:**
- COCO Dataset (Microsoft Common Objects in Context)
  - Purpose: Large-scale image dataset for self-supervised pretraining
  - SDK/Client: torchvision with COCO support
  - Usage: `datasets/coco_dataset.py` loads images from local COCO directories
  - No API keys required (local filesystem access)

**Public Test Datasets:**
- MEFB - Multi-exposure fusion benchmark
- SICE - Single Image Constrained Enhancement dataset
- Real-MFF - Real-world Multi-Focus Fusion dataset
- TNO - Thermal/IR-Visible fusion dataset
- RoadScene - Road scene thermal/visible dataset
  - Purpose: Evaluation benchmarks (Section 4)
  - SDK/Client: Manual image loading via PIL
  - No external API integration (local dataset directories)

## Data Storage

**Databases:**
- Not detected - No external database integration

**File Storage:**
- Local filesystem only
  - Training data: `configs/defusion.yaml` specifies `paths.coco_dir`
  - Output: Checkpoints saved to `paths.checkpoint_dir`
  - Results: Inference outputs saved as local PNG/image files
  - TensorBoard logs: Local file-based logging

**Caching:**
- PyTorch model checkpoint caching via `torch.load()`
- Streamlit @st.cache_resource decorator for model caching in `app.py` (lines 63-84)

## Authentication & Identity

**Auth Provider:**
- Custom / None
- Implementation: No authentication required
- All file access is local filesystem-based
- Streamlit web interface has no built-in authentication

## Monitoring & Observability

**Error Tracking:**
- Not detected - No external error tracking service

**Logs:**
- TensorBoard (`torch.utils.tensorboard.SummaryWriter`)
  - Location: `output_dir / 'tensorboard'` per `train.py` (line 364)
  - Metrics logged: Loss components, learning rate, epoch summaries
  - Config: `configs/defusion.yaml` enables tensorboard with `logging.tensorboard: true`
- Standard Python logging to file and console
  - File: `output_dir / 'training.log'` per `train.py` (lines 51-52)
  - Format: Timestamp, level, message
  - Usage: Training progress, model info, checkpoints

## CI/CD & Deployment

**Hosting:**
- Streamlit Cloud (optional, via `app.py`)
- Docker containerization available (`Dockerfile`)
- Local development via Python scripts

**CI Pipeline:**
- Not detected - No CI/CD configuration found (no .github/workflows, .gitlab-ci.yml, etc.)
- Manual execution of training and evaluation scripts

**Deployment Options:**
- Docker container (see `Dockerfile`)
- Streamlit web app: `streamlit run app.py`
- CLI scripts: `train.py`, `evaluate.py`, `test_fusion.py`
- Batch inference via `realtime_inference.py` or `test_fusion.py`

## Environment Configuration

**Required env vars:**
- No environment variables detected as critical
- All configuration via:
  - Command-line arguments (e.g., `--coco_path`, `--checkpoint`)
  - YAML config file (`configs/defusion.yaml`)
  - Streamlit sidebar (when using web interface)

**Secrets location:**
- Not applicable - No external API keys, database passwords, or secrets required
- All paths and parameters are explicit in config files or CLI args

## Data Flow

**Training Pipeline:**
```
COCO Images (local)
    ↓
COCODataset + CUDAugmentation
    ↓
DataLoader
    ↓
DeFusion Model
    ↓
CUDLoss (with masked regions)
    ↓
TensorBoard logs + Checkpoints (local)
```

**Inference Pipeline:**
```
Input Images (local file or uploaded)
    ↓
PIL Image.open() → Preprocessing
    ↓
DeFusion Model (forward_fusion)
    ↓
Output Tensor → Denormalization
    ↓
PIL Image (local save or download)
```

## Webhooks & Callbacks

**Incoming:**
- Not detected - No webhook endpoints

**Outgoing:**
- Not detected - No external callbacks

## Model Checkpoints & Weights

**Format:**
- PyTorch `.pth` format
- Structure: Contains `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `loss`
- Location: `checkpoint_dir / f'checkpoint_epoch{epoch:03d}.pth'`

**Loading:**
- `torch.load(path, map_location='cpu')`
- Automatic detection in `app.py` (lines 69-72):
  - Looks for `checkpoints/best_model.pth` relative to script location
  - Falls back to random initialization if not found

## Data Augmentation & Processing

**Preprocessing:**
- `torchvision.transforms` - Standard vision transforms
- PIL Image resize with `Image.BILINEAR` interpolation
- Normalization to [-1, 1] range for model input (per `app.py` lines 104-105)

**Augmentation (Training Only):**
- CUDAugmentation (`datasets/cud_augmentation.py`)
  - Non-overlapping mask generation (random rectangles)
  - Gaussian noise addition (sigma=0.1)
  - Mask coverage: 30-70% per image
  - Purpose: Self-supervised decomposition pretext task

**Post-processing:**
- Denormalization from [-1, 1] to [0, 1] via `(tensor + 1) / 2`
- Clamping to valid range [0, 1]
- PIL Image output in RGB format

## Evaluation Metrics Library

**Metrics Implementation:**
- All metrics computed locally in `utils/metrics.py`
- Scipy-based signal processing for metric calculation
- Supported metrics:
  - SSIM (Structural Similarity Index)
  - MEF-SSIM (Multi-Exposure Fusion SSIM)
  - Entropy
  - Mutual Information
  - QCV (Quality with No Reference)
  - SD (Standard Deviation)
  - Gradient magnitude
  - PSNR (Peak Signal-to-Noise Ratio)

---

*Integration audit: 2026-02-26*
