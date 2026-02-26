# Technology Stack

**Analysis Date:** 2026-02-26

## Languages

**Primary:**
- Python 3.8+ - All core application code, training, inference, and utilities

**Secondary:**
- YAML - Configuration management for training and model parameters
- Dockerfile - Container deployment specification

## Runtime

**Environment:**
- Python 3.8+ (per README badge)

**Package Manager:**
- pip - Standard Python package manager
- Lockfile: `requirements.txt` (present)

## Frameworks

**Core ML/DL:**
- PyTorch 1.9+ - Deep learning framework for model definition and training
  - `torch` - Core tensor operations
  - `torch.nn` - Neural network modules
  - `torch.optim` - Optimization algorithms (Adam optimizer used per paper)
  - `torch.utils.data` - DataLoader and Dataset utilities
  - `torch.utils.tensorboard` - Training visualization
- torchvision 0.10+ - Computer vision utilities
  - Image transformations and preprocessing

**Web/Visualization:**
- Streamlit - Interactive web interface for image fusion (`app.py`)

**Configuration:**
- PyYAML 5.4+ - Configuration file parsing (defusion.yaml)

**Logging & Monitoring:**
- tensorboard 2.5+ - Training metrics visualization and logging

## Key Dependencies

**Critical:**
- torch 1.9+ - Core deep learning framework
- torchvision 0.10+ - Vision utilities (image loading, transforms)
- numpy 1.19+ - Numerical computing and array operations
- Pillow 8.0+ - Image I/O and manipulation (PIL.Image)

**Scientific Computing:**
- scipy 1.6+ - Signal processing and metrics computation
  - `scipy.ndimage` - Image processing filters
  - `scipy.signal.convolve2d` - Convolution operations for metric calculation

**Utilities:**
- tqdm 4.60+ - Progress bars for training epochs and data loading
- matplotlib 3.3+ - Visualization and training curve plotting (optional but recommended)

**Optional/Development:**
- pytest 6.0+ - Testing framework (commented in requirements)
- black 21.0+ - Code formatting (commented in requirements)
- flake8 3.9+ - Linting (commented in requirements)
- pycocotools 2.0+ - COCO dataset tools (commented in requirements)

## Configuration

**Environment:**
- Configuration via command-line arguments and YAML config files
- Primary config: `configs/defusion.yaml`
- Key configs:
  - Model architecture parameters (feature_dim, base_channels, blocks)
  - Training parameters (epochs=50, batch_size=8, learning_rate=1e-3)
  - Dataset paths (COCO directory, output directories)
  - Hardware settings (device=cuda, num_workers=4)
  - Loss weights (weight_common, weight_unique, weight_recon)

**Build:**
- Dockerfile present for containerization
- No setup.py or pyproject.toml detected (pip requirements.txt only)

## Platform Requirements

**Development:**
- Python 3.8+ installed
- CUDA-capable GPU recommended (fallback to CPU available)
- Minimum 8GB RAM for typical training

**Production:**
- CUDA 10.2+ for GPU acceleration (optional)
- 4GB+ GPU VRAM for inference at 256x256 resolution
- CPU inference supported (~1.3 FPS per README)
- Container deployment via Docker available

## Model Parameters

**Architecture (per Section 3.3):**
- Input channels: 3 (RGB)
- Feature dimension (k): 256
- Base channels: 64
- Encoder blocks: [2, 2, 2]
- Decoder blocks: 2
- Projection hidden channels: 128

**Training (COCO dataset):**
- Image size: 256x256
- Max images: 50,000
- Epochs: 50
- Batch size: 8
- Optimizer: Adam
- Learning rate: 1e-3
- LR decay: 0.5 every 10 epochs
- Noise std: 0.1

## Output & Inference

**Model checkpoint format:**
- PyTorch `.pth` format with `model_state_dict` key
- Supports checkpoint resumption for training

**Inference performance:**
- FP32 CPU: ~1.3 FPS
- FP32 GPU: ~30 FPS
- FP16 GPU: ~60 FPS
- Output normalization: Tanh [-1, 1] to [0, 1]

---

*Stack analysis: 2026-02-26*
