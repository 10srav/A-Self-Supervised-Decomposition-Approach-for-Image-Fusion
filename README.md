# DeFusion: A Self-Supervised Decomposition Approach for Image Fusion

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/AI--Assisted-Development-blueviolet" alt="AI-Assisted">
</p>

PyTorch implementation of **"Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion"**

## Highlights

- **Self-supervised learning** - No paired training data required
- **Zero-shot fusion** - Train once on COCO, fuse any image pairs
- **Real-time inference** - TorchScript + FP16 optimization (~60 FPS on GPU)
- **Multiple fusion tasks** - MEF, MFF, IR-Visible fusion
- **Web interface** - Streamlit app for easy visualization

## Overview

| Property | Details |
|----------|---------|
| **Model** | DeFusion (17.7M parameters) |
| **Training** | CUD (Common and Unique Decomposition) pretext task |
| **Dataset** | COCO 2017 (50K images) |
| **Input Size** | 256 x 256 RGB |
| **Framework** | PyTorch 1.9+ |

## Architecture

```
Source Images (I1, I2)
        │
        ▼
┌───────────────────┐
│      DeNet        │
│  E + Ec + Du + Dc │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
Common (fc)  Unique (f1u, f2u)
    │           │
    └─────┬─────┘
          ▼
    Pr(fc, f1u, f2u)
          │
          ▼
    Fused Output
```

## Quick Start

### Installation

```bash
git clone https://github.com/10srav/A-Self-Supervised-Decomposition-Approach-for-Image-Fusion.git
cd A-Self-Supervised-Decomposition-Approach-for-Image-Fusion/defusion
pip install -r requirements.txt
```

### Training

**Demo Training (No dataset required):**
```bash
python train_demo.py --epochs 20 --batch_size 4 --num_samples 200
```

**Full Training (COCO dataset):**
```bash
python train.py --coco_path /path/to/coco --epochs 50 --batch_size 8
```

### Inference

**Command Line:**
```bash
python test_fusion.py --checkpoint checkpoints/best_model.pth \
    --i1 image1.png --i2 image2.png --output fused.png
```

**Web Interface:**
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

**Real-time Demo:**
```bash
python realtime_inference.py --demo
python realtime_inference.py --benchmark
```

## Project Structure

```
defusion/
├── models/
│   ├── resnest.py           # Split-attention blocks
│   ├── denet.py             # Encoder, Ensembler, Decoders
│   ├── projection_heads.py  # Pc, Pu, Pr heads
│   └── defusion.py          # Main model
├── datasets/
│   ├── cud_augmentation.py  # CUD pretext task
│   └── coco_dataset.py      # Dataset loader
├── utils/
│   ├── losses.py            # CUD loss (MAE)
│   ├── metrics.py           # SSIM, MI, entropy, etc.
│   └── visualization.py     # Feature visualization
├── train.py                 # Full training script
├── train_demo.py            # Demo training (synthetic)
├── test_fusion.py           # Inference script
├── realtime_inference.py    # Optimized inference
├── evaluate.py              # Evaluation metrics
├── app.py                   # Streamlit web app
└── requirements.txt         # Dependencies
```

## Model Components

| Component | Input | Output | Description |
|-----------|-------|--------|-------------|
| Encoder (E) | 3×256×256 | 256×32×32 | Feature extraction |
| Ensembler (Ec) | 512×32×32 | 256×32×32 | Feature combination |
| DecoderU (Du) | 512×32×32 | 256×32×32 | Unique features |
| DecoderC (Dc) | 256×32×32 | 256×32×32 | Common features |
| Projections | 256×32×32 | 3×256×256 | Image reconstruction |

## Training Details

**CUD Loss Function:**
```
L = L_common + L_unique1 + L_unique2 + L_reconstruction

L_common = MAE(Pc(fc), (M1 ∩ M2) ⊙ x)
L_unique1 = MAE(Pu(f1u), (M1 - M2) ⊙ x)
L_unique2 = MAE(Pu(f2u), (M2 - M1) ⊙ x)
L_recon = MAE(Pr(fc, f1u, f2u), x)
```

**Hyperparameters:**
- Optimizer: Adam (lr=1e-3)
- LR Schedule: Halve every 10 epochs
- Batch Size: 8
- Epochs: 50
- Noise σ: 0.1

## Performance

### Inference Speed

| Device | Precision | FPS | Latency |
|--------|-----------|-----|---------|
| CPU | FP32 | ~1 | ~1000ms |
| GPU | FP32 | ~30 | ~33ms |
| GPU | FP16 | ~60 | ~17ms |

### Benchmark Results (Paper)

| Dataset | Task | CE↓ | SSIM↑ | QCV↑ |
|---------|------|-----|-------|------|
| MEFB | Multi-Exposure | 2.881 | 0.608 | 262.3 |
| SICE | Multi-Exposure | 2.830 | 0.571 | 207.7 |
| Real-MFF | Multi-Focus | 0.971 | - | 33.88 |
| TNO | IR-Visible | 1.487 | 0.715 | 425.3 |

## Supported Fusion Tasks

| Task | Description | Example |
|------|-------------|---------|
| **Multi-Exposure (MEF)** | Combine over/under exposed images | HDR imaging |
| **Multi-Focus (MFF)** | Combine near/far focus images | Extended DOF |
| **IR-Visible** | Combine thermal + visible | Surveillance |

## API Usage

```python
from models.defusion import DeFusion, build_defusion

# Load model
model = build_defusion(pretrained='checkpoints/best_model.pth')
model.eval()

# Fuse images
with torch.no_grad():
    fused = model.forward_fusion(image1, image2)

# Or get decomposed features
fused, fc, f1u, f2u = model(image1, image2)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- numpy, Pillow, scipy
- streamlit (for web interface)
- tensorboard (for training visualization)

## Hardware Requirements

- **Training**: NVIDIA GPU with 6GB+ VRAM (RTX 3060+)
- **Inference**: CPU supported, GPU recommended for real-time
- **Memory**: 8GB+ RAM

## Citation

```bibtex
@article{defusion2023,
  title={Fusion from Decomposition: A Self-Supervised
         Decomposition Approach for Image Fusion},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with AI-assisted development tools</sub>
</p>
