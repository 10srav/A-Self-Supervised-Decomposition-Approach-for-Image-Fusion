# DeFusion: A Self-Supervised Decomposition Approach for Image Fusion

PyTorch implementation of **"Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion"**

## Project Overview

| Property | Details |
|----------|---------|
| **Name** | DeFusion (Fusion from Decomposition) |
| **Purpose** | Self-supervised image fusion via Common+Unique Decomposition (CUD) |
| **Framework** | PyTorch |
| **Key Innovation** | Zero-shot fusion after COCO pretraining (no paired data needed) |
| **Tasks** | Multi-Exposure (MEF), Multi-Focus (MFF), IR-Visible fusion |
| **Parameters** | 17.7M |

## Architecture

```
Original Scene (x) ──► CUD Augmentation ──► x1, x2 (masked+noise)
                              │
                              ▼
                       ┌──────────────┐
                       │    DeNet     │
                       │  (E + Ec + D)│
                       └──────┬───────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
         Common Features (fc)    Unique Features (f1u, f2u)
                │                           │
                ▼                           ▼
         Pc(fc) + Pu(f1u+f2u) ──► Pr(recon) ──► Loss (MAE)
                              │
                              ▼
                       FUSED IMAGE (Inference)
```

**Key Components:**
```
DeNet = Encoder(E) + Ensembler(Ec) + Decoder(Du+Dc)
Projections = Pc (common) + Pu (unique) + Pr (reconstruction)
```

## Project Structure

```
defusion/
├── models/
│   ├── resnest.py           # Split-attention blocks (ResNeSt)
│   ├── denet.py             # Encoder, Ensembler, Decoders
│   ├── projection_heads.py  # Pc, Pu, Pr projection heads
│   └── defusion.py          # Main DeFusion model
├── datasets/
│   ├── cud_augmentation.py  # CUD pretext task with masks
│   └── coco_dataset.py      # COCO dataset loader
├── utils/
│   ├── losses.py            # CUD loss function (MAE)
│   ├── metrics.py           # CE, QCV, SSIM, MI, etc.
│   └── visualization.py     # Feature visualization
├── configs/
│   └── defusion.yaml        # Hyperparameters
├── train.py                 # Full COCO training
├── train_demo.py            # Quick demo training (synthetic data)
├── test_fusion.py           # Zero-shot inference
├── realtime_inference.py    # TorchScript + FP16 optimization
├── evaluate.py              # Evaluation metrics
├── app.py                   # Streamlit web interface
└── requirements.txt         # Dependencies
```

## Quick Start

### 1. Installation

```bash
cd defusion
pip install -r requirements.txt
```

### 2. Training

**Quick Demo (no dataset required):**
```bash
python train_demo.py --epochs 20 --batch_size 4 --num_samples 200
```

**Full Training (COCO dataset):**
```bash
python train.py --coco_path /path/to/coco --epochs 50 --batch_size 8
```

### 3. Inference

**Command Line:**
```bash
python test_fusion.py \
    --checkpoint checkpoints/best_model.pth \
    --i1 image1.png \
    --i2 image2.png \
    --output fused.png
```

**Web Interface:**
```bash
streamlit run app.py
# Open http://localhost:8501
```

**Real-time Demo:**
```bash
python realtime_inference.py --demo
```

## Model Architecture

| Component | Input Size | Output Size | Description |
|-----------|-----------|-------------|-------------|
| Encoder (E) | 3×256×256 | 256×32×32 | 3 MaxPool + ResBlocks |
| Ensembler (Ec) | 512×32×32 | 256×32×32 | 1 ResBlock |
| DecoderU (Du) | 512×32×32 | 256×32×32 | Unique feature decoder |
| DecoderC (Dc) | 256×32×32 | 256×32×32 | Common feature decoder |
| Projection Heads | 256×32×32 | 3×256×256 | Upsample to image |

## CUD Training Loss

```python
# Common loss: reconstruct shared regions
loss_c = MAE(Pc(fc), (M1 ∩ M2) * x)

# Unique losses: reconstruct exclusive regions
loss_u1 = MAE(Pu(f1u), (M1 - M2) * x)
loss_u2 = MAE(Pu(f2u), (M2 - M1) * x)

# Reconstruction loss: full image
loss_r = MAE(Pr(concat(fc, f1u, f2u)), x)

# Total loss
loss = loss_c + loss_u1 + loss_u2 + loss_r
```

## Performance

### Inference Speed

| Device | Precision | FPS |
|--------|-----------|-----|
| CPU | FP32 | ~1 |
| GPU | FP32 | ~30 |
| GPU | FP16 | ~60 |

### Expected Results (Paper)

| Dataset | CE↓ | SSIM↑ | QCV↑ |
|---------|-----|-------|------|
| MEFB | 2.881 | 0.608 | 262.3 |
| SICE | 2.830 | 0.571 | 207.7 |
| Real-MFF | 0.971 | 33.88 | - |
| TNO (IR-Vis) | 1.487 | 0.715 | 425.3 |

## Supported Fusion Tasks

- **Multi-Exposure Fusion (MEF)**: Combine over/under exposed images → HDR
- **Multi-Focus Fusion (MFF)**: Combine near/far focus → extended depth of field
- **IR-Visible Fusion**: Combine thermal + visible → enhanced surveillance

## API Reference

```python
from models.defusion import DeFusion

# Initialize model
model = DeFusion().cuda()

# Training forward pass
outputs = model.forward_train(x1, x2)
# Returns: {'fc', 'f1u', 'f2u', 'xc', 'x1u', 'x2u', 'xr'}

# Inference forward pass
fused, fc, f1u, f2u = model(image1, image2)
# Returns: fused image + decomposed features
```

## Hardware Requirements

- **GPU**: RTX 3090/4090 recommended (batch_size=8)
- **VRAM**: ~6GB peak
- **Training**: 50 epochs ≈ 12 hours (GPU)
- **Inference**: ~0.03s/image (GPU), ~1s/image (CPU)

## Citation

```bibtex
@article{defusion2023,
  title={Fusion from Decomposition: A Self-Supervised
         Decomposition Approach for Image Fusion},
  author={...},
  journal={...},
  year={2023}
}
```

## License

MIT License
