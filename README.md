# DeFusion: A Self-Supervised Decomposition Approach for Image Fusion

PyTorch implementation of "Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion"

## Features

- **Self-supervised learning** using CUD (Common and Unique Decomposition) pretext task
- **Multiple fusion tasks**: Multi-exposure, Multi-focus, IR-Visible fusion
- **Real-time inference** with TorchScript and FP16 optimization
- **Streamlit web interface** for easy visualization

## Architecture

```
Source Images (I1, I2)
      │
      ▼
Encoder (E) - Shared weights
      │
      ▼
Ensembler (Ec) - Feature combination
      │
      ▼
Decoders (Du, Dc) - Unique & Common features
      │
      ▼
Projection Heads (Pc, Pu, Pr)
      │
      ▼
Fused Output
```

## Installation

```bash
cd defusion
pip install -r requirements.txt
```

## Quick Start

### Train the model
```bash
python train_demo.py --epochs 20 --batch_size 4 --num_samples 200
```

### Run inference
```bash
python test_fusion.py --checkpoint checkpoints/best_model.pth --i1 image1.png --i2 image2.png
```

### Web interface
```bash
streamlit run app.py
```
Then open http://localhost:8501

### Real-time demo
```bash
python realtime_inference.py --demo
```

## Project Structure

```
defusion/
├── models/
│   ├── resnest.py          # Split-attention blocks
│   ├── denet.py            # Encoder, Ensembler, Decoders
│   ├── projection_heads.py # Projection heads
│   └── defusion.py         # Main model
├── datasets/
│   ├── cud_augmentation.py # CUD pretext task
│   └── coco_dataset.py     # Dataset loader
├── utils/
│   ├── losses.py           # CUD loss function
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Visualization tools
├── configs/
│   └── defusion.yaml       # Configuration
├── train_demo.py           # Training script
├── test_fusion.py          # Inference script
├── realtime_inference.py   # Optimized inference
├── evaluate.py             # Evaluation script
├── app.py                  # Streamlit web app
└── requirements.txt        # Dependencies
```

## Model Specifications

| Component | Input Size | Output Size |
|-----------|-----------|-------------|
| Encoder | 3x256x256 | 256x32x32 |
| Ensembler | 512x32x32 | 256x32x32 |
| DecoderU | 512x32x32 | 256x32x32 |
| DecoderC | 256x32x32 | 256x32x32 |
| Projection Heads | 256x32x32 | 3x256x256 |

**Total Parameters**: 17.7M

## Training

The model uses CUD (Common and Unique Decomposition) self-supervised learning:
1. Generate non-overlapping masks M1, M2
2. Create augmented views with Gaussian noise
3. Train to decompose and reconstruct features

## Performance

| Device | Precision | FPS |
|--------|-----------|-----|
| CPU | FP32 | ~1 |
| GPU | FP32 | ~30 |
| GPU | FP16 | ~60 |

## Supported Fusion Tasks

- **Multi-Exposure Fusion** (HDR imaging)
- **Multi-Focus Fusion** (depth of field extension)
- **IR-Visible Fusion** (surveillance, autonomous driving)

## License

MIT License
