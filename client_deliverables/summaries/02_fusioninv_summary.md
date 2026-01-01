# FusionINV: Visible-Style Infrared-Visible Image Fusion
## Diffusion-Based Approach for ML-Compatible Fusion

---

### Paper Information
- **Title**: Visible-Style Infrared Image Fusion via Diffusion Model Inversion
- **Authors**: Pengwei Liang et al.
- **Venue**: IEEE TIP 2025 (Transactions on Image Processing)
- **Base Model**: Stable Diffusion v1.5

---

### Problem Statement
Traditional IR-Visible fusion produces images that:
- Look "unnatural" to human viewers
- Cause downstream ML models to fail
- Require retraining detection/segmentation models

**Challenge**: How to fuse IR thermal information while maintaining visible-light appearance?

---

### Key Innovation: Diffusion Inversion
**Leveraging Pre-trained Stable Diffusion**

The insight: Stable Diffusion "knows" what natural images look like.

**Process**:
1. **Invert** visible image into latent space
2. **Inject** IR information during denoising
3. **Generate** fusion that looks natural

```
Visible Image --> DDIM Inversion --> Latent Code
                                         |
IR Image -----> Feature Extraction ------+
                                         |
                                    Guided Denoising
                                         |
                                    Fused Output
                                  (Visible Style!)
```

---

### Technical Approach

**Stage 1: Latent Inversion**
- Use DDIM deterministic sampling
- Map visible image to noise latent
- Preserve structural information

**Stage 2: IR Feature Injection**
- Extract thermal features from IR
- Inject into cross-attention layers
- Control injection strength

**Stage 3: Guided Generation**
- Denoise with IR guidance
- Maintain visible-light statistics
- Output natural-looking fusion

---

### Results Comparison

| Aspect | Traditional Fusion | FusionINV |
|--------|-------------------|-----------|
| Visual Style | Mixed/Artificial | Natural Visible |
| Detection AP | -15% drop | No drop |
| Segmentation | Retraining needed | Works directly |
| Human Preference | 32% | **68%** |

---

### Why Visible-Style Matters

**For Downstream ML**:
- Detection models trained on RGB work directly
- No domain gap between training and inference
- Segmentation maintains accuracy

**For Human Operators**:
- Intuitive interpretation
- Natural color mapping
- Reduced cognitive load

---

### Client Applications

1. **Surveillance Systems**
   - Night vision with natural colors
   - Thermal detection + visible recognition
   - No model retraining needed

2. **Autonomous Vehicles**
   - Sensor fusion for perception
   - Works with existing detection stacks
   - All-weather capability

3. **Medical Imaging**
   - Multi-modal fusion
   - Familiar visual appearance
   - Radiologist-friendly output

---

### Deployment Considerations

**Requirements**:
- GPU with 8GB+ VRAM
- Stable Diffusion v1.5 weights
- CUDA 11.7+

**Performance**:
- ~2 seconds per image (RTX 3090)
- Batch processing supported
- Quality vs speed tradeoff available

---

### Comparison with DeFusion

| Aspect | DeFusion | FusionINV |
|--------|----------|-----------|
| Speed | 0.12s | 2.0s |
| Style | Mixed | Visible |
| Training | Self-supervised | Pre-trained SD |
| Best For | Speed/Universal | ML Compatibility |

**Recommendation**: Use DeFusion for speed, FusionINV for downstream ML

---

*Related to: DeFusion self-supervised approach*
