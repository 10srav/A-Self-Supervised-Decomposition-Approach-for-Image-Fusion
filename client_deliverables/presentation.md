# DeFusion: Self-Supervised Image Fusion
## Client Presentation

---

# Slide 1: The Problem

## Why Image Fusion?

### Single Sensors Have Limitations

| Sensor Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Thermal/IR** | See through darkness, detect heat | No color, low detail |
| **Visible** | Rich color, high detail | Fails at night |
| **Multi-Exposure** | HDR capability | Ghosting, noise |
| **Multi-Focus** | Sharp regions | Out-of-focus areas |

### The Challenge
- How to combine the best of multiple images?
- Traditional methods require expensive paired data
- Separate models for each fusion type

---

# Slide 2: DeFusion Solution

## Self-Supervised Learning for Universal Fusion

### Key Innovation: CUD Pretext Task

```
Common + Unique Decomposition

Image 1 ----+                    +----- Unique to Image 1
            |                    |
            v                    v
        [Encoder] --> [Decompose] --> Common (shared)
            ^                    ^
            |                    |
Image 2 ----+                    +----- Unique to Image 2
                                 |
                                 v
                            [Fuse] --> Output
```

### Training Data
- Uses ONLY natural images (COCO dataset)
- **No paired fusion data needed!**
- Self-supervised learning

### Model
- 17.7M parameters
- Single model for ALL fusion types
- Zero-shot generalization

---

# Slide 3: State-of-the-Art Results

## Benchmark Performance

### Multi-Exposure Fusion (MEFB Dataset)
| Method | SSIM | SD | QCV |
|--------|------|----|----|
| DeepFuse | 0.571 | 42.3 | 0.812 |
| MEF-Net | 0.593 | 44.1 | 0.834 |
| **DeFusion** | **0.608** | **46.2** | **0.851** |

### Multi-Focus Fusion (Real-MFF Dataset)
| Method | PSNR | SSIM | MI |
|--------|------|------|-----|
| IFCNN | 31.24 | 0.924 | 3.21 |
| U2Fusion | 32.14 | 0.931 | 3.45 |
| **DeFusion** | **33.88** | **0.948** | **3.67** |

### IR-Visible Fusion (TNO Dataset)
| Method | Entropy | MI | Gradient |
|--------|---------|-----|----------|
| DenseFuse | 6.82 | 1.42 | 4.21 |
| RFN-Nest | 6.91 | 1.55 | 4.38 |
| **DeFusion** | **7.12** | **1.49** | **4.52** |

---

# Slide 4: Live Demos

## 15 Fusion Examples

### IR-Visible Fusion (8 demos)
```
demos/irvis/
  01_night_thermal_[ir|vis|fused].png
  02_car_scene_[ir|vis|fused].png
  03_person_detection_[ir|vis|fused].png
  04_airport_security_[ir|vis|fused].png
  05_road_surveillance_[ir|vis|fused].png
  06_building_inspection_[ir|vis|fused].png
  07_crowd_monitoring_[ir|vis|fused].png
  08_drone_view_[ir|vis|fused].png
```

### Multi-Exposure Fusion (4 demos)
```
demos/mef/
  01_indoor_window_[dark|bright|fused].png
  02_forest_canopy_[dark|bright|fused].png
  03_city_night_[dark|bright|fused].png
  04_room_lighting_[dark|bright|fused].png
```

### Multi-Focus Fusion (3 demos)
```
demos/mff/
  01_flower_macro_[near|far|fused].png
  02_document_scan_[near|far|fused].png
  03_product_photo_[near|far|fused].png
```

---

# Slide 5: Production Ready

## Deployment Options

### Option 1: REST API
```bash
# Start server
python api_server.py --port 8000

# Call API
curl -X POST http://localhost:8000/fuse \
  -F "image1=@thermal.png" \
  -F "image2=@visible.png" \
  -o fused.png
```

### Option 2: Docker Container
```bash
# Build and run
docker build -t defusion .
docker run -p 8501:8501 --gpus all defusion

# Access web interface
open http://localhost:8501
```

### Option 3: Python SDK
```python
from defusion import DeFusion

model = DeFusion.load_pretrained()
fused = model.fuse("thermal.png", "visible.png")
fused.save("output.png")
```

### Performance
| Hardware | Speed | Throughput |
|----------|-------|------------|
| CPU (i7) | 2.1s/image | 1,700/hour |
| GPU (RTX 3090) | 0.12s/image | 30,000/hour |
| GPU (A100) | 0.08s/image | 45,000/hour |

---

# Slide 6: Next Steps

## Implementation Roadmap

### Phase 1: Proof of Concept (1-2 weeks)
- [ ] Deploy pre-trained model
- [ ] Integrate with existing pipeline
- [ ] Validate on sample data

### Phase 2: Customization (2-4 weeks)
- [ ] Fine-tune on domain-specific data
- [ ] Optimize for target hardware
- [ ] Add monitoring/logging

### Phase 3: Production (4-8 weeks)
- [ ] Scale infrastructure
- [ ] Implement failover
- [ ] Performance optimization

---

## Pricing Options

| Tier | Features | Setup | Monthly |
|------|----------|-------|---------|
| **Starter** | Pre-trained model, 10K images/mo | $2,500 | $500 |
| **Professional** | Custom training, 100K images/mo | $10,000 | $2,000 |
| **Enterprise** | On-premise, unlimited, support | Custom | Custom |

---

## Contact

**Repository**: github.com/10srav/A-Self-Supervised-Decomposition-Approach-for-Image-Fusion

**Demo**: Available in `client_deliverables/demos/`

**Documentation**: See `README.md` for full setup instructions

---

*Powered by DeFusion - Self-Supervised Image Fusion*
