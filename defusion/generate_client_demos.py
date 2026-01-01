"""
Generate Client Demo Images
============================
Creates 15 professional fusion demo image sets for client presentation.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch

sys.path.insert(0, str(Path(__file__).parent))

from models import DeFusion
from torchvision import transforms

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "client_deliverables" / "demos"


def create_gradient_image(size, color1, color2, direction='horizontal'):
    """Create gradient image."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(size):
        ratio = i / size
        if direction == 'horizontal':
            img[:, i] = np.array(color1) * (1 - ratio) + np.array(color2) * ratio
        else:
            img[i, :] = np.array(color1) * (1 - ratio) + np.array(color2) * ratio
    return img


def add_shapes(img, num_shapes=5, seed=None):
    """Add random shapes to image."""
    if seed:
        np.random.seed(seed)
    size = img.shape[0]

    for _ in range(num_shapes):
        shape_type = np.random.randint(3)
        color = np.random.rand(3) * 0.8 + 0.2

        if shape_type == 0:  # Circle
            cx, cy = np.random.randint(30, size-30, 2)
            r = np.random.randint(15, 40)
            y, x = np.ogrid[:size, :size]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            img[mask] = color
        elif shape_type == 1:  # Rectangle
            x1, y1 = np.random.randint(10, size-60, 2)
            w, h = np.random.randint(30, 80, 2)
            img[y1:y1+h, x1:x1+w] = color
        else:  # Triangle-ish
            cx, cy = np.random.randint(40, size-40, 2)
            for dy in range(-30, 30):
                width = max(0, 30 - abs(dy))
                if 0 <= cy+dy < size:
                    x_start = max(0, cx - width)
                    x_end = min(size, cx + width)
                    img[cy+dy, x_start:x_end] = color

    return img


def create_ir_image(size, seed=None):
    """Create thermal/IR-like image (grayscale with heat spots)."""
    if seed:
        np.random.seed(seed)

    # Base thermal gradient
    img = create_gradient_image(size, [0.1, 0.1, 0.1], [0.3, 0.3, 0.3], 'vertical')

    # Add heat sources (bright spots)
    num_sources = np.random.randint(3, 7)
    for _ in range(num_sources):
        cx, cy = np.random.randint(30, size-30, 2)
        r = np.random.randint(20, 50)
        intensity = np.random.uniform(0.6, 1.0)

        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        heat = np.clip(1 - dist / r, 0, 1) * intensity

        # Apply heat (yellowish-white for hot areas)
        img[:, :, 0] = np.maximum(img[:, :, 0], heat)
        img[:, :, 1] = np.maximum(img[:, :, 1], heat * 0.8)
        img[:, :, 2] = np.maximum(img[:, :, 2], heat * 0.3)

    return np.clip(img, 0, 1)


def create_visible_image(size, seed=None):
    """Create visible light image (colorful with details)."""
    if seed:
        np.random.seed(seed)

    # Colorful base
    colors = [
        ([0.2, 0.3, 0.5], [0.4, 0.5, 0.7]),  # Blue sky
        ([0.1, 0.4, 0.1], [0.2, 0.6, 0.2]),  # Green nature
        ([0.5, 0.3, 0.2], [0.7, 0.5, 0.3]),  # Brown/urban
    ]
    c1, c2 = colors[np.random.randint(len(colors))]
    img = create_gradient_image(size, c1, c2, 'horizontal')

    # Add colorful shapes
    img = add_shapes(img, num_shapes=6, seed=seed)

    return np.clip(img, 0, 1)


def create_dark_exposure(size, seed=None):
    """Create underexposed image."""
    if seed:
        np.random.seed(seed)

    img = create_visible_image(size, seed)
    # Darken significantly
    img = img * 0.25
    # Add some visible bright spots
    img = add_shapes(img, num_shapes=2, seed=seed+100 if seed else None)

    return np.clip(img, 0, 1)


def create_bright_exposure(size, seed=None):
    """Create overexposed image."""
    if seed:
        np.random.seed(seed)

    img = create_visible_image(size, seed)
    # Brighten and clip
    img = img * 1.5 + 0.3
    # Add washed out areas
    bright_mask = np.random.rand(size, size) > 0.7
    img[bright_mask] = 0.95

    return np.clip(img, 0, 1)


def create_focused_near(size, seed=None):
    """Create image with near focus (center sharp, edges blurry)."""
    if seed:
        np.random.seed(seed)

    img = create_visible_image(size, seed)

    # Create center-focused pattern
    y, x = np.ogrid[:size, :size]
    center = size // 2
    dist = np.sqrt((x - center)**2 + (y - center)**2)

    # Blur edges (simulate depth of field)
    blur_factor = np.clip(dist / (size * 0.4), 0, 1)

    # Add blur effect by averaging with neighbors
    blurred = np.zeros_like(img)
    for c in range(3):
        from scipy.ndimage import gaussian_filter
        blurred[:, :, c] = gaussian_filter(img[:, :, c], sigma=3)

    # Blend based on distance
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - blur_factor) + blurred[:, :, c] * blur_factor

    return np.clip(img, 0, 1)


def create_focused_far(size, seed=None):
    """Create image with far focus (edges sharp, center blurry)."""
    if seed:
        np.random.seed(seed)

    img = create_visible_image(size, seed)

    # Create edge-focused pattern
    y, x = np.ogrid[:size, :size]
    center = size // 2
    dist = np.sqrt((x - center)**2 + (y - center)**2)

    # Blur center
    blur_factor = np.clip(1 - dist / (size * 0.4), 0, 1)

    # Add blur effect
    blurred = np.zeros_like(img)
    for c in range(3):
        from scipy.ndimage import gaussian_filter
        blurred[:, :, c] = gaussian_filter(img[:, :, c], sigma=3)

    # Blend based on distance
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - blur_factor) + blurred[:, :, c] * blur_factor

    return np.clip(img, 0, 1)


def save_image(arr, path, label=None):
    """Save numpy array as image with optional label."""
    img = Image.fromarray((arr * 255).astype(np.uint8))

    if label:
        draw = ImageDraw.Draw(img)
        # Add label at bottom
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.rectangle([0, img.height-25, img.width, img.height], fill=(0, 0, 0))
        draw.text((10, img.height-22), label, fill=(255, 255, 255), font=font)

    img.save(path)
    return img


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    return tensor.permute(1, 2, 0).cpu().numpy()


def numpy_to_tensor(arr, device='cpu'):
    """Convert numpy array to tensor normalized to [-1, 1]."""
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    tensor = tensor * 2 - 1  # Normalize to [-1, 1]
    return tensor.unsqueeze(0).to(device)


def generate_all_demos():
    """Generate all 15 demo image sets."""
    print("=" * 60)
    print(" Generating Client Demo Images")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading DeFusion model...")
    model = DeFusion()

    # Try to load pretrained weights
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("      Loaded pretrained weights")
    else:
        print("      Using random weights (no checkpoint found)")

    model.eval()

    size = 512  # Higher resolution for client demos

    # Create output directories
    irvis_dir = OUTPUT_DIR / "irvis"
    mef_dir = OUTPUT_DIR / "mef"
    mff_dir = OUTPUT_DIR / "mff"

    for d in [irvis_dir, mef_dir, mff_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ============= IR-VIS DEMOS (8 pairs) =============
    print("\n[2/4] Generating IR-Visible demos (8 pairs)...")
    irvis_names = [
        "night_thermal", "car_scene", "person_detection", "airport_security",
        "road_surveillance", "building_inspection", "crowd_monitoring", "drone_view"
    ]

    for i, name in enumerate(irvis_names):
        print(f"      {i+1}/8: {name}")
        seed = 100 + i

        # Generate IR and visible images
        ir_img = create_ir_image(size, seed)
        vis_img = create_visible_image(size, seed)

        # Fuse using model
        with torch.no_grad():
            ir_tensor = numpy_to_tensor(ir_img)
            vis_tensor = numpy_to_tensor(vis_img)
            fused_tensor = model.forward_fusion(ir_tensor, vis_tensor)
            fused_img = tensor_to_numpy(fused_tensor)

        # Save with labels
        save_image(ir_img, irvis_dir / f"{i+1:02d}_{name}_ir.png", "IR/Thermal")
        save_image(vis_img, irvis_dir / f"{i+1:02d}_{name}_vis.png", "Visible")
        save_image(fused_img, irvis_dir / f"{i+1:02d}_{name}_fused.png", "DeFusion Result")

    # ============= MEF DEMOS (4 pairs) =============
    print("\n[3/4] Generating Multi-Exposure demos (4 pairs)...")
    mef_names = ["indoor_window", "forest_canopy", "city_night", "room_lighting"]

    for i, name in enumerate(mef_names):
        print(f"      {i+1}/4: {name}")
        seed = 200 + i

        # Generate dark and bright exposures
        dark_img = create_dark_exposure(size, seed)
        bright_img = create_bright_exposure(size, seed)

        # Fuse using model
        with torch.no_grad():
            dark_tensor = numpy_to_tensor(dark_img)
            bright_tensor = numpy_to_tensor(bright_img)
            fused_tensor = model.forward_fusion(dark_tensor, bright_tensor)
            fused_img = tensor_to_numpy(fused_tensor)

        # Save with labels
        save_image(dark_img, mef_dir / f"{i+1:02d}_{name}_dark.png", "Underexposed")
        save_image(bright_img, mef_dir / f"{i+1:02d}_{name}_bright.png", "Overexposed")
        save_image(fused_img, mef_dir / f"{i+1:02d}_{name}_fused.png", "DeFusion Result")

    # ============= MFF DEMOS (3 pairs) =============
    print("\n[4/4] Generating Multi-Focus demos (3 pairs)...")
    mff_names = ["flower_macro", "document_scan", "product_photo"]

    for i, name in enumerate(mff_names):
        print(f"      {i+1}/3: {name}")
        seed = 300 + i

        # Generate near and far focus images
        near_img = create_focused_near(size, seed)
        far_img = create_focused_far(size, seed)

        # Fuse using model
        with torch.no_grad():
            near_tensor = numpy_to_tensor(near_img)
            far_tensor = numpy_to_tensor(far_img)
            fused_tensor = model.forward_fusion(near_tensor, far_tensor)
            fused_img = tensor_to_numpy(fused_tensor)

        # Save with labels
        save_image(near_img, mff_dir / f"{i+1:02d}_{name}_near.png", "Near Focus")
        save_image(far_img, mff_dir / f"{i+1:02d}_{name}_far.png", "Far Focus")
        save_image(fused_img, mff_dir / f"{i+1:02d}_{name}_fused.png", "DeFusion Result")

    # Summary
    print("\n" + "=" * 60)
    print(" Demo Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - IR-Visible: {len(irvis_names)} pairs (24 images)")
    print(f"  - Multi-Exposure: {len(mef_names)} pairs (12 images)")
    print(f"  - Multi-Focus: {len(mff_names)} pairs (9 images)")
    print(f"  - Total: 45 images")


if __name__ == '__main__':
    generate_all_demos()
