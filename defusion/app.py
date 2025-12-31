"""
DeFusion Streamlit Web App
==========================
Interactive web interface for DeFusion image fusion.

Run with: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.defusion import DeFusion

# Page configuration
st.set_page_config(
    page_title="DeFusion - Image Fusion",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path=None):
    """Load DeFusion model (cached)."""
    model = DeFusion()

    # Auto-detect trained model if no path specified
    if not checkpoint_path:
        default_checkpoint = Path(__file__).parent / 'checkpoints' / 'best_model.pth'
        if default_checkpoint.exists():
            checkpoint_path = str(default_checkpoint)

    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        st.success(f"Loaded trained model from {checkpoint_path}")
    else:
        st.warning("Using untrained model (random weights). Train with: python train_demo.py --epochs 20")

    model.eval()
    return model


def preprocess_image(image, target_size=(256, 256)):
    """Preprocess uploaded image."""
    # Resize
    image = image.resize(target_size, Image.BILINEAR)

    # Convert to tensor
    img_array = np.array(image).astype(np.float32) / 255.0

    # Handle grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # To tensor [C, H, W]
    tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

    # Normalize to [-1, 1]
    tensor = tensor * 2 - 1

    return tensor


def postprocess_output(tensor):
    """Convert output tensor to displayable image."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)

    # To numpy [H, W, C]
    img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)

    return Image.fromarray(img_array)


def create_demo_images():
    """Create demo images for testing."""
    size = (256, 256)

    # Demo 1: Gradient
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img1[:, i, :] = i

    # Demo 2: Circle
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img2[i, :, :] = i
    y, x = np.ogrid[:256, :256]
    mask = (x - 128)**2 + (y - 128)**2 < 60**2
    img2[mask] = [200, 50, 80]

    return Image.fromarray(img1), Image.fromarray(img2)


def main():
    # Header
    st.markdown('<p class="main-header">🔬 DeFusion Image Fusion</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Self-Supervised Decomposition Approach for Image Fusion</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model settings
        st.subheader("Model")
        checkpoint_path = st.text_input(
            "Checkpoint Path (optional)",
            placeholder="path/to/model.pth",
            help="Leave empty to use untrained model"
        )

        # Image settings
        st.subheader("Processing")
        image_size = st.selectbox("Image Size", [256, 512], index=0)

        # Info
        st.divider()
        st.subheader("📊 Model Info")
        st.info("""
        **DeFusion Architecture:**
        - Parameters: 17.7M
        - Input: 256×256 RGB
        - Features: fc, f1u, f2u
        - Output: Fused image
        """)

        st.subheader("📝 Paper")
        st.markdown("""
        *"Fusion from Decomposition: A Self-Supervised
        Decomposition Approach for Image Fusion"*
        """)

    # Load model
    model = load_model(checkpoint_path if checkpoint_path else None)

    # Main content
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Fusion", "🎬 Demo", "📈 About"])

    with tab1:
        st.header("Upload Images for Fusion")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Source Image 1")
            uploaded_file1 = st.file_uploader(
                "Upload first image",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="img1"
            )

            if uploaded_file1:
                image1 = Image.open(uploaded_file1).convert('RGB')
                st.image(image1, caption="Source 1", use_container_width=True)

        with col2:
            st.subheader("Source Image 2")
            uploaded_file2 = st.file_uploader(
                "Upload second image",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="img2"
            )

            if uploaded_file2:
                image2 = Image.open(uploaded_file2).convert('RGB')
                st.image(image2, caption="Source 2", use_container_width=True)

        # Fuse button
        if uploaded_file1 and uploaded_file2:
            st.divider()

            if st.button("🔄 Fuse Images", type="primary", use_container_width=True):
                with st.spinner("Fusing images..."):
                    # Preprocess
                    t1 = preprocess_image(image1, (image_size, image_size))
                    t2 = preprocess_image(image2, (image_size, image_size))

                    # Add batch dimension
                    t1 = t1.unsqueeze(0)
                    t2 = t2.unsqueeze(0)

                    # Resize to 256 for model
                    if image_size != 256:
                        t1_model = F.interpolate(t1, (256, 256), mode='bilinear')
                        t2_model = F.interpolate(t2, (256, 256), mode='bilinear')
                    else:
                        t1_model, t2_model = t1, t2

                    # Fuse
                    start_time = time.time()
                    with torch.no_grad():
                        fused, fc, f1u, f2u = model(t1_model, t2_model)
                    inference_time = (time.time() - start_time) * 1000

                    # Resize back if needed
                    if image_size != 256:
                        fused = F.interpolate(fused, (image_size, image_size), mode='bilinear')

                    # Postprocess
                    fused_image = postprocess_output(fused)

                # Display result
                st.success(f"Fusion completed in {inference_time:.1f} ms!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(image1.resize((image_size, image_size)), caption="Source 1")

                with col2:
                    st.image(image2.resize((image_size, image_size)), caption="Source 2")

                with col3:
                    st.image(fused_image, caption="🎯 Fused Result")

                # Download button
                buf = io.BytesIO()
                fused_image.save(buf, format='PNG')
                st.download_button(
                    label="📥 Download Fused Image",
                    data=buf.getvalue(),
                    file_name="fused_result.png",
                    mime="image/png",
                    use_container_width=True
                )

                # Metrics
                st.divider()
                st.subheader("📊 Inference Metrics")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Inference Time", f"{inference_time:.1f} ms")
                m2.metric("FPS", f"{1000/inference_time:.1f}")
                m3.metric("Image Size", f"{image_size}×{image_size}")
                m4.metric("Model", "DeFusion")

    with tab2:
        st.header("🎬 Demo with Sample Images")
        st.write("Test the model with automatically generated demo images.")

        if st.button("🎲 Generate Demo Images & Fuse", use_container_width=True):
            with st.spinner("Generating and fusing..."):
                # Create demo images
                demo1, demo2 = create_demo_images()

                # Preprocess
                t1 = preprocess_image(demo1).unsqueeze(0)
                t2 = preprocess_image(demo2).unsqueeze(0)

                # Fuse
                start_time = time.time()
                with torch.no_grad():
                    fused, _, _, _ = model(t1, t2)
                inference_time = (time.time() - start_time) * 1000

                fused_image = postprocess_output(fused)

            st.success(f"Demo completed in {inference_time:.1f} ms!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(demo1, caption="Demo Source 1 (Horizontal Gradient)")

            with col2:
                st.image(demo2, caption="Demo Source 2 (Vertical Gradient + Circle)")

            with col3:
                st.image(fused_image, caption="🎯 Fused Result")

            st.info("""
            **Note:** The output may appear gray if using an untrained model.
            Train on COCO dataset for proper fusion results:
            ```bash
            python train.py --coco_path /path/to/coco --epochs 50
            ```
            """)

    with tab3:
        st.header("📈 About DeFusion")

        st.markdown("""
        ### Overview

        DeFusion is a self-supervised image fusion method that decomposes source images
        into **common** and **unique** features, then fuses them for the final output.

        ### Architecture

        ```
        Source Images (I1, I2)
              ↓
        Encoder (E) - Shared weights
              ↓
        Ensembler (Ec) - Combines features
              ↓
        Decoders (Du, Dc) - Extract unique & common
              ↓
        Projection Heads (Pc, Pu, Pr)
              ↓
        Fused Output
        ```

        ### Training (CUD Pretext Task)

        The model is trained using **Common and Unique Decomposition**:
        1. Generate non-overlapping masks M1, M2
        2. Create augmented views with Gaussian noise
        3. Train to reconstruct original from decomposed features

        ### Supported Fusion Tasks

        - **Multi-Exposure Fusion** (MEFB, SICE)
        - **Multi-Focus Fusion** (Real-MFF)
        - **IR-Visible Fusion** (TNO, RoadScene)

        ### Performance

        | Device | Precision | FPS |
        |--------|-----------|-----|
        | CPU | FP32 | ~1.3 |
        | GPU | FP32 | ~30 |
        | GPU | FP16 | ~60 |

        ### Citation

        ```bibtex
        @article{defusion2023,
          title={Fusion from Decomposition: A Self-Supervised
                 Decomposition Approach for Image Fusion},
          author={...},
          year={2023}
        }
        ```
        """)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        DeFusion - Self-Supervised Image Fusion | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
