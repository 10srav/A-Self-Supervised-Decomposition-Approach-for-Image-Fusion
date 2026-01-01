# DeFusion: Self-Supervised Image Fusion
# Production Docker Image

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY defusion/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY defusion/ /app/

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/data /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache/torch

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["python", "-c", "from models.defusion import DeFusion; print('DeFusion ready!')"]

# Usage examples:
# Build: docker build -t defusion .
# Train: docker run --gpus all -v ./data:/app/data defusion python train.py --data_path /app/data
# Streamlit: docker run --gpus all -p 8501:8501 defusion streamlit run app.py --server.address 0.0.0.0
# Inference: docker run --gpus all -v ./images:/app/images defusion python test_fusion.py --i1 /app/images/1.png --i2 /app/images/2.png
