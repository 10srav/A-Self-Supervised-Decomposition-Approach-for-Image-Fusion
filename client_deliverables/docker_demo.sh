#!/bin/bash
# ============================================================
#  DeFusion Client Demo Script
#  One-command setup for demonstration
# ============================================================

set -e

echo "============================================================"
echo "  DeFusion - Self-Supervised Image Fusion Demo"
echo "============================================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed!"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Configuration
IMAGE_NAME="defusion"
CONTAINER_NAME="defusion-demo"
PORT=8501

# Build if needed
echo "[1/4] Building Docker image..."
cd "$(dirname "$0")/.."
docker build -t $IMAGE_NAME . || {
    echo "Build failed. Trying without GPU..."
    docker build -t $IMAGE_NAME .
}

# Stop existing container
echo "[2/4] Stopping any existing demo..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run container
echo "[3/4] Starting DeFusion demo..."
if command -v nvidia-smi &> /dev/null; then
    echo "      GPU detected - running with CUDA"
    docker run -d \
        --name $CONTAINER_NAME \
        --gpus all \
        -p $PORT:8501 \
        -v "$(pwd)/client_deliverables/demos:/app/demo_images" \
        $IMAGE_NAME \
        streamlit run app.py --server.address 0.0.0.0
else
    echo "      No GPU - running on CPU"
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:8501 \
        -v "$(pwd)/client_deliverables/demos:/app/demo_images" \
        $IMAGE_NAME \
        streamlit run app.py --server.address 0.0.0.0
fi

# Wait for startup
echo "[4/4] Waiting for service to start..."
sleep 5

# Check health
if curl -s http://localhost:$PORT > /dev/null; then
    echo ""
    echo "============================================================"
    echo "  DeFusion Demo is Ready!"
    echo "============================================================"
    echo ""
    echo "  Web Interface: http://localhost:$PORT"
    echo ""
    echo "  Demo images available in: client_deliverables/demos/"
    echo "    - IR-Visible: 8 pairs"
    echo "    - Multi-Exposure: 4 pairs"
    echo "    - Multi-Focus: 3 pairs"
    echo ""
    echo "  Commands:"
    echo "    Stop:  docker stop $CONTAINER_NAME"
    echo "    Logs:  docker logs $CONTAINER_NAME"
    echo "    Shell: docker exec -it $CONTAINER_NAME bash"
    echo ""
    echo "============================================================"

    # Open browser (optional)
    if command -v open &> /dev/null; then
        open "http://localhost:$PORT"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$PORT"
    elif command -v start &> /dev/null; then
        start "http://localhost:$PORT"
    fi
else
    echo ""
    echo "WARNING: Service may still be starting..."
    echo "Check status with: docker logs $CONTAINER_NAME"
    echo "Access at: http://localhost:$PORT"
fi
