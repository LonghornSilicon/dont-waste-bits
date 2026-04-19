#!/bin/bash
# Setup script for NVIDIA Brev GPU environment (RTX 4090)
# Run this after provisioning a Brev instance

set -e

echo "=== Don't Waste Bits! Verification Setup ==="
echo "Hardware: NVIDIA RTX 4090 (24GB)"
echo ""

# System deps
pip install --upgrade pip

# Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers and evaluation
pip install transformers accelerate
pip install lm-eval  # EleutherAI lm-evaluation-harness

# KV quantization libraries
pip install bitsandbytes  # for static quantization baselines
pip install quanto         # HuggingFace quantization toolkit

# Research utilities
pip install pandas numpy matplotlib seaborn jupyter

# Clone original DWB repo
git clone https://github.com/SayedPedramHaeri/Dont-Waste-Bits ./src/dont-waste-bits-original
cd src/dont-waste-bits-original && pip install -r requirements.txt 2>/dev/null || true && cd ../..

echo ""
echo "=== Verifying GPU ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

echo ""
echo "=== Setup complete ==="
echo "Next: python research/src/run_baselines.py"
