#!/bin/bash
# Setup script for 2-simplex Triton kernels on CUDA cloud environment

echo "Installing/Updating dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install triton pyyaml pytest numpy

echo "Checking CUDA availability..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

echo "Setup complete. Run tests with: export PYTHONPATH=\$PYTHONPATH:. && pytest tests/test_cuda_kernels.py"
