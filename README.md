# 2-Simplex

Fast and Simple: **2-Simplicial Attention** implemented in PyTorch with high-performance **Triton GPU kernels**.

## Overview

This project implements a high-performance **2-Simplicial Attention** mechanism designed for geometric reasoning, as explored in recent AlphaGeometry distillations. Unlike traditional graph-based implementations, these **Triton GPU kernels** utilize an efficient **Sliding Window** approach.

Tokens in a sequence (e.g., geometric primitives, proof steps) naturally form 2-simplices with their physical neighbors. A token at position $i$ attends to pairs of tokens in its preceding windows $[i-w_1, i]$ and $[i-w_2, i]$, capturing local trilinear relationships without the overhead of explicit graph adjacency matrices.

### Key Features

- **Sliding Window 2-Simplicial Attention**: Trilinear attention over local temporal/spatial windows.
- **AlphaGeometry Optimized**: Designed to capture geometric relations (like `cong a b c d`) through sequence locality.
- **Triton Kernels**: Fully functional, ultra-fast Forward and Backward kernels.
- **Mathematical Parity**: Rigorously validated against PyTorch reference implementations.
- **Vanilla PyTorch fallback**: Transparent fallback to standard PyTorch for CPU or non-Triton environments.

## Project Structure

```
2-simplex/
├── configs/                  # YAML configuration files
├── src/                      # Source code
│   ├── config/               # Config loading utilities
│   ├── kernels/              # Triton GPU kernels (Forward/Backward/Launcher)
│   └── models/               # PyTorch model definitions
├── tests/                    # Comprehensive Test Suite
│   ├── core/                 # Model logic and math correctness
│   ├── edge_cases/           # Numerical stability and graph validation
│   └── triton/               # Triton kernel parity and performance tests
├── scripts/                  # Training and utility scripts
└── requirements.txt          # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/and-per-i/2-simplex.git
cd 2-simplex

# Install dependencies
pip install -r requirements.txt

# Triton requires an NVIDIA GPU with compatible drivers
pip install triton
```

## Usage

### Basic Example

```python
import torch
from src.models.two_simplicial_attention import TwoSimplicialAttention

# Initialize model with sliding window sizes w1, w2
model = TwoSimplicialAttention(
    in_dim=32,
    out_dim=64,
    num_heads=4,
    w1=8,
    w2=8,
    use_triton_kernel=True
).cuda()

# Input: Sequence of 1024 tokens, 32-dim features
x = torch.randn(1024, 32).cuda()

# Forward pass (uses optimized Sliding Window Triton kernels)
output = model(x)
print(output.shape)  # (1024, 64)
```

## Mathematical Foundation

The 2-simplicial attention mechanism computes trilinear scores over a sliding window:

- **Projections**: $Q = XW_Q, K = XW_K, V = XW_V, K' = XW_{K'}, V' = XW_{V'}$
- **Attention score**: $A_{ijk} = \frac{1}{\sqrt{d}} \langle q_i, k_j \odot k'_k \rangle$ where $j \in [i-w_1, i]$ and $k \in [i-w_2, i]$.
- **Softmax**: $S_{ijk} = \text{softmax}_{j,k}(A_{ijk})$
- **Output**: $v_i = \sum_{j,k} S_{ijk} (v_j \odot v'_k)$

## Performance & Validation

The implementation has been validated on NVIDIA hardware:
- **Numerical Parity**: Triton kernels match PyTorch reference within $10^{-2}$ absolute tolerance for BF16/TF32.
- **Autograd Integration**: Seamlessly integrated into PyTorch `autograd` via custom `torch.autograd.Function`.
- **Layout**: Optimized for `[Batch, Seq, Head, Dim]` memory layout.

## Running Tests

To run the full validation suite (requires CUDA for kernel tests):

```bash
export PYTHONPATH=$PYTHONPATH:.
pytest tests/ -v
```

To run only the core PyTorch logic (CPU compatible):

```bash
pytest tests/core/ -v
```

## Status

- [x] Sliding Window 2-Simplicial Attention (PyTorch)
- [x] Optimized Triton Forward Kernel (Sliding Window)
- [x] Optimized Triton Backward Kernel (Sliding Window)
- [x] Numerical Validation (CUDA)
- [ ] Multi-Batch Support (currently optimized for $B=1$)
- [ ] AlphaGeometry Distillation Pipeline

## License

MIT
