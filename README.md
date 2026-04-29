# 2-Simplex Geometry

Fast and Simple: **2-Simplicial Attention** implemented in PyTorch with high-performance **Triton GPU kernels**.

## Overview

This project implements a high-performance **2-Simplicial Attention** mechanism designed for geometric reasoning, as explored in recent AlphaGeometry distillations. Unlike traditional graph-based implementations, these **Triton GPU kernels** utilize an efficient **Sliding Window** approach.

Tokens in a sequence (e.g., geometric primitives, proof steps) naturally form 2-simplices with their physical neighbors. A token at position $i$ attends to pairs of tokens in its preceding windows $[i-w_1, i]$ and $[i-w_2, i]$, capturing local trilinear relationships without the overhead of explicit graph adjacency matrices.

### Key Features

- **Sliding Window 2-Simplicial Attention**: Trilinear attention over local temporal/spatial windows.
- **AlphaGeometry Optimized**: Designed to capture geometric relations (like `cong a b c d`) through sequence locality.
- **Triton GPU Kernels**: Ultra-fast Forward and Backward kernels optimized for TF32/BF16.
- **Strictly Causal**: Guaranteed zero future-token leakage during autoregressive training.
- **Native Batched Support**: Supports `Batch Size > 1` in all paths (Triton and PyTorch).
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
git clone https://github.com/and-per-i/2-simplex-geometry.git
cd 2-simplex-geometry

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

## Neuro-Symbolic Architecture & Test-Time Compute

This repository implements a state-of-the-art **Neuro-Symbolic** search pipeline inspired by AlphaGeometry, designed to solve complex International Mathematical Olympiad (IMO) geometry problems through **Test-Time Compute**.

The system pairs the neural intuition of our **2-Simplicial Transformer** with the infallible logical rigor of **Newclid/DDARN** (a fast symbolic deduction engine). 

### The Strategy: Massively Parallel Beam Search
1. **Pass@N (Massive Sampling):** The 2-Simplex neural model generates thousands of auxiliary constructions (e.g., $k=2048$) at a high temperature (`Temp=0.9`). This prompts the model to "hallucinate" creative geometric topologies.
2. **GPU Batch Generation:** Leveraging PyTorch batched inference, the GPU computes thousands of geometric hypotheses in milliseconds.
3. **Symbolic Sieve:** The hypotheses are fed to a massive multi-processing CPU cluster (e.g., 128-core AMD EPYC). The DDARN symbolic engine evaluates them in parallel, instantly discarding illegal or useless constructions, keeping only mathematically sound deduction paths.
4. **Iterative Deepening:** If a problem requires multiple auxiliary steps, the system feeds the successful intermediate points back into the Transformer, recursing up to `Depth 3`. This creates a dynamic search tree that evaluates thousands of complex logical chains.

## Status

- [x] Sliding Window 2-Simplicial Attention (PyTorch & Triton)
- [x] Optimized Triton Forward & Backward Kernels (Sliding Window)
- [x] Strictly Causal Sliding Window (Zero future-token leakage)
- [x] Native Multi-Batch Training Support (B > 1)
- [x] Optimized Backward Pass (Recompute-free)
- [x] HF-Compatible Tokenizer & Trainer Integration
- [x] Stable AlphaGeometry Distillation Pipeline
- [x] Human-Readable Geometry Translator
- [x] Batched Generation & Beam Search Inference Support
- [x] Newclid / DDARN Symbolic Engine Integration
- [x] Deep Test-Time Compute Beam Search (Multi-processing CPU)

## Future Work

To reach full IMO-level problem-solving capabilities, the following upgrades are planned:
- **150M+ Parameter Distillation:** Scale the `StudentModelProgressive` to 150M+ parameters and distill from a massive 1B+ parameter geometry model trained on 100M synthetic datasets. This will sharpen the geometric intuition for `Pass@2048`.
- **Rolling Buffer KV Cache:** Implement an $O(1)$ memory Rolling Buffer KV Cache directly inside the Triton kernels. Since 2-Simplicial Attention operates on a restricted sliding window, the KV Cache size is strictly bounded, enabling infinitely long generation without memory explosion.

## License

MIT
