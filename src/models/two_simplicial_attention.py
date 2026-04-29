import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoSimplicialAttention(nn.Module):
    """
    Minimal, PyTorch 2.0 vanilla implementation of the 2-simplicial attention layer
    (paper: Fast and Simplex: 2-Simplicial Attention in Triton). This MVP handles
    a single graph (batch size = 1) and expects input as a set of triangulations
    with adjacency provided as an edge_index-like structure.
    The core equations implemented (simplified to MVP) are:
      - Q = X W_Q, K = X W_K, V = X W_V
      - K' = X W_Kp, V' = X W_Vp
      - A_ijk^(2s) = (1 / sqrt(d)) < q_i, k_j, k'_k >
      - S_ijk^(2s) = softmax_{j,k} (A_ijk^(2s))
      - v~_i^(2s) = sum_{j,k} S_ijk^(2s) ( v_j ∘ v'_k )
      - y_i = W_O concat_heads(v~_i^(2s)) with final projection
    Notes:
      - edge_index must be provided as a tensor of shape (N, max_deg) with -1 padding
        for non-existing entries. Each row i contains indices of neighbors j of triangolo i.
      - Core (j,k) loops are vectorized via torch.einsum for performance;
        the per-node loop is kept since each node has a variable number of neighbors.
    """

    def __init__(self, in_dim, out_dim=None, num_heads=4, dropout=0.1, 
                 with_residual=True, with_norm=True, use_triton_kernel=True, w1=8, w2=8):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim) if out_dim is not None else self.in_dim
        self.num_heads = int(num_heads)
        assert self.out_dim % self.num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = self.out_dim // self.num_heads
        self.dropout = dropout
        self.with_residual = with_residual
        self.with_norm = with_norm
        self.use_triton_kernel = bool(use_triton_kernel)
        self.w1 = w1
        self.w2 = w2

        # Projections
        self.q_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.k_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.v_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.kp_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.vp_proj = nn.Linear(self.in_dim, self.out_dim, bias=True)

        self.out_proj = nn.Linear(self.out_dim, self.out_dim, bias=True)
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        x: Tensor of shape (B, S, in_dim) or (N, in_dim)
        Returns: Tensor of shape (B, S, out_dim) or (N, out_dim)
        """
        is_batched = x.dim() == 3
        if is_batched:
            B, S, _ = x.shape
        else:
            if x.dim() != 2:
                raise ValueError("input x must be (B, S, in_dim) or (N, in_dim)")
            N, _ = x.shape
            B, S = 1, N

        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)
        Kp = self.kp_proj(x).view(B, S, self.num_heads, self.head_dim)
        Vp = self.vp_proj(x).view(B, S, self.num_heads, self.head_dim)

        # Use optimized Triton path if requested and on CUDA
        if self.use_triton_kernel and x.is_cuda:
            try:
                from ..kernels.two_simplicial_autograd import TwoSimplicialAttentionFunction
                Z = TwoSimplicialAttentionFunction.apply(
                    x, Q, K, V, Kp, Vp, self.out_dim, self.num_heads, self.head_dim, self.w1, self.w2
                )
            except Exception as e:
                # Robust fallback for any Triton/CUDA error (OOM, shape mismatch, etc.)
                print(f"⚠️ [TwoSimplicialAttention] Triton kernel failed, falling back to PyTorch: {e}")
                Z = self._forward_pytorch(Q, K, V, Kp, Vp)
        else:
            Z = self._forward_pytorch(Q, K, V, Kp, Vp)

        Z_concat = Z.reshape(x.shape[0:-1] + (self.out_dim,))
        out = self.out_proj(Z_concat)
        
        if self.with_residual and out.shape == x.shape:
            out = out + x
        if self.with_norm:
            out = self.norm(out)
        return out

    def _forward_pytorch(self, Q, K, V, Kp, Vp):
        """Vectorized PyTorch implementation with Strictly Causal Sliding Window."""
        B, S, H, D = Q.shape
        device = Q.device
        dtype = Q.dtype

        def get_causal_windows(tensor, window_size):
            # To get [i-w+1, i] for token i:
            # 1. Pad at the START with w-1 zeros
            padded = torch.cat([
                torch.zeros(B, window_size - 1, H, D, device=device, dtype=dtype),
                tensor
            ], dim=1)
            # 2. Unfold on the sequence dimension (dim 1)
            # Result shape: (B, S, H, D, window_size)
            # Index i on dim 1 gives padded[:, i:i+window_size, :, :]
            # which ends at original tensor index i.
            return padded.unfold(1, window_size, 1).permute(0, 1, 4, 2, 3)

        K_win = get_causal_windows(K, self.w1)   # (B, S, w1, H, D)
        Kp_win = get_causal_windows(Kp, self.w2) # (B, S, w2, H, D)
        V_win = get_causal_windows(V, self.w1)   # (B, S, w1, H, D)
        Vp_win = get_causal_windows(Vp, self.w2) # (B, S, w2, H, D)

        # A_ijk[b, s, h, j, k] = sum_d Q[b, s, h, d] * K_win[b, s, j, h, d] * Kp_win[b, s, k, h, d]
        A = torch.einsum('bshd,bsjhd,bskhd->bshjk', Q, K_win, Kp_win) / (D ** 0.5)

        # Mask padding (zeros at the beginning of the sequence)
        # For sequence position s, the window contains w elements.
        # Elements at window index j are valid if s - (w - 1 - j) >= 0
        j_idx = torch.arange(self.w1, device=device).view(1, 1, self.w1)
        s_idx = torch.arange(S, device=device).view(1, S, 1)
        mask1 = j_idx >= (self.w1 - 1 - s_idx) # (1, S, w1)
        
        k_idx = torch.arange(self.w2, device=device).view(1, 1, self.w2)
        mask2 = k_idx >= (self.w2 - 1 - s_idx) # (1, S, w2)
        
        mask = (mask1.unsqueeze(3) & mask2.unsqueeze(2)).unsqueeze(2) # (1, S, 1, w1, w2)
        A = A.masked_fill(~mask, float('-inf'))

        S_attn = F.softmax(A.reshape(B, S, H, -1), dim=-1).reshape(B, S, H, self.w1, self.w2)
        S_attn = self.drop(S_attn)

        # Z[b, s, h, d] = sum_{j,k} S[b, s, h, j, k] * (V_win[b, s, j, h, d] * Vp_win[b, s, k, h, d])
        Z = torch.einsum('bshjk,bsjhd,bskhd->bshd', S_attn, V_win, Vp_win)
        
        return Z

