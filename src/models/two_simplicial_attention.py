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

    def __init__(self, in_dim, out_dim=None, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=True, w1=8, w2=8):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim) if out_dim is not None else self.in_dim
        self.num_heads = int(num_heads)
        assert self.out_dim % self.num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = self.out_dim // self.num_heads
        self.dropout = dropout
        self.with_residual = with_residual
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

    def forward(self, x, batch=None):
        """
        x: Tensor of shape (N, in_dim)
        batch: not used in MVP
        Returns: Tensor of shape (N, out_dim)
        """
        if x.dim() != 2:
            raise ValueError("input x must be (N, in_dim)")
        N, _ = x.shape
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)  # (N, H, D)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        Kp = self.kp_proj(x).view(N, self.num_heads, self.head_dim)
        Vp = self.vp_proj(x).view(N, self.num_heads, self.head_dim)

        # Use optimized Triton path if requested and on CUDA
        if self.use_triton_kernel and x.is_cuda:
            try:
                # Lazy import; tests/local envs may not have Triton kernel available yet
                from ..kernels.two_simplicial_autograd import TwoSimplicialAttentionFunction  # type: ignore
                Z_concat = TwoSimplicialAttentionFunction.apply(
                    x, Q, K, V, Kp, Vp, self.out_dim, self.num_heads, self.head_dim, self.w1, self.w2
                ).reshape(N, self.out_dim)
            except Exception:
                # Fall back to PyTorch MVP if Triton kernel fails or is unavailable
                Z_concat = self._forward_pytorch(N, Q, K, V, Kp, Vp)
        else:
            # Standard PyTorch path (CPU or explicitly requested)
            Z_concat = self._forward_pytorch(N, Q, K, V, Kp, Vp)

        out = self.out_proj(Z_concat)
        if self.with_residual and out.shape == x.shape:
            out = out + x
        out = self.norm(out)
        return out

    def _forward_pytorch(self, N, Q, K, V, Kp, Vp):
        """Vectorized PyTorch implementation with Sliding Window (No loops)."""
        device = Q.device
        dtype = Q.dtype
        H, D = self.num_heads, self.head_dim

        # 1. Create sliding window views for K and Kp
        # We want tensors of shape (N, W, H, D) where W is the window size.
        # We pad the sequences at the beginning to handle the first tokens.
        def get_windows(tensor, window_size):
            # Pad with zeros at the start: (window_size-1, H, D)
            padded = torch.cat([torch.zeros(window_size - 1, H, D, device=device, dtype=dtype), tensor], dim=0)
            # Use unfold to get sliding windows: (N, window_size, H, D)
            return padded.unfold(0, window_size, 1).permute(0, 3, 1, 2)

        K_win = get_windows(K, self.w1)   # (N, w1, H, D)
        Kp_win = get_windows(Kp, self.w2) # (N, w2, H, D)
        V_win = get_windows(V, self.w1)   # (N, w1, H, D)
        Vp_win = get_windows(Vp, self.w2) # (N, w2, H, D)

        # 2. Compute attention scores A_ijk = (qi * kj * kkp) / sqrt(d)
        # Q: (N, H, D) -> (N, 1, 1, H, D)
        # K_win: (N, w1, H, D) -> (N, w1, 1, H, D)
        # Kp_win: (N, w2, H, D) -> (N, 1, w2, H, D)
        
        # Optimized einsum: (N, H, w1, w2)
        # A_ijk[n, h, j, k] = sum_d Q[n, h, d] * K_win[n, j, h, d] * Kp_win[n, k, h, d]
        A = torch.einsum('nhd,njhd,nkhd->nhjk', Q, K_win, Kp_win) / (D ** 0.5)

        # 3. Masking: indices in the window that are actually padding (before the sequence start)
        # For each i, the window covers [i-w+1, i]. If i-w+1 < 0, some elements are padding.
        # We create a mask of shape (N, w1, w2)
        m1 = torch.arange(self.w1, device=device).unsqueeze(0) # (1, w1)
        m2 = torch.arange(self.w2, device=device).unsqueeze(0) # (1, w2)
        
        # Valid indices for i are those where (window_index) >= (w - 1 - i)
        row_idx = torch.arange(N, device=device).unsqueeze(1) # (N, 1)
        mask1 = m1 >= (self.w1 - 1 - row_idx) # (N, w1)
        mask2 = m2 >= (self.w2 - 1 - row_idx) # (N, w2)
        
        # Final mask (N, 1, w1, w2)
        mask = (mask1.unsqueeze(2) & mask2.unsqueeze(1)).unsqueeze(1)
        A = A.masked_fill(~mask, float('-inf'))

        # 4. Softmax and Value aggregation
        S = F.softmax(A.reshape(N, H, -1), dim=-1).reshape(N, H, self.w1, self.w2)
        S = self.drop(S)

        # Aggregation: v~_i = sum_{j,k} S_ijk (vj * vkp)
        # S: (N, H, w1, w2)
        # V_win: (N, w1, H, D)
        # Vp_win: (N, w2, H, D)
        # Result: (N, H, D)
        Z = torch.einsum('nhjk,njhd,nkhd->nhd', S, V_win, Vp_win)
        
        return Z.reshape(N, self.out_dim)
