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
        """Standard PyTorch implementation with Sliding Window (slow loop)."""
        Z_rows = []
        for i in range(N):
            # Sliding window indices for j and k
            # j in [i-w1+1, i], k in [i-w2+1, i] (or [i-w1, i] etc. depending on definition)
            # The Triton kernel uses (q_idx - w1) < kv_idx <= q_idx
            # So indices are [i-w1+1, i]
            
            j_start = max(0, i - self.w1 + 1)
            k_start = max(0, i - self.w2 + 1)
            
            neigh_j = torch.arange(j_start, i + 1, device=Q.device)
            neigh_k = torch.arange(k_start, i + 1, device=Q.device)
            
            K_j = K[neigh_j]    # (d1, H, D)
            V_j = V[neigh_j]
            Kp_k = Kp[neigh_k]
            Vp_k = Vp[neigh_k]

            q_i = Q[i]  # (H, D)
            head_outs = []
            for h in range(self.num_heads):
                qi = q_i[h]  # (D,)
                kj = K_j[:, h, :]   # (d1, D)
                kkp = Kp_k[:, h, :] # (d2, D)
                vj = V_j[:, h, :]   # (d1, D)
                vkp = Vp_k[:, h, :] # (d2, D)

                # A_ijk = (qi * kj * kkp) / sqrt(d)
                # This is (d1, d2)
                A_ijk = torch.einsum('d,jd,kd->jk', qi, kj, kkp) / (self.head_dim ** 0.5)

                S_flat = F.softmax(A_ijk.reshape(-1), dim=0)
                S = self.drop(S_flat.view(len(neigh_j), len(neigh_k)))

                # head_out = sum_{j,k} S_ijk (vj * vkp)
                head_outs.append(torch.einsum('jk,jd,kd->d', S, vj, vkp))
            Z_rows.append(torch.stack(head_outs))

        Z = torch.stack(Z_rows)
        return Z.reshape(N, self.out_dim)

