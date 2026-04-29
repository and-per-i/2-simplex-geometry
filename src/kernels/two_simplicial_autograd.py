"""
Autograd Function wrapper for 2-simplicial attention.
Enables PyTorch autograd integration with the Triton kernels.
"""
import torch
from . import triton_2s_forward, triton_2s_backward


class TwoSimplicialAttentionFunction(torch.autograd.Function):
    """Custom autograd function for 2-simplicial attention."""
    
    @staticmethod
    def forward(ctx, x, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1=8, w2=8):
        """Forward pass."""
        O, M = triton_2s_forward.forward(
            x, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1, w2
        )
        
        # Save tensors for backward
        ctx.save_for_backward(Q, K, V, Kp, Vp, M, O, x)
        ctx.out_dim = out_dim
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.w1 = w1
        ctx.w2 = w2
        
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        Q, K, V, Kp, Vp, M, O, x = ctx.saved_tensors
        
        # The backward kernel now accepts O and M to avoid recomputing forward
        dQ, dK, dV, dKp, dVp = triton_2s_backward.backward(
            grad_output,
            x,
            Q, K, V, Kp, Vp,
            O, M,
            ctx.out_dim, ctx.num_heads, ctx.head_dim,
            ctx.w1, ctx.w2
        )

        
        return (
            None, # x
            dQ,
            dK,
            dV,
            dKp,
            dVp,
            None, # out_dim
            None, # num_heads
            None, # head_dim
            None, # w1
            None  # w2
        )
