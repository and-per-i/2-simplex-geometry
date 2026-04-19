"""
Tests for the Triton backward kernels (2-simplicial attention).

These tests require Triton + CUDA GPU. They will be automatically skipped
in local environments without GPU support. Run on a cloud GPU instance.

Note: The backward wrapper (backward()) currently raises NotImplementedError.
These tests call the Triton kernels directly via the lower-level API,
adapting the layout to [batch, seq_len, num_heads, head_dim] in bfloat16.
"""
import torch
import pytest

from tests.triton.conftest import skip_no_triton


def _make_qkv(B, S, num_heads, head_dim, device, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    K1 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    K2 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    V1 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    V2 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    return Q, K1, K2, V1, V2


class _TritonAttnFn(torch.autograd.Function):
    """Thin autograd wrapper around the raw Triton forward/backward kernels for testing."""

    @staticmethod
    def forward(ctx, Q, K1, K2, V1, V2, w1, w2):
        from src.kernels.triton_2s_forward import _forward_kernel_call
        B, S, H, D = Q.shape
        # Detached output buffers — kernel fills them via tl.store
        O = torch.zeros(B, S, H, D, device=Q.device, dtype=Q.dtype)
        M = torch.zeros(B, H, S, dtype=torch.float32, device=Q.device)
        _forward_kernel_call(
            Q.detach(), K1.detach(), K2.detach(), V1.detach(), V2.detach(), O, M,
            B, S, H, D, w1, w2,
            *Q.stride(), *K1.stride(), *K2.stride(),
            *V1.stride(), *V2.stride(), *O.stride(), *M.stride(),
        )
        ctx.save_for_backward(Q, K1, K2, V1, V2, O, M)
        ctx.w1, ctx.w2 = w1, w2
        return O, M

    @staticmethod
    def backward(ctx, dO, dM):
        from src.kernels.triton_2s_backward import (
            two_simplicial_attn_bwd_kv1_kernel,
            two_simplicial_attn_bwd_kv2q_kernel,
        )
        import triton
        Q, K1, K2, V1, V2, O, M = ctx.saved_tensors
        B, S, H, D = Q.shape
        w1, w2 = ctx.w1, ctx.w2
        dO = dO.contiguous()

        # D_row = rowsum(dO * O): [B, H, S]
        D_row = (dO.float() * O.float()).sum(dim=3).permute(0, 2, 1).contiguous()

        dQ  = torch.zeros_like(Q)
        dK1 = torch.zeros_like(K1)
        dV1 = torch.zeros_like(V1)
        dK2 = torch.zeros_like(K2)
        dV2 = torch.zeros_like(V2)

        shared = {
            "q_stride_b":  Q.stride(0),  "q_stride_s":  Q.stride(1),  "q_stride_k":  Q.stride(2),  "q_stride_h":  Q.stride(3),
            "k1_stride_b": K1.stride(0), "k1_stride_s": K1.stride(1), "k1_stride_k": K1.stride(2), "k1_stride_h": K1.stride(3),
            "k2_stride_b": K2.stride(0), "k2_stride_s": K2.stride(1), "k2_stride_k": K2.stride(2), "k2_stride_h": K2.stride(3),
            "v1_stride_b": V1.stride(0), "v1_stride_s": V1.stride(1), "v1_stride_k": V1.stride(2), "v1_stride_h": V1.stride(3),
            "v2_stride_b": V2.stride(0), "v2_stride_s": V2.stride(1), "v2_stride_k": V2.stride(2), "v2_stride_h": V2.stride(3),
            "dO_stride_b": dO.stride(0), "dO_stride_s": dO.stride(1), "dO_stride_k": dO.stride(2), "dO_stride_h": dO.stride(3),
            "m_stride_b": 0, "m_stride_k": M.stride(1), "m_stride_s": M.stride(2),
            "d_stride_b": 0, "d_stride_k": D_row.stride(1), "d_stride_s": D_row.stride(2),
            "dq_stride_b": dQ.stride(0), "dq_stride_s": dQ.stride(1), "dq_stride_k": dQ.stride(2), "dq_stride_h": dQ.stride(3),
        }

        grid_kv1 = (triton.cdiv(S, 32), B * H)
        two_simplicial_attn_bwd_kv1_kernel[grid_kv1](
            Q, K1, K2, V1, V2, dO, M, D_row,
            dQ, dK1, dV1,
            B, S, H, D, w1, w2,
            **{**shared,
               "dk1_stride_b": dK1.stride(0), "dk1_stride_s": dK1.stride(1), "dk1_stride_k": dK1.stride(2), "dk1_stride_h": dK1.stride(3),
               "dv1_stride_b": dV1.stride(0), "dv1_stride_s": dV1.stride(1), "dv1_stride_k": dV1.stride(2), "dv1_stride_h": dV1.stride(3),
            },
            BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=32, HEAD_DIM=D,
            SM_SCALE=1.0 / (D ** 0.5), K2_BIAS=0.0, V2_BIAS=0.0,
            COMPUTE_DQ=True, num_stages=1, is_flipped=False,
        )

        grid_kv2q = (triton.cdiv(S, 64), B * H)
        kv2_strides = {**shared,
            "dk2_stride_b": dK2.stride(0), "dk2_stride_s": dK2.stride(1), "dk2_stride_k": dK2.stride(2), "dk2_stride_h": dK2.stride(3),
            "dv2_stride_b": dV2.stride(0), "dv2_stride_s": dV2.stride(1), "dv2_stride_k": dV2.stride(2), "dv2_stride_h": dV2.stride(3),
        }
        for is_second in [False, True]:
            two_simplicial_attn_bwd_kv2q_kernel[grid_kv2q](
                Q, K1, K2, V1, V2, dO, M, D_row,
                dQ, dK2, dV2,
                B, S, H, D, w1, w2,
                **kv2_strides,
                HEAD_DIM=D, SM_SCALE=1.0 / (D ** 0.5),
                K2_BIAS=0.0, V2_BIAS=0.0, IS_SECOND_PASS=is_second,
            )

        return dQ, dK1, dK2, dV1, dV2, None, None


def _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2):
    """Differentiable forward — wraps Triton kernels in an autograd Function."""
    from src.kernels.triton_2s_forward import _check_triton
    assert _check_triton()
    # _TritonAttnFn.apply returns (O, M); only O is differentiable
    O, M = _TritonAttnFn.apply(Q, K1, K2, V1, V2, w1, w2)
    return O, M



def _pytorch_fwd(Q, K1, K2, V1, V2, w1, w2, sm_scale):
    B, S, H, D = Q.shape
    O = torch.zeros(B, S, H, D, dtype=torch.float32, device=Q.device)
    M = torch.full((B, H, S), float("-inf"), dtype=torch.float32, device=Q.device)

    for b in range(B):
        for h in range(H):
            for i in range(S):
                q_i = Q[b, i, h].float()
                max_s = float("-inf")
                weighted_sum = torch.zeros(D, dtype=torch.float32, device=Q.device)
                denom = 0.0
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        s = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * sm_scale
                        if s.item() > max_s:
                            max_s = s.item()
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        s = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * sm_scale
                        e = torch.exp(s - max_s)
                        denom += e.item()
                        weighted_sum += e * (V1[b, j, h].float() * V2[b, k, h].float())
                if denom > 0:
                    O[b, i, h] = weighted_sum / denom
                    M[b, h, i] = max_s + torch.log(torch.tensor(denom, dtype=torch.float32, device=Q.device))
    return O, M


def _pytorch_bwd(dO, Q, K1, K2, V1, V2, O, M, w1, w2, sm_scale):
    B, S, H, D = Q.shape
    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dK1 = torch.zeros_like(K1, dtype=torch.float32)
    dK2 = torch.zeros_like(K2, dtype=torch.float32)
    dV1 = torch.zeros_like(V1, dtype=torch.float32)
    dV2 = torch.zeros_like(V2, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for i in range(S):
                q_i = Q[b, i, h].float()
                dO_i = dO[b, i, h].float()
                Di = torch.dot(dO_i, O[b, i, h].float())
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        s = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * sm_scale
                        p = torch.exp(s - M[b, h, i])
                        v1v2 = V1[b, j, h].float() * V2[b, k, h].float()
                        dp = torch.dot(dO_i, v1v2)
                        ds = p * (dp - Di)

                        dQ[b, i, h] += ds * K1[b, j, h].float() * K2[b, k, h].float() * sm_scale
                        dK1[b, j, h] += ds * q_i * K2[b, k, h].float() * sm_scale
                        dK2[b, k, h] += ds * q_i * K1[b, j, h].float() * sm_scale
                        dV1[b, j, h] += p * dO_i * V2[b, k, h].float()
                        dV2[b, k, h] += p * dO_i * V1[b, j, h].float()

    return dQ, dK1, dK2, dV1, dV2


@skip_no_triton
class TestTritonBackwardKernel:
    def test_backward_output_shapes(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=0)
        Q.requires_grad_(True)
        K1.requires_grad_(True)
        K2.requires_grad_(True)
        V1.requires_grad_(True)
        V2.requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        assert Q.grad is not None
        assert K1.grad is not None
        assert K2.grad is not None
        assert V1.grad is not None
        assert V2.grad is not None
        assert Q.grad.shape == Q.shape
        assert K1.grad.shape == K1.shape
        assert K2.grad.shape == K2.shape
        assert V1.grad.shape == V1.shape
        assert V2.grad.shape == V2.shape

    def test_backward_grads_finite(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=1)
        Q.requires_grad_(True)
        K1.requires_grad_(True)
        K2.requires_grad_(True)
        V1.requires_grad_(True)
        V2.requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, tensor in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(tensor.grad.float()).all(), f"Non-finite gradient for {name}"

    def test_backward_matches_pytorch_small(self, cuda_device):
        B, S, H, D = 1, 8, 1, 32
        w1, w2 = S, S
        sm_scale = 1.0 / (D ** 0.5)
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=2)

        O_triton, M_triton = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        O_ref, M_ref = _pytorch_fwd(Q, K1, K2, V1, V2, w1, w2, sm_scale)

        dO = torch.randn_like(O_triton)

        dQ_ref, dK1_ref, dK2_ref, dV1_ref, dV2_ref = _pytorch_bwd(
            dO, Q, K1, K2, V1, V2, O_ref, M_ref, w1, w2, sm_scale
        )

        Q2 = Q.clone().detach().requires_grad_(True)
        K1_2 = K1.clone().detach().requires_grad_(True)
        K2_2 = K2.clone().detach().requires_grad_(True)
        V1_2 = V1.clone().detach().requires_grad_(True)
        V2_2 = V2.clone().detach().requires_grad_(True)
        O2, M2 = _call_fwd_kernel(Q2, K1_2, K2_2, V1_2, V2_2, w1, w2)
        O2.backward(dO)

        torch.testing.assert_close(Q2.grad.float(), dQ_ref.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(K1_2.grad.float(), dK1_ref.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(V1_2.grad.float(), dV1_ref.float(), atol=5e-2, rtol=5e-2)

    def test_backward_local_window(self, cuda_device):
        B, S, H, D = 1, 16, 1, 32
        w1, w2 = 4, 8
        sm_scale = 1.0 / (D ** 0.5)
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=3)

        Q = Q.clone().detach().requires_grad_(True)
        K1 = K1.clone().detach().requires_grad_(True)
        K2 = K2.clone().detach().requires_grad_(True)
        V1 = V1.clone().detach().requires_grad_(True)
        V2 = V2.clone().detach().requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, t in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(t.grad.float()).all(), f"Non-finite grad for {name} with local window"

    def test_backward_large_input_no_nan(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=4)
        Q = Q * 50.0
        Q = Q.clone().detach().requires_grad_(True)
        K1 = K1.clone().detach().requires_grad_(True)
        K2 = K2.clone().detach().requires_grad_(True)
        V1 = V1.clone().detach().requires_grad_(True)
        V2 = V2.clone().detach().requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, t in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(t.grad.float()).all(), f"Non-finite grad for {name} with large input"

    def test_backward_multi_head(self, cuda_device):
        B, S, H, D = 1, 32, 4, 32
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=5)
        Q = Q.clone().detach().requires_grad_(True)
        K1 = K1.clone().detach().requires_grad_(True)
        K2 = K2.clone().detach().requires_grad_(True)
        V1 = V1.clone().detach().requires_grad_(True)
        V2 = V2.clone().detach().requires_grad_(True)

        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        dO = torch.randn_like(O)
        O.backward(dO)

        for name, t in [("Q", Q), ("K1", K1), ("K2", K2), ("V1", V1), ("V2", V2)]:
            assert torch.isfinite(t.grad.float()).all(), f"Non-finite grad for {name} in multi-head backward"
