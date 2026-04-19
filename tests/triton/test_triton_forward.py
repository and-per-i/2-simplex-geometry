"""
Tests for the Triton forward kernel (2-simplicial attention).

These tests require Triton + CUDA GPU. They will be automatically skipped
in local environments without GPU support. Run on a cloud GPU instance.
"""
import torch
import pytest

from tests.triton.conftest import skip_no_triton


def _pytorch_2s_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale):
    """PyTorch reference implementation of 2-simplicial attention (dense, no masking)."""
    B, S, H, D = Q.shape
    scale = sm_scale

    O = torch.zeros_like(Q)
    M = torch.zeros(B, H, S, dtype=torch.float32, device=Q.device)

    for b in range(B):
        for h in range(H):
            for i in range(S):
                q_i = Q[b, i, h].float()
                num = 0.0
                den = 0.0
                max_score = float("-inf")
                scores = []
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        score = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * scale
                        scores.append((score, j, k))
                        if score.item() > max_score:
                            max_score = score.item()

                exp_scores = []
                for score, j, k in scores:
                    e = torch.exp(score - max_score)
                    exp_scores.append((e, j, k))
                    den += e.item()
                    num = num + e * (V1[b, j, h].float() * V2[b, k, h].float())

                if den > 0:
                    O[b, i, h] = (num / den).to(Q.dtype)
                    M[b, h, i] = max_score + torch.log(torch.tensor(den, dtype=torch.float32, device=Q.device))
                else:
                    O[b, i, h] = 0
                    M[b, h, i] = float("-inf")

    return O, M


def _make_qkv(B, S, num_heads, head_dim, device, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    K1 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    K2 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    V1 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    V2 = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
    return Q, K1, K2, V1, V2


def _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2):
    from src.kernels.triton_2s_forward import _forward_kernel_call, _check_triton
    assert _check_triton(), "Triton not available"

    B, S, num_heads, head_dim = Q.shape
    O = torch.zeros_like(Q)
    M = torch.zeros(B, num_heads, S, dtype=torch.float32, device=Q.device)

    _forward_kernel_call(
        Q, K1, K2, V1, V2, O, M,
        B, S, num_heads, head_dim, w1, w2,
        *Q.stride(), *K1.stride(), *K2.stride(), *V1.stride(), *V2.stride(),
        *O.stride(), *M.stride(),
    )
    return O, M


@skip_no_triton
class TestTritonForwardKernel:
    def test_output_shape(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device)
        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1=S, w2=S)
        assert O.shape == (B, S, H, D)
        assert M.shape == (B, H, S)

    def test_output_no_nan_inf(self, cuda_device):
        B, S, H, D = 1, 128, 4, 64
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device)
        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1=S, w2=S)
        assert torch.isfinite(O.float()).all(), "Output contains non-finite values"
        assert torch.isfinite(M).all(), "Log-sum-exp contains non-finite values"

    def test_output_finite_with_large_input(self, cuda_device):
        B, S, H, D = 1, 64, 2, 64
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=7)
        Q = Q * 100.0
        O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1=S, w2=S)
        assert torch.isfinite(O.float()).all()

    def test_matches_pytorch_small(self, cuda_device):
        B, S, H, D = 1, 16, 1, 32
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=0)
        sm_scale = 1.0 / (D ** 0.5)

        O_triton, M_triton = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        O_ref, M_ref = _pytorch_2s_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

        torch.testing.assert_close(O_triton.float(), O_ref.float(), atol=1e-2, rtol=1e-2)

    def test_local_window_masking(self, cuda_device):
        B, S, H, D = 1, 32, 1, 32
        w1, w2 = 4, 8
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=1)
        sm_scale = 1.0 / (D ** 0.5)

        O_triton, M_triton = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        O_ref, M_ref = _pytorch_2s_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

        torch.testing.assert_close(O_triton.float(), O_ref.float(), atol=1e-2, rtol=1e-2)

    def test_multi_head(self, cuda_device):
        B, S, H, D = 1, 32, 4, 32
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=2)
        sm_scale = 1.0 / (D ** 0.5)

        O_triton, _ = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        O_ref, _ = _pytorch_2s_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

        torch.testing.assert_close(O_triton.float(), O_ref.float(), atol=1e-2, rtol=1e-2)

    def test_batch_size_2(self, cuda_device):
        B, S, H, D = 2, 32, 2, 32
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=3)
        sm_scale = 1.0 / (D ** 0.5)

        O_triton, _ = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        O_ref, _ = _pytorch_2s_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

        torch.testing.assert_close(O_triton.float(), O_ref.float(), atol=1e-2, rtol=1e-2)

    def test_log_sum_exp_values(self, cuda_device):
        B, S, H, D = 1, 16, 1, 32
        w1, w2 = S, S
        Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=4)
        sm_scale = 1.0 / (D ** 0.5)

        _, M_triton = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
        _, M_ref = _pytorch_2s_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

        torch.testing.assert_close(M_triton.float(), M_ref.float(), atol=1e-1, rtol=1e-1)

    def test_different_seq_lengths(self, cuda_device):
        for S in [64, 128, 256]:
            B, H, D = 1, 2, 64
            Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=5)
            O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1=S, w2=S)
            assert O.shape == (B, S, H, D)
            assert torch.isfinite(O.float()).all()

    def test_different_head_dims(self, cuda_device):
        for D in [32, 64, 128]:
            B, S, H = 1, 64, 2
            Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, cuda_device, seed=6)
            O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1=S, w2=S)
            assert O.shape == (B, S, H, D)
            assert torch.isfinite(O.float()).all()
