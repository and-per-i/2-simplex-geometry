"""
Tests for Triton kernels on CUDA environment.

Correctly compares the Triton forward kernel against a PyTorch reference
implementation of the SAME local-window 2-simplicial attention.

The PyTorch model (graph-based, uses edge_index) and the Triton kernel
(local-window, uses w1/w2 masking) have different semantics by design.
This test validates the kernel numerics, not the model-level equivalence.
"""
import torch
import pytest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _pytorch_local_window_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale):
    """PyTorch reference for local-window 2-simplicial attention.
    Uses strict bounds: i-w1 < j <= i and i-w2 < k <= i (matching the kernel mask).
    Q, K1, K2, V1, V2: [B, S, H, D]
    Returns O: [B, S, H, D]
    """
    B, S, H, D = Q.shape
    O = torch.zeros(B, S, H, D, dtype=torch.float32, device=Q.device)
    for b in range(B):
        for h in range(H):
            for i in range(S):
                q_i = Q[b, i, h].float()
                scores = []
                max_s = float("-inf")
                for j in range(S):
                    if not (i - w1 < j <= i):
                        continue
                    for k in range(S):
                        if not (i - w2 < k <= i):
                            continue
                        s = torch.dot(q_i, K1[b, j, h].float() * K2[b, k, h].float()) * sm_scale
                        scores.append((s, j, k))
                        max_s = max(max_s, s.item())
                if not scores:
                    continue
                denom = 0.0
                out = torch.zeros(D, dtype=torch.float32, device=Q.device)
                for s, j, k in scores:
                    e = torch.exp(s - max_s)
                    denom += e.item()
                    out += e * (V1[b, j, h].float() * V2[b, k, h].float())
                if denom > 0:
                    O[b, i, h] = out / denom
    return O


def _make_qkv(B, S, H, D, device, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    K1 = torch.randn(B, S, H, D, device=device, dtype=dtype)
    K2 = torch.randn(B, S, H, D, device=device, dtype=dtype)
    V1 = torch.randn(B, S, H, D, device=device, dtype=dtype)
    V2 = torch.randn(B, S, H, D, device=device, dtype=dtype)
    return Q, K1, K2, V1, V2


def _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2):
    from src.kernels.triton_2s_forward import _forward_kernel_call, _check_triton
    assert _check_triton(), "Triton not available"
    B, S, H, D = Q.shape
    O = torch.zeros_like(Q)
    M = torch.zeros(B, H, S, dtype=torch.float32, device=Q.device)
    _forward_kernel_call(
        Q, K1, K2, V1, V2, O, M,
        B, S, H, D, w1, w2,
        *Q.stride(), *K1.stride(), *K2.stride(),
        *V1.stride(), *V2.stride(),
        *O.stride(), *M.stride(),
    )
    return O, M


# ---------------------------------------------------------------------------
# Test 1: Output is finite
# ---------------------------------------------------------------------------
def test_triton_fwd_output_finite():
    B, S, H, D = 1, 64, 4, 16
    Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, "cuda")
    O, M = _call_fwd_kernel(Q, K1, K2, V1, V2, w1=S, w2=S)
    assert O.shape == (B, S, H, D)
    assert M.shape == (B, H, S)
    assert torch.isfinite(O.float()).all(), "Forward output contains non-finite values"
    assert torch.isfinite(M).all(), "Log-sum-exp M contains non-finite values"


# ---------------------------------------------------------------------------
# Test 2: Numerical match against PyTorch local-window reference
# ---------------------------------------------------------------------------
def test_triton_fwd_matches_pytorch_ref():
    B, S, H, D = 1, 16, 1, 32
    w1, w2 = S, S
    sm_scale = 1.0 / (D ** 0.5)
    Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, "cuda", seed=0)

    O_triton, _ = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
    O_ref = _pytorch_local_window_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

    torch.testing.assert_close(O_triton.float(), O_ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 3: Local window masking correctness
# ---------------------------------------------------------------------------
def test_triton_fwd_local_window():
    B, S, H, D = 1, 32, 1, 32
    w1, w2 = 4, 8
    sm_scale = 1.0 / (D ** 0.5)
    Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, "cuda", seed=1)

    O_triton, _ = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
    O_ref = _pytorch_local_window_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

    torch.testing.assert_close(O_triton.float(), O_ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 4: Multi-head
# ---------------------------------------------------------------------------
def test_triton_fwd_multi_head():
    B, S, H, D = 1, 32, 4, 32
    w1, w2 = S, S
    sm_scale = 1.0 / (D ** 0.5)
    Q, K1, K2, V1, V2 = _make_qkv(B, S, H, D, "cuda", seed=2)

    O_triton, _ = _call_fwd_kernel(Q, K1, K2, V1, V2, w1, w2)
    O_ref = _pytorch_local_window_ref(Q, K1, K2, V1, V2, w1, w2, sm_scale)

    torch.testing.assert_close(O_triton.float(), O_ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 5: Model runs without errors on CUDA (smoke test)
# ---------------------------------------------------------------------------
def test_model_triton_path_runs():
    from src.models.two_simplicial_attention import TwoSimplicialAttention
    device = "cuda"
    in_dim, out_dim, N, H = 32, 64, 64, 4

    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=H, use_triton_kernel=True).to(device)
    model.eval()

    tri_feats = torch.randn(N, in_dim, device=device)
    edge_index = torch.randint(-1, N, (N, 8), device=device)

    with torch.no_grad():
        out = model(tri_feats, edge_index)

    assert out.shape == (N, out_dim)
    assert torch.isfinite(out).all(), "Model Triton output contains non-finite values"
