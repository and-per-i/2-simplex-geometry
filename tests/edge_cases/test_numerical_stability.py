import torch

from src.models import TwoSimplicialAttention


def test_large_input_no_nan_or_inf():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()

    N = 4
    tri_feats = torch.randn(N, in_dim) * 1e4
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert torch.isfinite(out).all(), f"Non-finite values in output: {out}"


def test_very_small_input_no_underflow_issues():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()

    N = 4
    tri_feats = torch.randn(N, in_dim) * 1e-4
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert torch.isfinite(out).all()


def test_zero_input_no_nan():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()

    N = 3
    tri_feats = torch.zeros(N, in_dim)
    edge_index = torch.tensor([
        [1, -1],
        [0, 2],
        [1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert torch.isfinite(out).all()


def test_gradient_stability_with_moderate_input():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)

    N = 4
    tri_feats = torch.randn(N, in_dim, requires_grad=True) * 10.0
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    loss = out.sum()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"
