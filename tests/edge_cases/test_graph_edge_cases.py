import torch
import pytest

from src.models import TwoSimplicialAttention


def test_single_node_graph():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()

    tri_feats = torch.randn(1, in_dim)
    edge_index = torch.tensor([[-1]], dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert out.shape == (1, out_dim)
    assert torch.isfinite(out).all()


def test_isolated_node_in_connected_graph():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()

    N = 4
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 2, -1],
        [0, 1, -1],
        [-1, -1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)
    assert torch.isfinite(out).all()


def test_degree_one_node_softmax_is_one():
    in_dim = 8
    out_dim = 8
    num_heads = 1
    head_dim = out_dim // num_heads
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()

    N = 2
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([[1], [0]], dtype=torch.long)

    with torch.no_grad():
        Q = model.q_proj(tri_feats).view(N, num_heads, head_dim)
        K = model.k_proj(tri_feats).view(N, num_heads, head_dim)
        Kp = model.kp_proj(tri_feats).view(N, num_heads, head_dim)

        i = 0
        qi = Q[i, 0]
        kj = K[1, 0]
        kjp = Kp[1, 0]
        raw = torch.dot(qi, kj * kjp)
        scaled = raw / (head_dim ** 0.5)
        S = torch.softmax(scaled.reshape(-1), dim=0)
        assert torch.allclose(S, torch.tensor([1.0]), atol=1e-6)

    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)
    assert torch.isfinite(out).all()


def test_all_nodes_isolated():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()

    N = 3
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.full((N, 2), -1, dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)
    assert torch.isfinite(out).all()


def test_varying_degrees_in_same_graph():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()

    N = 5
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([
        [1, 2, 3],
        [0, -1, -1],
        [0, -1, -1],
        [0, 4, -1],
        [3, -1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)
    assert torch.isfinite(out).all()
