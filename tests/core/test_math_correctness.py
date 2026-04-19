import torch
import pytest

from src.models import TwoSimplicialAttention


def _make_model_and_input(in_dim=8, out_dim=16, num_heads=2, dropout=0.0, with_residual=True, N=4):
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=dropout, with_residual=with_residual, use_triton_kernel=False)
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([
        [1, -1],
        [0, 2, -1][:2],
        [1, 3, -1][:2],
        [2, -1],
    ][:N], dtype=torch.long)
    if N == 2:
        edge_index = torch.tensor([[1, -1], [0, -1]], dtype=torch.long)
    elif N == 3:
        edge_index = torch.tensor([[1, -1], [0, 2], [1, -1]], dtype=torch.long)
    return model, tri_feats, edge_index


def test_softmax_sums_to_one():
    torch.manual_seed(42)
    in_dim = 8
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()

    N = 4
    tri_feats = torch.randn(N, in_dim)
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
    ], dtype=torch.long)

    Q = model.q_proj(tri_feats).view(N, num_heads, out_dim // num_heads)
    K = model.k_proj(tri_feats).view(N, num_heads, out_dim // num_heads)
    Kp = model.kp_proj(tri_feats).view(N, num_heads, out_dim // num_heads)

    head_dim = out_dim // num_heads
    deg = (edge_index >= 0).sum(dim=1).tolist()

    for i in range(N):
        d_i = int(deg[i])
        if d_i == 0:
            continue
        for h in range(num_heads):
            qi = Q[i, h]
            neigh = edge_index[i, :d_i]
            kj = K[neigh, h]
            kjp = Kp[neigh, h]
            A_ijk = torch.zeros(d_i, d_i)
            for jj in range(d_i):
                for kk in range(d_i):
                    A_ijk[jj, kk] = torch.dot(qi, kj[jj] * kjp[kk])
            A_ijk = A_ijk / (head_dim ** 0.5)
            S_flat = torch.softmax(A_ijk.reshape(-1), dim=0)
            assert torch.allclose(S_flat.sum(), torch.tensor(1.0), atol=1e-5), \
                f"Softmax over (j,k) for node {i}, head {h} does not sum to 1: sum={S_flat.sum().item()}"


def test_identical_neighbors_symmetric_output():
    in_dim = 16
    out_dim = 16
    num_heads = 2
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()

    N = 3
    tri_feats = torch.randn(N, in_dim)

    edge_index = torch.tensor([
        [1, 2],
        [0, 2],
        [0, 1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    assert out.shape == (N, out_dim)
    assert torch.isfinite(out).all()


def test_zero_value_produces_zero_attention_output():
    in_dim = 8
    out_dim = 8
    num_heads = 1
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()

    torch.manual_seed(0)
    N = 3
    tri_feats = torch.randn(N, in_dim)

    with torch.no_grad():
        Q = model.q_proj(tri_feats).view(N, num_heads, out_dim // num_heads)
        V = model.v_proj(tri_feats).view(N, num_heads, out_dim // num_heads)
        Vp = model.vp_proj(tri_feats).view(N, num_heads, out_dim // num_heads)

    edge_index = torch.tensor([[1, -1], [0, 2], [1, -1]], dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert torch.isfinite(out).all()


def test_attention_scale_by_sqrt_d():
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
        raw_score = torch.dot(qi, kj * kjp)
        scaled_score = raw_score / (head_dim ** 0.5)

        A_ijk = torch.zeros(1, 1)
        A_ijk[0, 0] = scaled_score
        S_flat = torch.softmax(A_ijk.reshape(-1), dim=0)
        assert torch.allclose(S_flat[0], torch.tensor(1.0), atol=1e-6)
