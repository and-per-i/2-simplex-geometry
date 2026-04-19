import torch
import pytest

from src.models import TwoSimplicialAttention


def make_dummy_data(N=8, in_dim=32):
    tri_feats = torch.randn(N, in_dim)
    return tri_feats


def test_forward_shape_basic():
    N = 8
    in_dim = 32
    out_dim = 64
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False)
    tri_feats = make_dummy_data(N=N, in_dim=in_dim)
    out = model(tri_feats)
    assert out.shape == (N, out_dim)


def test_forward_with_no_neighbors():
    N = 6
    in_dim = 32
    out_dim = 64
    # With sliding window w1=0, w2=0, it's basically no neighbors (except self)
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False, w1=0, w2=0)
    tri_feats = torch.randn(N, in_dim)
    out = model(tri_feats)
    assert out.shape == (N, out_dim)


def test_forward_shape_in_dim_equals_out_dim():
    N = 6
    dim = 16
    model = TwoSimplicialAttention(dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, dim)
    out = model(tri_feats)
    assert out.shape == (N, dim)


def test_forward_shape_no_residual():
    N = 5
    in_dim = 16
    out_dim = 32
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=False, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, in_dim)
    out = model(tri_feats)
    assert out.shape == (N, out_dim)


def test_forward_shape_single_head():
    N = 4
    in_dim = 16
    out_dim = 16
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=1, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, in_dim)
    out = model(tri_feats)
    assert out.shape == (N, out_dim)


def test_forward_shape_many_heads():
    N = 4
    in_dim = 16
    out_dim = 32
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=8, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, in_dim)
    out = model(tri_feats)
    assert out.shape == (N, out_dim)
