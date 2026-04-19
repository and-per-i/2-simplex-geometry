import torch
import pytest

from src.models import TwoSimplicialAttention


def test_tri_feats_1d_raises():
    model = TwoSimplicialAttention(8, 16, num_heads=2)
    flat = torch.randn(16)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    with pytest.raises(ValueError, match="tri_feats must be"):
        model(flat, edge_index)


def test_tri_feats_3d_raises():
    model = TwoSimplicialAttention(8, 16, num_heads=2)
    tensor3d = torch.randn(2, 4, 8)
    edge_index = torch.tensor([[1], [0]], dtype=torch.long)
    with pytest.raises(ValueError, match="tri_feats must be"):
        model(tensor3d, edge_index)


def test_edge_index_none_raises():
    model = TwoSimplicialAttention(8, 16, num_heads=2)
    tri_feats = torch.randn(4, 8)
    with pytest.raises(ValueError, match="edge_index must be provided"):
        model(tri_feats, None)


def test_edge_index_not_tensor_raises():
    model = TwoSimplicialAttention(8, 16, num_heads=2)
    tri_feats = torch.randn(4, 8)
    edge_index = [[1, -1], [0, -1], [1, -1], [0, -1]]
    with pytest.raises(TypeError, match="edge_index must be a Tensor"):
        model(tri_feats, edge_index)


def test_out_dim_not_divisible_by_num_heads_raises():
    with pytest.raises(AssertionError, match="out_dim must be divisible by num_heads"):
        TwoSimplicialAttention(8, out_dim=10, num_heads=4)


def test_out_dim_none_defaults_to_in_dim():
    model = TwoSimplicialAttention(16, num_heads=4)
    assert model.out_dim == 16
    assert model.in_dim == 16


def test_negative_edge_index_padding_ignored():
    model = TwoSimplicialAttention(8, 16, num_heads=2)
    model.eval()
    tri_feats = torch.randn(3, 8)
    edge_index = torch.tensor([
        [1, -1, -1],
        [0, 2, -1],
        [1, -1, -1],
    ], dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()
