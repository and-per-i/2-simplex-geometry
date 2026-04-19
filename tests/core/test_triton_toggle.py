import torch
from src.models import TwoSimplicialAttention


def test_triton_toggle_fallback_to_pytorch():
    in_dim = 16
    out_dim = 32
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=True)
    tri_feats = torch.randn(4, in_dim)
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [1, -1, -1],
        [2, -1, -1],
    ], dtype=torch.long)
    # Since Triton kernel is not implemented, we expect a graceful fallback to PyTorch path
    out = model(tri_feats, edge_index)
    assert out.shape == (4, out_dim)
