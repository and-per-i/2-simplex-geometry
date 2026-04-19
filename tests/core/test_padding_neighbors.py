import torch
from src.models import TwoSimplicialAttention


def test_padding_handling_no_crash():
    in_dim = 32
    out_dim = 64
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=4, dropout=0.0, with_residual=True, use_triton_kernel=False)
    tri_feats = torch.randn(5, in_dim)
    edge_index = torch.tensor([
        [1, -1, -1],
        [0, 2, -1],
        [1, 3, -1],
        [2, 4, -1],
        [3, -1, -1],
    ], dtype=torch.long)
    out = model(tri_feats, edge_index)
    assert out.shape == (5, out_dim)
