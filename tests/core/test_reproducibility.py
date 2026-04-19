import torch

from src.models import TwoSimplicialAttention


def test_deterministic_output_same_seed():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    N = 4
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
    ], dtype=torch.long)

    torch.manual_seed(99)
    model1 = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model1.eval()
    tri_feats = torch.randn(N, in_dim)
    out1 = model1(tri_feats, edge_index)

    torch.manual_seed(99)
    model2 = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model2.eval()
    out2 = model2(tri_feats, edge_index)

    assert torch.equal(out1, out2), "Outputs differ for same seed and input"


def test_different_seed_different_output():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    N = 4
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
    ], dtype=torch.long)
    tri_feats = torch.randn(N, in_dim)

    torch.manual_seed(1)
    model1 = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model1.eval()
    out1 = model1(tri_feats, edge_index)

    torch.manual_seed(2)
    model2 = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.0, with_residual=True, use_triton_kernel=False)
    model2.eval()
    out2 = model2(tri_feats, edge_index)

    assert not torch.equal(out1, out2), "Outputs should differ with different seeds"


def test_eval_mode_dropout_is_noop():
    in_dim = 8
    out_dim = 16
    num_heads = 2
    N = 4
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [0, 1, -1],
        [1, -1, -1],
    ], dtype=torch.long)

    torch.manual_seed(42)
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=0.3, with_residual=True, use_triton_kernel=False)
    model.eval()
    tri_feats = torch.randn(N, in_dim)

    torch.manual_seed(100)
    out1 = model(tri_feats, edge_index)

    torch.manual_seed(200)
    out2 = model(tri_feats, edge_index)

    assert torch.equal(out1, out2), "Eval mode with dropout should be deterministic"
