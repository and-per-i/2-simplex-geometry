import torch
import torch.nn.functional as F
from src.models import TwoSimplicialAttention


def test_backward_through_mvp():
    torch.manual_seed(0)
    in_dim = 8
    out_dim = 16
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=2, dropout=0.0, with_residual=True, use_triton_kernel=False)
    tri_feats = torch.randn(6, in_dim, requires_grad=True)
    edge_index = torch.tensor([
        [1, 2, -1],
        [0, 3, -1],
        [1, 4, -1],
        [2, -1, -1],
        [3, -1, -1],
        [4, -1, -1],
    ], dtype=torch.long)
    out = model(tri_feats, edge_index)
    loss = out.sum()
    loss.backward()
    assert tri_feats.grad is not None


def test_all_model_params_receive_gradients():
    torch.manual_seed(1)
    in_dim = 8
    out_dim = 16
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=2, dropout=0.0, with_residual=True, use_triton_kernel=False)
    tri_feats = torch.randn(4, in_dim)
    edge_index = torch.tensor([
        [1, -1],
        [0, 2],
        [1, -1],
        [-1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    target = torch.randn_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()

    param_names = [name for name, _ in model.named_parameters() if model.get_parameter(name).requires_grad]
    for name in param_names:
        p = model.get_parameter(name)
        assert p.grad is not None, f"No gradient for parameter: {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for parameter: {name}"
        assert (p.grad.abs() > 0).any(), f"Zero gradient for parameter: {name}"


def test_gradient_no_residual():
    torch.manual_seed(2)
    in_dim = 8
    out_dim = 16
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=2, dropout=0.0, with_residual=False, use_triton_kernel=False)
    tri_feats = torch.randn(4, in_dim, requires_grad=True)
    edge_index = torch.tensor([
        [1, -1],
        [0, 2],
        [1, -1],
        [-1, -1],
    ], dtype=torch.long)

    out = model(tri_feats, edge_index)
    loss = out.sum()
    loss.backward()

    assert tri_feats.grad is not None
    assert torch.isfinite(tri_feats.grad).all()
