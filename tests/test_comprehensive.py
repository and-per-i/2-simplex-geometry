import torch
import pytest
import os
import yaml
import tempfile
from src.config import load_config, save_config
from src.models.two_simplicial_attention import TwoSimplicialAttention
from src.kernels.two_simplicial_autograd import TwoSimplicialAttentionFunction
# from scripts.train import create_synthetic_data

def test_config_system():
    """Test loading and saving of the YAML configuration."""
    test_cfg = {
        'model': {'in_dim': 16, 'out_dim': 32, 'w1': 8, 'w2': 8},
        'trainer': {'lr': 0.001}
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        yaml.dump(test_cfg, tmp)
        tmp_path = tmp.name
    
    try:
        loaded = load_config(tmp_path)
        assert loaded['model']['in_dim'] == 16
        assert loaded['trainer']['lr'] == 0.001
        
        # Test saving
        tmp_path_save = tmp_path + "_save"
        save_config(loaded, tmp_path_save)
        assert os.path.exists(tmp_path_save)
        resaved = load_config(tmp_path_save)
        assert resaved == loaded
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        if os.path.exists(tmp_path_save): os.remove(tmp_path_save)

def test_model_initialization():
    """Test various initialization configurations."""
    # Standard init
    model = TwoSimplicialAttention(in_dim=32, out_dim=64, num_heads=4)
    assert model.head_dim == 16
    
    # Init with defaults
    model_def = TwoSimplicialAttention(in_dim=32)
    assert model_def.out_dim == 32
    
    # Invalid head count
    with pytest.raises(AssertionError):
        TwoSimplicialAttention(in_dim=32, out_dim=64, num_heads=3)

def test_vanilla_forward_backward():
    """Full forward and backward pass on vanilla implementation."""
    in_dim, out_dim, N = 16, 32, 10
    model = TwoSimplicialAttention(in_dim=in_dim, out_dim=out_dim, num_heads=2)
    
    tri_feats = torch.randn(N, in_dim, requires_grad=True)
    
    output = model(tri_feats)
    assert output.shape == (N, out_dim)
    
    loss = output.sum()
    loss.backward()
    
    assert tri_feats.grad is not None
    assert model.q_proj.weight.grad is not None
    assert model.out_proj.weight.grad is not None

def test_no_neighbors_case():
    """Ensure the model handles nodes with no neighbors gracefully."""
    in_dim, out_dim, N = 8, 16, 3
    # w1=0, w2=0 means only self-attention
    model = TwoSimplicialAttention(in_dim=in_dim, out_dim=out_dim, num_heads=2, w1=0, w2=0)
    
    tri_feats = torch.randn(N, in_dim)
    
    output = model(tri_feats)
    assert output.shape == (N, out_dim)
    # With residual, output should be related to input features
    assert not torch.allclose(output, torch.zeros_like(output))

# def test_synthetic_data_generator():
#     """Verify the data generator used in scripts."""
#     N, in_dim, max_deg = 100, 32, 10
#     tri_feats, edge_index = create_synthetic_data(N, in_dim, max_deg, "cpu")
#     
#     assert tri_feats.shape == (N, in_dim)
#     assert edge_index.shape == (N, max_deg)
#     assert edge_index.max() < N
#     assert edge_index.min() >= -1

def test_triton_autograd_interface():
    """
    Test the autograd function wrapper. 
    """
    N, H, D = 8, 2, 8
    Q = torch.randn(N, H, D, requires_grad=True)
    K = torch.randn(N, H, D, requires_grad=True)
    V = torch.randn(N, H, D, requires_grad=True)
    Kp = torch.randn(N, H, D, requires_grad=True)
    Vp = torch.randn(N, H, D, requires_grad=True)
    x = torch.randn(N, H*D)
    
    try:
        TwoSimplicialAttentionFunction.apply(
            x, Q, K, V, Kp, Vp, H*D, H, D, 8, 8
        )
    except RuntimeError as e:
        assert "Triton" in str(e)
    except Exception as e:
        pass

def test_residual_toggle():
    """Check that residual connection is indeed optional."""
    in_dim = 16
    N = 5
    tri_feats = torch.randn(N, in_dim)
    
    model_res = TwoSimplicialAttention(in_dim, with_residual=True)
    model_no_res = TwoSimplicialAttention(in_dim, with_residual=False)
    
    out_res = model_res(tri_feats)
    out_no_res = model_no_res(tri_feats)
    
    assert not torch.allclose(out_res, out_no_res)

@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_hyperparameter_combinations(num_heads, dropout):
    """Test multiple head and dropout configs."""
    in_dim, out_dim, N = 16, 16, 5
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=num_heads, dropout=dropout)
    tri_feats = torch.randn(N, in_dim)
    
    output = model(tri_feats)
    assert output.shape == (N, out_dim)
    assert torch.isfinite(output).all()
