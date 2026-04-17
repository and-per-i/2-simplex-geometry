import torch
import pytest
from src.models.two_simplicial_attention import TwoSimplicialAttention

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponibile")
def test_triton_vs_pytorch_parity():
    """Verifica che l'output e i gradienti di Triton coincidano con PyTorch."""
    device = "cuda"
    in_dim, out_dim, N, H = 32, 64, 128, 4
    
    # Inizializza modello (usiamo con_residual=False per isolare il kernel)
    model = TwoSimplicialAttention(in_dim, out_dim, num_heads=H, with_residual=False).to(device)
    model.eval()
    
    # Dati di input
    tri_feats = torch.randn(N, in_dim, device=device, requires_grad=True)
    edge_index = torch.randint(-1, N, (N, 8), device=device)
    
    # 1. Path PyTorch
    model.use_triton_kernel = False
    out_py = model(tri_feats, edge_index)
    loss_py = out_py.sum()
    loss_py.backward()
    grad_py = tri_feats.grad.clone()
    
    tri_feats.grad.zero_()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
    
    # 2. Path Triton
    model.use_triton_kernel = True
    out_triton = model(tri_feats, edge_index)
    loss_triton = out_triton.sum()
    loss_triton.backward()
    grad_triton = tri_feats.grad.clone()
    
    # Verifiche numeriche
    torch.testing.assert_close(out_py, out_triton, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_py, grad_triton, atol=1e-2, rtol=1e-2)
    print("\n[OK] Triton vs PyTorch parity test passed!")

if __name__ == "__main__":
    test_triton_vs_pytorch_parity()
