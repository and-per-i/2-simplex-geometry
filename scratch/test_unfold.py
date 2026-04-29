import torch

def get_windows(tensor, window_size):
    N, H, D = tensor.shape
    device = tensor.device
    dtype = tensor.dtype
    # Pad with zeros at the start: (window_size-1, H, D)
    padded = torch.cat([torch.zeros(window_size - 1, H, D, device=device, dtype=dtype), tensor], dim=0)
    # Use unfold to get sliding windows: (N, window_size, H, D)
    return padded.unfold(0, window_size, 1).permute(0, 3, 1, 2)

T = torch.arange(5).view(5, 1, 1).float()
W = 3
win = get_windows(T, W)
print("Original:\n", T.squeeze())
print("Windows:\n", win.squeeze())
