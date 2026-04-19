import pytest


def triton_available():
    try:
        import triton
        import torch
        return triton is not None and torch.cuda.is_available()
    except Exception:
        return False


skip_no_triton = pytest.mark.skipif(
    not triton_available(),
    reason="Triton + CUDA GPU required for kernel tests"
)


@pytest.fixture(scope="session")
def cuda_device():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")
