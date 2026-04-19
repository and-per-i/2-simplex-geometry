import os
import pytest
import random
import numpy as np
import torch


def _load_yaml_config(path: str):
    try:
        import yaml  # PyYAML
    except Exception:
        pytest.skip("PyYAML is not installed; skipping YAML config tests.")
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def config():
    # Load the YAML config used by tests (model/environment/training/test_config)
    cfg = _load_yaml_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    # If the dedicated config.yaml in tests/config.yaml is preferred, fall back to it
    if cfg is None:
        cfg = _load_yaml_config(os.path.join(os.path.dirname(__file__), "../tests/config.yaml"))
    return cfg


@pytest.fixture(autouse=True)
def _set_global_seed():
    # Global deterministic seed for tests to ensure reproducibility where possible
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield
    # no teardown actions required for now


@pytest.fixture
def tiny_input_tensor():
    # Small synthetic input tensor for fast shape tests
    N, D = 4, 64
    return torch.randn(N, D)
