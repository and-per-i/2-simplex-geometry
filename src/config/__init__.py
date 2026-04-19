import yaml
from pathlib import Path


def load_config(path: str):
    """Load a YAML configuration file and return a dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(cfg, path: str):
    """Persist a configuration dict to a YAML file."""
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
