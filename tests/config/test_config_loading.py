import pytest


def test_config_loading(config):
    # Basic structural checks for the YAML config
    assert isinstance(config, dict), "Config should be a dictionary"
    required_top = ["model", "training", "test_config"]
    for key in required_top:
        assert key in config, f"Config missing required top-level key: {key}"
    assert isinstance(config.get("model"), dict)
    assert isinstance(config.get("training"), dict)
    assert isinstance(config.get("test_config"), dict)
