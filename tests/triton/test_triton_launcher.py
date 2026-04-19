"""
Tests for the Triton launcher utility.
"""
import pytest

from src.kernels.triton_launcher import ensure_kernel_loaded


def test_ensure_kernel_loaded_returns_bool():
    result = ensure_kernel_loaded()
    assert isinstance(result, bool)


def test_ensure_kernel_loaded_consistent():
    r1 = ensure_kernel_loaded()
    r2 = ensure_kernel_loaded()
    assert r1 == r2


def test_launcher_matches_triton_availability():
    try:
        import triton
        has_triton = True
    except Exception:
        has_triton = False
    assert ensure_kernel_loaded() == has_triton
