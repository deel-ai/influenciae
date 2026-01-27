# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Pytest configuration and fixtures for framework-agnostic testing.

This module provides:
- Backend detection and availability checks
- Pytest markers for backend-specific tests
- Fixtures for common test utilities
- Command-line options for backend selection
"""
import pytest

# ============================================================================
# Backend availability detection
# ============================================================================

def _check_tensorflow_available():
    """Check if TensorFlow is available and functional."""
    try:
        import tensorflow as tf
        # Verify it's actually usable
        _ = tf.constant([1.0])
        return True
    except (ImportError, OSError, Exception):
        return False


def _check_pytorch_available():
    """Check if PyTorch is available and functional."""
    try:
        import torch
        # Verify it's actually usable
        _ = torch.tensor([1.0])
        return True
    except (ImportError, OSError, Exception):
        return False


# Global availability flags (computed once at import time)
HAS_TENSORFLOW = _check_tensorflow_available()
HAS_PYTORCH = _check_pytorch_available()


# ============================================================================
# Pytest hooks and configuration
# ============================================================================

def pytest_addoption(parser):
    """Add command-line options for backend selection."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        choices=["tensorflow", "pytorch", "all", "minimal"],
        help=(
            "Select which backend to test: "
            "'tensorflow' (TF only), 'pytorch' (PyTorch only), "
            "'all' (both backends), 'minimal' (skip all backend-specific tests)"
        )
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "tensorflow: mark test as requiring TensorFlow backend"
    )
    config.addinivalue_line(
        "markers",
        "pytorch: mark test as requiring PyTorch backend"
    )
    config.addinivalue_line(
        "markers",
        "backend_agnostic: mark test as framework-agnostic (runs without any backend)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running"
    )
    config.addinivalue_line(
        "markers",
        "requires_both_backends: mark test as requiring both TensorFlow and PyTorch"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on backend availability and selection.

    This hook:
    1. Respects the --backend CLI option
    2. Automatically skips tests for unavailable backends
    3. Applies appropriate skip markers
    """
    backend_option = config.getoption("--backend")

    # Determine which backends to run based on CLI option and availability
    run_tensorflow = HAS_TENSORFLOW
    run_pytorch = HAS_PYTORCH

    if backend_option == "minimal":
        run_tensorflow = False
        run_pytorch = False
    elif backend_option == "tensorflow":
        run_pytorch = False
    elif backend_option == "pytorch":
        run_tensorflow = False
    # "all" or None: run whatever is available

    # Skip reasons
    skip_tf_not_installed = pytest.mark.skip(reason="TensorFlow is not installed")
    skip_tf_not_selected = pytest.mark.skip(reason="TensorFlow backend not selected (--backend)")
    skip_pt_not_installed = pytest.mark.skip(reason="PyTorch is not installed")
    skip_pt_not_selected = pytest.mark.skip(reason="PyTorch backend not selected (--backend)")
    skip_both_required = pytest.mark.skip(reason="Both TensorFlow and PyTorch required for this test")

    for item in items:
        # Check for tensorflow marker
        if "tensorflow" in item.keywords:
            if not HAS_TENSORFLOW:
                item.add_marker(skip_tf_not_installed)
            elif not run_tensorflow:
                item.add_marker(skip_tf_not_selected)

        # Check for pytorch marker
        if "pytorch" in item.keywords:
            if not HAS_PYTORCH:
                item.add_marker(skip_pt_not_installed)
            elif not run_pytorch:
                item.add_marker(skip_pt_not_selected)

        # Check for requires_both_backends marker
        if "requires_both_backends" in item.keywords:
            if not (HAS_TENSORFLOW and HAS_PYTORCH):
                item.add_marker(skip_both_required)

        # Auto-detect backend from test file name if no explicit marker
        # This handles existing tests that use pytestmark but aren't explicitly marked
        if not any(m in item.keywords for m in ["tensorflow", "pytorch", "backend_agnostic", "requires_both_backends"]):
            test_file = str(item.fspath)
            if "_pytorch" in test_file or "pytorch" in test_file.lower():
                if not HAS_PYTORCH:
                    item.add_marker(skip_pt_not_installed)
                elif not run_pytorch:
                    item.add_marker(skip_pt_not_selected)
            elif "_tensorflow" in test_file or "tensorflow" in test_file.lower():
                if not HAS_TENSORFLOW:
                    item.add_marker(skip_tf_not_installed)
                elif not run_tensorflow:
                    item.add_marker(skip_tf_not_selected)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def has_tensorflow():
    """Fixture that returns whether TensorFlow is available."""
    return HAS_TENSORFLOW


@pytest.fixture(scope="session")
def has_pytorch():
    """Fixture that returns whether PyTorch is available."""
    return HAS_PYTORCH


@pytest.fixture(scope="session")
def available_backends():
    """Fixture that returns a list of available backend names."""
    backends = []
    if HAS_TENSORFLOW:
        backends.append("tensorflow")
    if HAS_PYTORCH:
        backends.append("pytorch")
    return backends


@pytest.fixture
def skip_if_no_tensorflow():
    """Skip test if TensorFlow is not available."""
    if not HAS_TENSORFLOW:
        pytest.skip("TensorFlow is not available")


@pytest.fixture
def skip_if_no_pytorch():
    """Skip test if PyTorch is not available."""
    if not HAS_PYTORCH:
        pytest.skip("PyTorch is not available")


# ============================================================================
# Environment info (useful for debugging CI)
# ============================================================================

def pytest_report_header(config):
    """Add backend availability info to pytest header."""
    lines = [
        "Influenciae Backend Status:",
        f"  TensorFlow: {'available' if HAS_TENSORFLOW else 'NOT available'}",
        f"  PyTorch: {'available' if HAS_PYTORCH else 'NOT available'}",
    ]
    backend_option = config.getoption("--backend", default=None)
    if backend_option:
        lines.append(f"  Selected backend: {backend_option}")
    return lines
