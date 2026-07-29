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
import sys
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


def _check_pytorch_numpy_bridge_available():
    """Check whether torch tensor -> numpy conversion is functional."""
    try:
        import torch
        _ = torch.tensor([1.0]).detach().cpu().numpy()
        return True
    except (ImportError, OSError, Exception):
        return False


# Global availability flags (computed once at import time)
HAS_TENSORFLOW = _check_tensorflow_available()
HAS_PYTORCH = _check_pytorch_available()
HAS_PYTORCH_NUMPY_BRIDGE = _check_pytorch_numpy_bridge_available() if HAS_PYTORCH else False

# ============================================================================
# Collection ignore patterns based on backend availability
# This prevents ImportError during test collection when backends aren't installed
# ============================================================================

# Files/patterns that require PyTorch (will fail to import without it)
# Using glob patterns relative to the tests directory
_PYTORCH_PATTERNS = [
    # Pattern to match all *_pytorch.py files anywhere in tests
    "*_pytorch.py",
    "**/*_pytorch.py",
]

# Files/patterns that require TensorFlow (will fail to import without it)
# These files have top-level TF imports that will cause ImportError
_TENSORFLOW_PATTERNS = [
    # Benchmark tests - import tensorflow.keras directly
    "benchmark/test_bench.py",
    "benchmark/test_benchmark_base.py",
    # Boundary tests (non-pytorch versions)
    "boundary_based/test_sample_boundary.py",
    "boundary_based/test_weights_boundary.py",
    # Common module tests
    "common/test_ihvp.py",
    "common/test_ihvp_factory.py",
    "common/test_inf_abstract.py",
    "common/test_model_wrapper.py",
    "common/test_backend_tensorflow.py",
    # Influence tests
    "influence/test_first_order_influence_calculator.py",
    "influence/test_second_order_influence.py",
    "influence/test_arnoldi_influence_calculator.py",
    # RPS tests
    "rps/test_representer_point_l2.py",
    "rps/test_rps_lje.py",
    # TracIn tests
    "trac_in/test_tracin.py",
    # TrackStar tests (TF version imports tensorflow at module scope)
    "trackstar/test_optimizer_state.py",
    # Utils tests (TF-specific)
    "utils/test_nearest_neighbors.py",
    "utils/test_sorted_dict.py",
]

# Files that require both backends (import both TF and PyTorch at top level)
_BOTH_BACKENDS_PATTERNS = [
    "common/test_backend_parity.py",
    "utils/test_backtracking.py",
    "utils/test_cgd.py",
]

# Build collect_ignore_glob dynamically based on what's available
collect_ignore_glob = []

if not HAS_PYTORCH:
    collect_ignore_glob.extend(_PYTORCH_PATTERNS)

if not HAS_TENSORFLOW:
    collect_ignore_glob.extend(_TENSORFLOW_PATTERNS)

if not (HAS_TENSORFLOW and HAS_PYTORCH):
    collect_ignore_glob.extend(_BOTH_BACKENDS_PATTERNS)


# ============================================================================
# Pytest hooks and configuration
# ============================================================================

def pytest_ignore_collect(collection_path, config):
    """
    Hook to ignore test files that would fail to import due to missing backends.
    This is more reliable than collect_ignore_glob for complex cases.
    """
    path_str = str(collection_path)

    # Check PyTorch patterns
    if not HAS_PYTORCH:
        if "_pytorch.py" in path_str or "_pytorch" in path_str.lower():
            return True

    # Check TensorFlow patterns
    if not HAS_TENSORFLOW:
        # Check against known TF-specific files
        tf_files = [
            "test_bench.py", "test_benchmark_base.py",
            "test_sample_boundary.py", "test_weights_boundary.py",
            "test_ihvp.py", "test_ihvp_factory.py",
            "test_inf_abstract.py", "test_model_wrapper.py",
            "test_backend_tensorflow.py",
            "test_first_order_influence_calculator.py",
            "test_second_order_influence.py",
            "test_arnoldi_influence_calculator.py",
            "test_representer_point_l2.py", "test_rps_lje.py",
            "test_tracin.py",
            "test_nearest_neighbors.py", "test_sorted_dict.py",
            "test_optimizer_state.py",
        ]
        for tf_file in tf_files:
            if path_str.endswith(tf_file):
                # Make sure it's not a pytorch version
                if "_pytorch" not in path_str:
                    return True

    # Check both-backends patterns
    if not (HAS_TENSORFLOW and HAS_PYTORCH):
        both_files = ["test_backend_parity.py", "test_backtracking.py", "test_cgd.py"]
        for both_file in both_files:
            if path_str.endswith(both_file):
                return True

    return None  # Don't ignore, let pytest handle it


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
    skip_both_not_selected = pytest.mark.skip(reason="Both backends must be selected (--backend=all)")

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
            elif not (run_tensorflow and run_pytorch):
                item.add_marker(skip_both_not_selected)

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
        f"  PyTorch NumPy bridge: {'available' if HAS_PYTORCH_NUMPY_BRIDGE else 'NOT available'}",
        f"  Ignored patterns: {len(collect_ignore_glob)} pattern(s)",
    ]
    if HAS_PYTORCH and not HAS_PYTORCH_NUMPY_BRIDGE:
        lines.append("  Hint: this torch build needs numpy<2 for tensor.numpy() support")
    backend_option = config.getoption("--backend", default=None)
    if backend_option:
        lines.append(f"  Selected backend: {backend_option}")
    return lines
