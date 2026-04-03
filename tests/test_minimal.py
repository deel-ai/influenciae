# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Tests that run without any backend (minimal mode).
These tests verify the library can be imported and basic functionality works
without TensorFlow or PyTorch installed.

These tests MUST NOT import any backend-specific code at the module level.
"""


def test_import_influenciae():
    """Test that the main package can be imported."""
    import deel.influenciae

    assert hasattr(deel.influenciae, '__version__')


def test_version_string():
    """Test that version is a valid string."""
    import deel.influenciae

    assert isinstance(deel.influenciae.__version__, str)
    # Version should be in format X.Y.Z
    parts = deel.influenciae.__version__.split('.')
    assert len(parts) >= 2


def test_submodules_are_listed():
    """Test that submodules are discoverable via __dir__."""
    import deel.influenciae

    module_dir = dir(deel.influenciae)
    # These should be listed even if not yet imported (lazy loading)
    expected = ['common', 'influence', 'rps', 'trac_in', 'benchmark', 'plots']
    for submod in expected:
        assert submod in module_dir, f"Submodule '{submod}' should be in dir()"


def test_plots_import_without_backend_specific_helpers():
    """Test that the plots package can be imported without loading TensorFlow helpers."""
    from deel.influenciae import plots

    assert hasattr(plots, 'BenchmarkDisplay')
    assert 'plot_most_influential_images' in plots.__all__


def test_get_available_frameworks():
    """Test that get_available_frameworks works without error."""
    # Import directly from backend module to avoid full common import
    from deel.influenciae.common.backend import Framework, get_available_frameworks

    frameworks = get_available_frameworks()
    assert isinstance(frameworks, list)
    # All items should be Framework enum members
    for fw in frameworks:
        assert isinstance(fw, Framework)


def test_framework_enum():
    """Test Framework enum values."""
    from deel.influenciae.common.backend import Framework

    assert hasattr(Framework, 'TENSORFLOW')
    assert hasattr(Framework, 'PYTORCH')
    assert Framework.TENSORFLOW.value == 'tensorflow'
    assert Framework.PYTORCH.value == 'pytorch'


def test_types_import():
    """Test that types module can be imported."""
    from deel.influenciae import types

    # Should have common type aliases
    assert hasattr(types, 'Optional') or 'Optional' in dir(types)
