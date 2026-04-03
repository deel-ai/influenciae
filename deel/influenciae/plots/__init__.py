# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Visualizations module
"""
from typing import TYPE_CHECKING

from .._optional_imports import import_optional_attr

from .benchmark import BenchmarkDisplay

__all__ = [
    "BenchmarkDisplay",
    "plot_most_influential_images",
    "plot_datacentric_explanations",
]

_TF_IMAGE_ATTRS = {"plot_most_influential_images", "plot_datacentric_explanations"}


def __getattr__(name):
    """Lazy import TensorFlow-specific image plotting helpers."""
    if name in _TF_IMAGE_ATTRS:
        attr = import_optional_attr(".image", name, package=__package__, extra="tensorflow")
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .image import plot_datacentric_explanations, plot_most_influential_images
