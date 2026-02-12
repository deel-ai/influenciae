"""
Utilities for optional and lazy imports.

These helpers centralize optional dependency loading and keep import statements
at module top-level to satisfy linting, while preserving backend-agnostic lazy
behavior.
"""
from functools import lru_cache
import importlib
from typing import Any, Optional


@lru_cache(maxsize=None)
def _import_module_cached(module_name: str, package: Optional[str] = None) -> Any:
    """Import and cache a module by name."""
    return importlib.import_module(module_name, package=package)


def import_optional_module(
    module_name: str,
    package: Optional[str] = None,
    *,
    extra: Optional[str] = None,
) -> Any:
    """
    Import an optional module with a user-facing installation hint.

    Parameters
    ----------
    module_name
        Absolute or relative module name.
    package
        Package used to resolve relative imports.
    extra
        Optional pip extra name to suggest on import failure.

    Returns
    -------
    module
        Imported Python module.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    """
    try:
        return _import_module_cached(module_name, package)
    except ImportError as exc:
        if extra is None:
            hint = ""
        else:
            hint = f" Install with: pip install influenciae[{extra}]"
        raise ImportError(f"Could not import optional dependency '{module_name}'.{hint}") from exc


def import_optional_attr(
    module_name: str,
    attr_name: str,
    package: Optional[str] = None,
    *,
    extra: Optional[str] = None,
) -> Any:
    """
    Import and return an attribute from an optional module.

    Parameters
    ----------
    module_name
        Absolute or relative module name.
    attr_name
        Attribute to retrieve from the imported module.
    package
        Package used to resolve relative imports.
    extra
        Optional pip extra name to suggest on import failure.

    Returns
    -------
    attribute
        The requested attribute.
    """
    module = import_optional_module(module_name, package=package, extra=extra)

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module '{module.__name__}' does not expose attribute '{attr_name}'."
        ) from exc
