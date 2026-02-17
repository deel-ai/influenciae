# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import setuptools
from setuptools import setup

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Influenciae",
    version="0.3.0",
    description="A Framework-Agnostic Toolbox for Influence Functions",
    long_description=README,
    long_description_content_type="text/markdown",
    author="DEEL Core Team",
    author_email="agustin-martin.picard@irt-saintexupery.com",
    license="MIT",
    # Core dependencies - no specific backend required
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        # Backend-specific dependencies (user chooses one or both)
        "tensorflow": ["tensorflow>=2.11.0,<2.21.0"],
        "pytorch": ["torch>=1.13.0,<2.11.0"],
        "all": [
            "tensorflow>=2.11.0,<2.21.0",
            "torch>=1.13.0,<2.11.0",
        ],
        # Development dependencies
        "tests": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pylint>=2.15.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
            "numkdoc",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pylint>=2.15.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
            "numkdoc",
            "tox>=4.0.0",
            "bump2version>=1.0.0",
            "mypy>=1.0.0",
            "mypy-extensions>=1.0.0",
        ],
    },
    packages=setuptools.find_namespace_packages(include=["deel.*"]),
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
