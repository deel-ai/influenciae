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
    version="0.0.2",
    description="Influence Function toolbox for Tensorflow 2",
    long_description=README,
    long_description_content_type="text/markdown",
    author="DEEL Core Team",
    author_email="agustin-martin.picard@irt-saintexupery.com",
    license="MIT",
    install_requires=['tensorflow>=2.3.0', 'numpy', 'matplotlib'],
    extras_require={
        "tests": ["pytest", "pylint"],
        "docs": ["mkdocs", "mkdocs-material", "numkdoc"],
    },
    packages=setuptools.find_namespace_packages(include=["deel.*"]),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)