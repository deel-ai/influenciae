.PHONY: help prepare-dev prepare-dev-tf prepare-dev-pytorch prepare-dev-all test test-minimal test-tensorflow test-pytorch test-all test-disable-gpu test-quick lint typecheck doc serve-doc
.DEFAULT_GOAL := help

help:
	@echo "Influenciae Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make prepare-dev         Create and prepare development environment"
	@echo "  make prepare-dev-tf      Setup with TensorFlow backend"
	@echo "  make prepare-dev-pytorch Setup with PyTorch backend"
	@echo "  make prepare-dev-all     Setup with both backends"
	@echo ""
	@echo "Testing:"
	@echo "  make test                Run all tests (with available backends)"
	@echo "  make test-minimal        Run tests without any backend (interface tests only)"
	@echo "  make test-tensorflow     Run tests with TensorFlow backend"
	@echo "  make test-pytorch        Run tests with PyTorch backend"
	@echo "  make test-all            Run tests with both backends"
	@echo "  make test-disable-gpu    Run tests with GPU disabled"
	@echo ""
	@echo "Quality:"
	@echo "  make lint                Run pylint"
	@echo "  make typecheck           Run mypy type checking"
	@echo ""
	@echo "Documentation:"
	@echo "  make doc                 Build and deploy mkdocs documentation"
	@echo "  make serve-doc           Run documentation server for development"

# ============================================================================
# Development Environment Setup
# ============================================================================

prepare-dev:
	python3 -m venv influenciae_dev_env
	. influenciae_dev_env/bin/activate && pip install -e ".[dev]"

prepare-dev-tf:
	python3 -m venv influenciae_dev_env
	. influenciae_dev_env/bin/activate && pip install -e ".[dev,tensorflow]"

prepare-dev-pytorch:
	python3 -m venv influenciae_dev_env
	. influenciae_dev_env/bin/activate && pip install -e ".[dev,pytorch]"

prepare-dev-all:
	python3 -m venv influenciae_dev_env
	. influenciae_dev_env/bin/activate && pip install -e ".[dev,all]"

# ============================================================================
# Testing
# ============================================================================

test:
	tox

test-minimal:
	tox -e minimal

test-tensorflow:
	tox -e tensorflow

test-pytorch:
	tox -e pytorch

test-all:
	tox -e all

test-disable-gpu:
	CUDA_VISIBLE_DEVICES=-1 tox

# Quick test for development (fastest feedback)
test-quick:
	tox -e quick

# ============================================================================
# Code Quality
# ============================================================================

lint:
	tox -e lint

typecheck:
	tox -e typecheck

# ============================================================================
# Documentation
# ============================================================================

doc:
	mkdocs build
	mkdocs gh-deploy

serve-doc:
	CUDA_VISIBLE_DEVICES=-1 mkdocs serve
