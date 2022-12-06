.PHONY: help prepare-dev test test-disable-gpu doc serve-doc
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py37, py38, py39, py310"
	@echo "make test-disable-gpu"
	@echo "       run test with gpu disabled"
	@echo "make serve-doc"
	@echo "       run documentation server for development"
	@echo "make doc"
	@echo "       build mkdocs documentation"

prepare-dev:
	python3 -m pip install virtualenv
	python3 -m venv influenciae_dev_env
	. influenciae_dev_env/bin/activate && pip install -r requirements.txt
	. influenciae_dev_env/bin/activate && pip install -r requirements_dev.txt

test:
	tox

test-disable-gpu:
	CUDA_VISIBLE_DEVICES=-1 tox

doc:
	mkdocs build
	mkdocs gh-deploy

serve-doc:
	CUDA_VISIBLE_DEVICES=-1 mkdocs serve