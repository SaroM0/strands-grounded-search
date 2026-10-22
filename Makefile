VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.DEFAULT_GOAL := help

.PHONY: help install run test up down logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "%-10s %s\n", $$1, $$2}'

install: ## Create the venv and install the project (+ dev deps)
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -q -e ".[dev]"
	test -f .env || cp .env.example .env

up: ## Start local Qdrant (Docker)
	docker compose up -d

down: ## Stop local Qdrant
	docker compose down

logs: ## Follow Qdrant logs
	docker compose logs -f

run: ## Run the agent interactively (python -m grounded_search.cli)
	$(PYTHON) -m grounded_search.cli

test: ## Run the test suite
	$(PYTEST) -q

clean: ## Remove the venv and caches
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
