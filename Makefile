# ==============================================================================
# AI-Search Visibility Developer Automation Makefile
# ==============================================================================
.PHONY: help install test test-aiml test-devops lint format demo worker docker-build docker-up docker-down k8s-apply k8s-delete

PYTHON ?= python
PIP ?= pip
DOCKER ?= docker
DOCKER_COMPOSE ?= docker compose
KUBECTL ?= kubectl

IMAGE_NAME ?= ghcr.io/sriramkunamsetty/ai-search-visibility:latest

help:
	@echo "AI-Search Visibility - Command Automation"
	@echo "----------------------------------------"
	@echo "  make install        Install development package in editable mode"
	@echo "  make test           Run all 31 unit, delivery, ML, and DevOps tests"
	@echo "  make test-aiml      Run AI/ML and semantic evaluation tests"
	@echo "  make test-devops    Run DevOps, Prometheus metrics, and worker tests"
	@echo "  make lint           Check code quality using ruff"
	@echo "  make format         Auto-format codebase using ruff"
	@echo "  make demo           Launch dashboard with synthetic seeded simulator"
	@echo "  make worker         Launch background distributed audit worker"
	@echo "  make docker-build   Build multi-stage hardened OCI container"
	@echo "  make docker-up      Start API, Worker, Postgres, Redis, Prometheus & Grafana"
	@echo "  make docker-down    Stop and remove Docker Compose containers"
	@echo "  make k8s-apply      Deploy full Kubernetes production manifests"
	@echo "  make k8s-delete     Remove Kubernetes deployment from cluster"

install:
	$(PIP) install -e ".[postgres,ner]"

test:
	$(PYTHON) -m unittest discover -s tests -v

test-aiml:
	$(PYTHON) -m unittest tests/test_aiml.py -v

test-devops:
	$(PYTHON) -m unittest tests/test_devops_observability.py -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

demo:
	$(PYTHON) -m aivisibility.cli serve --demo

worker:
	$(PYTHON) -m aivisibility.cli worker --concurrency 4

audit-demo:
	$(PYTHON) -m aivisibility.cli audit list --demo

docker-build:
	$(DOCKER) build -t $(IMAGE_NAME) .

docker-up:
	$(DOCKER_COMPOSE) up -d --build

docker-down:
	$(DOCKER_COMPOSE) down -v

k8s-apply:
	$(KUBECTL) apply -f deploy/kubernetes/

k8s-delete:
	$(KUBECTL) delete -f deploy/kubernetes/
