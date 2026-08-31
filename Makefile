.DEFAULT_GOAL := help

.PHONY: help install dev test lint format format-check check \
        route bench docker-build docker-run compose-up compose-down clean

help:
	@echo "nashgate — common dev commands"
	@echo ""
	@echo "  make install        pip install -e ."
	@echo "  make dev            pip install -e '.[dev]'"
	@echo "  make test           run the test suite (pytest)"
	@echo "  make lint           ruff check ."
	@echo "  make format         ruff format . (rewrites files)"
	@echo "  make format-check   ruff format --check . (CI-safe, no writes)"
	@echo "  make check          lint + test — what CI runs"
	@echo "  make route          start the gateway against docs/example.config.yaml"
	@echo "  make bench          run the benchmark harness against the example config"
	@echo "  make docker-build   docker build -t nashgate ."
	@echo "  make docker-run     run the built image, mounting the example config"
	@echo "  make compose-up     docker compose up --build"
	@echo "  make compose-down   docker compose down"
	@echo "  make clean          remove caches/build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

check: lint test

route:
	nashgate route --config docs/example.config.yaml

bench:
	nashgate bench --config docs/example.config.yaml

docker-build:
	docker build -t nashgate .

docker-run:
	docker run --rm -p 8000:8000 \
		-v $(CURDIR)/docs/example.config.yaml:/config/config.yaml:ro \
		--env-file .env \
		nashgate

compose-up:
	docker compose up --build

compose-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info build dist
