set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

sync:
    uv sync --locked --all-extras --dev

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

typecheck:
    uv run ty check

test:
    uv run pytest -q

check: format-check lint typecheck test

build:
    uv build

verify-dist: build
    uv run python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz

ci: check verify-dist
