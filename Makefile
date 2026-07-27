.PHONY: install run debug clean lint lint-strict

install:
	@echo "Installing dependencies..."
	uv sync

run:
	@echo "Running the program"
	uv run python -m src

debug:
	@echo "Starting debug mode with pdb..."
	uv run python -m pdb -m src

clean:
	@echo "Cleaning temporary files..."
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	@echo "Running linter (flake8) and type checker (mypy)..."
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running linter (flake8) and type checker (mypy) strict..."
	uv run flake8 src
	uv run mypy src --strict