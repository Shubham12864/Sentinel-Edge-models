PYTHON ?= python

.PHONY: all setup test clean

all: check-structure

check-structure:
	@echo "Checking Tier 2 AI project structure..."
	@test -d app -a -d config -a -d data -a -d models -a -d tests
	@echo "Project structure is valid."

setup:
	$(PYTHON) -m venv .venv
	@echo "Create a virtual environment with: .venv\\Scripts\\activate"

test:
	pytest -q

clean:
	rm -rf .venv __pycache__ .pytest_cache
