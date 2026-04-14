PYTHON=python3
VENV=venv
VENV_PY=$(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

run:
	PYTHONPATH=. $(VENV_PY) scripts/run_demo.py --config configs/default.yaml

test:
	PYTHONPATH=. $(VENV_PY) -m pytest tests/ --cov=src --cov-report=term-missing

lint:
	PYTHONPATH=. $(VENV_PY) -m ruff check src tests scripts

benchmark:
	PYTHONPATH=. $(VENV_PY) scripts/benchmark.py --config configs/default.yaml

download-models:
	PYTHONPATH=. $(VENV_PY) scripts/download_models.py

download-data:
	PYTHONPATH=. $(VENV_PY) scripts/download_dataset.py