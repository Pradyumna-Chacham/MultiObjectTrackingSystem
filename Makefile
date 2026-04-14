PYTHON=python3
VENV=venv
VENV_PY=$(VENV)/bin/python
CONFIG?=configs/default.yaml
INPUT?=demo/sample_videos/sample.mp4
OUTPUT?=demo/sample_outputs/output.mp4
PRED?=demo/sample_outputs/output.mot.txt
SEQ?=MOT17-09-FRCNN

.PHONY: install dirs run test lint metrics download-models

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

dirs:
	mkdir -p demo/sample_videos demo/sample_outputs evaluation/results models

run:
	PYTHONPATH=. $(VENV_PY) scripts/run_demo.py --config $(CONFIG) --input $(INPUT) --output $(OUTPUT)

test:
	PYTHONPATH=. $(VENV_PY) -m pytest tests/ --cov=src --cov-report=term-missing

lint:
	PYTHONPATH=. $(VENV_PY) -m ruff check src tests scripts

metrics:
	PYTHONPATH=. $(VENV_PY) evaluation/evaluate_mot.py --pred $(PRED) --sequence-dir $(SEQ)

download-models:
	PYTHONPATH=. $(VENV_PY) scripts/download_models.py
