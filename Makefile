PYTHON ?= .venv/bin/python
PDF_PYTHON ?= $(PYTHON)
WORKERS ?= 4
RUN_DIR ?= runs/final-integration
RESULTS_DIR ?= runs/final-analysis
SKILL ?= CRUISE_RIGHT
SEED ?= 0
export PYTHONPATH := .
export PYTHONPYCACHEPREFIX := $(abspath ../cs642-driving-cache/python)

.PHONY: help install test lint typecheck verify smoke doctor check train study analyze report milestone metadrive-check calibrate clean

help:
	@echo "check: lint, types, tests, checkpoints, offline smoke"
	@echo "train: single-critic PPO (SKILL=CRUISE_RIGHT or OVERTAKE_LEFT, SEED=0)"
	@echo "study / analyze: final comparison (RUN_DIR, RESULTS_DIR, WORKERS)"
	@echo "report / milestone: rebuild recorded PDFs (PDF_PYTHON needs ReportLab)"
	@echo "metadrive-check / calibrate: physical integration and recorded skill calibration"
	@echo "install / test / lint / typecheck / verify / smoke / doctor / clean"

install:
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest -q --cov=driving --cov-report=term:skip-covered --cov-fail-under=60

lint:
	$(PYTHON) -m ruff check driving tests tools

typecheck:
	$(PYTHON) -m mypy

verify:
	$(PYTHON) -m driving verify --profile default

smoke:
	$(PYTHON) -m driving eval --profile ci --router scripted \
		--out-dir runs/ci --run-id ci-smoke

doctor:
	$(PYTHON) -m driving doctor

check: lint typecheck test verify smoke

train:
	$(PYTHON) -m driving.train --skill $(SKILL) --seed $(SEED)

study:
	$(PYTHON) tools/integrated_eval.py test --workers $(WORKERS) --output $(RUN_DIR)

analyze:
	PYTHONPATH=.:tools $(PYTHON) tools/integrated_analysis.py \
		--runs-dir $(RUN_DIR) --results-dir $(RESULTS_DIR) --audit-traces

report:
	$(PDF_PYTHON) tools/build_integrated_report.py

milestone:
	$(PDF_PYTHON) tools/build_milestone_report.py

metadrive-check:
	$(PYTHON) -m pytest -q tests/test_metadrive.py tests/test_layer_interactions.py tests/test_study.py
	$(PYTHON) -m driving eval --profile metadrive-study \
		--modes proposed --seeds 0 1 2 --run-id metadrive-smoke

calibrate:
	$(PYTHON) tools/calibrate_metadrive.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache build dist
	find driving tests tools models -type d -name __pycache__ -prune -exec rm -rf {} +
