PYTHON ?= .venv/bin/python
PDF_PYTHON ?= $(PYTHON)
WORKERS ?= 4
RUN_DIR ?= runs/final-integration
RESULTS_DIR ?= runs/final-analysis
SKILL ?= CRUISE_RIGHT
SEED ?= 0
export PYTHONPATH := .

.PHONY: help install test lint typecheck verify smoke doctor check train study analyze report milestone metadrive-check calibrate clean

help:
	@echo "check: lint, types, tests, checkpoints, offline smoke"
	@echo "train: single-critic PPO (SKILL=CRUISE_RIGHT or OVERTAKE_LEFT, SEED=0)"
	@echo "study / analyze: final comparison (RUN_DIR, RESULTS_DIR, WORKERS)"
	@echo "report / milestone: rebuild recorded PDFs (PDF_PYTHON needs ReportLab)"
	@echo "metadrive-check / calibrate: physical integration and recorded skill calibration"
	@echo "install / test / lint / typecheck / verify / smoke / doctor / clean"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check event_cbf_driving tests scripts

typecheck:
	$(PYTHON) -m mypy event_cbf_driving

verify:
	$(PYTHON) -m event_cbf_driving.cli verify --profile default

smoke:
	$(PYTHON) -m event_cbf_driving.cli eval --profile ci --router scripted \
		--out-dir runs/ci --run-id ci-smoke

doctor:
	$(PYTHON) -m event_cbf_driving.cli doctor

check: lint typecheck test verify smoke

train:
	$(PYTHON) -m event_cbf_driving.skills.train_ppo --skill $(SKILL) --seed $(SEED)

study:
	$(PYTHON) scripts/integrated_eval.py test --workers $(WORKERS) --output $(RUN_DIR)

analyze:
	PYTHONPATH=.:scripts $(PYTHON) scripts/integrated_analysis.py \
		--runs-dir $(RUN_DIR) --results-dir $(RESULTS_DIR) --audit-traces

report:
	$(PDF_PYTHON) scripts/build_integrated_report.py

milestone:
	$(PDF_PYTHON) ../docs/build_milestone_report.py

metadrive-check:
	$(PYTHON) -m pytest -q tests/test_metadrive.py tests/test_layer_interactions.py tests/test_study.py
	$(PYTHON) -m event_cbf_driving.cli eval --profile metadrive-study \
		--modes proposed --seeds 0 1 2 --no-figures --run-id metadrive-smoke

calibrate:
	$(PYTHON) scripts/calibrate_metadrive.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
