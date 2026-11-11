PYTHON ?= .venv/bin/python
PDF_PYTHON ?= $(PYTHON)
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
export PYTHONPATH := .

.PHONY: help install test lint format typecheck verify smoke eval eval-ci doctor check clean metadrive-check metadrive-eval calibrate study robustness study-live study-audit study-report

help:
	@echo "Targets:"
	@echo "  install    editable install with dev extras"
	@echo "  test       run the test suite"
	@echo "  lint       ruff check"
	@echo "  format     ruff format + import sorting"
	@echo "  typecheck  mypy across the package"
	@echo "  verify     validate checkpoints and learned diagnostics"
	@echo "  smoke      fast offline CI-profile evaluation"
	@echo "  eval       full production-profile evaluation + docs update"
	@echo "  doctor     dependency/config/checkpoint diagnostics"
	@echo "  check      lint + typecheck + test + verify + smoke"
	@echo "  metadrive-check  real physical integration tests + randomized smoke"
	@echo "  metadrive-eval   full randomized MetaDrive comparison"
	@echo "  calibrate / study / robustness / study-live   expanded physical experiments"
	@echo "  study-audit / study-report   verify saved final runs / build measured PDF"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) -q

lint:
	$(RUFF) check event_cbf_driving tests scripts

format:
	$(RUFF) check --fix event_cbf_driving tests scripts
	$(RUFF) format event_cbf_driving tests scripts

typecheck:
	$(MYPY) event_cbf_driving

verify:
	$(PYTHON) -m event_cbf_driving.cli verify --profile default

smoke:
	$(PYTHON) -m event_cbf_driving.cli eval --profile ci --router scripted \
		--out-dir runs/ci --run-id ci-smoke

eval:
	$(PYTHON) -m event_cbf_driving.cli eval --profile production --router scripted \
		--out-dir runs/production --update-docs

eval-ci:
	$(PYTHON) -m event_cbf_driving.cli eval --profile ci --router scripted \
		--out-dir runs/ci --run-id ci-smoke

doctor:
	$(PYTHON) -m event_cbf_driving.cli doctor

check: lint typecheck test verify smoke

metadrive-check:
	$(PYTEST) -q tests/test_metadrive.py tests/test_layer_interactions.py tests/test_study.py
	$(PYTHON) -m event_cbf_driving.cli eval --profile metadrive --modes proposed \
		--seeds 0 1 2 --no-figures --run-id metadrive-smoke

metadrive-eval:
	$(PYTHON) -m event_cbf_driving.cli eval --profile metadrive

calibrate:
	$(PYTHON) scripts/calibrate_metadrive.py

study:
	$(PYTHON) -m event_cbf_driving.cli eval --profile metadrive-study

robustness:
	$(PYTHON) scripts/run_robustness.py --output runs/robustness-final

study-live:
	OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $(PYTHON) -m event_cbf_driving.cli eval \
		--profile metadrive-local --require-live --no-fail-on-regression

study-audit:
	$(PYTHON) scripts/audit_study.py

study-report:
	$(PDF_PYTHON) scripts/build_research_report.py

clean:
	rm -rf runs .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
