# XVoxel-FCM Makefile — Unified build entry points (Principle P3)
#
# Purpose: Ensure multi-artifact reports (text + figures + tables) are
# regenerated atomically from a single command. Eliminates the "updated text
# but forgot figures" failure observed in the 2026-06-25 session.
#
# Usage:
#   make fig7        # Run FCM L-shape canonical example (Fig 7)
#   make fluid       # Run fluid cylinder cross-flow validation example
#   make baseline    # Regenerate golden baseline (ONLY on intentional change)
#   make test        # Run full test suite
#   make test-fidelity  # Run fidelity tests only (regression + bit-level)
#   make test-quick  # Run fast unit tests (exclude slow integration)
#   make clean       # Remove generated outputs
#
# NOTE: On Windows, run via `mingw32-make` or use the VS Code tasks.json
# equivalent. Python invocations use the project venv.

PYTHON = .venv_xvoxel/Scripts/python.exe

.PHONY: fig7 fluid baseline test test-fidelity test-quick clean help

help:  ## Show available targets
	@echo "XVoxel-FCM build targets:"
	@echo "  make fig7           Run FCM L-shape canonical example (Fig 7)"
	@echo "  make fluid          Run fluid cylinder cross-flow validation example"
	@echo "  make baseline       Regenerate golden baseline (intentional changes only)"
	@echo "  make test           Run full test suite"
	@echo "  make test-fidelity  Run regression + bit-level fidelity tests"
	@echo "  make test-quick     Run fast unit tests (exclude slow integration)"
	@echo "  make clean          Remove generated outputs"

fig7:  ## Run FCM L-shape canonical example (Paper Fig 7-9)
	$(PYTHON) examples/fig7_lshape_v2.py
	@echo "✓ Fig 7 L-shape example complete"

fluid:  ## Run fluid cylinder cross-flow validation example
	$(PYTHON) examples/cylinder_flow_validation.py
	@echo "✓ Fluid cylinder validation complete: figures/"

baseline:  ## Regenerate golden baseline (ONLY on intentional algorithm change)
	$(PYTHON) tests/baselines/generate_fig7_baseline.py
	@echo "✓ Golden baseline regenerated: tests/baselines/fig7_golden.json"
	@echo "  WARNING: Only run this when algorithm logic intentionally changes."

test:  ## Run full test suite
	$(PYTHON) -m pytest tests/ -v

test-fidelity:  ## Run regression + bit-level fidelity tests (Principle P1 gate)
	$(PYTHON) -m pytest tests/test_regression_fig7.py tests/test_vectorization_bit_level.py -v

test-quick:  ## Run fast unit tests (exclude slow integration)
	$(PYTHON) -m pytest tests/ -v -m "not slow"

clean:  ## Remove generated outputs
	@if exist "figures\*.png" del /Q "figures\*.png"
	@echo "✓ Generated outputs removed"
