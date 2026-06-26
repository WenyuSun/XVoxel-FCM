# XVoxel-FCM Makefile — Unified build entry points (Principle P3)
#
# Purpose: Ensure multi-artifact reports (text + figures + tables) are
# regenerated atomically from a single command. Eliminates the "updated text
# but forgot figures" failure observed in the 2026-06-25 session.
#
# Usage:
#   make report          # Regenerate Fig 7 comparison report + all figures
#   make baseline        # Regenerate golden baseline (ONLY on intentional change)
#   make test            # Run full test suite
#   test-fidelity        # Run fidelity tests only (regression + bit-level)
#   make clean           # Remove generated outputs
#
# NOTE: On Windows, run via `mingw32-make` or use the VS Code tasks.json
# equivalent. Python invocations use the project venv.

PYTHON = .venv_xvoxel/Scripts/python.exe

.PHONY: report baseline test test-fidelity test-quick clean help

help:  ## Show available targets
	@echo "XVoxel-FCM build targets:"
	@echo "  make report        Regenerate Fig 7 comparison report + figures"
	@echo "  make baseline      Regenerate golden baseline (intentional changes only)"
	@echo "  make test          Run full test suite"
	@echo "  make test-fidelity Run regression + bit-level fidelity tests"
	@echo "  make clean         Remove generated outputs"

report:  ## Regenerate Fig 7 comparison report + all figures (atomic)
	$(PYTHON) examples/compare_fig7_paper.py
	@echo "✓ Report and figures regenerated: output/paper_compare/"

baseline:  ## Regenerate golden baseline (ONLY on intentional algorithm change)
	$(PYTHON) tests/baselines/generate_fig7_baseline.py
	@echo "✓ Golden baseline regenerated: tests/baselines/fig7_golden.json"
	@echo "  WARNING: Only run this when algorithm logic intentionally changes."

test:  ## Run full test suite
	$(PYTHON) -m pytest tests/ -v

test-fidelity:  ## Run regression + bit-level fidelity tests (Principle P1 gate)
	$(PYTHON) -m pytest tests/test_regression_fig7.py tests/test_vectorization_bit_level.py -v

test-quick:  ## Run fast unit tests (exclude slow integration)
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_nitsche_hex32.py -k "not cantilever"

clean:  ## Remove generated outputs
	@if exist "output\paper_compare\*.png" del /Q "output\paper_compare\*.png"
	@if exist "output\paper_compare\COMPARISON_REPORT.md" del /Q "output\paper_compare\COMPARISON_REPORT.md"
	@if exist "figures\*.png" del /Q "figures\*.png"
	@echo "✓ Generated outputs removed"
