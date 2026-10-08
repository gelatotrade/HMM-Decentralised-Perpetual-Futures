# Hyperliquid Regime Engine — task runner
# Every target in this file runs offline. The REST and S3 fetch paths are not
# targets here: they live in `regime_engine.fetch_official` / `fetch_prices`
# and are run by hand (see README.md).

# src-layout: put src/ on the path so `python -m regime_engine.…` resolves
# without installing. `pip install -e .` works too and makes the prefix
# unnecessary.
PY := PYTHONPATH=src python3 -m
PANEL ?= data/processed/panel_BTC_subsample.csv
OUT   ?= output

.PHONY: help setup analyse robustness benchmarks diagnostics bootstrap \
        test paper clean

BOOT_REPS ?= 500

help:
	@echo "setup      - install pinned deps (and the package, editable)"
	@echo "analyse    - fit both models on \$$PANEL (default: the shipped subsample)"
	@echo "robustness - winsorisation robustness check on \$$PANEL"
	@echo "benchmarks - out-of-family benchmarks (single Gaussian, GARCH) vs the HMMs"
	@echo "diagnostics- residual diagnostics for the HMM and the MS-AR"
	@echo "bootstrap  - parametric bootstrap of the K=3 HMM (\$$BOOT_REPS reps, ~35 min)"
	@echo "test       - run the unit test suite"
	@echo "paper      - compile paper/paper.tex (pdflatex, falling back to tectonic)"
	@echo "clean      - remove interim data and generated figures (keeps ALL of data/processed)"

setup:
	pip install -r requirements.txt
	@# Editable install is optional and needs pip >= 21.3 (PEP 660); it is allowed
	@# to fail, because every target puts src/ on PYTHONPATH.
	-pip install -e .

# ---- what actually runs offline, today, on the shipped panel ----
analyse:
	$(PY) regime_engine.analysis --panel $(PANEL) --out $(OUT)

robustness:
	$(PY) regime_engine.robustness --panel $(PANEL) --out $(OUT)

# The three deliverables of regime_engine.diagnostics. `benchmarks` and
# `diagnostics` run in a couple of minutes; `bootstrap` is the slow one.
benchmarks:
	$(PY) regime_engine.diagnostics --benchmarks --panel $(PANEL) --out $(OUT)

diagnostics:
	$(PY) regime_engine.diagnostics --diagnostics --panel $(PANEL) --out $(OUT)

bootstrap:
	$(PY) regime_engine.diagnostics --bootstrap --reps $(BOOT_REPS) --out $(OUT)

test:
	python3 -m pytest tests/ -q

# Uses pdflatex if available, otherwise tectonic.
paper:
	@if command -v pdflatex >/dev/null 2>&1; then \
	    echo "[paper] pdflatex"; \
	    cd paper && pdflatex -interaction=nonstopmode paper.tex >/dev/null && \
	                pdflatex -interaction=nonstopmode paper.tex >/dev/null && \
	                echo "paper/paper.pdf built"; \
	elif command -v tectonic >/dev/null 2>&1; then \
	    echo "[paper] tectonic"; \
	    cd paper && tectonic paper.tex && echo "paper/paper.pdf built"; \
	elif [ -x "$$HOME/.local/bin/tectonic" ]; then \
	    echo "[paper] tectonic ($$HOME/.local/bin)"; \
	    cd paper && "$$HOME/.local/bin/tectonic" paper.tex && echo "paper/paper.pdf built"; \
	else \
	    echo "ERROR: neither pdflatex nor tectonic found." >&2; exit 1; \
	fi

# Removes only regenerable files: untracked interim pulls, generated figures and
# the fit cache. data/processed, data/raw, output/tables and output/*.json are
# never removed.
clean:
	@# data/interim is not a scratch directory: the tracked series are the exact
	@# inputs of the paper (later fetches return longer files, and the 2h price
	@# window recedes by a day per day). Only untracked files are removed.
	git ls-files --others --exclude-standard -z -- data/interim \
	  | xargs -0 -r rm -f
	rm -rf output/figures/*
	rm -rf output/cache/*
	rm -f paper/*.aux paper/*.log paper/*.out
	@echo "cleaned (data/processed, data/raw, output/tables and output/*.json preserved)"
