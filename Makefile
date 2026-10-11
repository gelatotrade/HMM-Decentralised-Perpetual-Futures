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
        censoring longhistory premiumgap msartables oos figures a31 \
        test paper clean

BOOT_REPS ?= 500

help:
	@echo "setup      - install pinned deps (and the package, editable)"
	@echo "analyse    - fit both models on \$$PANEL (default: the shipped subsample)"
	@echo "robustness - winsorisation robustness check on \$$PANEL"
	@echo "benchmarks - out-of-family benchmarks (single Gaussian, GARCH) vs the HMMs"
	@echo "diagnostics- residual diagnostics for the HMM and the MS-AR"
	@echo "bootstrap  - parametric bootstrap of the K=3 HMM (\$$BOOT_REPS reps, ~35 min)"
	@echo "censoring  - funding at the floor on three venues (Table 3, Figure 2)"
	@echo "longhistory- censoring since December 2023, BTC and ETH (Table 4, Figure 3)"
	@echo "premiumgap - the premium gap and its split (Tables 1, 5, C1, C2, Figure 4)"
	@echo "msartables - the MS-AR tables from diagnostics and oos (Tables 6, C4, C5)"
	@echo "oos        - rolling windows and upper-tail sensitivity of the MS-AR"
	@echo "figures    - the explanatory figures (Figures 1, 5, 6, 7, 8)"
	@echo "a31        - evidence on the high-to-low transition (\$$A31_PART; all: ~2 h)"
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

# The three deliverables of regime_engine.diagnostics. `benchmarks` runs in well
# under a minute and `diagnostics` in about three minutes; `bootstrap` is the
# slow one (about 35 minutes).
benchmarks:
	$(PY) regime_engine.diagnostics --benchmarks --panel $(PANEL) --out $(OUT)

diagnostics:
	$(PY) regime_engine.diagnostics --diagnostics --panel $(PANEL) --out $(OUT)

bootstrap:
	$(PY) regime_engine.diagnostics --bootstrap --reps $(BOOT_REPS) --out $(OUT)

# ---- the cross-venue comparison: offline from data/, seconds each ----
censoring:
	$(PY) regime_engine.censoring

longhistory:
	$(PY) regime_engine.longhistory

premiumgap:
	$(PY) regime_engine.premium_gap

# ---- MS-AR robustness and its tables ----
oos:
	$(PY) regime_engine.oos

msartables:
	$(PY) regime_engine.msar_tables

figures:
	$(PY) regime_engine.paper_figures

# One block of output/a31_evidence.json (profile, profile_calibrated, power,
# power_em, garch, convergence) or all of them; the others are kept. "profile"
# keeps its "calibrated" sub-block only while the bound and its calibration are
# unchanged; otherwise rerun "profile_calibrated" (about 40 min) after it.
A31_PART ?= all
a31:
	PYTHONPATH=src python3 scripts/a31_evidence.py --part $(A31_PART)

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
