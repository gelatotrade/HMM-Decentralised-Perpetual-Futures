"""How the MS-AR multi-start split depends on the seed block.

Section 5.3 says that the split between the two optima of the premium likelihood
varies with the seed while their ranking does not. This script is the record
behind that sentence. It runs the paper's protocol (40 starts, 40 search
repetitions) on the paper window six times, each time on a disjoint block of
40 consecutive seeds, and writes every optimum found to
output/msar_seed_variation.json.

Run from the repository root:  PYTHONPATH=src python3 scripts/msar_seed_variation.py
Runtime: about ten minutes (macOS/arm64, CPython 3.9.6).
"""
import json
import warnings

from regime_engine.hmm import MSAR_SEED, fit_ms_ar1_multistart
from regime_engine.panel import load_paper_window_premium
from regime_engine.paths import DEFAULT_OUT

warnings.filterwarnings("ignore")

N_STARTS = 40
SEARCH_REPS = 40
N_BLOCKS = 6

y = load_paper_window_premium()
runs = []
for block in range(N_BLOCKS):
    seed_base = MSAR_SEED + block * N_STARTS
    _, res, best_seed, diag = fit_ms_ar1_multistart(
        y, n_starts=N_STARTS, seed=seed_base, search_reps=SEARCH_REPS, verbose=False)
    runs.append({
        "seed_base": seed_base,
        "seeds": [seed_base, seed_base + N_STARTS - 1],
        "optima_found": diag["optima_found"],
        "failures": diag["failures"],
        "best_llf": round(float(res.llf), 2),
        "best_seed": int(best_seed),
        "modal_is_best": diag["modal_is_best"],
    })
    print(f"block {block}: seeds {seed_base}-{seed_base + N_STARTS - 1}  "
          f"{diag['optima_found']}  failures {diag['failures']}", flush=True)

inferior = [next((o["count"] for o in r["optima_found"] if abs(o["llf"] + 4828.67) < 0.05), 0)
            for r in runs]
payload = {
    "_note": ("MS-AR(1) on the paper-window premium (T = 5,001), the paper's protocol "
              "repeated on six disjoint blocks of 40 consecutive seeds. Records how many "
              "starts reach each optimum. Backs the seed sentence of Section 5.3."),
    "protocol": {"n_starts": N_STARTS, "search_reps": SEARCH_REPS, "n_blocks": N_BLOCKS},
    "T": int(len(y)),
    "runs": runs,
    "starts_at_inferior_optimum": {"min": min(inferior), "max": max(inferior),
                                   "per_block": inferior},
    "best_llf_every_block": sorted({r["best_llf"] for r in runs}),
}
(DEFAULT_OUT / "msar_seed_variation.json").write_text(json.dumps(payload, indent=1) + "\n")
print(f"inferior optimum reached by {min(inferior)} to {max(inferior)} of {N_STARTS} starts",
      flush=True)
