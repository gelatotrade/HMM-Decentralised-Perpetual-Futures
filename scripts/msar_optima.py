"""What each optimum of the paper-window premium likelihood implies.

Section 5.3 reports that the paper's multi-start protocol finds two optima on the
paper window and that the more frequent one is the worse one. This script records
the half-lives and their ratio at each: it refits single starts from the first
seed block of scripts/msar_seed_variation.py, in seed order, until every optimum
that block found has been reached once, and writes them to output/msar_optima.json.

Run from the repository root:  PYTHONPATH=src python3 scripts/msar_optima.py
Runtime: about a minute (macOS/arm64, CPython 3.9.6).
"""
import json
import warnings

import numpy as np

from regime_engine.hmm import fit_ms_ar1
from regime_engine.msar import half_life
from regime_engine.panel import load_paper_window_premium
from regime_engine.paths import DEFAULT_OUT

warnings.filterwarnings("ignore")

block = json.loads((DEFAULT_OUT / "msar_seed_variation.json").read_text())["runs"][0]
search_reps = json.loads((DEFAULT_OUT / "msar_seed_variation.json").read_text())[
    "protocol"]["search_reps"]
wanted = sorted((o["llf"] for o in block["optima_found"]), reverse=True)

y = load_paper_window_premium()
found = {}
for seed in range(block["seeds"][0], block["seeds"][1] + 1):
    try:
        _, res, _ = fit_ms_ar1(y, seed=seed, search_reps=search_reps, seed_retries=1)
    except Exception:
        continue
    llf = round(float(res.llf), 2)
    if llf not in wanted or llf in found:
        continue
    names = list(res.model.param_names)
    p = np.asarray(res.params, dtype=float)
    sigma2 = np.array([p[names.index(f"sigma2[{k}]")] for k in range(2)])
    phi = np.array([p[names.index(f"ar.L1[{k}]")] for k in range(2)])
    tau = half_life(phi[np.argsort(sigma2)])          # 0 = calm (low variance)
    found[llf] = {"llf": llf, "first_seed": seed, "tau_calm": float(tau[0]),
                  "tau_stressed": float(tau[1]), "ratio": float(tau[0] / tau[1])}
    print(found[llf], flush=True)
    if len(found) == len(wanted):
        break

payload = {
    "_note": ("Half-lives and their ratio at each optimum of the MS-AR(1) likelihood on "
              "the paper-window premium (T = 5,001), from single starts of the first "
              "seed block of msar_seed_variation.json. Regimes ordered by sigma2."),
    "T": int(len(y)),
    "search_reps": search_reps,
    "optima": [found[llf] for llf in wanted],
    "logL_gap": round(wanted[0] - wanted[-1], 2),
}
(DEFAULT_OUT / "msar_optima.json").write_text(json.dumps(payload, indent=1) + "\n")
