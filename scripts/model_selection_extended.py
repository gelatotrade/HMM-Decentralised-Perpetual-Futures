"""Information criteria for K = 2 ... 5 on the reproducible sub-sample.

Backs the model-order paragraph of Section 4.3: every K converges from 40
starts, K = 4 attains the best BIC, and the paper keeps K = 3 for the reasons it
gives there. Writes output/model_selection_extended.json.

Run from the repository root:  PYTHONPATH=src python3 scripts/model_selection_extended.py
"""
import json
import warnings

import pandas as pd

from regime_engine.paths import DEFAULT_OUT, DEFAULT_PANEL
from regime_engine.robustness import (
    SELECTION_KS,
    SELECTION_N_STARTS,
    model_selection,
    preferred_k,
)

warnings.filterwarnings("ignore")

pan = pd.read_csv(DEFAULT_PANEL)
r = pan["log_return"].dropna().values          # raw returns, as in summary.json
print(f"Panel T={len(r)} (raw log-returns, the summary.json convention)", flush=True)
recs = model_selection(r, verbose=True)
best = preferred_k(recs)
prev = json.load(open(DEFAULT_OUT / "summary.json")).get("model_selection", {})
payload = {
    "_note": ("Information criteria for K in {2,3,4,5} on the reproducible "
              "sub-sample panel, raw log-returns. All four converge from 40 "
              "starts and K=4 attains the best BIC; the paper retains K=3 on "
              "the grounds given in Section 4.3, not on BIC alone."),
    "panel": str(DEFAULT_PANEL.relative_to(DEFAULT_PANEL.parents[2])),
    "T": int(len(r)),
    "protocol": {"n_starts": SELECTION_N_STARTS, "ks": list(SELECTION_KS)},
    "records": recs,
    "preferred_by_bic": best,
    "previous_k2_k3_from_summary_json": prev,
}
(DEFAULT_OUT / "model_selection_extended.json").write_text(json.dumps(payload, indent=1))
print(f"\nBIC optimum: K={best}", flush=True)
print("\nCheck against summary.json (8 starts, seed 42+K):", flush=True)
for k in ("2", "3"):
    if k in prev:
        got = next(x for x in recs if x["K"] == int(k))
        print(f"  K={k}: summary.json BIC={prev[k]['BIC']:.2f}  "
              f"new (40 starts) BIC={got['BIC']:.2f}  "
              f"difference {got['BIC'] - prev[k]['BIC']:+.2f}", flush=True)
print("DONE", flush=True)
