"""Why the paper models the premium: the MS-AR fitted to the headline rate.

Section 3.2 states that a two-state MS-AR(1) fitted to Hyperliquid's hourly
funding rate collapses onto a regime of numerically zero variance pinned to the
floor, leaving one ordinary regime to carry all variation. This fits it on the
paper window with the package's estimator from the first four seeds of the
paper's seed block and records each fit in output/msar_on_rate.json. The
likelihood is unbounded as that variance goes to zero, so the optimiser reports
non-convergence and the log-likelihood differs between seeds; the collapse and
the ordinary regime's coefficient do not.

Run from the repository root:  PYTHONPATH=src python3 scripts/msar_on_rate.py
"""
import json
import warnings

import numpy as np
import pandas as pd

from regime_engine.hmm import MSAR_SEED, fit_ms_ar1
from regime_engine.paths import DEFAULT_OUT, DEFAULT_PREMIUM_FULL, PAPER_WINDOW

warnings.filterwarnings("ignore")

d = pd.read_csv(DEFAULT_PREMIUM_FULL)
d["time"] = pd.to_datetime(d["time"], utc=True, format="ISO8601")
start = pd.Timestamp(PAPER_WINDOW[0], tz="UTC")
end = pd.Timestamp(PAPER_WINDOW[1])
y = d.loc[(d["time"] >= start) & (d["time"] <= end), "fundingRate"].to_numpy(float) * 1e4

fits = []
for seed in range(MSAR_SEED, MSAR_SEED + 4):
    _, res, _ = fit_ms_ar1(y, seed=seed, search_reps=40, seed_retries=1)
    names = list(res.model.param_names)
    p = np.asarray(res.params, dtype=float)
    sigma2 = [float(p[names.index(f"sigma2[{k}]")]) for k in range(2)]
    phi = [float(p[names.index(f"ar.L1[{k}]")]) for k in range(2)]
    order = list(np.argsort(sigma2))            # 0 = the collapsed regime
    fits.append({"seed": seed, "llf": float(res.llf),
                 "sigma2": [sigma2[k] for k in order], "phi": [phi[k] for k in order]})
    print(fits[-1], flush=True)

payload = {
    "_note": ("Two-state MS-AR(1) on the hourly funding rate (bps/h), paper window "
              "(T = 5,001). Regimes ordered by sigma2: index 0 is the collapsed one."),
    "T": int(len(y)),
    "fits": fits,
    "max_collapsed_sigma2": max(f["sigma2"][0] for f in fits),
    "ordinary_phi": [f["phi"][1] for f in fits],
}
(DEFAULT_OUT / "msar_on_rate.json").write_text(json.dumps(payload, indent=1) + "\n")
