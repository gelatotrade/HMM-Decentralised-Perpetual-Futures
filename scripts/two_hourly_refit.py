"""The two-hourly K = 3 refit over the whole paper window (Section 4.3).

Hourly candles for October and November 2025 are no longer served, but
two-hourly candles for the whole window are. This refits the paper's K = 3
Gaussian HMM, with the settings of regime_engine.analysis, to the two-hourly
log-returns of 2 October 2025 -- 28 April 2026 and writes the state
volatilities (annualised with sqrt(12 * 365)), stationary masses and kernel to
output/two_hourly_refit.json.

Run from the repository root:  PYTHONPATH=src python3 scripts/two_hourly_refit.py
"""
import json

import numpy as np
import pandas as pd

from regime_engine.analysis import HMM_N_ITER, HMM_N_STARTS, HMM_SEED_BASE
from regime_engine.hmm import GaussianHMM
from regime_engine.paths import DEFAULT_OUT, PAPER_WINDOW, REPO_ROOT

K = 3
px = pd.read_csv(REPO_ROOT / "data" / "interim" / "prices_BTC_2h_full.csv")
px["time"] = pd.to_datetime(px["time"], utc=True, format="ISO8601")
start = pd.Timestamp(PAPER_WINDOW[0], tz="UTC")
end = pd.Timestamp(PAPER_WINDOW[1])
window = px[(px["time"] >= start) & (px["time"] < end)]
r = window["log_return"].dropna().to_numpy(float)

hmm = GaussianHMM(K=K, n_starts=HMM_N_STARTS, n_iter=HMM_N_ITER,
                  random_state=HMM_SEED_BASE + K).fit(r)
annualise = np.sqrt(12 * 365)
payload = {
    "_note": ("K = 3 Gaussian HMM on two-hourly log-returns over the paper window, "
              "states ordered by sigma ascending; settings of regime_engine.analysis."),
    "bars": int(len(window)),
    "T": int(len(r)),
    "first_bar": str(window["time"].iloc[0]),
    "last_bar": str(window["time"].iloc[-1]),
    "seed": HMM_SEED_BASE + K,
    "annualised_sigma_pct": [float(100 * s * annualise) for s in hmm.sigma],
    "stationary": [float(p) for p in hmm.stationary_distribution()],
    "A": [[float(a) for a in row] for row in hmm.A],
    "logL": float(hmm.log_likelihood_),
}
(DEFAULT_OUT / "two_hourly_refit.json").write_text(json.dumps(payload, indent=1) + "\n")
print({k: payload[k] for k in ("T", "annualised_sigma_pct", "stationary")},
      "A31", payload["A"][2][0])
