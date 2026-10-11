"""Information criteria for K = 2 ... 5 on the reproducible sub-sample.

Backs the model-order paragraph of Section 6.2. Every K is fitted by the
converged-fit protocol of ``hmm_batch.fit_protocol`` from a wide set of
starting points (``hmm_batch.wide_multistart``): the 40 starts of
``GaussianHMM.fit``, 160 starts with state volatilities spread log-uniformly
over 0.15-3.5 times the sample standard deviation, and the starts made by
splitting each state of the best K-1 fit. Each record reports the best optimum
found, how many starts reach it, and the fit the 40 default starts alone
select. K = 4 attains the best BIC; the paper keeps K = 3 for the reasons it
gives there. Writes output/model_selection_extended.json; its "seconds" field
is the wall-clock run time, so every rerun changes that field and nothing else.

Run from the repository root:  PYTHONPATH=src python3 scripts/model_selection_extended.py
Runtime: about nine minutes (macOS/arm64, CPython 3.9.6).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from regime_engine import hmm_batch as hb  # noqa: E402
from regime_engine.diagnostics import _stationary  # noqa: E402
from regime_engine.paths import DEFAULT_OUT, DEFAULT_PANEL  # noqa: E402
from regime_engine.robustness import (  # noqa: E402
    HMM_SEED,
    SELECTION_KS,
    SELECTION_N_STARTS,
    n_free_parameters,
    preferred_k,
)

warnings.filterwarnings("ignore")

#: Starts with spread volatilities per K, and the seed they are drawn from
#: (SPREAD_SEED_BASE + K).
N_SPREAD = 160
SPREAD_SEED_BASE = 20261009
#: A state "holds" an hour when its smoothed probability exceeds this.
HOLD = 0.5


def _criteria(ll: float, K: int, T: int) -> dict:
    k = n_free_parameters(K)
    return {"logL": ll, "n_params": k, "AIC": 2 * k - 2 * ll, "BIC": k * np.log(T) - 2 * ll}


def _states(p: dict, x: np.ndarray, times: pd.Series) -> list:
    """Per state: volatility, mass, duration, and when it holds the hours it holds."""
    bp = hb.BatchParams.stack([p])
    gamma = hb.e_step(x, bp)[0][0]
    pi = _stationary(np.asarray(p["A"]))
    out = []
    for k in range(len(p["mu"])):
        held = np.flatnonzero(gamma[:, k] > HOLD)
        rec = {"sigma_bps_per_hour": float(p["sigma"][k] * 1e4),
               "mu_bps_per_hour": float(p["mu"][k] * 1e4),
               "stationary": float(pi[k]),
               "expected_duration_h": float(1.0 / (1.0 - p["A"][k][k])),
               "hours_held": int(held.size),
               "smoothed_mass_h": float(gamma[:, k].sum())}
        if held.size:
            rec["first_hour_held"] = str(times.iloc[held[0]])
            rec["last_hour_held"] = str(times.iloc[held[-1]])
            mass = gamma[:, k]
            c = np.cumsum(mass) / mass.sum()
            lo, hi = int(np.searchsorted(c, 0.05)), int(np.searchsorted(c, 0.95))
            rec["central_90pct_of_mass"] = [str(times.iloc[lo]), str(times.iloc[hi])]
        out.append(rec)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT / "model_selection_extended.json"),
                    help="where to write the JSON (default: %(default)s)")
    out = pathlib.Path(ap.parse_args(argv).out)
    t0 = time.time()
    pan = pd.read_csv(DEFAULT_PANEL)
    pan = pan[pan["log_return"].notna()].reset_index(drop=True)
    r = pan["log_return"].to_numpy(float)       # raw returns, as in summary.json
    times = pan["time"]
    T = len(r)
    print(f"Panel T={T} (raw log-returns, the summary.json convention)", flush=True)
    recs = []
    parent = {"pi": np.array([1.0]), "A": np.array([[1.0]]),
              "mu": np.array([r.mean()]), "sigma": np.array([r.std()])}
    with ProcessPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 2))) as ex:
        for K in SELECTION_KS:
            fit = hb.wide_multistart(r, K, n_default=SELECTION_N_STARTS,
                                     default_seed=HMM_SEED + K, n_spread=N_SPREAD,
                                     spread_seed=SPREAD_SEED_BASE + K, parent=parent,
                                     executor=ex)
            p = fit["params"]
            dp = fit["default_protocol"]
            rec = {"K": int(K), **_criteria(fit["ll"], K, T),
                   "sigma_bps_per_hour": [float(s) * 1e4 for s in p["sigma"]],
                   "seed": int(HMM_SEED + K), "spread_seed": int(SPREAD_SEED_BASE + K),
                   "converged": bool(fit["record"]["converged"]),
                   "n_starts": fit["record"]["n_starts"],
                   "n_starts_reaching_best": fit["record"]["n_starts_reaching_best"],
                   "convergence": fit["record"],
                   "params": {k: np.asarray(v).tolist() for k, v in p.items()},
                   "states": _states(p, r, times),
                   "default_protocol": {
                       **_criteria(dp["ll"], K, T),
                       "n_starts": SELECTION_N_STARTS, "seed": int(HMM_SEED + K),
                       "selected_start": dp["selected_start"],
                       "sigma_bps_per_hour": [float(s) * 1e4 for s in dp["params"]["sigma"]],
                       "converged": dp["converged"], "polish": dp["polish"],
                       "is_best_optimum": bool(abs(dp["ll"] - fit["ll"]) < hb.REACH_TOL)}}
            recs.append(rec)
            parent = p
            c = fit["record"]
            print(f"  K={K}  logL={rec['logL']:.4f}  BIC={rec['BIC']:.2f}  "
                  f"reached by {c['n_starts_reaching_best']}/{c['n_starts']} "
                  f"{c['n_reaching_best_by_family']}  default 40 starts logL="
                  f"{dp['ll']:.4f}  sigma={[round(s, 1) for s in rec['sigma_bps_per_hour']]}"
                  f"  converged={rec['converged']}  ({time.time() - t0:.0f}s)", flush=True)
    base = next(x["BIC"] for x in recs if x["K"] == 2)
    for rec in recs:
        rec["delta_BIC"] = rec["BIC"] - base
    best = preferred_k(recs)
    prev = json.load(open(DEFAULT_OUT / "summary.json")).get("model_selection", {})
    by_k = {x["K"]: x for x in recs}
    payload = {
        "_note": ("Information criteria for K in {2,3,4,5} on the reproducible "
                  "sub-sample panel, raw log-returns, each at the best optimum a wide "
                  "multi-start finds (protocol). records[*].n_starts_reaching_best "
                  "counts the starts that end within reach_tol of it; "
                  "records[*].default_protocol is the fit the 40 starts of "
                  "GaussianHMM.fit alone select. For K = 5 the 40 default starts "
                  "stop in a lower local optimum; the best optimum's extra state is "
                  "described in records[*].states. K=4 attains the best BIC; the "
                  "paper retains K=3 on the grounds given in Section 6.2, not on BIC "
                  "alone."),
        "panel": str(DEFAULT_PANEL.relative_to(DEFAULT_PANEL.parents[2])),
        "T": int(T),
        "protocol": {
            "ks": list(SELECTION_KS),
            "n_default_starts": SELECTION_N_STARTS,
            "default_seed": f"{HMM_SEED} + K",
            "n_spread_starts": N_SPREAD,
            "spread_seed": f"{SPREAD_SEED_BASE} + K",
            "spread_vol_range_x_sd": list(hb.SPREAD_VOL_RANGE),
            "split_factors": list(hb.SPLIT_FACTORS),
            "split_parent": "the best fit with K-1 states (a single Gaussian for K=2)",
            "iteration_cap": hb.FIT_CAP,
            "reach_tol": hb.REACH_TOL,
            "rule": ("every start to logL gain < 1e-6; the first start with the "
                     "largest final logL continued to |dlogL| < 1e-9 and parameter "
                     "step < 1e-10 (hmm_batch.fit_protocol's rule)"),
            "state_holds_hour_when_smoothed_probability_above": HOLD,
        },
        "all_converged": all(x["converged"] for x in recs),
        "records": recs,
        "preferred_by_bic": best,
        "logL_gain_K5_over_K4": by_k[5]["logL"] - by_k[4]["logL"],
        "BIC_K5_minus_K3": by_k[5]["BIC"] - by_k[3]["BIC"],
        "BIC_K5_minus_K4": by_k[5]["BIC"] - by_k[4]["BIC"],
        "previous_k2_k3_from_summary_json": prev,
        "seconds": time.time() - t0,
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nBIC optimum: K={best}", flush=True)
    print("\nCheck against summary.json (8 starts, seed 42+K):", flush=True)
    for k in ("2", "3"):
        if k in prev:
            got = by_k[int(k)]
            print(f"  K={k}: summary.json BIC={prev[k]['BIC']:.2f}  "
                  f"new BIC={got['BIC']:.2f}  "
                  f"difference {got['BIC'] - prev[k]['BIC']:+.2f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
