"""Evidence on the high-to-low transition A_31 of the K = 3 return HMM.

Writes output/a31_evidence.json with these blocks:

  profile_likelihood  profile log-likelihood of A_31 on the hourly sub-period
                      (T = 3,507) by constrained EM, the one-sided 95% upper
                      bound and the two-sided 95% bound, a multi-start check of
                      the profile at the bound, and a parametric-bootstrap check
                      of the size of the one-sided likelihood-ratio test there
                      (part "profile"); its sub-block "calibrated" inverts the
                      bootstrap test over a grid of a (part "profile_calibrated");
  power               how often the maximum-likelihood estimate of A_31 is
                      exactly zero when the true value is 0, 0.003, 0.01 or
                      0.02, under the full-sample (T = 5,001) and the
                      sub-period (T = 3,507) parameters, decided at the
                      constrained maximum by the Kuhn--Tucker condition;
  power_em            on the first paths of each power cell: the published EM
                      rule, the same EM run on, and the estimate at the maximum;
  garch_generic       how often a K = 3 Gaussian HMM fitted to paths simulated
                      from single-regime GARCH(1,1) processes puts A_31 at zero,
                      at the best of the paper's 8 starts and at the best
                      optimum of a wider multi-start;
  convergence         every sub-period HMM fit behind a printed number, refitted
                      with the same seeds and starts by the shipped protocol
                      (GaussianHMM.fit) and read off at the earlier cap as well,
                      checked against the shipped artifacts, with the printed
                      numbers the converged fits change, and the model-selection
                      numbers at the wide multi-start's best optimum.

Each block can be rebuilt on its own (``--part``); the others are kept.
"profile" keeps its "calibrated" sub-block only while the bound and its
calibration at the bound are unchanged; otherwise it drops the sub-block and
says so, and "profile_calibrated" (about 40 min) has to be rerun.
"profile_calibrated" reads the "profile" block, so it runs after it.

Run times with 8 jobs (macOS/arm64, CPython 3.9.6; each block records its
own under "seconds"): profile about 9 min, profile_calibrated about 40 min,
power about 11 min, power_em about 23 min, garch about 32 min (the wide
multi-start on the Gaussian-GARCH paths), convergence about 6 min; "all"
takes about two hours.

Run from the repository root:
    PYTHONPATH=src python3 scripts/a31_evidence.py [--part all|profile|...]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from regime_engine import a31 as A31  # noqa: E402
from regime_engine import diagnostics as dg  # noqa: E402
from regime_engine import hmm_batch as hb  # noqa: E402
from regime_engine.hmm import GaussianHMM  # noqa: E402
from regime_engine.paths import DEFAULT_OUT, DEFAULT_PANEL, PAPER_WINDOW, REPO_ROOT  # noqa: E402

OUT_JSON = DEFAULT_OUT / "a31_evidence.json"
ANN_1H = np.sqrt(24 * 365)
ANN_2H = np.sqrt(12 * 365)
ZERO = 1e-8                      # "exactly zero": the threshold of bootstrap_hmm.json

# ---- power simulation -----------------------------------------------------
POWER_ALTERNATIVES = (0.0, 0.003, 0.01, 0.02)
POWER_REPS = 1000
POWER_CHUNK = 125
POWER_EM_REPS = 200
POWER_EM_CHUNK = 50
DECISION_TOL = 1e-7               # |dlogL| of the fit with A_31 held at zero
DECISION_CAP = 5000
POWER_SEED_FULL = 20260820       # boundary_power's base: seed + int(a * 1e6)
POWER_SEED_SUB = 20261820
PUBLISHED_N_ITER = dg.BOOT_N_ITER  # 400
PUBLISHED_TOL = dg.BOOT_TOL        # 1e-6, rule of GaussianHMM.fit
LONG_CAP = 2000                  # cap of the EM continued to |dlogL| < 1e-10

# ---- profile ----------------------------------------------------------------
CAL_REPS = 400
CAL_CHUNK = 25
CAL_SEED = 20261008

# ---- GARCH ------------------------------------------------------------------
#: paths per specification. The Gaussian GARCH, whose zero share is near one
#: half, gets four times as many so that the share has a standard error of about
#: three points; the seeds of its first 60 paths are those of the other two.
GARCH_PATHS = {"garch_t_fitted": 60, "garch_t_conventional": 60, "garch_normal_fitted": 240}
GARCH_CHUNK = 15                 # paths per job; each path is 8 starts (+ GARCH_SPREAD)
GARCH_SEED = 20261009
#: Starts with spread volatilities (hmm_batch.spread_starts) added to the 8
#: seeded starts of a path, and the number of paths per specification that get
#: them. The 8-start optimum is a local one on a third of the Gaussian-GARCH
#: paths, so that specification is refitted on every path; on the Student-t
#: specifications the wider search is a check on the first 30 paths.
GARCH_SPREAD = 24
GARCH_WIDE_PATHS = {"garch_t_fitted": 30, "garch_t_conventional": 30,
                    "garch_normal_fitted": 240}
GARCH_SPREAD_SEED_OFFSET = 7919  # spread starts of a path: seed (path seed + offset)
#: A wide fit counts as a higher optimum when it beats the 8-start one by more than this.
GARCH_HIGHER_TOL = 1e-3
FIT_CAP = 20000                  # iteration cap of a multi-start fit run to tol


# =============================================================================
#  helpers
# =============================================================================
def _jsonable(obj):
    return dg._jsonable(obj)


def _load_sub_returns() -> np.ndarray:
    return pd.read_csv(DEFAULT_PANEL)["log_return"].to_numpy(float)


def _summary_params(block: dict) -> dict:
    return {"pi": np.asarray(block.get("initial_pi", dg._stationary(np.asarray(block["A"]))),
                             float),
            "A": np.asarray(block["A"], float),
            "mu": np.asarray(block["mu_bps_per_hour"], float) / 1e4,
            "sigma": np.asarray(block["sigma_bps_per_hour"], float) / 1e4}


def _full_sample_params_mu_order() -> dict:
    """The reconstructed full-sample kernel, in the archive's mu-ascending order."""
    doc = json.loads((DEFAULT_OUT / dg.FULLSAMPLE_JSON).read_text())
    full = doc["return_K3"]
    A = dg.reconstruct_transition_matrix(full["diag_A"], full["stationary"], (0, 1))
    return {"A": A, "mu": np.asarray(full["mu_bps_per_hour"]) / 1e4,
            "sigma": np.asarray(full["sigma_bps_per_hour"]) / 1e4, "T": int(doc["n_obs"])}


def _high_low(sigma) -> tuple:
    s = np.asarray(sigma)
    return int(np.argmax(s)), int(np.argmin(s))


def _alternative_kernel(A0, cell, a: float) -> np.ndarray:
    """``diagnostics.boundary_power``'s construction: mass moves from the cell's row-mate."""
    A = np.array(A0, dtype=float, copy=True)
    i, j = cell
    other = [c for c in range(A.shape[0]) if c not in (i, j)][0]
    A[i, j] = a
    A[i, other] = max(A0[i, other] - a, 0.0)
    A[i, i] = 1.0 - A[i, j] - A[i, other]
    return A


def _seeds(base: int, n: int) -> list:
    ss = np.random.SeedSequence(base)
    return [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(n)]


def _cell_after_sort(p: hb.BatchParams) -> np.ndarray:
    """A_31 of every member once its states are sorted by sigma."""
    return hb.sort_by_sigma(p).A[:, -1, 0]


def _pool(n_jobs):
    return ProcessPoolExecutor(max_workers=n_jobs)


def _quantiles(v, qs=(0.025, 0.25, 0.5, 0.75, 0.975)) -> dict:
    v = np.asarray(v, dtype=float)
    return {f"q{int(round(q * 1000)):03d}": float(np.quantile(v, q)) for q in qs}


# =============================================================================
#  (a) profile likelihood on the sub-period
# =============================================================================
def _calibration_chunk(args):
    """Size of the one-sided LR test of A_31 = a_b at data simulated with A_31 = a_b."""
    seeds, truth, T, a_b = args
    hi, lo = 2, 0
    xs, states = [], []
    for s in seeds:
        x, st = dg.simulate_gaussian_hmm(truth["A"], truth["mu"], truth["sigma"], T,
                                         np.random.default_rng(s))
        xs.append(x)
        states.append(st)
    X = np.array(xs)
    n = len(seeds)
    p_true = hb.BatchParams.stack([{**truth, "pi": dg._stationary(truth["A"])}] * n)
    con = A31.fit_converged(X, p_true, fixed_cell=(hi, lo), fixed_value=np.full(n, a_b),
                            tol=A31.CONV_TOL, n_iter=A31.CONV_CAP)
    unc = hb.run_em(X, p_true, LONG_CAP, A31.CONV_TOL, rule="abs")
    unc_sorted = hb.sort_by_sigma(unc.params)
    bd = A31.boundary_decision(X, unc_sorted, (hi, lo))
    ll_unc = hb.loglik(X, unc.params)
    ll_max = np.maximum(ll_unc, bd["ll0"])
    a_hat = np.where(bd["at_boundary"], 0.0, unc_sorted.A[:, hi, lo])
    lr = 2.0 * (ll_max - con.ll)
    n_true = [int(np.sum((st[:-1] == hi) & (st[1:] == lo))) for st in states]
    return {"seeds": list(seeds), "lr": lr.tolist(), "a_hat": a_hat.tolist(),
            "at_boundary": bd["at_boundary"].tolist(), "n_true_31": n_true,
            "con_stopped": con.stopped.tolist(), "unc_stopped": unc.stopped.tolist()}


def _multistart_profile_chunk(args):
    """Constrained EM at A_31 = a from the paper's random starts.

    Each start is relabelled sigma-ascending first, so the cell is held between
    the start's own highest- and lowest-sigma states.
    """
    x, a, K, n_starts, random_state, starts = args
    init = hb.sort_by_sigma(hb._start_params(x, K, n_starts, random_state).take(starts))
    res = A31.fit_converged(x, init, fixed_cell=A31.HIGH_TO_LOW,
                            fixed_value=np.full(init.R, a), tol=1e-9, n_iter=FIT_CAP)
    order_kept = np.all(np.diff(res.params.sigma, axis=1) > 0, axis=1)
    return [{"start": int(s), "ll": float(res.ll[r]), "n_iter": int(res.n_iter[r]),
             "stopped": bool(res.stopped[r]), "cell_still_high_to_low": bool(order_kept[r])}
            for r, s in enumerate(starts)]


def part_profile(n_jobs: int, cal_reps: int) -> dict:
    t0 = time.time()
    x = _load_sub_returns()
    T = x.size
    summ = json.loads((DEFAULT_OUT / "summary.json").read_text())["return_K3"]
    start = _summary_params(summ)
    cell = A31.HIGH_TO_LOW

    unc = A31.fit_converged(x, start)
    bd = A31.boundary_decision(x, unc.params, cell)
    ll_unc = float(hb.loglik(x, unc.params)[0])
    ll_max = max(ll_unc, float(bd["ll0"][0]))
    mle = (bd["params0"] if bd["at_boundary"][0] else unc.params).member(0)
    a_hat = 0.0 if bd["at_boundary"][0] else float(unc.params.A[0, 2, 0])
    pb = A31.profile_upper_bound(x, mle, ll_max, [A31.CHIBAR_95, A31.CHI2_1_95], cell,
                                 a_hat=a_hat)
    b1, b2 = pb["bound"]
    print(f"[profile] a_hat={a_hat}  ratio={bd['ratio'][0]:.4f}  "
          f"bound one-sided={b1:.5f} two-sided={b2:.5f}  ({time.time() - t0:.0f}s)",
          flush=True)

    stationary = dg._stationary(np.asarray(mle["A"]))
    n_high = float(stationary[2] * T)
    ll_at_bound = float(A31.profile_likelihood(x, mle, [b1], cell)["ll"][0])

    # ---- the profile at the bound from the paper's own random starts --------
    jobs = [(x, b1, 3, 8, 45, list(range(8)))]
    jobs += [(x, b1, 3, 40, 48, list(range(k, k + 8))) for k in range(0, 40, 8)]
    with _pool(n_jobs) as ex:
        ms = list(ex.map(_multistart_profile_chunk, jobs))
    ms8 = ms[0]
    ms40 = [r for chunk in ms[1:] for r in chunk]
    valid8 = [r for r in ms8 if r["cell_still_high_to_low"]]
    valid40 = [r for r in ms40 if r["cell_still_high_to_low"]]
    best_ms = max(r["ll"] for r in valid8 + valid40)
    print(f"[profile] multistart at bound: best {best_ms:.4f} vs profile "
          f"{ll_at_bound:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- size of the one-sided test at the bound -----------------------------
    p_b = A31.profile_likelihood(x, mle, [b1], cell)["params"].member(0)
    truth = {"A": p_b["A"], "mu": p_b["mu"], "sigma": p_b["sigma"]}
    seeds = _seeds(CAL_SEED, cal_reps)
    chunks = [(seeds[k:k + CAL_CHUNK], truth, T, b1) for k in range(0, cal_reps, CAL_CHUNK)]
    with _pool(n_jobs) as ex:
        cal = list(ex.map(_calibration_chunk, chunks))
    lr = np.concatenate([c["lr"] for c in cal])
    ah = np.concatenate([c["a_hat"] for c in cal])
    signed = _signed_statistic(lr, ah, b1)
    reject = signed >= A31.CHIBAR_95
    print(f"[profile] calibration: size {reject.mean():.3f} ({time.time() - t0:.0f}s)",
          flush=True)

    return {
        "sample": "hourly sub-period 3 Dec 2025 - 28 Apr 2026",
        "T": int(T),
        "cell": "A_31, high- to low-volatility state (sigma-ascending labels)",
        "method": (
            "Profile log-likelihood l_p(a) = max over all other parameters of logL "
            "subject to A_31 = a, computed by constrained EM: the E-step is unchanged "
            "and the M-step sets A_31 = a and A_3k proportional to the expected "
            "transition counts out of state 3 for k = 2, 3 (the exact maximiser of "
            "the expected complete-data log-likelihood under the constraint). Each "
            "constrained run starts at the unconstrained maximum and stops at "
            "|dlogL| < 1e-10. LR(a) = 2 (l_max - l_p(a)). The maximum lies on the "
            "boundary, a_hat = 0, so only an upper bound is informative. The 95% "
            "upper bound is the a at which LR(a) = 2.706: the signed-root LR "
            "statistic is asymptotically standard normal for an interior true value, "
            "so a one-sided 95% bound uses z_0.95^2 = 2.706, and at a true value on "
            "the boundary the LR statistic is a 50:50 mixture of chi2_0 and chi2_1 "
            "(Chernoff 1954; Self and Liang 1987), whose 95% quantile is also 2.706. "
            "The chi2_1 crossing at 3.841 is the upper end of the two-sided 95% "
            "interval, equivalently a one-sided 97.5% bound, and is reported as the "
            "conservative figure."),
        "start": "output/summary.json return_K3 (the converged sub-period fit)",
        "unconstrained": {
            "logL": ll_unc, "n_iter": int(unc.n_iter[0]),
            "A31": float(unc.params.A[0, 2, 0]),
            "annualised_sigma_pct": (unc.params.sigma[0] * ANN_1H * 100).tolist(),
        },
        "boundary": {
            "logL_A31_zero": float(bd["ll0"][0]),
            "score_ratio_at_zero": float(bd["ratio"][0]),
            "mle_at_boundary": bool(bd["at_boundary"][0]),
            "note": ("score ratio G_31 / n_3 at the maximum with A_31 held at zero; "
                     "below one the unconstrained maximum is on the boundary"),
        },
        "a_hat": a_hat,
        "logL_max": ll_max,
        "upper_bound_one_sided_95": b1,
        "upper_bound_two_sided_95": b2,
        "upper_bound_one_sided_95_pct_per_hour": 100 * b1,
        "upper_bound_two_sided_95_pct_per_hour": 100 * b2,
        "LR_at": {f"{a:g}": v for a, v in zip(pb["grid_a"], pb["grid_LR"])
                  if a in (0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02)},
        "LR_slope_per_0.001_below_0.005": float(
            np.interp(0.005, pb["grid_a"], pb["grid_LR"]) / 5.0),
        "profile_grid": {"a": pb["grid_a"], "LR": pb["grid_LR"]},
        "rule_of_three_same_sample": {
            "high_state_hours_from_stationary_mass": n_high,
            "bound": 3.0 / n_high,
            "note": ("3/n treats the latent high-state hours as observed Bernoulli "
                     "trials; it is not a likelihood-based bound for a hidden chain"),
        },
        "multistart_at_bound": {
            "a": b1,
            "profile_logL_at_bound": ll_at_bound,
            "best_logL_paper_starts_seed45": max(r["ll"] for r in valid8),
            "best_logL_selection_starts_seed48": max(r["ll"] for r in valid40),
            "max_gain_over_profile": best_ms - ll_at_bound,
            "n_starts": len(ms8) + len(ms40),
            "n_starts_order_kept": len(valid8) + len(valid40),
            "n_starts_at_profile_value": int(sum(abs(r["ll"] - ll_at_bound) < 1e-4
                                                 for r in valid8 + valid40)),
            "best_logL_order_reversed": max([r["ll"] for r in ms8 + ms40
                                             if not r["cell_still_high_to_low"]],
                                            default=None),
            "n_stopped_by_tol": int(sum(r["stopped"] for r in ms8 + ms40)),
            "note": ("constrained EM at A_31 = bound from the 8 starts of the paper's "
                     "K=3 fit (seed 45) and the 40 starts of its model-selection fit "
                     "(seed 48), the cell held between each start's highest- and "
                     "lowest-sigma states, tolerance 1e-9. Starts whose converged "
                     "states are no longer in sigma-ascending order have held a "
                     "different cell (in practice low-to-high) and are excluded; "
                     "a gain above zero among the rest would mean the profile "
                     "started at the MLE is only local"),
        },
        "calibration_at_bound": {
            "a": b1,
            "design": ("paths of T = 3,507 hours simulated from the constrained fit "
                       "at A_31 = bound (stationary start); on each, the LR statistic "
                       "for A_31 = bound with a_hat from the boundary decision and the "
                       "converged unconstrained EM; the one-sided test rejects when "
                       "a_hat < bound and LR >= 2.706"),
            "n_reps": int(lr.size),
            "seed": CAL_SEED,
            "rejection_rate": A31.binomial_share(reject),
            "q95_one_sided_statistic": float(np.quantile(signed, 0.95)),
            "share_a_hat_zero": A31.binomial_share(np.concatenate(
                [c["at_boundary"] for c in cal])),
            "mean_true_31_transitions": float(np.mean(np.concatenate(
                [c["n_true_31"] for c in cal]))),
        },
        "seconds": time.time() - t0,
    }


CAL_GRID = (0.0005, 0.001, 0.003, 0.004, 0.005)
CALIBRATED_NOTE = (
    "Both bounds use the Gaussian HMM likelihood, and the chi2 calibration of the "
    "profile block assumes that model as well, which the residual diagnostics "
    "reject. Within the model the chi2 bound is conservative: its simulated size at "
    "the bound is profile_likelihood.calibration_at_bound.rejection_rate, against a "
    "nominal 5%. The bootstrap inversion here rejects every A_31 > 0 on the grid, "
    "because LR(a) rises almost linearly from zero, so the p-value of a small a is "
    "close to the probability that a path simulated from the fitted HMM has a "
    "score ratio at zero as low as the observed one, and the observed ratio "
    "(profile_likelihood.boundary.score_ratio_at_zero) lies in the lower tail of "
    "the ratios simulated under the fitted HMM (power.sub_period.*.score_ratio). "
    "That is a misspecification signal, not a bound: upper_bound_bootstrap_95 is "
    "then null. Among three-state fits to GARCH(1,1) paths the observed ratio is "
    "typical of the conventional Student-t GARCH and within the range of the fitted "
    "one (garch_generic.*.share_paths_ratio_at_most_observed).")


def _signed_statistic(lr, a_hat, a):
    """The one-sided statistic for H0: A_31 = a against A_31 < a."""
    return np.where(np.asarray(a_hat) < a, np.maximum(np.asarray(lr), 0.0), 0.0)


def part_profile_calibrated(n_jobs: int, reps: int, profile: dict) -> dict:
    """A parametric-bootstrap upper bound: the largest a not rejected at 5%.

    For each a on a grid, paths are simulated from the constrained maximum at
    A_31 = a, and the p-value of the observed statistic LR_obs(a) is the share of
    simulated one-sided statistics at least as large. The bound interpolates
    linearly in a between the last grid point with p > 0.05 and the first with
    p <= 0.05; the asymptotic bound of the profile block, whose simulated
    rejection rate is already stored there, closes the grid.
    """
    t0 = time.time()
    x = _load_sub_returns()
    T = x.size
    summ = json.loads((DEFAULT_OUT / "summary.json").read_text())["return_K3"]
    unc = A31.fit_converged(x, _summary_params(summ))
    bd = A31.boundary_decision(x, unc.params, A31.HIGH_TO_LOW)
    ll_max = max(float(hb.loglik(x, unc.params)[0]), float(bd["ll0"][0]))
    mle = bd["params0"].member(0)
    if not bd["at_boundary"][0]:
        raise RuntimeError("the calibrated bound assumes the estimate is on the boundary")
    prof = A31.profile_likelihood(x, mle, list(CAL_GRID))
    lr_obs = 2.0 * (ll_max - prof["ll"])
    jobs = []
    for g, a in enumerate(CAL_GRID):
        p = prof["params"].member(g)
        truth = {"A": p["A"], "mu": p["mu"], "sigma": p["sigma"]}
        seeds = _seeds(CAL_SEED + 1 + g, reps)
        jobs += [(g, (seeds[k:k + CAL_CHUNK], truth, T, a)) for k in range(0, reps, CAL_CHUNK)]
    with _pool(n_jobs) as ex:
        res = list(ex.map(_calibration_chunk, [j for _, j in jobs]))
    rows = []
    for g, a in enumerate(CAL_GRID):
        chunks = [r for (gg, _), r in zip(jobs, res) if gg == g]
        st = _signed_statistic(np.concatenate([c["lr"] for c in chunks]),
                               np.concatenate([c["a_hat"] for c in chunks]), a)
        rows.append({"a": a, "LR_observed": float(lr_obs[g]),
                     "p_value": A31.binomial_share(st >= lr_obs[g]),
                     "q95_simulated_statistic": float(np.quantile(st, 0.95)),
                     "rejection_rate_at_2.706": float(np.mean(st >= A31.CHIBAR_95)),
                     "n_reps": int(st.size), "seed": CAL_SEED + 1 + g})
    cal = profile["calibration_at_bound"]
    rows.append({"a": profile["upper_bound_one_sided_95"], "LR_observed": A31.CHIBAR_95,
                 "p_value": cal["rejection_rate"],
                 "q95_simulated_statistic": cal["q95_one_sided_statistic"],
                 "rejection_rate_at_2.706": cal["rejection_rate"]["share"],
                 "n_reps": cal["n_reps"], "seed": cal["seed"]})
    bound = None
    for lo, hi in zip(rows[:-1], rows[1:]):
        p_lo, p_hi = lo["p_value"]["share"], hi["p_value"]["share"]
        if p_lo > 0.05 >= p_hi:
            bound = lo["a"] + (p_lo - 0.05) * (hi["a"] - lo["a"]) / (p_lo - p_hi)
            break
    rejects_all = bool(all(r["p_value"]["share"] <= 0.05 for r in rows))
    print(f"[calibrated] bound {bound}, rejects every grid value {rejects_all} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {
        "method": (
            "Parametric-bootstrap test inversion. At each a, paths of T = 3,507 hours "
            "are simulated from the constrained maximum-likelihood fit with A_31 = a; "
            "on each the one-sided statistic (LR for A_31 = a when a_hat < a, else 0) "
            "is computed as in calibration_at_bound. The p-value of a is the share of "
            "simulated statistics at least as large as the observed LR(a); the 95% "
            "bound is the a at which the p-value falls to 0.05, linear in a between "
            "grid points."),
        "grid": rows,
        "upper_bound_bootstrap_95": bound,
        "rejects_every_grid_value": rejects_all,
        "smallest_grid_value": float(min(r["a"] for r in rows)),
        "note": CALIBRATED_NOTE,
        "seconds": time.time() - t0,
    }


# =============================================================================
#  (b) power: is the estimate exactly zero when the truth is not?
# =============================================================================
def _simulate_batch(seeds, A, mu, sigma, T, cell):
    xs, n_true = [], []
    for s in seeds:
        x, st = dg.simulate_gaussian_hmm(A, mu, sigma, T, np.random.default_rng(s))
        xs.append(x)
        n_true.append(int(np.sum((st[:-1] == cell[0]) & (st[1:] == cell[1]))))
    init = hb.BatchParams.stack([{"pi": dg._stationary(A), "A": A, "mu": mu,
                                  "sigma": sigma}] * len(seeds))
    return np.array(xs), n_true, init


def _power_chunk(args):
    """The boundary decision on every path, from the data-generating parameters."""
    seeds, A, mu, sigma, T, cell = args
    X, n_true, init = _simulate_batch(seeds, A, mu, sigma, T, cell)
    bd = A31.boundary_decision(X, hb.sort_by_sigma(init), A31.HIGH_TO_LOW,
                               tol=DECISION_TOL, n_iter=DECISION_CAP)
    return {"seeds": list(seeds), "n_true": n_true, "ratio": bd["ratio"].tolist(),
            "at_boundary": bd["at_boundary"].tolist(), "n_iter": bd["n_iter"].tolist(),
            "stopped": bd["stopped"].tolist()}


def _power_em_chunk(args):
    """The published EM rule, the same EM run on, and the estimate at the maximum."""
    seeds, A, mu, sigma, T, cell = args
    X, n_true, init = _simulate_batch(seeds, A, mu, sigma, T, cell)
    pub = hb.run_em(X, init, PUBLISHED_N_ITER, PUBLISHED_TOL, rule="fit")
    a_pub = _cell_after_sort(pub.params)
    long = hb.run_em(X, pub.params, LONG_CAP, A31.CONV_TOL, rule="abs")
    a_long = _cell_after_sort(long.params)
    m = A31.cell_mle(X, hb.sort_by_sigma(long.params), A31.HIGH_TO_LOW,
                     tol=A31.CONV_TOL, n_iter=LONG_CAP)
    return {"seeds": list(seeds), "n_true": n_true, "a_pub": a_pub.tolist(),
            "a_long": a_long.tolist(), "long_n_iter": long.n_iter.tolist(),
            "long_stopped": long.stopped.tolist(), "a_hat": m["a_hat"].tolist(),
            "ratio": m["ratio"].tolist(), "at_boundary": m["at_boundary"].tolist(),
            "lr0": (2.0 * (m["ll_max"] - m["ll0"])).tolist()}


def _cat(rows, key):
    return np.concatenate([np.asarray(r[key]) for r in rows])


def _power_cell(rows: list, a: float, seed_base: int) -> dict:
    ratio = _cat(rows, "ratio")
    at_b = _cat(rows, "at_boundary").astype(bool)
    n_true = _cat(rows, "n_true")
    return {
        "alternative_A31": a,
        "n_reps": int(at_b.size),
        "seed_base": seed_base + int(a * 1e6),
        "share_mle_zero": A31.binomial_share(at_b),
        "share_score_ratio_within_0.001_of_one": float(np.mean(np.abs(ratio - 1.0) < 1e-3)),
        "score_ratio": _quantiles(ratio),
        "n_decision_fits_stopped_by_tol": int(_cat(rows, "stopped").sum()),
        "max_decision_iterations": int(_cat(rows, "n_iter").max()),
        "share_paths_without_true_31_transition": A31.binomial_share(n_true == 0),
        "mean_true_31_transitions": float(n_true.mean()),
    }


def _power_em_cell(rows: list, a: float, decision_rows: list, published=None) -> dict:
    a_pub, a_long, a_hat = _cat(rows, "a_pub"), _cat(rows, "a_long"), _cat(rows, "a_hat")
    at_b = _cat(rows, "at_boundary").astype(bool)
    n = at_b.size
    at_b_decision = _cat(decision_rows, "at_boundary").astype(bool)[:n]
    out = {
        "alternative_A31": a,
        "n_reps": int(n),
        "share_zero_published_rule": A31.binomial_share(a_pub < ZERO),
        "share_zero_em_continued": A31.binomial_share(a_long < ZERO),
        "share_below_1e-6_em_continued": A31.binomial_share(a_long < 1e-6),
        "n_em_continued_stopped_by_tol": int(_cat(rows, "long_stopped").sum()),
        "share_mle_zero": A31.binomial_share(at_b),
        "agreement_with_decision_from_truth": float(np.mean(at_b == at_b_decision)),
        "agreement_mle_zero_vs_em_continued_zero": float(np.mean(at_b == (a_long < ZERO))),
        "mle_A31": {"mean": float(a_hat.mean()), **_quantiles(a_hat)},
        "share_LR_zero_vs_free_at_least_2.706": float(
            np.mean(_cat(rows, "lr0") >= A31.CHIBAR_95)),
    }
    if published is not None:
        k = min(int(published["n_reps"]), n)
        out["published_rule_reproduction"] = {
            "n": k,
            "share_zero_here": float(np.mean(a_pub[:k] < ZERO)),
            "share_zero_in_bootstrap_hmm_json": published["share"],
        }
    return out


def _power_designs(reps: int, chunk: int) -> list:
    full = _full_sample_params_mu_order()
    summ = json.loads((DEFAULT_OUT / "summary.json").read_text())
    sub = _summary_params(summ["return_K3"])
    designs = []
    for name, P, T, base in (("full_sample", full, full["T"], POWER_SEED_FULL),
                             ("sub_period", sub, int(summ["n_obs"]), POWER_SEED_SUB)):
        cell = _high_low(P["sigma"])     # (0, 1) in the archive's mu order, (2, 0) in sigma order
        for a in POWER_ALTERNATIVES:
            A = _alternative_kernel(P["A"], cell, a)
            seeds = _seeds(base + int(a * 1e6), reps)
            designs.append((name, a, base, [(seeds[k:k + chunk], A, P["mu"], P["sigma"],
                                             T, cell) for k in range(0, reps, chunk)]))
    return designs


def _run_designs(designs, worker, n_jobs):
    jobs = [(d, c) for d, design in enumerate(designs) for c in design[3]]
    order = sorted(range(len(jobs)), key=lambda k: -jobs[k][1][4] * len(jobs[k][1][0]))
    with _pool(n_jobs) as ex:
        done = list(ex.map(worker, [jobs[k][1] for k in order]))
    results = [None] * len(jobs)
    for k, r in zip(order, done):
        results[k] = r
    return [[r for (dd, _), r in zip(jobs, results) if dd == d] for d in range(len(designs))]


POWER_DESIGN_TEXT = (
    "Paths simulated from the K=3 Gaussian HMM with the high-to-low cell set to each "
    "alternative (mass moved from the high-to-moderate cell, as in "
    "diagnostics.boundary_power), starting from the stationary distribution. "
    "share_mle_zero: the likelihood is maximised with A_31 held at zero, starting "
    "from the data-generating parameters (tol 1e-7 on |dlogL|), and the score ratio "
    "G_31 / n_3 is evaluated there; the maximum-likelihood estimate of A_31 is "
    "exactly zero when the ratio is at most one. This is the estimator at "
    "convergence; it does not depend on how fast EM approaches the boundary.")


def part_power(n_jobs: int, reps: int) -> dict:
    t0 = time.time()
    designs = _power_designs(reps, POWER_CHUNK)
    grouped = _run_designs(designs, _power_chunk, n_jobs)
    out = {"full_sample": {}, "sub_period": {}}
    for (name, a, base, _), rows in zip(designs, grouped):
        c = _power_cell(rows, a, base)
        out[name][f"{a:g}"] = c
        print(f"[power] {name} a={a:g}: MLE zero {c['share_mle_zero']['share']:.3f} "
              f"+- {c['share_mle_zero']['se']:.3f}", flush=True)
    return {
        "design": POWER_DESIGN_TEXT,
        "parameters": {
            "full_sample": ("output/summary_FULLSAMPLE_2025-10-02_to_2026-04-28.json, "
                            "kernel reconstructed by diagnostics.reconstruct_transition_"
                            "matrix, T = 5,001; seeds as diagnostics.boundary_power "
                            "(base 20260820 + int(a * 1e6)), so the first 200 paths at "
                            "0.003 and 0.01 are those behind output/bootstrap_hmm.json"),
            "sub_period": ("output/summary.json return_K3, T = 3,507; seeds base "
                           "20261820 + int(a * 1e6)"),
        },
        "reps_per_cell": reps,
        **out,
        "seconds": time.time() - t0,
    }


def part_power_em(n_jobs: int, reps: int) -> dict:
    t0 = time.time()
    boot = json.loads((DEFAULT_OUT / "bootstrap_hmm.json").read_text())
    published = {round(r["alternative_A_ij"], 6): {
        "n_reps": r["n_reps"], "share": r["frac_reps_MLE_exactly_zero_sigma_labelling"]}
        for r in boot["boundary_power_full_sample"]}
    designs = _power_designs(reps, POWER_EM_CHUNK)
    grouped = _run_designs(designs, _power_em_chunk, n_jobs)
    dec_grouped = _run_designs(designs, _power_chunk, n_jobs)
    out = {"full_sample": {}, "sub_period": {}}
    for (name, a, _, _), rows, dec_rows in zip(designs, grouped, dec_grouped):
        pub = published.get(round(a, 6)) if name == "full_sample" else None
        c = _power_em_cell(rows, a, dec_rows, pub)
        out[name][f"{a:g}"] = c
        print(f"[power-em] {name} a={a:g}: published {c['share_zero_published_rule']['share']:.3f}"
              f"  continued {c['share_zero_em_continued']['share']:.3f}"
              f"  MLE zero {c['share_mle_zero']['share']:.3f}", flush=True)
    return {
        "design": (
            "The first paths of each cell of the power block. (1) The published rule: "
            "Baum-Welch from the data-generating parameters with the stopping rule of "
            "GaussianHMM.fit, tol 1e-6, cap 400, A_31 < 1e-8 counted as zero -- the "
            "procedure behind output/bootstrap_hmm.json boundary_power_full_sample. "
            "(2) The same EM continued to |dlogL| < 1e-10, cap 2,000. (3) The "
            "maximum-likelihood estimate: the boundary decision at the converged fit "
            "and, where the estimate is interior, EM restarted from the constrained "
            "maximum with A_31 = 1e-3 (EM cannot leave an exact zero, so at a true "
            "A_31 = 0 the rules (1) and (2) return zero on every path by "
            "construction). EM approaches a boundary maximum geometrically at the rate "
            "of the score ratio, so (1) and (2) undercount zeros whenever the ratio "
            "is close to one."),
        "reps_per_cell": reps,
        "zero_threshold_em": ZERO,
        **out,
        "seconds": time.time() - t0,
    }


# =============================================================================
#  (c) GARCH paths: is the boundary zero generic?
# =============================================================================
def _garch_specs(x: np.ndarray) -> dict:
    bench = json.loads((DEFAULT_OUT / "benchmarks.json").read_text())
    rows = {r["model"]: r["params"] for r in bench["rows"]}
    gt, gn = rows["GARCH(1,1)-Student-t"], rows["GARCH(1,1)-Normal"]
    v = float(np.var(x))
    return {
        "garch_t_fitted": {
            "source": "output/benchmarks.json GARCH(1,1)-Student-t (BIC winner)",
            "mu": gt["mu"], "omega": gt["omega"], "alpha": gt["alpha"],
            "beta": gt["beta"], "nu": gt["nu"]},
        "garch_t_conventional": {
            "source": ("alpha 0.10, beta 0.88, nu 4, omega matching the sample "
                       "variance of the sub-period returns"),
            "mu": 0.0, "omega": v * (1 - 0.10 - 0.88), "alpha": 0.10, "beta": 0.88,
            "nu": 4.0},
        "garch_normal_fitted": {
            "source": "output/benchmarks.json GARCH(1,1)-Normal",
            "mu": gn["mu"], "omega": gn["omega"], "alpha": gn["alpha"],
            "beta": gn["beta"], "nu": None},
    }


def _garch_chunk(args):
    """K=3 fits to GARCH paths, all starts of all paths in one batch.

    Every path gets the paper's protocol (8 seeded starts); with ``n_spread`` > 0
    it also gets that many starts with spread volatilities, and the boundary
    decision is taken a second time at the best optimum over all starts.
    """
    spec, seeds, T, h0, n_spread = args
    n_st = 8
    n_all = n_st + n_spread
    xs = [A31.simulate_garch11(spec["mu"], spec["omega"], spec["alpha"], spec["beta"],
                               T, np.random.default_rng(s), nu=spec["nu"], h0=h0)
          for s in seeds]
    inits = []
    for s, x in zip(seeds, xs):
        inits.append(hb._start_params(x, 3, n_st, int(s) % (2 ** 31)))
        if n_spread:
            inits.append(hb.spread_starts(x, 3, n_spread,
                                          (int(s) + GARCH_SPREAD_SEED_OFFSET) % (2 ** 31)))
    init = hb.BatchParams(*(np.concatenate([getattr(q, k) for q in inits])
                            for k in ("pi", "A", "mu", "sigma")))
    X = np.repeat(np.array(xs), n_all, axis=0)
    res = hb.run_em(X, init, FIT_CAP, 1e-6, rule="fit", snapshot_iters=(150,))
    snap_p, snap_ll = res.snapshots[150]
    best, best150, wide = [], [], []
    for i in range(len(seeds)):
        sl = slice(i * n_all, i * n_all + n_st)
        best.append(i * n_all + int(np.argmax(res.ll[sl])))
        best150.append(i * n_all + int(np.argmax(snap_ll[sl])))
        wide.append(i * n_all + int(np.argmax(res.ll[i * n_all:(i + 1) * n_all])))
    pb = hb.sort_by_sigma(res.params.take(best))
    p150 = hb.sort_by_sigma(snap_p.take(best150))
    bd = A31.boundary_decision(np.array(xs), pb, A31.HIGH_TO_LOW, tol=1e-8,
                               n_iter=DECISION_CAP)
    if n_spread:
        pw = hb.sort_by_sigma(res.params.take(wide))
        bw = A31.boundary_decision(np.array(xs), pw, A31.HIGH_TO_LOW, tol=1e-8,
                                   n_iter=DECISION_CAP)
    out = []
    for i, (s, x) in enumerate(zip(seeds, xs)):
        sl = slice(i * n_all, i * n_all + n_st)
        row = {
            "seed": int(s), "sd_bps": float(x.std() * 1e4),
            "A31_150": float(p150.A[i, 2, 0]), "A31_conv": float(pb.A[i, 2, 0]),
            "A13_conv": float(pb.A[i, 0, 2]), "A32_conv": float(pb.A[i, 2, 1]),
            "ratio": float(bd["ratio"][i]), "at_boundary": bool(bd["at_boundary"][i]),
            "ann_sigma_pct": (pb.sigma[i] * ANN_1H * 100).tolist(),
            "stationary": dg._stationary(pb.A[i]).tolist(),
            "best_n_iter": int(res.n_iter[best[i]]),
            "n_starts_stopped": int(res.stopped[sl].sum()),
            "logL": float(res.ll[best[i]]),
        }
        if n_spread:
            gain = float(res.ll[wide[i]] - res.ll[best[i]])
            row["wide"] = {
                "n_starts": n_all, "logL": float(res.ll[wide[i]]), "logL_gain": gain,
                "higher_optimum": bool(gain > GARCH_HIGHER_TOL),
                "best_start_family": "seeded" if wide[i] - i * n_all < n_st else "spread",
                "n_starts_stopped": int(res.stopped[i * n_all:(i + 1) * n_all].sum()),
                "A31_conv": float(pw.A[i, 2, 0]),
                "ratio": float(bw["ratio"][i]), "at_boundary": bool(bw["at_boundary"][i]),
                "ann_sigma_pct": (pw.sigma[i] * ANN_1H * 100).tolist(),
                "stationary": dg._stationary(pw.A[i]).tolist(),
            }
        out.append(row)
    return out


def _wide_summary(rows: list) -> dict | None:
    """The 8-start protocol against the wider multi-start, on the paths that got both."""
    w = [r for r in rows if "wide" in r]
    if not w:
        return None
    gains = np.array([r["wide"]["logL_gain"] for r in w])
    higher = gains > GARCH_HIGHER_TOL
    ratios = np.array([r["wide"]["ratio"] for r in w])
    return {
        "n_paths": len(w),
        "n_starts_per_path": w[0]["wide"]["n_starts"],
        "share_mle_zero_8_starts": A31.binomial_share([r["at_boundary"] for r in w]),
        "share_mle_zero_best_optimum": A31.binomial_share(
            [r["wide"]["at_boundary"] for r in w]),
        "n_paths_higher_optimum": int(higher.sum()),
        "logL_gain_on_those_paths": _quantiles(gains[higher]) if higher.any() else None,
        "n_decisions_changed": int(sum(r["at_boundary"] != r["wide"]["at_boundary"]
                                       for r in w)),
        "n_zero_to_interior": int(sum(r["at_boundary"] and not r["wide"]["at_boundary"]
                                      for r in w)),
        "n_interior_to_zero": int(sum(r["wide"]["at_boundary"] and not r["at_boundary"]
                                      for r in w)),
        "score_ratio_best_optimum": _quantiles(ratios),
        "median_ann_sigma_pct_best_optimum": np.median(
            [r["wide"]["ann_sigma_pct"] for r in w], axis=0).tolist(),
        "median_high_state_mass_best_optimum": float(np.median(
            [r["wide"]["stationary"][2] for r in w])),
    }


def _observed_score_ratio(x: np.ndarray) -> float:
    """Score ratio of A_31 at zero on the data, as in the profile block."""
    summ = json.loads((DEFAULT_OUT / "summary.json").read_text())["return_K3"]
    unc = A31.fit_converged(x, _summary_params(summ))
    return float(A31.boundary_decision(x, unc.params, A31.HIGH_TO_LOW)["ratio"][0])


def part_garch(n_jobs: int, n_paths=None) -> dict:
    """``n_paths`` (an int) overrides GARCH_PATHS for every specification."""
    t0 = time.time()
    x = _load_sub_returns()
    T = x.size
    h0 = float(np.var(x))
    specs = _garch_specs(x)
    counts = {name: int(n_paths) if n_paths else GARCH_PATHS[name] for name in specs}
    seeds = _seeds(GARCH_SEED, max(counts.values()))
    wide = {name: min(counts[name], GARCH_WIDE_PATHS[name]) for name in specs}
    jobs = [(name, (specs[name], seeds[k:min(k + GARCH_CHUNK, counts[name])], T, h0,
                    GARCH_SPREAD if k < wide[name] else 0))
            for name in specs for k in range(0, counts[name], GARCH_CHUNK)]
    jobs.sort(key=lambda j: -len(j[1][1]) * (8 + j[1][4]))
    observed_ratio = _observed_score_ratio(x)
    with _pool(n_jobs) as ex:
        res = list(ex.map(_garch_chunk, [j for _, j in jobs]))
    out = {}
    for name in specs:
        rows = [r for (nm, _), chunk in zip(jobs, res) if nm == name for r in chunk]
        a150 = np.array([r["A31_150"] for r in rows])
        ac = np.array([r["A31_conv"] for r in rows])
        atb = np.array([r["at_boundary"] for r in rows])
        a13 = np.array([r["A13_conv"] for r in rows])
        ratios = np.array([r["ratio"] for r in rows])
        sp = specs[name]
        out[name] = {
            "spec": sp,
            "persistence": sp["alpha"] + sp["beta"],
            "largest_hourly_variance_fall_factor_beta": sp["beta"],
            "n_paths": len(rows),
            "share_A31_zero_paper_protocol_150_iter": A31.binomial_share(a150 < ZERO),
            "share_A31_zero_converged": A31.binomial_share(ac < ZERO),
            "share_mle_zero": A31.binomial_share(atb),
            "share_A31_below_1e-4_converged": A31.binomial_share(ac < 1e-4),
            "share_A13_above_1e-3": A31.binomial_share(a13 > 1e-3),
            "A31_converged": _quantiles(ac),
            "median_ann_sigma_pct": np.median([r["ann_sigma_pct"] for r in rows],
                                              axis=0).tolist(),
            "median_high_state_mass": float(np.median([r["stationary"][2] for r in rows])),
            "median_return_sd_bps": float(np.median([r["sd_bps"] for r in rows])),
            "score_ratio": _quantiles(ratios),
            "share_paths_ratio_at_most_observed": A31.binomial_share(ratios <= observed_ratio),
            "wide_multistart": _wide_summary(rows),
            "paths": rows,
        }
        ws = out[name]["wide_multistart"]
        if ws is not None:
            ws["share_paths_ratio_at_most_observed"] = A31.binomial_share(
                [r["wide"]["ratio"] <= observed_ratio for r in rows if "wide" in r])
        print(f"[garch] {name}: MLE zero {out[name]['share_mle_zero']['share']:.3f}, "
              f"150-iter zero {out[name]['share_A31_zero_paper_protocol_150_iter']['share']:.3f}"
              + ("" if ws is None else
                 f", wide: {ws['share_mle_zero_best_optimum']['k']}/{ws['n_paths']} zero, "
                 f"{ws['n_paths_higher_optimum']} higher optima"),
              flush=True)
    return {
        "design": (
            "Paths of T = 3,507 hourly returns simulated from single-regime GARCH(1,1) "
            "processes, the variance recursion started at the sample variance of the "
            "sub-period returns (the backcast of diagnostics.fit_garch11). A K=3 "
            "Gaussian HMM is fitted to each path with the paper's protocol (8 random "
            "starts, seeded per path), once capped at 150 iterations as the paper's "
            "fits were and once run to tol 1e-6 (cap 20,000); for the converged fit the "
            "boundary decision of the power block says whether the maximum-"
            "likelihood A_31 is exactly zero (share_mle_zero: at the best of the 8 "
            "starts). The 8-start optimum can be a local one, so the paths named in "
            "wide_paths_per_specification are also fitted from 24 further starts with "
            "volatilities spread log-uniformly over 0.15-3.5 x the path's s.d. "
            "(hmm_batch.spread_starts) and the decision is repeated at the best optimum "
            "over all 32 starts (wide_multistart; per path under paths[*].wide). For the "
            "Gaussian GARCH that is every path, and its zero share is "
            "wide_multistart.share_mle_zero_best_optimum; for the Student-t "
            "specifications it is a check on the first 30 paths. Every specification "
            "draws its path seeds "
            "from the same sequence, so the first 60 paths share seeds across "
            "specifications. A GARCH(1,1) variance can fall by at most the factor beta "
            "per hour; the zero share follows beta (largest_hourly_variance_fall_factor_"
            "beta) more closely than persistence. In these fits the high state is rarer "
            "and more extreme than in the data's (median_ann_sigma_pct, "
            "median_high_state_mass)."),
        "T": int(T),
        "seed_base": GARCH_SEED,
        "paths_per_specification": counts,
        "wide_paths_per_specification": wide,
        "wide_spread_starts": GARCH_SPREAD,
        "wide_spread_seed": f"path seed + {GARCH_SPREAD_SEED_OFFSET}",
        "higher_optimum_tol": GARCH_HIGHER_TOL,
        **out,
        "observed_score_ratio_at_zero": observed_ratio,
        "data_for_reference": ("the sub-period fit has A_31 on the boundary (profile block); "
                               "observed_score_ratio_at_zero is its score ratio"),
        "seconds": time.time() - t0,
    }


# =============================================================================
#  (d) convergence report
# =============================================================================
def _winsorise(r: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(r, [0.0005, 0.9995])
    return np.clip(r, lo, hi)


def _two_hourly_returns() -> np.ndarray:
    px = pd.read_csv(REPO_ROOT / "data" / "interim" / "prices_BTC_2h_full.csv")
    px["time"] = pd.to_datetime(px["time"], utc=True, format="ISO8601")
    start = pd.Timestamp(PAPER_WINDOW[0], tz="UTC")
    end = pd.Timestamp(PAPER_WINDOW[1])
    w = px[(px["time"] >= start) & (px["time"] < end)]
    return w["log_return"].dropna().to_numpy(float)


#: name -> (series, K, n_starts, seed, cap of the earlier artifacts, shipped artifact)
FITS = {
    "analysis_K2": ("hourly", 2, 8, 44, 150, "summary.json return_K2"),
    "analysis_K3": ("hourly", 3, 8, 45, 150, "summary.json return_K3"),
    "winsorised_K3": ("winsorised", 3, 8, 45, 150, "robustness.json winsorised"),
    "selection_K2": ("hourly", 2, 40, 47, 300, "model_selection_extended.json K=2"),
    "selection_K3": ("hourly", 3, 40, 48, 300, "model_selection_extended.json K=3"),
    "selection_K4": ("hourly", 4, 40, 49, 300, "model_selection_extended.json K=4"),
    "selection_K5": ("hourly", 5, 40, 50, 300, "model_selection_extended.json K=5"),
    "two_hourly_K3": ("two_hourly", 3, 8, 45, 150, "two_hourly_refit.json"),
}


def _conv_job(args):
    name, x, K, n_starts, seed, old_cap = args
    out = hb.fit_protocol(x, K, n_starts, seed, snapshot_iters=(old_cap,))
    out.pop("polish_result", None)
    ms = out.pop("multistart")
    return name, {"converged": out["params"], "record": out["record"],
                  "capped": ms["snapshots"][old_cap], "tol_only": ms["best"]}


def _describe(p: dict, x: np.ndarray, ann: float) -> dict:
    """The quantities the paper prints from a fit, from sigma-sorted parameters."""
    K = len(p["mu"])
    pi = dg._stationary(p["A"])
    ll = float(hb.loglik(x, hb.BatchParams.stack([p]))[0])
    k = K * K + 2 * K - 1
    return {"logL": ll, "AIC": 2 * k - 2 * ll, "BIC": k * np.log(x.size) - 2 * ll,
            "sigma_bps": (p["sigma"] * 1e4).tolist(),
            "ann_sigma_pct": (p["sigma"] * ann * 100).tolist(),
            "mu_bps": (p["mu"] * 1e4).tolist(),
            "stationary_pct": (pi * 100).tolist(), "A": np.asarray(p["A"]).tolist(),
            "expected_duration_h": (1.0 / (1.0 - np.diag(p["A"]))).tolist()}


def _hmm_object(p: dict) -> GaussianHMM:
    h = GaussianHMM(K=len(p["mu"]))
    h.pi, h.A, h.mu, h.sigma = (np.asarray(p[k], float) for k in ("pi", "A", "mu", "sigma"))
    return h


def _residual_numbers(p: dict, x: np.ndarray) -> dict:
    h = _hmm_object(p)
    out = {}
    for use in ("filtered", "smoothed"):
        b = dg.residual_battery(dg.hmm_standardised_residuals(h, x, use=use), use)
        row = {"ARCH_LM_12": b["arch_lm"]["LM"], "ARCH_LM_12_p": b["arch_lm"]["p"],
               "JB": b["jarque_bera"]["JB"], "JB_p": b["jarque_bera"]["p"],
               "skew": b["jarque_bera"]["skew"],
               "excess_kurtosis": b["jarque_bera"]["excess_kurtosis"]}
        for tag, block in (("z", "ljung_box_z"), ("z2", "ljung_box_z2"),
                           ("absz", "ljung_box_abs_z")):
            for r in b[block]:
                row[f"Q_{tag}_{r['lag']}"] = r["Q"]
                row[f"Q_{tag}_{r['lag']}_p"] = r["p"]
        out[use] = row
    return out


def _shipped(name: str, ann: float) -> dict | None:
    """The shipped artifact's numbers for one fit, in the units of ``_describe``."""
    def load(f):
        path = DEFAULT_OUT / f
        return json.loads(path.read_text()) if path.exists() else None

    if name.startswith("analysis_"):
        doc = load("summary.json")
        if doc is None:
            return None
        b = doc[f"return_K{name[-1]}"]
        return {"sigma_bps": b["sigma_bps_per_hour"], "A": b["A"], "logL": b["logL"]}
    if name == "winsorised_K3":
        doc = load("robustness.json")
        if doc is None:
            return None
        b = doc["winsorised"]
        return {"sigma_bps": b["sigma_bps"], "A": b["A"], "logL": b["logL"]}
    if name.startswith("selection_"):
        doc = load("model_selection_extended.json")
        if doc is None:
            return None
        b = next(r for r in doc["records"] if r["K"] == int(name[-1]))
        # the 40-start protocol refitted here is the artifact's default_protocol;
        # its headline record is the best optimum of a wider multi-start
        b = b.get("default_protocol", b)
        return {"sigma_bps": b["sigma_bps_per_hour"], "logL": b["logL"]}
    doc = load("two_hourly_refit.json")
    if doc is None:
        return None
    return {"sigma_bps": [v / (ann * 100) * 1e4 for v in doc["annualised_sigma_pct"]],
            "A": doc["A"], "logL": doc["logL"]}


def _shipped_agreement(conv: dict, shipped: dict | None) -> dict:
    if shipped is None:
        return {"available": False}
    d_sig = float(np.max(np.abs(np.array(conv["sigma_bps"]) - np.array(shipped["sigma_bps"]))))
    d_ll = abs(conv["logL"] - shipped["logL"])
    d_A = (float(np.max(np.abs(np.array(conv["A"]) - np.array(shipped["A"]))))
           if "A" in shipped else None)
    # robustness.json rounds sigma to 4 decimals in bps, A to 6 and logL to 2
    ok = d_sig < 1e-4 and d_ll < 1e-2 and (d_A is None or d_A < 1e-6)
    return {"available": True, "max_abs_diff_sigma_bps": d_sig, "abs_diff_logL": d_ll,
            "max_abs_diff_A": d_A, "matches": bool(ok)}


#: Version 1.2.0 of the paper (8 October 2026) printed its sub-period figures
#: under the file numbers of the figure files; this is where they are now.
V120_FIGURES = {"Figure 2(a) and Figure 1(a) legends":
                "Figures 4(a) and 5(a) legends (now Figures C1(a), C2(a))"}


def _v120_where(where: str) -> str:
    """Label a place in version 1.2.0 of the paper as such."""
    return "v1.2.0 " + V120_FIGURES.get(where, where)


#: The printed sub-period numbers that depend on these fits, as version 1.2.0
#: of the paper printed them (from the 150- and 300-iteration fits): where in
#: that version, quantity, printed value, decimals, and how to read the value
#: from a described fit.
def _printed_rows(d: dict) -> list:
    a3, w3, h2, a2 = d["analysis_K3"], d["winsorised_K3"], d["two_hourly_K3"], d["analysis_K2"]
    sel = {K: d[f"selection_K{K}"] for K in (2, 3, 4, 5)}
    rows = []

    def add(where, quantity, printed, dec, get):
        rows.append((_v120_where(where), quantity, printed, dec, get))

    for k, (pv, wv) in enumerate(zip((12.9, 31.0, 87.7), (12.6, 30.4, 85.4))):
        add("Table A1 raw row; Sec. 4.3 sub-period fit", f"annualised sigma state {k + 1} (%)",
            pv, 1, lambda f, k=k: a3[f]["ann_sigma_pct"][k])
        add("Table A1 winsorised row", f"annualised sigma state {k + 1} (%)",
            wv, 1, lambda f, k=k: w3[f]["ann_sigma_pct"][k])
    add("Sec. 4.3 robustness paragraph ('from 87.7% to 85.4%')",
        "annualised sigma state 3, raw (%)", 87.7, 1, lambda f: a3[f]["ann_sigma_pct"][2])
    add("Sec. 4.3 ('87.7% lies inside the full sample's [83.0, 94.0]')",
        "annualised sigma state 3, raw (%)", 87.7, 1, lambda f: a3[f]["ann_sigma_pct"][2])
    add("Figure 2(a) and Figure 1(a) legends", "sigma state 1 (bps/h)", 13.7, 1,
        lambda f: a3[f]["sigma_bps"][0])
    add("Figure 2(a) and Figure 1(a) legends", "sigma state 2 (bps/h)", 33.1, 1,
        lambda f: a3[f]["sigma_bps"][1])
    add("Figure 2(a) and Figure 1(a) legends", "sigma state 3 (bps/h)", 93.7, 1,
        lambda f: a3[f]["sigma_bps"][2])
    add("Table A1 caption", "stationary mass high, raw (%)", 21.2, 1,
        lambda f: a3[f]["stationary_pct"][2])
    add("Table A1 caption", "stationary mass high, winsorised (%)", 22.4, 1,
        lambda f: w3[f]["stationary_pct"][2])
    add("Table A1 caption", "stationary mass low, raw (%)", 18.3, 1,
        lambda f: a3[f]["stationary_pct"][0])
    add("Table A1 caption", "stationary mass low, winsorised (%)", 17.3, 1,
        lambda f: w3[f]["stationary_pct"][0])
    add("Table A1 caption", "largest move of a stationary mass (pp)", 1.2, 1,
        lambda f: max(abs(w3[f]["stationary_pct"][k] - a3[f]["stationary_pct"][k])
                      for k in range(3)))
    add("Sec. 4.3 sub-period fit", "A_32 high to moderate", 0.26, 2, lambda f: a3[f]["A"][2][1])
    add("Sec. 4.3 sub-period fit; Table A1 caption", "A_31 high to low (printed as zero)",
        0.0, 8, lambda f: a3[f]["A"][2][0])
    add("Sec. 4.3 winsorised fit; Table A1 caption", "A_31 high to low (printed as zero)",
        0.0, 8, lambda f: w3[f]["A"][2][0])
    for K, pv in zip((2, 3, 4, 5), (-28356.8, -28533.4, -28557.0, -28482.2)):
        add("Sec. 4.3 more than three states", f"BIC K={K}", pv, 1,
            lambda f, K=K: sel[K][f]["BIC"])
    add("Sec. 4.3", "BIC margin of K=4 over K=3", 23.6, 1,
        lambda f: sel[3][f]["BIC"] - sel[4][f]["BIC"])
    add("Sec. 4.3", "BIC gain of K=3 over K=2", 176.6, 1,
        lambda f: sel[2][f]["BIC"] - sel[3][f]["BIC"])
    add("Sec. 4.3", "logL gain of K=5 over K=4", 7.5, 1,
        lambda f: sel[5][f]["logL"] - sel[4][f]["logL"])
    for k, pv in enumerate((11.1, 24.8, 55.8, 134.9)):
        add("Sec. 4.3", f"K=4 sigma state {k + 1} (bps/h)", pv, 1,
            lambda f, k=k: sel[4][f]["sigma_bps"][k])

    def ladder(f):
        s = np.array(sel[4][f]["sigma_bps"])
        return s[1:] / s[:-1]

    add("Sec. 4.3", "K=4 smallest ladder ratio", 2.2, 1, lambda f: min(ladder(f)))
    add("Sec. 4.3", "K=4 largest ladder ratio", 2.4, 1, lambda f: max(ladder(f)))
    for q, pv in (("logL", 14323.8), ("AIC", -28619.7), ("BIC", -28533.4)):
        add("Table A3 Gaussian HMM K=3 row", q, pv, 1, lambda f, q=q: a3[f][q])
    for q, pv in (("logL", 14207.0), ("AIC", -28400.0), ("BIC", -28356.8)):
        add("Table A3 Gaussian HMM K=2 row", q, pv, 1, lambda f, q=q: a2[f][q])
    add("Sec. 4.5 text", "HMM K=3 BIC (integer)", -28533, 0, lambda f: a3[f]["BIC"])
    add("Sec. 4.5 text", "HMM K=3 AIC (integer)", -28620, 0, lambda f: a3[f]["AIC"])
    for k, pv in enumerate((11.0, 31.6, 88.8)):
        add("Sec. 4.3 two-hourly refit (T = 2,499, whole window)",
            f"annualised sigma state {k + 1} (%)", pv, 1,
            lambda f, k=k: h2[f]["ann_sigma_pct"][k])
    add("Sec. 4.3 two-hourly refit", "stationary mass high (%)", 21.0, 1,
        lambda f: h2[f]["stationary_pct"][2])
    add("Sec. 4.3 two-hourly refit", "A_31 (printed as approx. 0.01)", 0.01, 2,
        lambda f: h2[f]["A"][2][0])
    table_a2 = (
        ("filtered", "Q_z_10", 7.9, 1), ("filtered", "Q_z_10_p", 0.64, 2),
        ("filtered", "Q_z_24", 18.3, 1), ("filtered", "Q_z_24_p", 0.79, 2),
        ("filtered", "Q_z2_10", 2.4, 1), ("filtered", "Q_z2_10_p", 0.99, 2),
        ("filtered", "Q_z2_24", 5.7, 1), ("filtered", "Q_z2_24_p", 1.00, 2),
        ("filtered", "Q_absz_10", 20.6, 1), ("filtered", "Q_absz_10_p", 0.024, 3),
        ("filtered", "Q_absz_24", 96.0, 1),
        ("filtered", "ARCH_LM_12", 2.4, 1), ("filtered", "ARCH_LM_12_p", 0.9985, 4),
        ("filtered", "JB", 31101, 0),
        ("filtered", "skew", -1.02, 2), ("filtered", "excess_kurtosis", 14.45, 2),
        ("smoothed", "Q_z_10", 14.2, 1), ("smoothed", "Q_z_10_p", 0.16, 2),
        ("smoothed", "Q_z_24", 28.2, 1), ("smoothed", "Q_z_24_p", 0.25, 2),
        ("smoothed", "Q_z2_10", 53.3, 1), ("smoothed", "Q_z2_24", 92.3, 1),
        ("smoothed", "Q_absz_10", 32.8, 1), ("smoothed", "Q_absz_10_p", 0.0003, 4),
        ("smoothed", "Q_absz_24", 78.4, 1),
        ("smoothed", "ARCH_LM_12", 46.1, 1), ("smoothed", "JB", 13, 0),
        ("smoothed", "JB_p", 0.0017, 4),
        ("smoothed", "skew", -0.06, 2), ("smoothed", "excess_kurtosis", 0.27, 2))
    for use, stat, pv, dec in table_a2:
        add(f"Table A2 HMM K=3 {use} row", stat, pv, dec,
            lambda f, use=use, stat=stat: a3["residuals_" + f][use][stat])
    add("Sec. 4.5 text ('an excess kurtosis of 14.4 against 6.9')",
        "filtered excess kurtosis", 14.4, 1,
        lambda f: a3["residuals_" + f]["filtered"]["excess_kurtosis"])
    add("Sec. 4.5 text ('Q(z^2)(10) = 53.3' for the smoothed posterior)",
        "smoothed Q_z2_10", 53.3, 1, lambda f: a3["residuals_" + f]["smoothed"]["Q_z2_10"])
    add("Sec. 4.5 text ('Q(z^2)(10) falls to 2.4 and ARCH-LM ... to 2.4')",
        "filtered Q_z2_10", 2.4, 1, lambda f: a3["residuals_" + f]["filtered"]["Q_z2_10"])
    add("Sec. 4.5 text ('Q(z^2)(10) falls to 2.4 and ARCH-LM ... to 2.4')",
        "filtered ARCH_LM_12", 2.4, 1,
        lambda f: a3["residuals_" + f]["filtered"]["ARCH_LM_12"])
    add("Sec. 4.5 text ('the tail-robust Q(|z|)(24) = 96.0')", "filtered Q_absz_24", 96.0, 1,
        lambda f: a3["residuals_" + f]["filtered"]["Q_absz_24"])
    return rows


def _printed_comparison(d: dict) -> list:
    """Every printed sub-period number: the capped fits against the converged ones."""
    out = []
    for where, quantity, printed, dec, get in _printed_rows(d):
        old, new = float(get("capped")), float(get("converged"))
        half = 10 ** -dec / 2
        n = round(new, dec)
        out.append({"where": where, "quantity": quantity, "printed": printed,
                    "decimals": dec, "capped_unrounded": old, "converged_unrounded": new,
                    "converged_at_printed_precision": n,
                    "capped_reproduces_printed": bool(abs(round(old, dec) - printed) < half),
                    "changes": bool(abs(n - printed) > half)})
    return out


def _selection_best_optimum(d: dict) -> dict | None:
    """Model-selection numbers at the 40-start optimum against the wide multi-start's."""
    path = DEFAULT_OUT / "model_selection_extended.json"
    if not path.exists():
        return None
    recs = {r["K"]: r for r in json.loads(path.read_text())["records"]}
    if any("default_protocol" not in r for r in recs.values()):
        return None
    sel = {K: d[f"selection_K{K}"]["converged"] for K in (2, 3, 4, 5)}
    rows = []

    def add(quantity, here, best):
        rows.append({"quantity": quantity, "converged_40_starts": here,
                     "best_optimum_wide_multistart": best,
                     "changes_at_one_decimal": bool(round(here, 1) != round(best, 1))})

    for K in (2, 3, 4, 5):
        add(f"BIC K={K}", sel[K]["BIC"], recs[K]["BIC"])
    add("logL gain of K=5 over K=4", sel[5]["logL"] - sel[4]["logL"],
        recs[5]["logL"] - recs[4]["logL"])
    add("BIC K=5 minus BIC K=3", sel[5]["BIC"] - sel[3]["BIC"],
        recs[5]["BIC"] - recs[3]["BIC"])
    add("BIC margin of K=4 over K=3", sel[3]["BIC"] - sel[4]["BIC"],
        recs[3]["BIC"] - recs[4]["BIC"])
    return {
        "source": ("output/model_selection_extended.json records[*] "
                   "(scripts/model_selection_extended.py)"),
        "note": ("The 40 starts of GaussianHMM.fit reach the best optimum for K = 2, 3 "
                 "and 4 but not for K = 5, where they stop in a lower local optimum. "
                 "The paper prints the best optimum."),
        "n_starts_reaching_best": {K: recs[K]["n_starts_reaching_best"] for K in recs},
        "n_starts": {K: recs[K]["n_starts"] for K in recs},
        "rows": rows,
    }


CONVERGENCE_PROTOCOL = (
    "GaussianHMM.fit, as now shipped (regime_engine.hmm_batch.fit_protocol): every "
    "random start (the same seeds and starting points as before) runs Baum-Welch "
    "until the log-likelihood rises by less than 1e-6, with a safeguard cap of "
    "100,000 iterations; the first start with the largest final log-likelihood is "
    "selected and continued until |dlogL| < 1e-9 and no parameter moves by more "
    "than 1e-10 per iteration (means and volatilities in units of the state's "
    "volatility, transition probabilities as they are). 'capped' is the same run "
    "read off at the cap of the earlier artifacts (150 iterations, 300 for model "
    "selection) with the selection made there; it reproduces the numbers printed "
    "in version 1.2.0 of the paper (8 October 2026); 'where' gives their place in "
    "that version. 'tol_only' is the selected start where the 1e-6 rule left "
    "it, before the polish. Information criteria are evaluated at the returned "
    "parameters.")


def part_convergence(n_jobs: int) -> dict:
    t0 = time.time()
    hourly = _load_sub_returns()
    series = {"hourly": hourly, "winsorised": _winsorise(hourly),
              "two_hourly": _two_hourly_returns()}
    jobs = [(name, series[ser], K, n_starts, seed, cap)
            for name, (ser, K, n_starts, seed, cap, _) in FITS.items()]
    jobs.sort(key=lambda j: -j[2] * j[3])        # the slowest (largest K x starts) first
    with _pool(n_jobs) as ex:
        res = dict(ex.map(_conv_job, jobs))
    print(f"[convergence] fits done ({time.time() - t0:.0f}s)", flush=True)

    d = {}
    for name, (ser, K, n_starts, seed, cap, shipped_in) in FITS.items():
        r = res[name]
        x = series[ser]
        ann = ANN_2H if ser == "two_hourly" else ANN_1H
        rec = r["record"]
        n_it = np.array(rec["per_start_iterations"])
        conv = _describe(r["converged"], x, ann)
        d[name] = {
            "series": ser, "T": int(x.size), "K": K, "n_starts": n_starts, "seed": seed,
            "old_cap": cap,
            "n_starts_at_old_cap": int(np.sum(n_it >= cap)),
            "capped_best_start": r["capped"]["best_start"],
            "selected_start": rec["selected_start"],
            "max_start_iterations": int(n_it.max()),
            "record": rec,
            "converged_flag": rec["converged"],
            "capped": _describe(r["capped"], x, ann),
            "tol_only": _describe(r["tol_only"], x, ann),
            "converged": conv,
            "shipped_artifact": shipped_in,
            "shipped_agreement": _shipped_agreement(conv, _shipped(name, ann)),
        }
        if name == "analysis_K3":
            d[name]["residuals_capped"] = _residual_numbers(r["capped"], x)
            d[name]["residuals_converged"] = _residual_numbers(r["converged"], x)
        print(f"[convergence] {name}: at old cap {d[name]['n_starts_at_old_cap']}/{n_starts}, "
              f"start {r['capped']['best_start']}->{rec['selected_start']}, polish "
              f"{rec['polish']['iterations']} it, converged {rec['converged']}, BIC "
              f"{d[name]['capped']['BIC']:.2f} -> {conv['BIC']:.2f}, shipped "
              f"{d[name]['shipped_agreement'].get('matches')}", flush=True)
    rows = _printed_comparison(d)
    a3 = d["analysis_K3"]
    gap_ll = a3["converged"]["logL"] - a3["capped"]["logL"]
    gap_vol = max(abs(u - v) for u, v in zip(a3["converged"]["ann_sigma_pct"],
                                             a3["capped"]["ann_sigma_pct"]))
    return {
        "protocol": CONVERGENCE_PROTOCOL,
        "fits": d,
        "all_converged": bool(all(f["converged_flag"] for f in d.values())),
        "shipped_artifacts_are_the_converged_fits": bool(
            all(f["shipped_agreement"].get("matches", False) for f in d.values())),
        "printed_vs_converged": rows,
        "selection_best_optimum": _selection_best_optimum(d),
        "changed_at_printed_precision": [r for r in rows if r["changes"]],
        "sub_period_cap_effect": {"logL_converged_minus_capped": gap_ll,
                                  "max_abs_change_annualised_sigma_pct": gap_vol},
        "full_sample_note": (
            "The full-sample K=3 fit (T = 5,001; Table 7, Figure 8) used a 150-iteration "
            "cap, but its hourly returns are no longer available, so it cannot be "
            "refitted and whether it converged cannot be checked. On the sub-period the "
            f"same cap left the log-likelihood {gap_ll:.1e} below the converged value "
            f"and moved no annualised volatility by more than {gap_vol:.3f} points "
            "(sub_period_cap_effect)."),
        "seconds": time.time() - t0,
    }


# =============================================================================
def _software() -> dict:
    import scipy
    import statsmodels
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "pandas": pd.__version__}


def _keep_calibrated(old: dict, new: dict) -> dict:
    """Carry the "calibrated" sub-block over to a rebuilt profile block.

    The sub-block inverts the bootstrap test around the profile's bound and
    copies the bound and its calibration into its last grid row, so it is kept
    only while those two are unchanged (compared as they are written to JSON).
    """
    if "calibrated" not in old:
        return new

    def as_json(x):
        return json.loads(json.dumps(_jsonable(x)))

    keys = ("upper_bound_one_sided_95", "calibration_at_bound")
    if all(as_json(old.get(k)) == as_json(new.get(k)) for k in keys):
        new["calibrated"] = old["calibrated"]
    else:
        print("[profile] calibrated sub-block dropped; rerun --part profile_calibrated "
              "(about 40 min)", flush=True)
    return new


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", default="all",
                    choices=["all", "profile", "profile_calibrated", "power", "power_em",
                             "garch", "convergence"])
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--power-reps", type=int, default=POWER_REPS)
    ap.add_argument("--power-em-reps", type=int, default=POWER_EM_REPS)
    ap.add_argument("--cal-reps", type=int, default=CAL_REPS)
    ap.add_argument("--garch-paths", type=int, default=None,
                    help="paths for every GARCH specification (default: GARCH_PATHS)")
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc["generated_by"] = "scripts/a31_evidence.py"
    doc["software"] = _software()
    parts = (["profile", "profile_calibrated", "power", "power_em", "garch", "convergence"]
             if a.part == "all"
             else [a.part])
    for part in parts:
        if part == "profile":
            doc["profile_likelihood"] = _keep_calibrated(
                doc.get("profile_likelihood", {}), part_profile(a.jobs, a.cal_reps))
        elif part == "profile_calibrated":
            doc["profile_likelihood"]["calibrated"] = part_profile_calibrated(
                a.jobs, a.cal_reps, doc["profile_likelihood"])
        elif part == "power":
            doc["power"] = part_power(a.jobs, a.power_reps)
        elif part == "power_em":
            doc["power_em"] = part_power_em(a.jobs, a.power_em_reps)
        elif part == "garch":
            doc["garch_generic"] = part_garch(a.jobs, a.garch_paths)
        elif part == "convergence":
            doc["convergence"] = part_convergence(a.jobs)
        out.write_text(json.dumps(_jsonable(doc), indent=1) + "\n")
        print(f"[saved] {out} ({part})", flush=True)
    return doc


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
