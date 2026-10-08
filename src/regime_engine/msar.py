"""Markov-Switching AR(1) on the funding premium: fit, inference, half-lives.

Kept separate from ``analysis.py`` so it can be unit-tested (``tests/test_msar.py``)
and so that the inference the paper needs -- standard errors, z-statistics,
95% confidence intervals, and delta-method intervals for the mean-reversion
half-lives and their ratio -- lives next to the estimation.

Parameterisation (statsmodels ``MarkovAutoregression`` with ``switching_trend``):

    y_t - mu_{S_t} = phi_{S_t} (y_{t-1} - mu_{S_{t-1}}) + sigma_{S_t} eps_t

i.e. the model is *mean-adjusted*: ``const[k]`` is the regime MEAN mu_k, not an
intercept, and the likelihood conditions on the pair (S_t, S_{t-1}).  Anything
that draws a regression line must therefore use, within regime k
(S_{t-1} = S_t = k), ``mu_k (1 - phi_k) + phi_k * y_{t-1}``.

Regimes are relabelled by ``sigma2`` ascending, so index 0 is always the calm
regime and index 1 the stressed one.
"""
from __future__ import annotations

import numpy as np

from .hmm import MSAR_SEED, fit_ms_ar1, fit_ms_ar1_multistart

__all__ = [
    "MSAR_SEED",
    "REGIME_LABELS",
    "Z95",
    "fit_msar_premium",
    "half_life",
    "half_life_derivative",
]

Z95 = 1.959963984540054      # two-sided normal 95% critical value
REGIME_LABELS = ("calm", "stressed")


def half_life(phi) -> np.ndarray:
    """tau = ln(0.5) / ln|phi|, in units of the sampling interval (hours)."""
    p = np.abs(np.asarray(phi, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(0.5) / np.log(p)


def half_life_derivative(phi) -> np.ndarray:
    """dtau/dphi = -ln(0.5) / (phi * (ln|phi|)^2)."""
    p = np.asarray(phi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(0.5) / (p * np.log(np.abs(p)) ** 2)


def _finite(x):
    """JSON-safe float: non-finite values become None."""
    if x is None:
        return None
    v = float(x)
    return v if np.isfinite(v) else None


def fit_msar_premium(y, seed: int = MSAR_SEED, n_pad: int = 0,
                     n_starts: int = 40, search_reps: int = 40) -> dict:
    """Fit the MS-AR(1) on ``y`` (premium in bps) and return everything needed.

    Uses the multi-start protocol by default, and it is not optional in spirit:
    this likelihood is multi-modal, and on the paper's 2 Oct 2025 -- 28 Apr 2026
    window the *dominant* basin of attraction is 20.1 log-likelihood points BELOW
    the maximum.  A single start returns the inferior optimum roughly three times
    in five, stably enough to look converged, and reports a half-life ratio of
    24.6 instead of 11.2.  See ``hmm.fit_ms_ar1_multistart``.

    Parameters
    ----------
    y : array_like
        Premium series in basis points.
    seed : int
        Base seed; start ``i`` uses ``seed + i`` (see hmm.MSAR_SEED).
    n_pad : int
        If > 0, pad the smoothed/filtered probability matrices at the front with
        this many rows of NaN so they align with a panel of that length.  The
        AR(1) lag costs the first observation.
    n_starts : int
        Independent starts; the default is the paper's protocol.  With 1 the
        fit is a single start, which on the paper window stops at the inferior
        optimum roughly three times in five.
    search_reps : int
        Random start-parameter replications inside each start.
    """
    y = np.asarray(y, dtype=float)
    if n_starts and n_starts > 1:
        model, res, seed_used, msar_diagnostics = fit_ms_ar1_multistart(
            y, n_starts=n_starts, seed=seed, search_reps=search_reps,
            k_regimes=2, switching_variance=True, switching_ar=True)
    else:
        model, res, seed_used = fit_ms_ar1(y, k_regimes=2, switching_variance=True,
                                           switching_ar=True, seed=seed)
        msar_diagnostics = {"n_starts": 1, "warning": "single start; likelihood is multi-modal"}

    pnames = list(model.param_names)
    pvals = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)

    def _by_regime(prefix):
        return np.array([float(pvals[pnames.index(f"{prefix}[{k}]")]) for k in range(2)])

    sigma2 = _by_regime("sigma2")
    mu = _by_regime("const")          # regime MEAN, not an intercept
    phi = _by_regime("ar.L1")

    order = np.argsort(sigma2)        # 0 = calm (low variance), 1 = stressed
    inv_order = np.argsort(order)

    smoothed = np.asarray(res.smoothed_marginal_probabilities, dtype=float)
    filtered = np.asarray(res.filtered_marginal_probabilities, dtype=float)
    if n_pad > 0:
        pad = n_pad - smoothed.shape[0]
        if pad > 0:
            smoothed = np.vstack([np.full((pad, 2), np.nan), smoothed])
            filtered = np.vstack([np.full((pad, 2), np.nan), filtered])

    # ---- per-parameter inference table (raw statsmodels names) -------------
    z = np.divide(pvals, bse, out=np.full_like(pvals, np.nan), where=bse > 0)
    lo = pvals - Z95 * bse
    hi = pvals + Z95 * bse
    param_table = [
        {
            "name": pnames[i],
            "coef": _finite(pvals[i]),
            "se": _finite(bse[i]),
            "z": _finite(z[i]),
            "ci95_low": _finite(lo[i]),
            "ci95_high": _finite(hi[i]),
        }
        for i in range(len(pnames))
    ]

    # ---- half-lives with delta-method inference ---------------------------
    phi_ord = phi[order]
    se_phi_raw = np.array([float(bse[pnames.index(f"ar.L1[{k}]")]) for k in range(2)])
    se_phi = se_phi_raw[order]
    tau = half_life(phi_ord)
    dtau = half_life_derivative(phi_ord)
    se_tau = np.abs(dtau) * se_phi

    # Monotone-transform interval: tau(phi) is increasing in phi on (0, 1), so
    # transforming the endpoints of phi's Wald interval gives the more reliable
    # interval when the upper endpoint approaches (or exceeds) unity.
    phi_lo = phi_ord - Z95 * se_phi
    phi_hi = phi_ord + Z95 * se_phi
    tau_from_phi_lo = half_life(phi_lo)
    tau_from_phi_hi = half_life(phi_hi)

    half_lives = []
    for k in range(2):
        contains_unit_root = bool(phi_hi[k] >= 1.0)
        half_lives.append({
            "regime": REGIME_LABELS[k],
            "phi": _finite(phi_ord[k]),
            "phi_se": _finite(se_phi[k]),
            "phi_ci95": [_finite(phi_lo[k]), _finite(phi_hi[k])],
            "half_life_hours": _finite(tau[k]),
            "d_halflife_d_phi": _finite(dtau[k]),
            "half_life_se_delta": _finite(se_tau[k]),
            "half_life_ci95_delta": [
                _finite(tau[k] - Z95 * se_tau[k]),
                _finite(tau[k] + Z95 * se_tau[k]),
            ],
            # None on the upper end == unbounded (phi's CI reaches a unit root)
            "half_life_ci95_from_phi": [
                _finite(tau_from_phi_lo[k]),
                None if contains_unit_root else _finite(tau_from_phi_hi[k]),
            ],
            "phi_ci95_contains_unit_root": contains_unit_root,
        })

    # ---- ratio of half-lives, delta method with the full covariance --------
    # R = tau_calm / tau_stressed = ln|phi_stressed| / ln|phi_calm|
    ratio = float(tau[0] / tau[1])
    ratio_block = {
        "definition": "tau_calm / tau_stressed",
        "value": _finite(ratio),
        "se_delta": None,
        "ci95_delta": [None, None],
    }
    try:
        cov = np.asarray(res.cov_params(), dtype=float)
        idx = [pnames.index(f"ar.L1[{k}]") for k in range(2)]
        idx_ord = [idx[order[0]], idx[order[1]]]
        v = cov[np.ix_(idx_ord, idx_ord)]
        p0, p1 = phi_ord[0], phi_ord[1]
        l0, l1 = np.log(np.abs(p0)), np.log(np.abs(p1))
        grad = np.array([-l1 / (p0 * l0 ** 2), 1.0 / (p1 * l0)])
        var_ratio = float(grad @ v @ grad)
        if np.isfinite(var_ratio) and var_ratio >= 0:
            se_ratio = float(np.sqrt(var_ratio))
            ratio_block["se_delta"] = _finite(se_ratio)
            ratio_block["ci95_delta"] = [
                _finite(ratio - Z95 * se_ratio),
                _finite(ratio + Z95 * se_ratio),
            ]
            ratio_block["gradient"] = [_finite(grad[0]), _finite(grad[1])]
    except Exception as exc:                       # pragma: no cover - defensive
        ratio_block["error"] = f"{type(exc).__name__}: {exc}"

    # ---- regime transition matrix, reordered ------------------------------
    try:
        P = np.asarray(res.regime_transition, dtype=float)
        P = P[:, :, 0] if P.ndim == 3 else P
        # statsmodels stores P[j, i] = P(S_t = j | S_{t-1} = i); transpose to
        # the row-stochastic "from -> to" convention used elsewhere in the repo.
        P = P.T[np.ix_(order, order)]
        transition = [[_finite(P[i, j]) for j in range(2)] for i in range(2)]
    except Exception:                              # pragma: no cover - defensive
        transition = None

    nobs = int(getattr(res, "nobs", len(y) - 1))
    return {
        "result": res,
        "model": model,
        "seed": int(seed_used),
        "multistart": msar_diagnostics,
        "order": order,
        "inv_order": inv_order,
        "filtered": filtered[:, order],
        "smoothed": smoothed[:, order],
        "mu_const": mu[order],           # regime MEANS (see module docstring)
        "phi": phi_ord,
        "sigma2": sigma2[order],
        "sigma": np.sqrt(sigma2[order]),
        "half_life_hours": tau,
        "param_table": param_table,
        "half_lives": half_lives,
        "half_life_ratio": ratio_block,
        "transition_matrix": transition,
        "llf": _finite(res.llf),
        "aic": _finite(res.aic),
        "bic": _finite(res.bic),
        "nobs": nobs,
        "param_names": pnames,
    }
