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

Refinements of the inference sit beside the fit (``msar_robust_inference``):

* **Covariance.**  statsmodels 0.14.6 offers three estimators for a
  Markov-switching fit: ``approx`` (the inverse negative Hessian, by complex
  step; the default of ``fit``), ``opg`` (the inverse outer product of the
  per-observation scores) and ``robust`` (the sandwich H^-1 (S'S) H^-1).  The
  first two are valid only when the Gaussian likelihood is the true one; the
  sandwich stays valid under misspecification.  Every quantity below is
  reported under all three.

* **Half-lives under switching.**  tau_k = ln 0.5 / ln|phi_k| is the half-life
  of a dislocation *conditional on remaining in regime k*.  Under the fitted
  chain the deviation x_t = y_t - mu_{S_t} decays in expectation as
  E[x_{t+h} | x_t = 1, S_t = k] = e_k' (P o Phi)^h 1, with P row-stochastic and
  (P o Phi)_ij = P_ij phi_j; the first hour at which that path reaches one half
  is the switching-adjusted half-life.

* **Interval forms for R = tau_calm / tau_stressed.**  Beside the linear delta
  interval, a log-scale delta interval (R is positive and right-skewed) and
  Fieller's set for the ratio ln|phi_stressed| / ln|phi_calm|, each with the
  side of unity it falls on.  These sets are descriptive.

* **The side of unity.**  R = 1 exactly when phi_calm = phi_stressed, so the
  verdict is the Wald test of that linear restriction (``ratio_wald_test``),
  which needs no delta step; the log-scale interval, by contrast, excludes
  unity too often when phi_stressed's interval approaches one.  With a
  persistent chain the switching-adjusted ratio falls on the same side of
  unity (``switching_ratio_side``); its interval is simulated from the
  estimate's asymptotic distribution (``switching_ratio_simulation``).

The multi-start scan behind every fit is guarded (``fit_msar_multistart``):
a start with a non-finite likelihood, a collapsed regime variance or a
singular Hessian cannot be the reported optimum.
"""
from __future__ import annotations

import math

import numpy as np

from .hmm import MSAR_SEED, fit_ms_ar1

__all__ = [
    "COV_TYPES",
    "EXCLUSIONS",
    "MSAR_SEED",
    "REGIME_LABELS",
    "SIGMA2_FLOOR_REL",
    "SIM_DRAWS",
    "SIM_SEED",
    "Z95",
    "chain_eigenvalue",
    "expected_durations",
    "fieller_interval",
    "fit_msar_multistart",
    "fit_msar_premium",
    "half_life",
    "half_life_derivative",
    "msar_covariances",
    "msar_robust_inference",
    "ratio_intervals",
    "ratio_sandwich_influence",
    "ratio_wald_test",
    "row_stochastic_transition",
    "side_of_unity",
    "stationary_distribution",
    "summarise_msar_fit",
    "switching_decay_closed_form",
    "switching_decay_path",
    "switching_delta",
    "switching_half_life",
    "switching_half_lives_batch",
    "switching_half_lives_from_raw",
    "switching_ratio_side",
    "switching_ratio_simulation",
]

Z95 = 1.959963984540054      # two-sided normal 95% critical value
REGIME_LABELS = ("calm", "stressed")

#: The covariance estimators of a statsmodels Markov-switching fit, keyed by the
#: name the artifacts use; the values are statsmodels' own ``cov_type`` strings.
COV_TYPES = {"hessian": "approx", "opg": "opg", "robust": "robust"}


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


#: A start whose smaller regime variance lies below this fraction of var(y) has
#: collapsed onto a handful of hours, where the Gaussian likelihood is unbounded.
SIGMA2_FLOOR_REL = 1e-6

#: The ways a converged start can fail to be admissible as the optimum.
EXCLUSIONS = ("non_finite_llf", "variance_floor", "singular_hessian")


def _exclusion(model, res, floor: float):
    """Why a converged start cannot be the reported optimum, or None if it can."""
    if not np.isfinite(float(res.llf)):
        return "non_finite_llf"
    params = np.asarray(res.params, dtype=float)
    sigma2 = np.array([params[i] for i, name in enumerate(model.param_names)
                       if name.startswith("sigma2")])
    if sigma2.size and not np.all(sigma2 >= floor):
        return "variance_floor"
    try:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cov = np.asarray(res.cov_params_approx, dtype=float)
    except (np.linalg.LinAlgError, ValueError):
        return "singular_hessian"
    if not (np.all(np.isfinite(cov)) and np.all(np.diag(cov) > 0)):
        return "singular_hessian"
    return None


def fit_msar_multistart(y, n_starts: int = 40, seed: int = MSAR_SEED,
                        search_reps: int = 40, verbose: bool = True,
                        sigma2_floor: float = SIGMA2_FLOOR_REL, **kwargs):
    """``hmm.fit_ms_ar1_multistart`` with a guard on which start may be the optimum.

    Runs the same starts (``seed``, ``seed + 1``, ...) and returns the same
    ``(model, res, seed_used, diagnostics)``, but the optimum is the best
    *admissible* start.  Three kinds of converged start are not admissible and
    are counted in ``diagnostics["excluded"]``:

    * ``non_finite_llf``: the log-likelihood is NaN or infinite.  An unguarded
      scan keeps such a start when it comes first, since ``llf > nan`` is
      False for every later start.
    * ``variance_floor``: a regime variance below ``sigma2_floor`` x var(y)
      (``diagnostics["sigma2_floor"]`` holds the absolute floor).  The
      Gaussian likelihood is unbounded as one regime's variance shrinks onto a
      single hour, so such a start can beat the proper maximum.
    * ``singular_hessian``: the inverse-Hessian covariance is not finite with
      a positive diagonal, so the fit has no standard errors.

    ``optima_found`` still lists every converged start, admissible or not.
    ``hmm.fit_ms_ar1`` is looked up when the scan runs, so a test can count or
    script the starts.
    """
    from collections import Counter

    from . import hmm

    y = np.asarray(y, dtype=float)
    floor = float(sigma2_floor) * float(np.var(y))
    best = None
    optima: Counter = Counter()
    failures = 0
    excluded = dict.fromkeys(EXCLUSIONS, 0)
    for i in range(max(1, n_starts)):
        s = int(seed) + i
        try:
            model, res, seed_used = hmm.fit_ms_ar1(
                y, seed=s, search_reps=search_reps, seed_retries=1, **kwargs)
        except Exception:
            failures += 1
            continue
        optima[round(float(res.llf), 2)] += 1
        why = _exclusion(model, res, floor)
        if why is not None:
            excluded[why] += 1
            continue
        if best is None or res.llf > best[1].llf:
            best = (model, res, seed_used)

    if best is None:
        raise RuntimeError(f"MS-AR multi-start found no admissible optimum in {n_starts} "
                           f"starts ({failures} failures, excluded {excluded})")

    ranked = optima.most_common()
    best_llf = round(float(best[1].llf), 2)
    diagnostics = {
        "n_starts": int(n_starts),
        "search_reps": int(search_reps),
        "seed_base": int(seed),
        "failures": int(failures),
        "optima_found": [{"llf": llf, "count": c} for llf, c in ranked],
        "best_llf": float(best[1].llf),
        "best_seed": int(best[2]),
        "modal_llf": ranked[0][0],
        "modal_is_best": bool(abs(ranked[0][0] - best_llf) < 1e-9),
        "excluded": excluded,
        "sigma2_floor": floor,
    }
    if verbose:
        print(f"[MS-AR] multi-start: {len(ranked)} distinct optima over "
              f"{n_starts} starts ({failures} failures; excluded: "
              + ", ".join(f"{k} {v}" for k, v in excluded.items()) + ")")
        for llf, c in ranked:
            mark = "  <-- reported (max)" if llf == best_llf else ""
            print(f"         logL {llf:12.2f}  x{c:3d}{mark}")
        if not diagnostics["modal_is_best"]:
            print("[MS-AR] WARNING: the most frequent optimum is NOT the maximum. "
                  "A single-start fit would have reported the inferior one.")
    return best[0], best[1], best[2], diagnostics


def fit_msar_premium(y, seed: int = MSAR_SEED, n_pad: int = 0,
                     n_starts: int = 40, search_reps: int = 40) -> dict:
    """Fit the MS-AR(1) on ``y`` (premium in bps) and return everything needed.

    Uses the multi-start protocol by default, and it is not optional in spirit:
    this likelihood is multi-modal, and on the paper's 2 Oct 2025 -- 28 Apr 2026
    window the *dominant* basin of attraction is 20.1 log-likelihood points BELOW
    the maximum.  A single start returns the inferior optimum roughly three times
    in five, stably enough to look converged, and reports a half-life ratio of
    24.6 instead of 11.2.  See ``hmm.fit_ms_ar1_multistart``; the scan here is
    its guarded form, :func:`fit_msar_multistart`.

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
        model, res, seed_used, msar_diagnostics = fit_msar_multistart(
            y, n_starts=n_starts, seed=seed, search_reps=search_reps,
            k_regimes=2, switching_variance=True, switching_ar=True)
    else:
        model, res, seed_used = fit_ms_ar1(y, k_regimes=2, switching_variance=True,
                                           switching_ar=True, seed=seed)
        msar_diagnostics = {"n_starts": 1, "warning": "single start; likelihood is multi-modal"}
    return summarise_msar_fit(y, model, res, seed_used, msar_diagnostics, n_pad=n_pad)


def summarise_msar_fit(y, model, res, seed_used: int, multistart: dict,
                       n_pad: int = 0) -> dict:
    """The inference dict of :func:`fit_msar_premium`, for a fit already in hand.

    ``model``, ``res`` and ``seed_used`` are what ``hmm.fit_ms_ar1`` or
    ``hmm.fit_ms_ar1_multistart`` return, and ``multistart`` is the latter's
    diagnostics.  Callers that ran the multi-start scan themselves pass its
    winner here instead of fitting again.
    """
    y = np.asarray(y, dtype=float)
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
        P = row_stochastic_transition(res.regime_transition, order)
        transition = [[_finite(P[i, j]) for j in range(2)] for i in range(2)]
    except Exception:                              # pragma: no cover - defensive
        transition = None

    nobs = int(getattr(res, "nobs", len(y) - 1))
    return {
        "result": res,
        "model": model,
        "seed": int(seed_used),
        "multistart": multistart,
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


# ---------------------------------------------------------------------------
# The fitted chain
# ---------------------------------------------------------------------------
def row_stochastic_transition(regime_transition, order) -> np.ndarray:
    """statsmodels' transition array as a row-stochastic matrix in ``order``.

    statsmodels stores ``regime_transition[i, j] = P(S_t = i | S_{t-1} = j)``
    (columns sum to one; the parameter ``p[j->i]`` is that probability).  The
    result is ``P[a, b] = P(S_t = order[b] | S_{t-1} = order[a])``: rows are
    "from", columns "to", both in the sigma2-ascending order of ``order``.
    """
    P = np.asarray(regime_transition, dtype=float)
    P = P[:, :, 0] if P.ndim == 3 else P
    order = np.asarray(order, dtype=int)
    return P.T[np.ix_(order, order)]


def expected_durations(P) -> np.ndarray:
    """Mean sojourn 1 / (1 - p_kk) of each regime, in hours, for a row-stochastic P."""
    p = np.diag(np.asarray(P, dtype=float))
    with np.errstate(divide="ignore"):
        return 1.0 / (1.0 - p)


def chain_eigenvalue(P) -> float:
    """p_cc + p_ss - 1, the second eigenvalue of a row-stochastic two-state ``P``.

    Positive for a persistent chain, one in which the regime an hour ahead is
    more likely to be the current one than it would be under independence.
    Then a chain started in the calm regime is calm at least as often, at every
    later hour, as one started in the stressed regime (a monotone coupling),
    so with positive coefficients the switching-adjusted ratio lies on the
    same side of unity as phi_calm - phi_stressed.  For a negative value the
    regime alternates and that ordering can reverse.
    """
    P = np.asarray(P, dtype=float)
    if P.shape != (2, 2):
        raise ValueError(f"P has shape {P.shape}, expected a two-state chain")
    return float(np.trace(P) - 1.0)


def stationary_distribution(P) -> np.ndarray:
    """The distribution pi with pi P = pi and sum(pi) = 1."""
    P = np.asarray(P, dtype=float)
    k = P.shape[0]
    a = np.vstack([P.T - np.eye(k), np.ones(k)])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    return pi


def _chain(P, phi, start):
    P = np.asarray(P, dtype=float)
    phi = np.asarray(phi, dtype=float)
    k = phi.size
    if P.shape != (k, k):
        raise ValueError(f"P has shape {P.shape}, expected {(k, k)}")
    start = int(start)
    if not 0 <= start < k:
        raise ValueError(f"start {start} is not a regime index of a {k}-state chain")
    return P, phi, start


def switching_decay_path(P, phi, start: int, horizon: int) -> np.ndarray:
    """Expected decay of a unit deviation under the chain, m_k(h) for h = 0..horizon.

    With x_t = y_t - mu_{S_t}, the model gives x_{t+1} = phi_{S_{t+1}} x_t +
    noise, so E[x_{t+h} | x_t = 1, S_t = k] = e_k' (P o Phi)^h 1 where
    (P o Phi)_ij = P_ij phi_j and ``P`` is row-stochastic ("from" in rows).
    """
    P, phi, k = _chain(P, phi, start)
    M = P * phi[None, :]
    out = np.empty(int(horizon) + 1)
    out[0] = 1.0
    v = np.zeros(phi.size)
    v[k] = 1.0
    for h in range(1, int(horizon) + 1):
        v = v @ M
        out[h] = v.sum()
    return out


def switching_decay_closed_form(P, phi, start: int, hours) -> np.ndarray:
    """Two-state closed form of :func:`switching_decay_path` at integer ``hours``.

    With r_k = sum_j P_kj phi_j, the expected one-step factor from regime k,
    and lambda_1, lambda_2 the eigenvalues of M = P o Phi, Sylvester's formula
    M^h = [lambda_1^h (M - lambda_2 I) - lambda_2^h (M - lambda_1 I)] /
    (lambda_1 - lambda_2) gives

        m_k(h) = [(r_k - lambda_2) lambda_1^h + (lambda_1 - r_k) lambda_2^h]
                 / (lambda_1 - lambda_2),

    and for a repeated eigenvalue lambda, m_k(h) = lambda^h + h lambda^(h-1)
    (r_k - lambda).
    """
    P, phi, k = _chain(P, phi, start)
    if phi.size != 2:
        raise ValueError("the closed form is for a two-state chain")
    h = np.asarray(hours)
    if not np.all(np.equal(np.mod(h, 1), 0)):
        raise ValueError("hours must be integers")
    h = h.astype(int)
    M = P * phi[None, :]
    tr, det = float(np.trace(M)), float(np.linalg.det(M))
    r = float(M[k].sum())
    disc = tr * tr - 4.0 * det
    scale = max(tr * tr, 1e-300)
    if abs(disc) <= 1e-14 * scale:
        lam = tr / 2.0
        return lam ** h + h * lam ** (h - 1) * (r - lam)
    root = np.sqrt(complex(disc))
    l1, l2 = (tr + root) / 2.0, (tr - root) / 2.0
    if disc > 0:
        l1, l2 = l1.real, l2.real
        return ((r - l2) * np.power(l1, h) + (l1 - r) * np.power(l2, h)) / (l1 - l2)
    out = ((r - l2) * np.power(l1, h) + (l1 - r) * np.power(l2, h)) / (l1 - l2)
    return np.real(out)


def switching_half_life(P, phi, start: int, level: float = 0.5,
                        max_hours: int = 100_000) -> dict:
    """Hours until the expected deviation from regime ``start`` first reaches ``level``.

    Returns ``{"hours": h, "interpolated": h_star}``: ``h`` is the first integer
    hour with m_k(h) <= level and ``h_star`` interpolates log-linearly between
    hours h - 1 and h, which is exact when the path is geometric (no switching,
    or equal phi).  Both are ``None`` if the level is not reached within
    ``max_hours``.
    """
    P, phi, k = _chain(P, phi, start)
    M = P * phi[None, :]
    v = np.zeros(phi.size)
    v[k] = 1.0
    prev = 1.0
    for h in range(1, int(max_hours) + 1):
        v = v @ M
        m = float(v.sum())
        if m <= level:
            if prev > 0 and m > 0:
                frac = (np.log(prev) - np.log(level)) / (np.log(prev) - np.log(m))
            else:
                frac = (prev - level) / (prev - m)
            return {"hours": h, "interpolated": (h - 1) + float(frac)}
        prev = m
    return {"hours": None, "interpolated": None}


def switching_half_lives_from_raw(theta, order) -> list:
    """Interpolated switching-adjusted half-lives (calm, stressed) from raw parameters.

    ``theta`` is (p[0->0], p[1->0], ar.L1[0], ar.L1[1]) in statsmodels' raw
    regime order and ``order`` the sigma2-ascending permutation, so that
    ``order[0]`` is the raw index of the calm regime.
    """
    p00, p10, f0, f1 = (float(v) for v in theta)
    raw = np.array([[p00, 1.0 - p00], [p10, 1.0 - p10]])     # row-stochastic, raw order
    order = np.asarray(order, dtype=int)
    P = raw[np.ix_(order, order)]
    phi = np.array([f0, f1])[order]
    return [switching_half_life(P, phi, start=k)["interpolated"] for k in range(2)]


def switching_delta(theta, order, cov, z: float = Z95, rel_step: float = 1e-6) -> dict:
    """Delta-method inference for the interpolated switching-adjusted half-lives.

    ``theta`` and ``order`` are as in :func:`switching_half_lives_from_raw` and
    ``cov`` is the 4x4 covariance of ``theta``.  The gradient of (h_calm,
    h_stressed, h_calm / h_stressed) is taken by central differences; the
    ratio gets a linear and a log-scale 95% interval with its side of unity.
    """
    theta = np.asarray(theta, dtype=float)
    V = np.asarray(cov, dtype=float)

    def f(t):
        hc, hs = switching_half_lives_from_raw(t, order)
        if hc is None or hs is None or hs <= 0:
            return np.full(3, np.nan)
        return np.array([hc, hs, hc / hs])

    base = f(theta)
    J = np.empty((3, theta.size))
    for j in range(theta.size):
        step = rel_step * max(abs(theta[j]), 1e-3)
        up, dn = theta.copy(), theta.copy()
        up[j] += step
        dn[j] -= step
        J[:, j] = (f(up) - f(dn)) / (2.0 * step)
    with np.errstate(invalid="ignore"):
        var = np.einsum("ij,jk,ik->i", J, V, J)
    se = np.where(np.isfinite(var) & (var >= 0), np.sqrt(np.abs(var)), np.nan)
    r, se_r = float(base[2]), float(se[2])
    if np.isfinite(r) and np.isfinite(se_r):
        linear = _form("bounded", [r - z * se_r, r + z * se_r])
    else:
        linear = _form("undefined", [None, None])
    if np.isfinite(r) and np.isfinite(se_r) and r > 0:
        g = float(np.exp(z * se_r / r))
        log = _form("bounded", [r / g, r * g])
    else:
        log = _form("undefined", [None, None])
    return {
        "half_lives": [_finite(base[0]), _finite(base[1])],
        "half_life_se": [_finite(se[0]), _finite(se[1])],
        "ratio": _finite(r),
        "ratio_se": _finite(se_r),
        "ratio_ci95": {"linear": linear, "log": log},
    }


def switching_half_lives_batch(p_stay, phi, level: float = 0.5,
                               max_hours: int = 10_000, return_end: bool = False) -> tuple:
    """:func:`switching_half_life` for many two-state chains at once.

    ``p_stay`` and ``phi`` are (n, 2) arrays of staying probabilities and
    autoregressive coefficients, calm first.  Returns ``(hours, interpolated)``,
    two (n, 2) arrays whose column k belongs to a start in regime k: the first
    integer hour at which the expected path falls to ``level`` and the
    interpolated crossing, computed as in :func:`switching_half_life`.  Both
    are ``inf`` where the path does not get there within ``max_hours``.  With
    ``return_end`` a third (n, 2) array holds such a path's value at
    ``max_hours`` (NaN for the paths that do halve).
    """
    p_stay = np.asarray(p_stay, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if p_stay.ndim != 2 or p_stay.shape[1] != 2 or phi.shape != p_stay.shape:
        raise ValueError("p_stay and phi must both be (n, 2) arrays")
    n = p_stay.shape[0]
    # M = P o Phi, row-stochastic P with "from" in rows: M[i, j] = P[i, j] phi_j
    m_all = (p_stay[:, 0] * phi[:, 0], (1.0 - p_stay[:, 0]) * phi[:, 1],
             (1.0 - p_stay[:, 1]) * phi[:, 0], p_stay[:, 1] * phi[:, 1])
    hours = np.full((n, 2), np.inf)
    interp = np.full((n, 2), np.inf)
    end = np.full((n, 2), np.nan)
    for k in range(2):
        idx = np.arange(n)
        a = np.full(n, 1.0 if k == 0 else 0.0)      # the row vector e_k' M^h, (a, b)
        b = np.full(n, 0.0 if k == 0 else 1.0)
        prev = np.ones(n)
        m00, m01, m10, m11 = m_all
        for h in range(1, int(max_hours) + 1):
            if idx.size == 0:
                break
            # A path that explodes (a coefficient above one that the chain
            # rarely leaves) overflows to inf and never halves.
            with np.errstate(over="ignore", invalid="ignore"):
                a, b = a * m00 + b * m10, a * m01 + b * m11
                m = a + b
            hit = m <= level
            if not hit.any():
                prev = m
                continue
            mh, ph = m[hit], prev[hit]
            with np.errstate(divide="ignore", invalid="ignore"):
                loglin = (np.log(ph) - np.log(level)) / (np.log(ph) - np.log(mh))
                linear = (ph - level) / (ph - mh)
            hours[idx[hit], k] = h
            interp[idx[hit], k] = (h - 1) + np.where((ph > 0) & (mh > 0), loglin, linear)
            keep = ~hit
            idx, a, b, prev = idx[keep], a[keep], b[keep], m[keep]
            m00, m01, m10, m11 = m00[keep], m01[keep], m10[keep], m11[keep]
        with np.errstate(over="ignore", invalid="ignore"):
            end[idx, k] = a + b
    return (hours, interp, end) if return_end else (hours, interp)


#: Draws, seed and horizon of :func:`switching_ratio_simulation`.
SIM_DRAWS = 50_000
SIM_SEED = 20261009
SIM_MAX_HOURS = 10_000


def switching_ratio_simulation(theta, order, cov, draws: int = SIM_DRAWS,
                               seed: int = SIM_SEED, max_hours: int = SIM_MAX_HOURS,
                               level: float = 0.95) -> dict:
    """A simulated 95% interval for the switching-adjusted half-life ratio.

    ``theta`` is (p[0->0], p[1->0], ar.L1[0], ar.L1[1]) in statsmodels' raw
    regime order, ``order`` the sigma2-ascending permutation and ``cov`` the
    4x4 covariance of ``theta``.  The parameters are drawn from the normal
    distribution the estimate's asymptotic theory gives them (Krinsky and
    Robb, 1986), the two transition probabilities on the logit scale (their
    covariance carried over by the delta method) so that every draw is a
    chain, and every draw goes through the exact, non-linear switching
    half-lives.  The interval is the pair of 2.5% and 97.5% quantiles of
    their ratio, so R_sw itself is not linearised; the draws still rest on
    the first-order normal approximation of the estimate.

    A path that does not halve within ``max_hours`` has an unbounded
    half-life, which makes R_sw infinite (calm) or zero (stressed).  When
    neither path halves the ratio is ordered by the two paths at
    ``max_hours`` (infinite if the calm path is the higher, zero if the
    lower) and is undefined only on a tie.  ``infinite`` is the share of draws
    with an unbounded half-life, ``undefined`` the share left out.
    ``ci95`` is the interval of the ratio of the interpolated crossings,
    ``ci95_hours`` that of the ratio of whole hours, which moves in steps; an
    upper end ``None`` is unbounded.  With a persistent chain and positive
    coefficients a draw has R_sw > 1 exactly when its phi_calm > phi_stressed
    (:func:`chain_eigenvalue`), so the interval lies on the side of unity
    :func:`ratio_wald_test` gives, up to the draws that violate those two
    conditions and simulation error.
    """
    theta = np.asarray(theta, dtype=float)
    V = np.asarray(cov, dtype=float)
    order = np.asarray(order, dtype=int)
    out = {"draws": int(draws), "seed": int(seed), "max_hours": int(max_hours),
           "transform": "logit", "infinite": None, "undefined": None,
           "ci95": [None, None], "ci95_hours": [None, None]}
    p = theta[:2]
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(V))
            and np.all((p > 0) & (p < 1))):
        return out
    # the probabilities on the logit scale: d logit(p) / dp = 1 / (p (1 - p))
    jac = np.concatenate([1.0 / (p * (1.0 - p)), np.ones(theta.size - 2)])
    centre = np.concatenate([np.log(p / (1.0 - p)), theta[2:]])
    w, Q = np.linalg.eigh(jac[:, None] * ((V + V.T) / 2.0) * jac[None, :])
    root = Q * np.sqrt(np.clip(w, 0.0, None))           # root @ root.T == the covariance
    rng = np.random.default_rng(int(seed))
    # Accelerate's BLAS raises spurious FP flags inside this matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sample = centre + rng.standard_normal((int(draws), theta.size)) @ root.T
    p00, p10 = (1.0 / (1.0 + np.exp(-sample[:, i])) for i in range(2))
    stay = np.column_stack([p00, 1.0 - p10])[:, order]
    phi = sample[:, 2:][:, order]
    hours, interp, end = switching_half_lives_batch(stay, phi, max_hours=max_hours,
                                                    return_end=True)
    neither = np.isinf(interp).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = {"ci95": interp[:, 0] / interp[:, 1], "ci95_hours": hours[:, 0] / hours[:, 1]}
    for r in ratios.values():
        r[neither] = np.where(end[neither, 0] > end[neither, 1], np.inf,
                              np.where(end[neither, 0] < end[neither, 1], 0.0, np.nan))
    out["infinite"] = float(np.isinf(interp).any(axis=1).mean())
    out["undefined"] = float(np.isnan(ratios["ci95"]).mean())
    q = [(1.0 - level) / 2.0, (1.0 + level) / 2.0]
    for key, r in ratios.items():
        r = r[~np.isnan(r)]
        if r.size:
            out[key] = [_finite(v) for v in np.quantile(r, q, method="inverted_cdf")]
    return out


# ---------------------------------------------------------------------------
# Interval forms for the half-life ratio
# ---------------------------------------------------------------------------
def fieller_interval(a, b, var_a, var_b, cov_ab, z: float = Z95) -> dict:
    """Fieller's confidence set for a / b from two jointly normal estimates.

    The set is {r : (a - r b)^2 <= z^2 (var_a - 2 r cov_ab + r^2 var_b)}, the
    inversion of a pivot that is exactly normal when a and b are.  In
    :func:`ratio_intervals` a and b are ln|phi_k|, whose variances come from
    the delta method, so there the set is first-order too.  With
    A = b^2 - z^2 var_b it is

    * ``bounded``    [lo, hi]                  when A > 0 (b is significant);
    * ``two_rays``   (-inf, lo] U [hi, inf)    when A < 0 and the quadratic has
      real roots;
    * ``whole_line`` otherwise (also for the measure-zero case A = 0).

    Non-finite inputs give ``undefined``.
    """
    vals = (a, b, var_a, var_b, cov_ab)
    if not all(v is not None and np.isfinite(v) for v in vals):
        return {"shape": "undefined", "bounds": [None, None]}
    z2 = z * z
    qa = b * b - z2 * var_b
    qb = -2.0 * (a * b - z2 * cov_ab)
    qc = a * a - z2 * var_a
    disc = qb * qb - 4.0 * qa * qc
    if qa > 0:
        root = np.sqrt(max(disc, 0.0))
        lo, hi = sorted(((-qb - root) / (2 * qa), (-qb + root) / (2 * qa)))
        return {"shape": "bounded", "bounds": [float(lo), float(hi)]}
    if qa < 0 and disc > 0:
        root = np.sqrt(disc)
        lo, hi = sorted(((-qb - root) / (2 * qa), (-qb + root) / (2 * qa)))
        return {"shape": "two_rays", "bounds": [float(lo), float(hi)]}
    return {"shape": "whole_line", "bounds": [None, None]}


def side_of_unity(shape: str, bounds) -> str:
    """Where a confidence set for a positive ratio lies relative to one.

    Only the positive part of the set is admissible: a half-life ratio cannot
    be negative.  Returns ``above`` or ``below`` when that part lies entirely
    on one side of unity, ``ns`` when it contains unity, and ``undetermined``
    when it is empty, undefined, or straddles unity without containing it
    (two rays (0, lo] U [hi, inf) with lo < 1 < hi).
    """
    if shape == "whole_line":
        return "ns"
    lo, hi = (None, None) if bounds is None else (bounds[0], bounds[1])
    if (shape not in ("bounded", "two_rays") or lo is None or hi is None
            or not (np.isfinite(lo) and np.isfinite(hi))):
        return "undetermined"
    if shape == "bounded":
        if hi <= 0.0:
            return "undetermined"
        if max(lo, 0.0) > 1.0:
            return "above"
        if hi < 1.0:
            return "below"
        return "ns"
    if lo <= 0.0:                       # positive part is [hi, inf)
        return "above" if hi > 1.0 else "ns"
    if lo >= 1.0 or hi <= 1.0:          # (0, lo] U [hi, inf) contains unity
        return "ns"
    return "undetermined"


def _form(shape: str, bounds) -> dict:
    b = [_finite(v) for v in bounds]
    return {"shape": shape, "bounds": b, "significance": side_of_unity(shape, b)}


def ratio_intervals(phi_calm, phi_stressed, cov, z: float = Z95) -> dict:
    """R = tau_calm / tau_stressed with linear, log-scale and Fieller 95% sets.

    ``cov`` is the 2x2 covariance of (phi_calm, phi_stressed).  The linear set
    is the delta method on R, the log-scale set the delta method on ln R,
    R exp(+-z se / R), and Fieller's set treats R as the ratio
    ln|phi_stressed| / ln|phi_calm| of two jointly normal estimates.
    """
    pc, ps = float(phi_calm), float(phi_stressed)
    V = np.asarray(cov, dtype=float)
    r = float(half_life(pc) / half_life(ps))
    lc, ls = np.log(abs(pc)), np.log(abs(ps))
    grad = np.array([-ls / (pc * lc ** 2), 1.0 / (ps * lc)])
    with np.errstate(invalid="ignore"):
        var = float(grad @ V @ grad)
    se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else float("nan")
    out = {"value": _finite(r), "se": _finite(se)}
    if np.isfinite(se):
        out["linear"] = _form("bounded", [r - z * se, r + z * se])
    else:
        out["linear"] = _form("undefined", [None, None])
    if np.isfinite(se) and r > 0:
        with np.errstate(over="ignore"):        # an infinite factor leaves the ends open
            f = float(np.exp(z * se / r))
        out["log"] = _form("bounded", [r / f, r * f])
    else:
        out["log"] = _form("undefined", [None, None])
    fl = fieller_interval(ls, lc, V[1, 1] / ps ** 2, V[0, 0] / pc ** 2,
                          V[0, 1] / (ps * pc), z)
    out["fieller"] = _form(fl["shape"], fl["bounds"])
    return out


def ratio_wald_test(phi_calm, phi_stressed, cov, z: float = Z95) -> dict:
    """Wald test of R = 1, which holds exactly when |phi_calm| = |phi_stressed|.

    R = tau_calm / tau_stressed = ln|phi_stressed| / ln|phi_calm| is one exactly
    when the two coefficients are equal in absolute value.  Given their signs
    (both positive in every fit of the paper) that is a linear restriction,
    phi_calm = phi_stressed (or phi_calm = -phi_stressed for opposite signs),
    so the statistic

        z = (|phi_calm| - |phi_stressed|) / sqrt(g' V g),
        g = (sgn phi_calm, -sgn phi_stressed),

    is exactly normal under normal estimates and needs no delta step.  The
    delta intervals of :func:`ratio_intervals` do need one; the log-scale form
    linearises ln R, which is singular at |phi_stressed| = 1, and excludes
    unity too often when phi_stressed's interval approaches one.

    ``cov`` is the 2x2 covariance of (phi_calm, phi_stressed).  A positive z
    means the calm coefficient is the larger in absolute value, so R > 1.
    ``significance`` is ``above`` or ``below`` when |z| exceeds ``z`` (the 5%
    two-sided critical value by default), ``ns`` when it does not and
    ``undetermined`` when the statistic is not defined; ``p`` is two-sided.
    """
    pc, ps = float(phi_calm), float(phi_stressed)
    V = np.asarray(cov, dtype=float)
    sc = 1.0 if pc >= 0 else -1.0
    ss = 1.0 if ps >= 0 else -1.0
    g = np.array([sc, -ss])
    diff = abs(pc) - abs(ps)
    with np.errstate(invalid="ignore"):
        var = float(g @ V @ g)
    se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = float(np.float64(diff) / np.float64(se))
    if np.isnan(stat):
        p, side = None, "undetermined"
    else:
        p = math.erfc(abs(stat) / math.sqrt(2.0))
        side = "above" if stat > z else "below" if stat < -z else "ns"
    return {
        "restriction": "phi_calm = phi_stressed" if sc == ss else "phi_calm = -phi_stressed",
        "difference": _finite(diff),
        "se": _finite(se),
        "z": _finite(stat),
        "p": p,
        "significance": side,
    }


def switching_ratio_side(wald_side: str, lam, lam_se, phi, z: float = Z95) -> str:
    """The side of unity of the switching-adjusted ratio R_sw.

    phi_calm = phi_stressed makes R_sw = 1 whatever the chain, so where the
    Wald test of :func:`ratio_wald_test` does not reject (``ns``) neither does
    R_sw.  Where it does, the side carries over when the chain is persistent,
    lambda = p_cc + p_ss - 1 > 0 (:func:`chain_eigenvalue`), and both
    coefficients are positive: R_sw - 1 then has the sign of phi_calm -
    phi_stressed, as R - 1 has.  The persistence must itself be significant,
    lambda > z se(lambda), so that both conditions are established at the
    same level (an intersection-union test); otherwise the side is
    ``undetermined``.
    """
    if wald_side == "ns":
        return "ns"
    if wald_side not in ("above", "below"):
        return "undetermined"
    if lam is None or lam_se is None or not (np.isfinite(lam) and np.isfinite(lam_se)):
        return "undetermined"
    if lam - z * lam_se <= 0 or not np.all(np.asarray(phi, dtype=float) > 0):
        return "undetermined"
    return wald_side


# ---------------------------------------------------------------------------
# Covariances and the switching-aware inference block
# ---------------------------------------------------------------------------
def msar_covariances(res) -> dict:
    """The Hessian, OPG and sandwich covariance matrices of a statsmodels fit.

    These are the matrices behind ``cov_type='approx'``, ``'opg'`` and
    ``'robust'`` (``COV_TYPES``), read from the fitted results object without
    smoothing again.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return {
            "hessian": np.asarray(res.cov_params_approx, dtype=float),
            "opg": np.asarray(res.cov_params_opg, dtype=float),
            "robust": np.asarray(res.cov_params_robust, dtype=float),
        }


def _sqrt_diag(cov, idx) -> np.ndarray:
    d = np.diag(cov)[idx]
    return np.where(np.isfinite(d) & (d >= 0), np.sqrt(np.abs(d)), np.nan)


def msar_robust_inference(fit: dict, z: float = Z95) -> dict:
    """Chain summaries, switching-adjusted half-lives and robust ratio inference.

    ``fit`` is the dict of :func:`fit_msar_premium` / :func:`summarise_msar_fit`.
    Every per-regime list is ordered (calm, stressed).  Under each covariance in
    ``inference`` the block gives the standard errors of phi_k and of the
    staying probabilities p_kk, the t-ratios of phi_k against unity, the
    half-life interval obtained by transforming phi_k's Wald interval (upper
    end ``None`` when that interval reaches a unit root), and the standard
    error and three 95% sets for R = tau_calm / tau_stressed, each with the
    side of unity it lies on, and delta-method standard errors and intervals
    for the interpolated switching-adjusted half-lives and their ratio.

    The verdict on R's side of unity is ``ratio_wald``, the Wald test of
    phi_calm = phi_stressed (:func:`ratio_wald_test`); the interval forms are
    descriptive.  ``ratio_switching_side`` carries that verdict over to the
    switching-adjusted ratio where the chain is significantly persistent
    (``chain_eigenvalue`` = p_cc + p_ss - 1 and ``chain_eigenvalue_se``; see
    :func:`switching_ratio_side`), and ``ratio_switching_sim`` is a simulated
    95% interval for that ratio (:func:`switching_ratio_simulation`).
    """
    res = fit["result"]
    names = list(fit["param_names"])
    order = np.asarray(fit["order"], dtype=int)
    if order.size != 2:
        raise ValueError("msar_robust_inference is defined for two regimes")
    params = np.asarray(res.params, dtype=float)

    P = row_stochastic_transition(res.regime_transition, order)
    p_stay = np.diag(P)
    durations = expected_durations(P)
    pi = stationary_distribution(P)
    lam = chain_eigenvalue(P)

    i_phi = [names.index(f"ar.L1[{order[k]}]") for k in range(2)]
    # staying probability of raw regime 0 is p[0->0]; of raw regime 1, 1 - p[1->0]
    i_stay = [names.index("p[0->0]") if order[k] == 0 else names.index("p[1->0]")
              for k in range(2)]
    phi = params[i_phi]

    sw = [switching_half_life(P, phi, start=k) for k in range(2)]
    hours = [s["hours"] for s in sw]
    interp = [s["interpolated"] for s in sw]

    def ratio_of(a, b):
        return _finite(a / b) if a is not None and b is not None and b > 0 else None

    i_raw = [names.index(n) for n in ("p[0->0]", "p[1->0]", "ar.L1[0]", "ar.L1[1]")]

    inference = {}
    for name, cov in msar_covariances(res).items():
        se_phi = _sqrt_diag(cov, i_phi)
        se_p = _sqrt_diag(cov, i_stay)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_unit = (phi - 1.0) / se_phi
            se_dur = se_p / (1.0 - p_stay) ** 2
        lo, hi = phi - z * se_phi, phi + z * se_phi
        hl_ci = []
        for k in range(2):
            lower = 0.0 if lo[k] <= 0 else float(half_life(lo[k]))
            upper = None if not np.isfinite(hi[k]) or hi[k] >= 1.0 else float(half_life(hi[k]))
            hl_ci.append([_finite(lower) if np.isfinite(lo[k]) else None, _finite(upper)])
        ratio = ratio_intervals(phi[0], phi[1], cov[np.ix_(i_phi, i_phi)], z)
        wald = ratio_wald_test(phi[0], phi[1], cov[np.ix_(i_phi, i_phi)], z)
        v_raw = cov[np.ix_(i_raw, i_raw)]
        # lambda = p[0->0] - p[1->0] in the raw parameters: linear, so its
        # standard error is exact
        with np.errstate(invalid="ignore"):
            lam_var = float(v_raw[0, 0] + v_raw[1, 1] - 2.0 * v_raw[0, 1])
        lam_se = math.sqrt(lam_var) if np.isfinite(lam_var) and lam_var >= 0 else None
        inference[name] = {
            "phi_se": [_finite(v) for v in se_phi],
            "phi_t_unit": [_finite(v) for v in t_unit],
            "phi_ci95": [[_finite(lo[k]), _finite(hi[k])] for k in range(2)],
            "half_life_ci95_from_phi": hl_ci,
            "p_stay_se": [_finite(v) for v in se_p],
            "expected_duration_se": [_finite(v) for v in se_dur],
            "ratio_se": ratio["se"],
            "ratio_ci95": {form: ratio[form] for form in ("linear", "log", "fieller")},
            "ratio_wald": wald,
            "chain_eigenvalue_se": _finite(lam_se),
        }
        sw_delta = switching_delta(params[i_raw], order, v_raw, z)
        inference[name].update({
            "half_life_switching_interp_se": sw_delta["half_life_se"],
            "ratio_switching_interp_se": sw_delta["ratio_se"],
            "ratio_switching_interp_ci95": sw_delta["ratio_ci95"],
            "ratio_switching_side": switching_ratio_side(wald["significance"], lam, lam_se,
                                                         phi, z),
            "ratio_switching_sim": switching_ratio_simulation(params[i_raw], order, v_raw),
        })

    return {
        "p_stay": [_finite(v) for v in p_stay],
        "expected_duration_hours": [_finite(v) for v in durations],
        "stationary_share": [_finite(v) for v in pi],
        "chain_eigenvalue": _finite(lam),
        "half_life_switching_hours": hours,
        "half_life_switching_interp_hours": [_finite(v) for v in interp],
        "ratio_switching": ratio_of(hours[0], hours[1]),
        "ratio_switching_interp": ratio_of(interp[0], interp[1]),
        "inference": inference,
    }


def ratio_sandwich_influence(fit: dict, top: int = 3) -> dict:
    """Each hour's share of the sandwich variance of R = tau_calm / tau_stressed.

    The sandwich variance of a smooth function of the parameters is
    sum_t (s_t' H^-1 g)^2, with s_t the score of observation t, H the Hessian
    and g the gradient of R, so it splits exactly into one non-negative
    contribution per hour.  Returns that variance, every hour's share
    (``shares``, aligned to the scores, i.e. to y[1:]) and the ``top`` hours,
    each with its index into the fitted series, the observation, its
    predecessor and its share.
    """
    model, res = fit["model"], fit["result"]
    names = list(fit["param_names"])
    order = np.asarray(fit["order"], dtype=int)
    params = np.asarray(res.params, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        H = np.asarray(model.hessian(params, transformed=True), dtype=float)
        S = np.asarray(model.score_obs(params, transformed=True), dtype=float)
    i_c = names.index(f"ar.L1[{order[0]}]")
    i_s = names.index(f"ar.L1[{order[1]}]")
    pc, ps = params[i_c], params[i_s]
    lc, ls = np.log(abs(pc)), np.log(abs(ps))
    g = np.zeros(params.size)
    g[i_c] = -ls / (pc * lc ** 2)
    g[i_s] = 1.0 / (ps * lc)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        contrib = (S @ (np.linalg.pinv(H) @ g)) ** 2
    variance = float(contrib.sum())
    shares = contrib / variance
    y = np.asarray(model.orig_endog, dtype=float).ravel()
    lag = int(model.order)
    best = np.argsort(-contrib)[:int(top)]
    return {
        "variance": _finite(variance),
        "shares": [float(v) for v in shares],
        "top": [{"index": int(i) + lag, "y": float(y[i + lag]), "y_prev": float(y[i + lag - 1]),
                 "share": float(shares[i])} for i in best],
    }
