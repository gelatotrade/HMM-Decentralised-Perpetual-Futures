"""
Inference on a transition probability estimated on the boundary.

The K = 3 Gaussian HMM of the paper estimates the hourly probability of a
direct move from the high- to the low-volatility state, A_31, at zero.  This
module supplies the pieces that turn that zero into a statement with known
properties:

* :func:`profile_likelihood` and :func:`profile_upper_bound` -- the profile
  log-likelihood of one transition cell, maximised over every other parameter
  by constrained EM, and the upper confidence bound where the likelihood-ratio
  statistic crosses a critical value;
* :func:`boundary_decision` -- whether the maximum-likelihood estimate of the
  cell is exactly zero, decided by the Kuhn--Tucker condition at the
  constrained maximum rather than by how far EM happened to drive the cell
  before it stopped;
* :func:`simulate_garch11` -- paths from a GARCH(1,1) with Gaussian or
  standardised Student-t innovations, for asking how often a Gaussian HMM
  fitted to a single-regime volatility process puts the cell on the boundary;
* :func:`binomial_share` -- a share with its binomial standard error.

States are labelled sigma-ascending throughout (index 0 low, K-1 high), so the
high-to-low cell of a K = 3 model is ``(2, 0)``.
"""
from __future__ import annotations

import numpy as np

from .hmm_batch import BatchParams, boundary_score_ratio, loglik, move_cell_mass, run_em

__all__ = [
    "CHI2_1_95",
    "CHIBAR_95",
    "HIGH_TO_LOW",
    "binomial_share",
    "boundary_decision",
    "cell_mle",
    "fit_converged",
    "profile_likelihood",
    "profile_upper_bound",
    "simulate_garch11",
]

#: 95% critical value of the signed-root likelihood-ratio test of a one-sided
#: hypothesis, z_{0.95}^2; it is also the 95% quantile of the 50:50 mixture of
#: chi2_0 and chi2_1 that the statistic follows when the true value is on the
#: boundary (Chernoff 1954; Self and Liang 1987).
CHIBAR_95 = 2.705543454095404

#: 95% quantile of chi2_1: the two-sided interval, a one-sided 97.5% bound.
CHI2_1_95 = 3.841458820694124

HIGH_TO_LOW = (2, 0)

CONV_TOL = 1e-10          # |delta logL| at which a fit counts as converged
CONV_CAP = 20000


def _as_batch(params, R: int) -> BatchParams:
    if isinstance(params, BatchParams):
        return params.copy()
    return BatchParams.stack([params] * R)


def fit_converged(x, init, *, fixed_cell=None, fixed_value=None,
                  tol: float = CONV_TOL, n_iter: int = CONV_CAP):
    """EM from ``init`` until |delta logL| < ``tol``; returns the ``EMResult``.

    ``init`` is a :class:`BatchParams` or a single parameter dict, which is then
    replicated once per entry of ``fixed_value``.  With ``fixed_cell`` the cell
    is first set to ``fixed_value`` (rescaling the rest of its row) and held
    there.
    """
    if fixed_cell is not None:
        fv = np.atleast_1d(np.asarray(fixed_value, dtype=float))
        p = _as_batch(init, fv.size)
        p.A = move_cell_mass(p.A, fixed_cell, fv)
        return run_em(x, p, n_iter, tol, fixed_cell=fixed_cell, fixed_value=fv,
                      rule="abs")
    return run_em(x, _as_batch(init, 1), n_iter, tol, rule="abs")


def boundary_decision(x, init, cell=HIGH_TO_LOW, *, tol: float = CONV_TOL,
                      n_iter: int = CONV_CAP) -> dict:
    """Is the maximum-likelihood estimate of ``A[cell]`` exactly zero?

    Maximises the likelihood with the cell held at zero, starting from
    ``init`` (a batch: one decision per member), and evaluates the score ratio
    of :func:`regime_engine.hmm_batch.boundary_score_ratio` there.  A ratio of
    at most one means the constrained maximum is also a local maximum of the
    unconstrained problem, so the estimate is zero; above one the likelihood
    rises as soon as the cell leaves zero and the estimate is interior.
    """
    p = _as_batch(init, 1)
    zero = np.zeros(p.R)
    p.A = move_cell_mass(p.A, cell, zero)
    res = run_em(x, p, n_iter, tol, fixed_cell=cell, fixed_value=zero, rule="abs")
    ratio = boundary_score_ratio(x, res.params, cell)
    return {"ratio": ratio, "at_boundary": ratio <= 1.0, "ll0": res.ll,
            "params0": res.params, "n_iter": res.n_iter, "stopped": res.stopped}


def cell_mle(x, init, cell=HIGH_TO_LOW, *, seed_value: float = 1e-3,
             tol: float = CONV_TOL, n_iter: int = CONV_CAP) -> dict:
    """Maximum-likelihood estimate of ``A[cell]`` for every member of a batch.

    ``init`` should be close to the maximum (a converged EM fit, states sorted
    by sigma).  The boundary decision comes first.  Where the estimate is
    interior, EM is restarted from the constrained maximum with the cell moved
    to ``seed_value``: EM cannot leave an exact zero, because the expected
    count of a transition with probability zero is zero, so a fit started at
    zero stays there whatever the data say.  Returns the estimate ``a_hat``,
    the maximised log-likelihood ``ll_max`` (the larger of the two maxima),
    the constrained maximum ``ll0`` and the parameters at the estimate.
    """
    bd = boundary_decision(x, init, cell, tol=tol, n_iter=n_iter)
    p0 = bd["params0"]
    a_hat = np.zeros(p0.R)
    ll_max = np.array(bd["ll0"], dtype=float)
    params = p0.copy()
    inner = np.nonzero(~bd["at_boundary"])[0]
    if inner.size:
        q = p0.take(inner)
        q.A = move_cell_mass(q.A, cell, np.full(inner.size, seed_value))
        xb = np.asarray(x, dtype=float)
        xs = xb if xb.ndim == 1 else xb[inner]
        res = run_em(xs, q, n_iter, tol, rule="abs")
        ll_in = loglik(xs, res.params)
        better = ll_in > ll_max[inner]
        idx = inner[better]
        a_hat[idx] = res.params.A[better, cell[0], cell[1]]
        ll_max[idx] = ll_in[better]
        params.put(idx, res.params.take(np.nonzero(better)[0]))
    return {"a_hat": a_hat, "ll_max": ll_max, "ll0": np.asarray(bd["ll0"]),
            "ratio": bd["ratio"], "at_boundary": bd["at_boundary"], "params": params,
            "params0": p0}


def profile_likelihood(x, mle, values, cell=HIGH_TO_LOW, *, tol: float = CONV_TOL,
                       n_iter: int = CONV_CAP) -> dict:
    """Profile log-likelihood of ``A[cell]`` at each of ``values``.

    Each value is a constrained EM run started from ``mle`` (a parameter dict)
    with the cell moved to the value.  Starting every run at the unconstrained
    maximum follows the profile continuously from it; a multi-start check of
    the profile at the bound is left to the caller.
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    res = fit_converged(x, mle, fixed_cell=cell, fixed_value=values, tol=tol,
                        n_iter=n_iter)
    return {"values": values, "ll": res.ll, "n_iter": res.n_iter,
            "stopped": res.stopped, "params": res.params}


def profile_upper_bound(x, mle, ll_max: float, crit, cell=HIGH_TO_LOW, *,
                        a_hat: float = 0.0, grid=None, rounds: int = 3,
                        points_per_round: int = 8, tol: float = CONV_TOL,
                        n_iter: int = CONV_CAP) -> dict:
    """Upper confidence bound where LR(a) = 2 (ll_max - ll(a)) first reaches ``crit``.

    A grid above ``a_hat`` brackets each crossing; each bracket is then refined
    ``rounds`` times by evaluating ``points_per_round`` equally spaced interior
    points, and the bound is the linear interpolation of LR inside the final
    bracket.  Several critical values are handled in one batch.
    """
    crit = np.atleast_1d(np.asarray(crit, dtype=float))
    if grid is None:
        grid = a_hat + np.array([0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
                                 0.008, 0.010, 0.0125, 0.015, 0.02, 0.03, 0.05])
    grid = np.asarray(grid, dtype=float)
    prof = profile_likelihood(x, mle, grid, cell, tol=tol, n_iter=n_iter)
    a_all = np.r_[a_hat, grid]
    lr_all = np.r_[0.0, 2.0 * (ll_max - prof["ll"])]
    evaluated = {float(a): float(v) for a, v in zip(a_all, lr_all)}

    def bracket(c):
        a_sorted = np.array(sorted(evaluated))
        lr = np.array([evaluated[a] for a in a_sorted])
        hit = np.nonzero(lr >= c)[0]
        if hit.size == 0:
            return None
        k = int(hit[0])
        return (float(a_sorted[max(k - 1, 0)]), float(a_sorted[k]))

    for _ in range(rounds):
        pts = []
        for c in crit:
            br = bracket(c)
            if br is None or br[1] - br[0] <= 0:
                continue
            pts.extend(np.linspace(br[0], br[1], points_per_round + 2)[1:-1])
        pts = [p for p in pts if float(p) not in evaluated]
        if not pts:
            break
        pr = profile_likelihood(x, mle, pts, cell, tol=tol, n_iter=n_iter)
        for a, ll in zip(pr["values"], pr["ll"]):
            evaluated[float(a)] = float(2.0 * (ll_max - ll))

    bounds = []
    for c in crit:
        br = bracket(c)
        if br is None:
            bounds.append(None)
            continue
        lo, hi = br
        f_lo, f_hi = evaluated[lo], evaluated[hi]
        bounds.append(float(lo + (c - f_lo) * (hi - lo) / (f_hi - f_lo))
                      if f_hi > f_lo else float(hi))
    a_sorted = sorted(evaluated)
    return {
        "crit": crit.tolist(),
        "bound": bounds,
        "grid_a": [float(a) for a in a_sorted],
        "grid_LR": [evaluated[a] for a in a_sorted],
        "grid_n_iter_max": int(np.max(prof["n_iter"])),
        "grid_all_stopped_by_tol": bool(np.all(prof["stopped"])),
    }


def simulate_garch11(mu: float, omega: float, alpha: float, beta: float, T: int,
                     rng: np.random.Generator, nu: float | None = None,
                     h0: float | None = None) -> np.ndarray:
    """r_t = mu + sqrt(h_t) z_t with h_t = omega + alpha (r_{t-1} - mu)^2 + beta h_{t-1}.

    ``z_t`` is standard normal or, with ``nu``, Student-t scaled to unit
    variance.  ``h0`` initialises the recursion -- the estimation code backcasts
    with the sample variance, and the caller passes the same here -- and
    defaults to the unconditional variance, which requires alpha + beta < 1.
    """
    if nu is None:
        z = rng.standard_normal(T)
    else:
        z = rng.standard_t(nu, size=T) * np.sqrt((nu - 2.0) / nu)
    if h0 is None:
        h0 = omega / (1.0 - alpha - beta)
    r = np.empty(T)
    h = float(h0)
    for t in range(T):
        if t > 0:
            e = r[t - 1] - mu
            h = omega + alpha * e * e + beta * h
        r[t] = mu + np.sqrt(h) * z[t]
    return r


def binomial_share(flags) -> dict:
    """Share of True in ``flags`` with its binomial standard error."""
    f = np.asarray(flags, dtype=bool)
    n = int(f.size)
    k = int(f.sum())
    p = k / n if n else float("nan")
    se = float(np.sqrt(p * (1.0 - p) / n)) if n else float("nan")
    return {"k": k, "n": n, "share": float(p), "se": se}
