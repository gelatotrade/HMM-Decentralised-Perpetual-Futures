"""Inference, residual diagnostics and out-of-family benchmarks.

Three analyses that accompany the fits reported in the paper:

1. **Inference on the return HMM.**  ``bootstrap_hmm`` supplies standard errors
   and percentile intervals for the regime parameters
   (``paper/tables/table2_regime_params.tex``) by *parametric* bootstrap.
   This is the form of inference available for the paper's headline
   full-sample fit: a parametric bootstrap conditions on the fitted
   parameters and the sample length, never on the data, and the full-sample
   parameters survive in
   ``output/summary_FULLSAMPLE_2025-10-02_to_2026-04-28.json`` even though
   the full-sample returns do not.

2. **Residual diagnostics.**  The Gaussian HMM assumes conditional independence
   given the state; that is testable and it is not free.  ``residual_battery``
   runs Ljung--Box on the standardised residuals and their squares, ARCH-LM and
   Jarque--Bera, on the HMM residuals, on the raw returns (so the reader can see
   how much clustering the HMM absorbs) and on the MS-AR(1) residuals.

3. **Out-of-family benchmarks.**  ``run_benchmarks`` compares the HMMs with
   models outside their family: it fits a single Gaussian, a Gaussian
   GARCH(1,1) and a Student-*t* GARCH(1,1) on *identical* data in *identical*
   units and reports logL / k / AIC / BIC on one scale.

Everything writes to ``output/``:  ``bootstrap_hmm.json``, ``diagnostics.json``,
``benchmarks.json``, ``tables/table6_diagnostics.tex``, ``tables/table7_benchmarks.tex``.

Usage
-----
    PYTHONPATH=src python3 -m regime_engine.diagnostics --all
    PYTHONPATH=src python3 -m regime_engine.diagnostics --bootstrap --reps 500

Author: Gregor Albiez, 2026
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

from .hmm import GaussianHMM
from .paths import DEFAULT_OUT, DEFAULT_PANEL

__all__ = [
    "ANNUALISATION",
    "arch_lm",
    "bootstrap_hmm",
    "fit_garch11",
    "fit_single_gaussian",
    "garch11_variance",
    "hmm_predictive_moments",
    "hmm_standardised_residuals",
    "jarque_bera",
    "ljung_box",
    "msar_standardised_residuals",
    "reconstruct_transition_matrix",
    "residual_battery",
    "run_benchmarks",
    "run_bootstrap",
    "run_diagnostics",
    "simulate_gaussian_hmm",
]

# sqrt(hours per year) -- the paper's annualisation of an hourly sigma
ANNUALISATION = np.sqrt(24 * 365)

FULLSAMPLE_JSON = "summary_FULLSAMPLE_2025-10-02_to_2026-04-28.json"

# Refit hyper-parameters for a bootstrap replication.  ``n_iter`` is generous
# because the EM is started AT the data-generating parameters and we want the
# stopping rule, not the iteration budget, to bind.
BOOT_N_ITER = 400
BOOT_TOL = 1e-6

# Hyper-parameters of the paper's own K=3 fit.  Duplicated from analysis.py
# rather than imported, because importing analysis pulls in matplotlib and every
# bootstrap worker process would pay for it;
# ``tests/test_diagnostics.py::test_paper_fit_hyperparameters_match_analysis``
# fails the moment the two drift apart.
HMM_SEED_BASE = 42
HMM_N_STARTS = 8
HMM_N_ITER = 150


# ==========================================================================
#  0.  Small shared helpers
# ==========================================================================
def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    # bool BEFORE int: bool is a subclass of int in Python, so testing int first
    # silently writes `true` out as `1`.
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _stationary(A: np.ndarray) -> np.ndarray:
    """Left eigenvector of ``A`` for eigenvalue 1, normalised to a distribution."""
    evals, evecs = np.linalg.eig(np.asarray(A, dtype=float).T)
    v = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
    return v / v.sum()


def reconstruct_transition_matrix(diag_A, stationary, zero_cell) -> np.ndarray:
    """Recover a full 3x3 transition matrix from what the archive actually kept.

    ``output/summary_FULLSAMPLE_*.json`` stores only ``diag_A`` and the
    stationary distribution -- the full kernel of the paper's headline fit is
    not stored.  For K=3 that is nevertheless enough,
    because the unknowns are exactly identified:

      * each row has two off-diagonal cells whose sum is fixed at ``1 - A_kk``,
        leaving one free split per row, i.e. three unknowns;
      * ``pi A = pi`` supplies two independent linear equations;
      * the boundary estimate ``A[zero_cell] = 0`` supplies the third.

    ``zero_cell`` is the ``(i, j)`` index of the cell the fit put on the
    boundary.  It is a statement about the *stored* array, not about the
    estimator: the arrays in ``summary_FULLSAMPLE_*.json`` are in mu-ascending
    order, which puts the high-to-low transition at ``(0, 1)``.  Artifacts
    written by the current code are sigma-ascending (``hmm.fit``), which puts
    the same transition at ``(2, 0)``.  Callers must pass the cell that
    matches the array they hand in; locating it as ``argmin`` is the
    convention-independent way to do that.

    The construction is exact, not approximate: ``tests/test_diagnostics.py``
    checks that it reproduces the *archived* sub-sample kernel (which was
    persisted in full) to machine precision from its diagonal and stationary
    distribution alone.
    """
    d = np.asarray(diag_A, dtype=float)
    pi = np.asarray(stationary, dtype=float)
    if d.size != 3 or pi.size != 3:
        raise ValueError("reconstruct_transition_matrix is only defined for K=3")
    zi, zj = (int(zero_cell[0]), int(zero_cell[1]))
    # The algebra below is written for an arbitrary boundary cell, because the
    # cell depends on the labelling: under a mu-ascending rule the
    # high-volatility state is index 0, under the sigma-ascending rule the
    # paper adopts it is index 2.
    if not (0 <= zi < 3 and 0 <= zj < 3):
        raise ValueError(f"zero_cell {zero_cell} is out of range for K=3")
    if zi == zj:
        raise ValueError("the boundary cell cannot be on the diagonal")

    A = np.zeros((3, 3))
    np.fill_diagonal(A, d)
    other = 3 - zi - zj                      # the remaining column of row zi
    A[zi, zj] = 0.0
    A[zi, other] = 1.0 - d[zi]

    # Column equation for column `other`:  pi_other = sum_i pi_i A[i, other].
    # Row `other` contributes its diagonal; row zi contributes A[zi, other];
    # the third row -- call it `third` -- is the only unknown.
    third = 3 - zi - other
    A[third, other] = (pi[other] * (1.0 - d[other]) - pi[zi] * A[zi, other]) / pi[third]
    A[third, zi] = (1.0 - d[third]) - A[third, other]
    # Column equation for column zi closes row `other`.
    A[other, zi] = (pi[zi] * (1.0 - d[zi]) - pi[third] * A[third, zi]) / pi[other]
    A[other, third] = (1.0 - d[other]) - A[other, zi]
    return A


# ==========================================================================
#  1.  Parametric bootstrap for the Gaussian HMM
# ==========================================================================
def simulate_gaussian_hmm(A, mu, sigma, T: int, rng, pi0=None):
    """Draw a state path from ``A`` and emit r_t ~ N(mu_{S_t}, sigma_{S_t}^2).

    The path starts from ``pi0``, defaulting to the *stationary* distribution of
    ``A``.  Starting from the fitted initial distribution ``pi`` would be wrong
    here: Baum--Welch on a single sequence drives ``pi`` to a one-hot vector
    (it is estimated from one observation), so it carries no information and
    would put every replication in the same opening state.
    """
    A = np.asarray(A, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    K = mu.size
    p0 = _stationary(A) if pi0 is None else np.asarray(pi0, dtype=float)

    cum_A = np.cumsum(A, axis=1)
    u = rng.random(T)
    states = np.empty(T, dtype=int)
    states[0] = int(np.searchsorted(np.cumsum(p0), u[0]))
    for t in range(1, T):
        states[t] = int(np.searchsorted(cum_A[states[t - 1]], u[t]))
    states = np.clip(states, 0, K - 1)
    x = rng.normal(mu[states], sigma[states])
    return x, states


def _em_from(x, init, n_iter=BOOT_N_ITER, tol=BOOT_TOL):
    """Run Baum--Welch from a *given* starting point, using the paper's own E/M steps.

    A parametric bootstrap conventionally initialises each refit at the
    data-generating parameters rather than repeating the multi-start search:
    the simulated sample is drawn from that point, so it is the natural (and by
    far the cheapest) start.  ``bootstrap_hmm(multistart_check=n)`` verifies on
    a subset of replications that this does not bias the answer.
    """
    hmm = GaussianHMM(K=len(init["mu"]), n_iter=n_iter, tol=tol)
    hmm.pi = np.asarray(init["pi"], dtype=float).copy()
    hmm.A = np.asarray(init["A"], dtype=float).copy()
    hmm.mu = np.asarray(init["mu"], dtype=float).copy()
    hmm.sigma = np.asarray(init["sigma"], dtype=float).copy()

    prev = -np.inf
    ll = -np.inf
    used = 0
    for it in range(n_iter):
        gamma, xi_sum, ll = hmm._e_step(x)
        hmm._m_step(x, gamma, xi_sum)
        used = it + 1
        if ll - prev < tol and it > 5:
            break
        prev = ll
    # Report the likelihood OF THE RETURNED PARAMETERS, not of the parameters
    # one M-step earlier (which is what GaussianHMM.fit stores).  The gap is
    # bounded by `tol` but it is free to remove.
    _, _, ll = hmm._e_step(x)
    hmm.log_likelihood_ = float(ll)
    return hmm, used


def _order_by(mu, sigma, A, by: str):
    """Permutation that sorts the states by ``mu`` or by ``sigma``, ascending."""
    key = mu if by == "mu" else sigma
    order = np.argsort(key)
    return order, mu[order], sigma[order], A[np.ix_(order, order)]


def _sigma_rank_signature(mu, sigma) -> tuple:
    """Rank of each sigma once the states are put in mu-ascending order.

    This is the object that makes label switching measurable.  Two labelling
    rules -- "sort by mu" and "sort by sigma" -- induce the *same* map from
    fitted components to indices if and only if this signature is the same.
    On the paper's full-sample fit the signature is (2, 0, 1): the lowest-mu
    state has the largest sigma, the middle-mu state the smallest.  Any
    replication with a different signature is one where a mu-ascending rule has
    assigned the indices differently from the sigma-ascending rule the paper
    adopts.
    """
    order = np.argsort(mu)
    s = np.asarray(sigma)[order]
    return tuple(int(r) for r in np.argsort(np.argsort(s)))


def _summarise(samples: np.ndarray, point: float) -> dict:
    """bootstrap mean / bias / SE / 2.5-97.5 percentile interval for one scalar."""
    a = np.asarray(samples, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {"point": float(point), "boot_mean": None, "bias": None,
                "se": None, "ci95": [None, None], "n_finite": 0}
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "point": float(point),
        "boot_mean": float(finite.mean()),
        "bias": float(finite.mean() - point),
        "se": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "ci95": [float(lo), float(hi)],
        "n_finite": int(finite.size),
    }


def _rep_worker(args):
    """One bootstrap replication.  Module-level so it survives `spawn` pickling."""
    (seed, A, mu, sigma, T, multistart, n_starts, n_iter_ms) = args
    rng = np.random.default_rng(seed)
    A = np.asarray(A)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    x, states = simulate_gaussian_hmm(A, mu, sigma, T, rng)

    init = {"pi": _stationary(A), "A": A, "mu": mu, "sigma": sigma}
    hmm, n_it = _em_from(x, init)
    out = {
        "seed": int(seed),
        "mu": hmm.mu.copy(),
        "sigma": hmm.sigma.copy(),
        "A": hmm.A.copy(),
        "ll": float(hmm.log_likelihood_),
        "em_iters": int(n_it),
        "n_state_hours": np.bincount(states, minlength=len(mu)).astype(float),
    }
    if multistart:
        ms = GaussianHMM(K=len(mu), n_starts=n_starts, n_iter=n_iter_ms,
                         random_state=int(seed) % (2 ** 31)).fit(x)
        # GaussianHMM.fit stores the likelihood of the parameters ONE M-step
        # before the ones it returns (it breaks after recording `ll`, then the
        # M-step has already run). The gap is bounded by `tol`, but comparing it
        # against _em_from's likelihood -- which is evaluated at the returned
        # parameters -- would be an apples-to-oranges comparison, so re-evaluate.
        _, _, ms_ll = ms._e_step(x)
        out["ms_mu"] = ms.mu.copy()
        out["ms_sigma"] = ms.sigma.copy()
        out["ms_A"] = ms.A.copy()
        out["ms_ll"] = float(ms_ll)
        out["ms_ll_as_reported_by_fit"] = float(ms.log_likelihood_)
    return out


def bootstrap_hmm(params: dict, T: int, n_reps: int = 500, seed: int = 20260819,
                  n_jobs: int | None = None, multistart_check: int = 0,
                  ms_n_starts: int = HMM_N_STARTS, ms_n_iter: int = HMM_N_ITER,
                  label: str = "", progress: bool = True) -> dict:
    """Parametric bootstrap of a fitted K-state Gaussian HMM.

    Parameters
    ----------
    params : dict with keys ``mu``, ``sigma`` (raw log-return units) and ``A``.
    T : int
        Sample length to simulate -- the length of the sample the fit came from.
    n_reps : int
        Number of replications.  500 for the reported run.
    multistart_check : int
        Additionally refit the first ``multistart_check`` replications with the
        full multi-start search, so the "initialise at the truth" shortcut can
        be shown not to bias the result.

    Returns
    -------
    dict with, for every mu_k, sigma_k, annualised sigma_k, A_ij, E[tau_k] and
    stationary pi_k: the bootstrap mean, bias, standard error and 2.5/97.5
    percentiles -- under a mu-ascending labelling *and* under the paper's
    sigma-ascending labelling, plus an explicit label-switching diagnostic.
    """
    mu = np.asarray(params["mu"], dtype=float)
    sigma = np.asarray(params["sigma"], dtype=float)
    A = np.asarray(params["A"], dtype=float)
    K = mu.size
    pi_stat = _stationary(A)

    ss = np.random.SeedSequence(seed)
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(n_reps)]
    jobs = [(seeds[i], A, mu, sigma, T, i < multistart_check, ms_n_starts, ms_n_iter)
            for i in range(n_reps)]

    t0 = time.time()
    if n_jobs is None:
        import os
        n_jobs = max(1, min(8, (os.cpu_count() or 2) - 2))
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            reps = list(ex.map(_rep_worker, jobs, chunksize=4))
    else:
        reps = [_rep_worker(j) for j in jobs]
    elapsed = time.time() - t0
    if progress:
        print(f"[bootstrap{' ' + label if label else ''}] {n_reps} reps, T={T}, "
              f"{n_jobs} workers, {elapsed:.1f}s "
              f"({elapsed / max(1, n_reps):.2f}s/rep)")

    # ---- collect, under two labelling rules -------------------------------
    truth_sig = _sigma_rank_signature(mu, sigma)
    store = {rule: {"mu": [], "sigma": [], "A": []} for rule in ("mu", "sigma")}
    signatures = []
    lls, iters, n_state1 = [], [], []
    for rep in reps:
        for rule in ("mu", "sigma"):
            _, m, s, a = _order_by(rep["mu"], rep["sigma"], rep["A"], rule)
            store[rule]["mu"].append(m)
            store[rule]["sigma"].append(s)
            store[rule]["A"].append(a)
        signatures.append(_sigma_rank_signature(rep["mu"], rep["sigma"]))
        lls.append(rep["ll"])
        iters.append(rep["em_iters"])
        n_state1.append(rep["n_state_hours"][0])

    # Point estimates in each labelling, for the bias column.
    _, mu_mu, sig_mu, A_mu = _order_by(mu, sigma, A, "mu")
    _, mu_sg, sig_sg, A_sg = _order_by(mu, sigma, A, "sigma")
    point = {"mu": (mu_mu, sig_mu, A_mu), "sigma": (mu_sg, sig_sg, A_sg)}

    def _block(rule):
        M = np.array(store[rule]["mu"])
        S = np.array(store[rule]["sigma"])
        AA = np.array(store[rule]["A"])
        p_mu, p_sig, p_A = point[rule]
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = 1.0 / np.maximum(1.0 - np.einsum("rkk->rk", AA), 1e-12)
            p_tau = 1.0 / np.maximum(1.0 - np.diag(p_A), 1e-12)
        stat = np.array([_stationary(a) for a in AA])
        p_stat = _stationary(p_A)
        return {
            "mu_bps_per_hour": [_summarise(M[:, k] * 1e4, p_mu[k] * 1e4) for k in range(K)],
            "sigma_bps_per_hour": [_summarise(S[:, k] * 1e4, p_sig[k] * 1e4) for k in range(K)],
            "annualised_sigma_pct": [
                _summarise(S[:, k] * ANNUALISATION * 100, p_sig[k] * ANNUALISATION * 100)
                for k in range(K)],
            "A": [[_summarise(AA[:, i, j], p_A[i, j]) for j in range(K)] for i in range(K)],
            "expected_duration_hours": [_summarise(tau[:, k], p_tau[k]) for k in range(K)],
            "stationary": [_summarise(stat[:, k], p_stat[k]) for k in range(K)],
        }

    # ---- boundary behaviour of the off-diagonal cells ---------------------
    A_mu_reps = np.array(store["mu"]["A"])
    A_sg_reps = np.array(store["sigma"]["A"])
    zero_frac_mu = [[float(np.mean(A_mu_reps[:, i, j] < 1e-8)) for j in range(K)]
                    for i in range(K)]
    zero_frac_sg = [[float(np.mean(A_sg_reps[:, i, j] < 1e-8)) for j in range(K)]
                    for i in range(K)]
    boundary_cells = [(i, j) for i in range(K) for j in range(K)
                      if i != j and A_mu[i, j] < 1e-8]
    n1 = float(np.mean(n_state1))
    boundary = {
        "definition": "a cell counts as at the boundary when the refit returns < 1e-8",
        "point_boundary_cells_1indexed": [[i + 1, j + 1] for i, j in boundary_cells],
        "frac_reps_at_zero_mu_labelling": zero_frac_mu,
        "frac_reps_at_zero_sigma_labelling": zero_frac_sg,
        "mean_state1_hours_per_rep": n1,
        "rule_of_three_bound_per_hour": 3.0 / n1 if n1 > 0 else None,
        "note": ("A percentile interval for a cell estimated at the boundary is "
                 "degenerate: the parametric bootstrap simulates from A_ij = 0, so "
                 "no replication can produce a positive count and the interval "
                 "collapses to [0, 0]. The rule-of-three bound 3/n is the primary "
                 "statement about that cell; the bootstrap says only that the "
                 "estimator is not noisy *given* the null, not that the true value "
                 "is zero. See `boundary_power` for the complementary calculation."),
    }

    # ---- label-switching diagnostic ---------------------------------------
    from collections import Counter
    sig_counts = Counter(signatures)
    label_block = {
        "signature_definition": ("rank of sigma_k after sorting the states by mu "
                                 "ascending; equal signatures mean the mu-ascending "
                                 "and sigma-ascending labelling rules agree"),
        "point_signature": list(truth_sig),
        "disagreement_rate": float(np.mean([s != truth_sig for s in signatures])),
        "signature_counts": [{"signature": list(s), "count": int(c)}
                             for s, c in sig_counts.most_common()],
    }

    return {
        "T": int(T),
        "K": int(K),
        "n_reps": int(n_reps),
        "seed": int(seed),
        "seconds": float(elapsed),
        "n_jobs": int(n_jobs),
        "refit": "Baum-Welch initialised at the data-generating parameters",
        "point_mu_bps_per_hour": (mu_mu * 1e4).tolist(),
        "point_sigma_bps_per_hour": (sig_mu * 1e4).tolist(),
        "point_A": A_mu.tolist(),
        "point_stationary": pi_stat.tolist(),
        "mu_ascending": _block("mu"),
        "sigma_ascending": _block("sigma"),
        "label_switching": label_block,
        "boundary": boundary,
        "em_iterations": {
            "mean": float(np.mean(iters)),
            "max": int(np.max(iters)),
            "cap": int(BOOT_N_ITER),
            "n_reps_at_cap": int(np.sum(np.asarray(iters) >= BOOT_N_ITER)),
            "frac_reps_at_cap": float(np.mean(np.asarray(iters) >= BOOT_N_ITER)),
            "note": ("Replications that hit the cap stopped on the iteration budget "
                     "rather than on the tolerance (1e-6 in absolute log-likelihood). "
                     "They are the label-ambiguous ones: the likelihood is flat along "
                     "the direction that swaps the two upper means, which is exactly "
                     "the variability the sigma-ascending block isolates."),
        },
        "loglik": {"mean": float(np.mean(lls)), "min": float(np.min(lls)),
                   "max": float(np.max(lls))},
        "multistart_check": _multistart_report(reps, multistart_check),
    }


def _multistart_report(reps, n_checked: int) -> dict | None:
    """Compare the cheap refit with a full multi-start refit on the same samples."""
    checked = [r for r in reps if "ms_ll" in r]
    if not checked:
        return None
    d_ll = np.array([r["ms_ll"] - r["ll"] for r in checked])
    # Both parameter vectors are put in mu-ascending order before comparison.
    d_mu, d_sig = [], []
    for r in checked:
        _, m1, s1, _ = _order_by(r["mu"], r["sigma"], r["A"], "mu")
        _, m2, s2, _ = _order_by(r["ms_mu"], r["ms_sigma"], r["ms_A"], "mu")
        d_mu.append(np.abs(m2 - m1) * 1e4)
        d_sig.append(np.abs(s2 - s1) * 1e4)
    d_mu = np.array(d_mu)
    d_sig = np.array(d_sig)
    return {
        "n_checked": int(len(checked)),
        "requested": int(n_checked),
        "delta_logL_multistart_minus_at_truth": {
            "mean": float(d_ll.mean()), "min": float(d_ll.min()),
            "max": float(d_ll.max()),
            "n_multistart_strictly_better_by_1e-3": int(np.sum(d_ll > 1e-3)),
            "n_at_truth_strictly_better_by_1e-3": int(np.sum(d_ll < -1e-3)),
        },
        "max_abs_diff_mu_bps": float(d_mu.max()),
        "max_abs_diff_sigma_bps": float(d_sig.max()),
        "mean_abs_diff_mu_bps": float(d_mu.mean()),
        "mean_abs_diff_sigma_bps": float(d_sig.mean()),
    }


def boundary_power(params: dict, T: int, cell, alternatives=(0.003, 0.01),
                   n_reps: int = 200, seed: int = 20260820,
                   n_jobs: int | None = None) -> list:
    """How often does the MLE return exactly zero when the truth is *not* zero?

    The bootstrap under the fitted null is uninformative about a boundary cell
    (it can only ever return zero).  The informative question is the
    converse: if the true hourly probability were as large as the rule-of-three
    bound, would this sample size have detected it?  We simulate under
    ``A[cell] = a`` for each alternative ``a`` (renormalising the row against the
    cell's row-mate) and report the share of replications whose MLE is still
    exactly zero.
    """
    mu = np.asarray(params["mu"], dtype=float)
    sigma = np.asarray(params["sigma"], dtype=float)
    A0 = np.asarray(params["A"], dtype=float)
    i, j = cell
    K = mu.size
    other = [c for c in range(K) if c not in (i, j)][0]

    out = []
    for a in alternatives:
        A = A0.copy()
        A[i, j] = a
        A[i, other] = max(A0[i, other] - a, 0.0)
        A[i, i] = 1.0 - A[i, j] - A[i, other]
        rep = bootstrap_hmm({"mu": mu, "sigma": sigma, "A": A}, T,
                            n_reps=n_reps, seed=seed + int(a * 1e6),
                            n_jobs=n_jobs, label=f"power a={a}", progress=True)
        # Under a sigma-ascending labelling the cell moves; sigma is the stably
        # identified coordinate, so that is the labelling the power statement
        # should be read off. Both are reported.
        inv = np.argsort(np.argsort(sigma))
        si, sj = int(inv[i]), int(inv[j])
        est_mu = rep["mu_ascending"]["A"][i][j]
        est_sg = rep["sigma_ascending"]["A"][si][sj]
        out.append({
            "alternative_A_ij": float(a),
            "cell_1indexed_mu_labelling": [i + 1, j + 1],
            "cell_1indexed_sigma_labelling": [si + 1, sj + 1],
            "n_reps": int(n_reps),
            "frac_reps_MLE_exactly_zero_sigma_labelling": float(
                rep["boundary"]["frac_reps_at_zero_sigma_labelling"][si][sj]),
            "frac_reps_MLE_exactly_zero_mu_labelling": float(
                rep["boundary"]["frac_reps_at_zero_mu_labelling"][i][j]),
            "boot_mean_A_ij_sigma_labelling": est_sg["boot_mean"],
            "boot_ci95_A_ij_sigma_labelling": est_sg["ci95"],
            "boot_mean_A_ij_mu_labelling": est_mu["boot_mean"],
            "boot_ci95_A_ij_mu_labelling": est_mu["ci95"],
            "label_disagreement_rate": rep["label_switching"]["disagreement_rate"],
        })
    return out


# ==========================================================================
#  2.  Residual diagnostics
# ==========================================================================
def ljung_box(z, lags) -> list:
    """Ljung--Box Q at each lag in ``lags`` (statsmodels, no df adjustment)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    res = acorr_ljungbox(z, lags=list(lags), return_df=True)
    return [{"lag": int(lag), "Q": float(res.loc[lag, "lb_stat"]),
             "p": float(res.loc[lag, "lb_pvalue"])} for lag in lags]


def arch_lm(z, lag: int = 12) -> dict:
    """Engle's ARCH-LM test at ``lag`` lags."""
    from statsmodels.stats.diagnostic import het_arch
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    lm, lmp, f, fp = het_arch(z, nlags=lag)
    return {"lag": int(lag), "LM": float(lm), "p": float(lmp),
            "F": float(f), "F_p": float(fp)}


def jarque_bera(z) -> dict:
    """Jarque--Bera normality test, with the moments that drive it."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    jb, p = stats.jarque_bera(z)
    return {"JB": float(jb), "p": float(p),
            "skew": float(stats.skew(z)),
            "excess_kurtosis": float(stats.kurtosis(z, fisher=True))}


def residual_battery(z, name: str, lb_lags=(10, 24), arch_lag: int = 12) -> dict:
    """Ljung--Box on z and z^2, ARCH-LM on z, Jarque--Bera on z, plus robust variants.

    ``ljung_box_abs_z`` and ``ljung_box_log_z2`` are not decoration.  A
    Ljung--Box statistic on z^2 divides the sample autocovariances by the
    *sample variance of z^2*, which is a fourth-moment object; on a residual
    with excess kurtosis of 14 a handful of observations dominate that
    denominator and the statistic loses essentially all its power.  On this
    paper's causally standardised residuals that failure mode is not
    hypothetical -- Q^2(10) = 2.4 (p = 0.99) while Q(10) on |z| is 20.6
    (p = 0.024) and on log z^2 is 33.3 (p = 2.4e-4) on the same series.  The
    z^2 row alone would therefore overstate how much clustering the model
    absorbs.
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    trimmed = _trim_extremes(z, 5)
    return {
        "series": name,
        "n": int(z.size),
        "mean": float(z.mean()),
        "sd": float(z.std(ddof=1)),
        "ljung_box_z": ljung_box(z, lb_lags),
        "ljung_box_z2": ljung_box(z ** 2, lb_lags),
        "ljung_box_abs_z": ljung_box(np.abs(z), lb_lags),
        "ljung_box_log_z2": ljung_box(np.log(z ** 2 + 1e-300), lb_lags),
        "arch_lm": arch_lm(z, arch_lag),
        "arch_lm_trimmed5": arch_lm(trimmed, arch_lag),
        "ljung_box_z2_trimmed5": ljung_box(trimmed ** 2, lb_lags),
        "jarque_bera": jarque_bera(z),
    }


def _trim_extremes(z, n_drop: int):
    """Drop the ``n_drop`` largest |z| (used only for the robustness columns)."""
    z = np.asarray(z, dtype=float)
    if n_drop <= 0 or n_drop >= z.size:
        return z
    keep = np.ones(z.size, dtype=bool)
    keep[np.argsort(-np.abs(z))[:n_drop]] = False
    return z[keep]


def hmm_predictive_moments(hmm: GaussianHMM, x, use: str = "filtered"):
    r"""One-step-ahead conditional mean and variance of the HMM, for t = 2..T.

    Let :math:`\tilde p_{j,t} = P(S_t = j \mid \mathcal F_{t-1})
    = \sum_k P(S_{t-1}=k \mid \mathcal F_{t-1}) A_{kj}`.  Then

        E[r_t | F_{t-1}]   = sum_j p~_j mu_j
        Var[r_t | F_{t-1}] = sum_j p~_j sigma_j^2                (within-regime)
                             + sum_j p~_j (mu_j - E)^2           (between-regime)

    which is exactly the decomposition the paper writes down in its
    conditional-moments subsection (``eq:cond-mean`` and ``eq:cond-var``).

    **Filtered, not smoothed, and deliberately.**  The residuals are used for
    serial-correlation tests, and a Ljung--Box statistic is only interpretable
    if the conditioning set is the past.  Smoothed probabilities
    :math:`P(S_t\mid\mathcal F_T)` use the whole sample, including
    :math:`r_t` itself, so smoothed residuals are shrunk towards zero by
    construction and their autocorrelations are not the autocorrelations of a
    forecast error.  Standardising by the smoothed mixture is the *generous*
    choice -- it makes the model look better than a forecast would -- so the
    causal filtered version is the headline and the smoothed version
    (``use="smoothed"``) is reported alongside it for contrast.

    The first observation has no predecessor, so the residual series has length
    T-1 and is aligned to x[1:].
    """
    x = np.asarray(x, dtype=float)
    # The posterior rows are frequently exact 0/1 in this fit (the regimes are
    # well separated), and Accelerate's BLAS raises spurious divide/overflow
    # flags on such matmuls. The results are finite and are asserted so below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        post = (hmm.filtered_posterior(x) if use == "filtered"
                else hmm.smoothed_posterior(x))
        if use == "filtered":
            pred = post[:-1] @ hmm.A           # P(S_t | F_{t-1}) for t = 2..T
        else:
            # "smoothed" variant: condition the mixture on the two-sided posterior
            # of the *contemporaneous* state -- the generous standardisation.
            pred = post[1:]
        mean = pred @ hmm.mu
        second = pred @ (hmm.sigma ** 2 + hmm.mu ** 2)
    var = second - mean ** 2
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var))):
        raise FloatingPointError("non-finite predictive moments")
    return mean, np.maximum(var, 1e-300)


def hmm_standardised_residuals(hmm: GaussianHMM, x, use: str = "filtered"):
    """z_t = (r_t - E[r_t|F_{t-1}]) / sqrt(Var[r_t|F_{t-1}]), t = 2..T."""
    x = np.asarray(x, dtype=float)
    mean, var = hmm_predictive_moments(hmm, x, use=use)
    return (x[1:] - mean) / np.sqrt(var)


def msar_standardised_residuals(fit: dict, y):
    """Standardised one-step-ahead residuals of the MS-AR(1) on the premium.

    ``fit`` is the dict returned by ``regime_engine.msar.fit_msar_premium``.

    The mean-adjusted specification subtracts the level of the *previous*
    regime from the lagged observation,

        y_t - mu_{S_t} = phi_{S_t} (y_{t-1} - mu_{S_{t-1}}) + sigma_{S_t} eps_t,

    so the regime-conditional one-step forecast is indexed by the state *pair*:

        m_{ij} = mu_j + phi_j (y_{t-1} - mu_i),   i = S_{t-1},  j = S_t.

    Marginalising over the pair needs its predicted joint law, which factorises
    as P(S_{t-1}=i | F_{t-1}) P_{ij} -- the filtered probability one step back,
    propagated through the transition kernel. Building the forecast from
    ``predicted_marginal_probabilities`` alone instead can only put mu_j on both
    sides, i.e. it assumes S_{t-1} = S_t and drops phi_j (mu_j - mu_i) on every
    transition. That term is not small here: on the paper's window it reaches
    0.43 bps against a calm-regime sigma of 0.53 bps, with about 5% of the
    predicted mass off the diagonal.

    The AR(1) costs the first observation and the pair construction costs one
    more (the earliest usable t has no filtered predecessor inside the array),
    so the residuals align to ``y[2:]``.
    """
    y = np.asarray(y, dtype=float)
    filt = np.asarray(fit["filtered"], dtype=float)
    P = np.asarray(fit["transition_matrix"], dtype=float)
    mu = np.asarray(fit["mu_const"], dtype=float)
    phi = np.asarray(fit["phi"], dtype=float)
    s2 = np.asarray(fit["sigma2"], dtype=float)

    n = filt.shape[0]
    # q[t, i, j] = P(S_{t-1} = i, S_t = j | F_{t-1})
    q = filt[:-1, :, None] * P[None, :, :]
    y_win = y[-n:]
    y_lag = y_win[:-1]
    y_now = y_win[1:]
    m = mu[None, None, :] + phi[None, None, :] * (y_lag[:, None, None] - mu[None, :, None])
    mean = (q * m).sum(axis=(1, 2))
    var = (q * (m ** 2 + s2[None, None, :])).sum(axis=(1, 2)) - mean ** 2
    return (y_now - mean) / np.sqrt(np.maximum(var, 1e-300))


# ==========================================================================
#  3.  Out-of-family benchmarks
# ==========================================================================
def fit_single_gaussian(r) -> dict:
    """i.i.d. N(mu, sigma^2).  Closed-form MLE -- no optimiser, no seed."""
    r = np.asarray(r, dtype=float)
    T = r.size
    mu = float(r.mean())
    var = float(r.var(ddof=0))                     # ML, not the unbiased estimator
    ll = -0.5 * T * (np.log(2 * np.pi) + np.log(var) + 1.0)
    return {"model": "Gaussian i.i.d.", "params": {"mu": mu, "sigma": float(np.sqrt(var))},
            "logL": float(ll), "k": 2, "T": int(T)}


def garch11_variance(r, mu, omega, alpha, beta, h1=None):
    """Conditional-variance recursion h_t = omega + alpha e_{t-1}^2 + beta h_{t-1}.

    ``h1`` initialises the recursion; the default is the sample variance of ``r``
    (a fixed, data-determined backcast, not an estimated parameter -- which is
    why the parameter counts below do not include it).
    """
    r = np.asarray(r, dtype=float)
    T = r.size
    e = r - mu
    e2 = e * e
    h = np.empty(T)
    h[0] = float(np.var(r, ddof=0)) if h1 is None else float(h1)
    for t in range(1, T):
        h[t] = omega + alpha * e2[t - 1] + beta * h[t - 1]
    return h, e


def _garch_negloglik(theta, r, dist):
    if dist == "normal":
        mu, log_omega, alpha, beta = theta
        nu = None
    else:
        mu, log_omega, alpha, beta, nu = theta
        if not (2.05 < nu < 500):
            return 1e10
    if not (0.0 <= alpha < 1.0 and 0.0 <= beta < 1.0 and alpha + beta < 0.99999):
        return 1e10
    omega = np.exp(log_omega)
    h, e = garch11_variance(r, mu, omega, alpha, beta)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        return 1e10
    if dist == "normal":
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + e * e / h)
    else:
        c = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)))
        ll = np.sum(c - 0.5 * np.log(h)
                    - (nu + 1) / 2 * np.log1p(e * e / (h * (nu - 2))))
    return -ll if np.isfinite(ll) else 1e10


def fit_garch11(r, dist: str = "normal", scale: float = 1e4) -> dict:
    """GARCH(1,1) with Gaussian or standardised-Student-t innovations, by MLE.

    ``arch`` is not a dependency of this package (whether it is installed is
    recorded at run time in ``benchmarks.json["software"]``), so this is a
    self-contained scipy implementation.

    The optimisation runs on ``scale * r`` for conditioning (hourly log-returns
    are O(1e-3), so the variance parameters are O(1e-7) and a numerical gradient
    is hopeless), and the returned log-likelihood is then re-evaluated *in the
    original units* by the recursion, so it is directly comparable with the
    HMM's.  The Jacobian identity logL(r) = logL(c r) + T log c is asserted at
    the end as a self-check and is reported.
    """
    r = np.asarray(r, dtype=float)
    T = r.size
    y = r * scale
    v = float(np.var(y, ddof=0))

    starts = []
    for (a, b) in ((0.05, 0.90), (0.10, 0.85), (0.02, 0.95), (0.15, 0.70)):
        base = [float(y.mean()), float(np.log(max(v * (1 - a - b), 1e-12))), a, b]
        starts.append(base if dist == "normal" else base + [6.0])

    bounds = [(None, None), (-40, 20), (0.0, 0.999), (0.0, 0.999)]
    if dist != "normal":
        bounds = bounds + [(2.05, 200.0)]
    cons = ({"type": "ineq", "fun": lambda th: 0.99999 - th[2] - th[3]},)

    best = None
    for th0 in starts:
        try:
            res = optimize.minimize(_garch_negloglik, th0, args=(y, dist),
                                    method="SLSQP", bounds=bounds,
                                    constraints=cons,
                                    options={"maxiter": 800, "ftol": 1e-12})
        except Exception:
            continue
        if res.fun < 1e9 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError(f"GARCH(1,1)-{dist} failed to converge from any start")

    th = best.x
    ll_scaled = -float(best.fun)

    # ---- re-express in the original return units and re-evaluate -----------
    if dist == "normal":
        mu_s, log_om_s, alpha, beta = th
        nu = None
        k = 4
    else:
        mu_s, log_om_s, alpha, beta, nu = th
        k = 5
    mu = mu_s / scale
    omega = np.exp(log_om_s) / scale ** 2
    theta_raw = ([mu, np.log(omega), alpha, beta] if dist == "normal"
                 else [mu, np.log(omega), alpha, beta, nu])
    ll_raw = -_garch_negloglik(theta_raw, r, dist)

    # Change of variables y = c r: f_r(r) = c f_y(c r), so logL_r = logL_y + T log c.
    # At c = 1e4 and T = 3,507 that is 32,300.7 nats -- larger than any difference
    # in the benchmark table, which is exactly why every row must share one scale.
    jacobian_gap = float(ll_raw - (ll_scaled + T * np.log(scale)))
    h, _ = garch11_variance(r, mu, omega, alpha, beta)
    return {
        "model": f"GARCH(1,1)-{'Normal' if dist == 'normal' else 'Student-t'}",
        "params": {"mu": float(mu), "omega": float(omega), "alpha": float(alpha),
                   "beta": float(beta), **({} if nu is None else {"nu": float(nu)})},
        "persistence": float(alpha + beta),
        "implied_annualised_vol_pct": float(
            np.sqrt(omega / (1 - alpha - beta)) * ANNUALISATION * 100),
        "logL": float(ll_raw),
        "logL_in_bps_units": float(ll_scaled),
        "jacobian_check_gap": jacobian_gap,
        "mean_conditional_vol_annualised_pct": float(
            np.mean(np.sqrt(h)) * ANNUALISATION * 100),
        "k": k,
        "T": int(T),
        "converged": bool(best.success),
    }


def _ic(block: dict) -> dict:
    """AIC and BIC in the lower-is-better convention, from logL / k / T."""
    ll, k, T = block["logL"], block["k"], block["T"]
    return {**block,
            "AIC": float(2 * k - 2 * ll),
            "BIC": float(k * np.log(T) - 2 * ll)}


# ==========================================================================
#  Runners
# ==========================================================================
def _load_panel_returns(panel_path):
    from .panel import load_panel
    panel = load_panel(panel_path)
    return panel


def _fit_paper_hmm(r, K: int) -> GaussianHMM:
    """Refit exactly the model analysis.py reports: same seed, starts, iterations."""
    return GaussianHMM(K=K, n_starts=HMM_N_STARTS, n_iter=HMM_N_ITER,
                       random_state=HMM_SEED_BASE + K).fit(r)


def run_bootstrap(out_dir, n_reps: int = 500, seed: int = 20260819,
                  n_jobs: int | None = None, multistart_check: int = 25,
                  power_reps: int = 200) -> dict:
    out_dir = pathlib.Path(out_dir)
    full_doc = json.loads((out_dir / FULLSAMPLE_JSON).read_text())
    full, T_full = full_doc["return_K3"], int(full_doc["n_obs"])
    sub_doc = json.loads((out_dir / "summary.json").read_text())
    sub, T_sub = sub_doc["return_K3"], int(sub_doc["n_obs"])

    A_full = reconstruct_transition_matrix(full["diag_A"], full["stationary"], (0, 1))
    params_full = {"mu": np.array(full["mu_bps_per_hour"]) / 1e4,
                   "sigma": np.array(full["sigma_bps_per_hour"]) / 1e4,
                   "A": A_full}
    params_sub = {"mu": np.array(sub["mu_bps_per_hour"]) / 1e4,
                  "sigma": np.array(sub["sigma_bps_per_hour"]) / 1e4,
                  "A": np.array(sub["A"])}

    res_full = bootstrap_hmm(params_full, T=T_full, n_reps=n_reps, seed=seed,
                             n_jobs=n_jobs, multistart_check=multistart_check,
                             label="full-sample")
    res_full["source"] = FULLSAMPLE_JSON
    res_full["A_provenance"] = (
        "The archived full-sample summary stores only diag(A) and the stationary "
        "distribution. The full kernel is exactly identified by those two plus the "
        "boundary estimate A_12 = 0 (in the archive's mu-ascending order) and is "
        "reconstructed here by "
        "reconstruct_transition_matrix(); the same routine reproduces the archived "
        "sub-sample kernel to machine precision from its diagonal and stationary "
        "distribution alone (tests/test_diagnostics.py::test_reconstruction_"
        "recovers_archived_subsample_kernel).")

    res_sub = bootstrap_hmm(params_sub, T=T_sub, n_reps=n_reps, seed=seed + 1,
                            n_jobs=n_jobs, multistart_check=multistart_check,
                            label="sub-sample")
    res_sub["source"] = "summary.json (full kernel archived; no reconstruction needed)"

    power = boundary_power(params_full, T=T_full, cell=(0, 1),
                           alternatives=(0.003, 0.01), n_reps=power_reps,
                           n_jobs=n_jobs)

    payload = {
        "generated_by": "regime_engine.diagnostics.run_bootstrap",
        "method": (
            "Parametric bootstrap: simulate a state path from the fitted A starting "
            "at its stationary distribution, emit r_t ~ N(mu_{S_t}, sigma^2_{S_t}), "
            "refit the K=3 Gaussian HMM by Baum-Welch initialised at the "
            "data-generating parameters, relabel by sigma ascending (the paper's "
            "rule) and, separately, by mu ascending, so the two can be "
            "compared."),
        "full_sample": res_full,
        "sub_sample": res_sub,
        "boundary_power_full_sample": power,
        "software": _software(),
    }
    (out_dir / "bootstrap_hmm.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"[saved] {out_dir / 'bootstrap_hmm.json'}")
    write_bootstrap_tables(payload, out_dir)
    return payload


def write_bootstrap_tables(payload: dict, out_dir) -> None:
    """Emit the four bootstrap tables (two samples x two labelling rules)."""
    tab = pathlib.Path(out_dir) / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    for tag, stem in (("full_sample", "table8_bootstrap_fullsample"),
                      ("sub_sample", "table8b_bootstrap_subsample")):
        for rule, suffix in (("sigma_ascending", "sigma"), ("mu_ascending", "mu")):
            _write_table8(payload[tag], tab / f"{stem}_{suffix}.tex", labelling=rule)


def run_diagnostics(panel_path, out_dir, msar_starts: int = 40,
                    premium_full_path=None) -> dict:
    from .paths import REPO_ROOT

    out_dir = pathlib.Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    panel = _load_panel_returns(panel_path)
    r = panel["log_return"].values
    T = len(r)

    print(f"[diagnostics] refitting the paper's K=3 HMM on {T} returns ...")
    hmm3 = _fit_paper_hmm(r, 3)
    # The archived log-likelihood is a REFERENCE value that lives with the
    # repository, not something the caller's --out directory is expected to
    # contain, so it is looked up in DEFAULT_OUT first. This keeps
    # `make diagnostics OUT=<fresh dir>` -- the override the Makefile
    # documents -- working on an empty output directory.
    archived = None
    for candidate in (pathlib.Path(DEFAULT_OUT) / "summary.json",
                      out_dir / "summary.json"):
        if candidate.exists():
            archived = json.loads(candidate.read_text())["return_K3"]["logL"]
            break
    if archived is None:
        print(f"[diagnostics] logL {hmm3.log_likelihood_:.6f} "
              "(no archived summary.json to compare against; run `make analyse` first)")
    else:
        print(f"[diagnostics] logL {hmm3.log_likelihood_:.6f} "
              f"(archived {archived:.6f}, gap {hmm3.log_likelihood_ - archived:.2e})")

    z_filt = hmm_standardised_residuals(hmm3, r, use="filtered")
    z_smooth = hmm_standardised_residuals(hmm3, r, use="smoothed")

    blocks = [
        residual_battery(r, "raw log-returns"),
        residual_battery(z_filt, "HMM K=3 standardised residuals (filtered)"),
        residual_battery(z_smooth, "HMM K=3 standardised residuals (smoothed)"),
    ]

    # ---- MS-AR(1) on the premium ------------------------------------------
    msar_blocks = []
    windows = [("sub-sample panel (T=3,507)", panel["premium"].values * 1e4)]
    if premium_full_path is None:
        premium_full_path = REPO_ROOT / "data" / "interim" / "premium_BTC_full.csv"
    premium_full_path = pathlib.Path(premium_full_path)
    if premium_full_path.exists():
        from .panel import load_paper_window_premium
        y_full = load_paper_window_premium(premium_full_path)
        windows.append((f"paper window 2 Oct 2025 - 28 Apr 2026 (T={len(y_full):,})",
                        y_full))
    else:
        print(f"[diagnostics] {premium_full_path} absent; MS-AR diagnostics on the "
              "sub-sample premium only")

    for name, y in windows:
        print(f"[diagnostics] MS-AR(1) multi-start on the premium, {name} ...")
        fit = _fit_msar_multistart(y, n_starts=msar_starts)
        z = msar_standardised_residuals(fit, y)
        block = residual_battery(z, f"MS-AR(1) standardised residuals -- {name}",
                                 lb_lags=(1, 2, 10))
        # Persist the whole inference block, not just the point estimates.
        # The funding-parameter table (paper/tables/table3_funding_params.tex)
        # quotes regime levels, autoregressive coefficients WITH standard
        # errors, half-lives WITH intervals and the ratio WITH its delta-method
        # interval; storing them here gives that table a machine-readable
        # source a replicator can diff against.
        hl = fit["half_lives"]
        ratio = fit["half_life_ratio"]
        block["msar"] = {
            "window": name,
            "regime_order": "sigma2 ascending (0 = calm, 1 = stressed)",
            "parameterisation": "mean-adjusted: mu_bps[k] is the regime MEAN, not an intercept",
            "n_obs": int(fit["nobs"]),
            "logL": fit["llf"],
            "AIC": fit["aic"],
            "BIC": fit["bic"],
            "mu_bps": [float(m) for m in fit["mu_const"]],
            "sigma_bps": [float(s) for s in fit["sigma"]],
            "phi": [float(p) for p in fit["phi"]],
            "phi_se": [float(h["phi_se"]) for h in hl],
            "half_life_hours": [float(h) for h in fit["half_life_hours"]],
            "half_life_se": [float(h["half_life_se_delta"]) for h in hl],
            "half_life_ci95": [[float(c) for c in h["half_life_ci95_delta"]] for h in hl],
            "half_life_ratio": ratio["value"],
            "half_life_ratio_se": ratio["se_delta"],
            "half_life_ratio_ci95": [float(c) for c in ratio["ci95_delta"]],
            "params": fit["param_table"],
            "multistart": fit["multistart_diagnostics"],
        }
        msar_blocks.append(block)

    payload = {
        "generated_by": "regime_engine.diagnostics.run_diagnostics",
        "panel": str(panel_path),
        "T": int(T),
        "hmm_K3_logL": float(hmm3.log_likelihood_),
        "hmm_K3_logL_archived": float(archived),
        "standardisation": (
            "z_t = (r_t - E[r_t|F_{t-1}]) / sqrt(Var[r_t|F_{t-1}]) with the "
            "one-step-ahead predicted regime distribution p~ = P(S_{t-1}|F_{t-1}) A; "
            "Var uses the full mixture variance, within-regime plus between-regime. "
            "FILTERED probabilities are the headline because a Ljung-Box statistic "
            "on a residual standardised by a two-sided (smoothed) posterior is not a "
            "test of forecast-error independence: the smoothed posterior conditions "
            "on r_t itself. The smoothed variant is reported alongside, for "
            "contrast, because it is the generous choice."),
        "tests": blocks + msar_blocks,
        "software": _software(),
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"[saved] {out_dir / 'diagnostics.json'}")
    _write_table6(payload, out_dir / "tables" / "table6_diagnostics.tex")
    return payload


def _fit_msar_multistart(y, n_starts: int = 40) -> dict:
    """The repo's documented multi-start protocol, returning the full inference block.

    The MS-AR likelihood on this series is multi-modal and its *dominant* basin
    is not its maximum, so a single-start fit is not admissible (see
    ``hmm.fit_ms_ar1_multistart``).  We run exactly that protocol to locate the
    winning seed, then rebuild through ``msar.fit_msar_premium`` at that seed so
    the standard errors, half-lives and regime ordering come from the code the
    paper already uses.  Re-running the winning seed is deterministic: the retry
    ladder inside ``fit_ms_ar1`` only advances on a numerical failure, and by
    construction that seed did not fail.
    """
    from .hmm import fit_ms_ar1_multistart
    from .msar import fit_msar_premium

    _, res, best_seed, diag = fit_ms_ar1_multistart(
        y, n_starts=n_starts, search_reps=40, verbose=True)
    fit = fit_msar_premium(y, seed=best_seed)
    gap = float(fit["llf"] - float(res.llf))
    # The two calls differ only in ``search_reps`` (40 in the multi-start scan, the
    # fit_ms_ar1 default of 20 in fit_msar_premium), so the perturbed start values
    # differ and the optimiser stops a fraction of a millionth of a nat apart. A
    # gap of that size is numerical noise; anything larger means the two calls
    # landed in different basins and the reported inference would not belong to
    # the reported optimum.
    if abs(gap) > 1e-2:
        raise RuntimeError(
            f"re-fit at the winning seed {best_seed} did not reproduce the "
            f"multi-start optimum ({fit['llf']} vs {res.llf})")
    diag = dict(diag)
    diag["refit_logL_gap"] = gap
    fit["multistart_diagnostics"] = diag
    return fit


def run_benchmarks(panel_path, out_dir) -> dict:
    out_dir = pathlib.Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    panel = _load_panel_returns(panel_path)
    r = panel["log_return"].values
    T = len(r)

    rows = []
    print("[benchmarks] single Gaussian ...")
    rows.append(_ic(fit_single_gaussian(r)))
    print("[benchmarks] GARCH(1,1)-Normal ...")
    rows.append(_ic(fit_garch11(r, "normal")))
    print("[benchmarks] GARCH(1,1)-Student-t ...")
    rows.append(_ic(fit_garch11(r, "t")))
    for K in (2, 3):
        print(f"[benchmarks] Gaussian HMM K={K} ...")
        h = _fit_paper_hmm(r, K)
        rows.append(_ic({
            "model": f"Gaussian HMM, K={K}",
            "params": {"mu_bps_per_hour": (h.mu * 1e4).tolist(),
                       "sigma_bps_per_hour": (h.sigma * 1e4).tolist(),
                       "diag_A": np.diag(h.A).tolist()},
            "logL": float(h.log_likelihood_),
            "k": int(h.n_params()),
            "T": int(T),
            "seed": HMM_SEED_BASE + K,
        }))
        # HMM-favourable alternative count: treat the initial distribution as
        # fixed at the stationary distribution rather than freely estimated.
        rows[-1]["k_without_initial_distribution"] = int(h.n_params() - (K - 1))
        rows[-1]["BIC_without_initial_distribution"] = float(
            rows[-1]["k_without_initial_distribution"] * np.log(T)
            - 2 * h.log_likelihood_)

    for row in rows:
        if row.get("persistence", 0) > 0.999:
            row["caveat"] = (
                "alpha + beta sits at the 0.99999 bound imposed in estimation: the "
                "process is near-integrated, its unconditional variance barely "
                "finite, and `implied_annualised_vol_pct` is not interpretable.")
            row.pop("implied_annualised_vol_pct", None)
        if row.get("params", {}).get("nu", 99) < 4.0:
            row["caveat_tails"] = (
                f"nu = {row['params']['nu']:.2f} < 4, so the fitted innovation "
                "distribution has infinite kurtosis; the model wins on likelihood "
                "by allowing tails the Gaussian mixture cannot reach, not by "
                "describing the regime structure better.")

    rows.sort(key=lambda d: d["BIC"])
    by_aic = sorted(rows, key=lambda d: d["AIC"])
    scale_gap = float(T * np.log(1e4))

    payload = {
        "generated_by": "regime_engine.diagnostics.run_benchmarks",
        "panel": str(panel_path),
        "T": int(T),
        "units": ("Every log-likelihood is evaluated on the RAW hourly log-return "
                  "series, never on returns in basis points. A log-likelihood is not "
                  f"scale-free: logL(c r) = logL(r) - T log c, which at c = 1e4 and "
                  f"T = {T} is {scale_gap:,.1f} nats -- three orders of magnitude "
                  "larger than any model difference in this table. Mixing units "
                  "across rows would decide the ranking by itself."),
        "log_scale_gap_nats": scale_gap,
        "convention": ("AIC = 2k - 2 logL and BIC = k log T - 2 logL for every row, "
                       "lower is better. GaussianHMM.aic/.bic use the same convention, "
                       "so the sign convention matches the model-selection table "
                       "(table1_model_selection.tex)."),
        "likelihood_conditioning": (
            f"Every row evaluates all T = {T} observations. The GARCH recursion is "
            "initialised at the sample variance -- a fixed backcast, not an estimated "
            "parameter -- so no row conditions on a different number of observations "
            "than another."),
        "parameter_count_note": (
            "The HMM rows follow the paper's count |theta_K| = K(K-1) + 2K + (K-1), "
            "which treats the initial distribution as K-1 free parameters. The GARCH "
            "rows have no analogous term. `BIC_without_initial_distribution` gives "
            "the most HMM-favourable alternative count; it is a lower bound on the "
            "BIC of a properly stationary-initialised HMM, because constraining pi "
            "to the stationary distribution can only lower the log-likelihood."),
        "arch_package_available": _have_arch(),
        "garch_implementation": (
            "self-contained scipy MLE in this module (arch is not a dependency of "
            "this package). Cross-checked against arch 7.2.0 in "
            "tests/test_diagnostics.py::test_garch_matches_the_arch_package_when_it_"
            "is_installed, which skips when arch is absent."),
        "rows": rows,
        "ranking_by_BIC": [rw["model"] for rw in rows],
        "ranking_by_AIC": [rw["model"] for rw in by_aic],
        "best_by_BIC": rows[0]["model"],
        "best_by_AIC": by_aic[0]["model"],
        "criteria_agree": bool(rows[0]["model"] == by_aic[0]["model"]),
        "headline": _benchmark_headline(rows, by_aic),
        "software": _software(),
    }
    (out_dir / "benchmarks.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"[saved] {out_dir / 'benchmarks.json'}")
    _write_table7(payload, out_dir / "tables" / "table7_benchmarks.tex")
    return payload


def _benchmark_headline(by_bic: list, by_aic: list) -> dict:
    """State the K=3 HMM's comparison with each benchmark, including where it loses."""
    def find(name_fragment, rows):
        return next(r for r in rows if name_fragment in r["model"])

    hmm3 = find("K=3", by_bic)
    gauss = find("i.i.d.", by_bic)
    gn = find("Normal", by_bic)
    gt = find("Student-t", by_bic)
    return {
        "hmm3_vs_single_gaussian_delta_BIC": float(hmm3["BIC"] - gauss["BIC"]),
        "hmm3_vs_garch_normal_delta_BIC": float(hmm3["BIC"] - gn["BIC"]),
        "hmm3_vs_garch_t_delta_BIC": float(hmm3["BIC"] - gt["BIC"]),
        "hmm3_vs_garch_t_delta_AIC": float(hmm3["AIC"] - gt["AIC"]),
        "statement": (
            "The K=3 Gaussian HMM beats the single Gaussian by "
            f"{gauss['BIC'] - hmm3['BIC']:,.0f} BIC points and the Gaussian GARCH(1,1) "
            f"by {gn['BIC'] - hmm3['BIC']:,.0f}. It LOSES to the Student-t "
            f"GARCH(1,1) on BIC by {hmm3['BIC'] - gt['BIC']:,.1f} points while "
            f"WINNING on AIC by {gt['AIC'] - hmm3['AIC']:,.1f}: the two criteria "
            "disagree because they price the HMM's nine extra parameters "
            "differently."),
    }


def _have_arch() -> bool:
    try:
        import arch  # noqa: F401
        return True
    except ImportError:
        return False


def _software() -> dict:
    import scipy
    import statsmodels
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "arch": "not installed" if not _have_arch() else __import__("arch").__version__}


# ==========================================================================
#  LaTeX tables (booktabs, matching the existing output/tables/*.tex style)
# ==========================================================================
def _p(p: float) -> str:
    """Format a p-value for a booktabs table."""
    if p is None or not np.isfinite(p):
        return "--"
    if p < 1e-4:
        return "$<$0.0001"
    return f"{p:.4f}"


def _tex_escape(s: str) -> str:
    return (s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
             .replace("--", "---"))


def _write_table6(payload: dict, path) -> None:
    rows = payload["tests"]
    tex = [
        "% Generated by regime_engine.diagnostics -- do not edit by hand.",
        "% z_t is standardised by the one-step-ahead regime-mixture mean and the",
        "% FULL conditional variance (within-regime + between-regime).  The headline",
        "% HMM row uses FILTERED (causal) regime probabilities; the smoothed row is",
        "% the two-sided (non-causal, generous) standardisation, shown because a",
        "% Ljung-Box statistic on a smoothed residual is not a test of forecast-error",
        "% independence.  Q(|z|) is reported because Q(z^2) has almost no power",
        "% against ARCH when the residual is strongly leptokurtic.  See diagnostics.json.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\multicolumn{5}{l}{\\emph{Panel A: serial correlation --- Ljung--Box "
        "$Q(\\ell)$, $p$-value in brackets}} \\\\[2pt]",
        "Series & $\\ell$ & $Q(z)$ & $Q(z^2)$ & $Q(|z|)$ \\\\",
        "\\midrule",
    ]
    for b in rows:
        name = _tex_escape(b["series"])
        lags = [d["lag"] for d in b["ljung_box_z"]]
        for i, lag in enumerate(lags):
            label = name if i == 0 else ""
            q = b["ljung_box_z"][i]
            q2 = b["ljung_box_z2"][i]
            qa = b["ljung_box_abs_z"][i]
            tex.append(
                f"{label} & {lag} & "
                f"{q['Q']:,.1f} [{_p(q['p'])}] & "
                f"{q2['Q']:,.1f} [{_p(q2['p'])}] & "
                f"{qa['Q']:,.1f} [{_p(qa['p'])}] \\\\")
        tex.append("\\addlinespace")
    tex += [
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Panel B: conditional heteroskedasticity and "
        "normality}} \\\\[2pt]",
        "Series & ARCH-LM(12) & Jarque--Bera & Skewness & Excess kurt. \\\\",
        "\\midrule",
    ]
    for b in rows:
        a, j = b["arch_lm"], b["jarque_bera"]
        tex.append(
            f"{_tex_escape(b['series'])} & {a['LM']:,.1f} [{_p(a['p'])}] & "
            f"{j['JB']:,.0f} [{_p(j['p'])}] & {j['skew']:.2f} & "
            f"{j['excess_kurtosis']:.2f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    pathlib.Path(path).write_text("\n".join(tex))
    print(f"[LaTeX] wrote {path}")


def _write_table8(boot: dict, path, labelling: str = "sigma_ascending") -> None:
    """The regime parameters of table2_regime_params.tex, with parametric-bootstrap inference.

    ``labelling`` selects which sorting rule indexes the states.  Both files are
    written, and the contrast between them is itself a result: under a
    mu-ascending rule the standard errors are dominated by relabelling rather
    than by estimation error, which is why the paper labels by sigma.
    """
    b = boot[labelling]
    rule = "mu ascending" if labelling == "mu_ascending" \
        else "sigma ascending (the paper's rule)"
    tex = [
        "% Generated by regime_engine.diagnostics -- do not edit by hand.",
        f"% Parametric bootstrap, {boot['n_reps']} replications, T = {boot['T']}, "
        f"seed {boot['seed']}.",
        f"% States are labelled by {rule}.",
        "% The two rules disagree on "
        f"{boot['label_switching']['disagreement_rate']:.1%} of replications, so under",
        "% mu ascending a large part of every standard error below is relabelling,",
        "% not estimation error. Compare the two files before quoting either.",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Parameter & Estimate & Boot.\\ s.e. & 95\\% percentile interval \\\\",
        "\\midrule",
    ]

    def row(name, cell, fmt="{:.3f}"):
        lo, hi = cell["ci95"]
        se = "--" if cell["se"] is None else fmt.format(cell["se"])
        tex.append(f"{name} & {fmt.format(cell['point'])} & {se} & "
                   f"[{fmt.format(lo)}, {fmt.format(hi)}] \\\\")

    for k in range(boot["K"]):
        row(f"$\\hat\\mu_{{{k+1}}}$ (bps/h)", b["mu_bps_per_hour"][k])
    tex.append("\\addlinespace")
    for k in range(boot["K"]):
        row(f"$\\hat\\sigma_{{{k+1}}}$ (bps/h)", b["sigma_bps_per_hour"][k], "{:.2f}")
    tex.append("\\addlinespace")
    for k in range(boot["K"]):
        row(f"Annualised $\\hat\\sigma_{{{k+1}}}$ (\\%)",
            b["annualised_sigma_pct"][k], "{:.1f}")
    tex.append("\\addlinespace")
    for i in range(boot["K"]):
        for j in range(boot["K"]):
            row(f"$\\hat A_{{{i+1}{j+1}}}$", b["A"][i][j], "{:.4f}")
    tex.append("\\addlinespace")
    for k in range(boot["K"]):
        row(f"$\\mathbb{{E}}[\\tau_{{{k+1}}}]$ (h)",
            b["expected_duration_hours"][k], "{:.2f}")
    tex.append("\\addlinespace")
    for k in range(boot["K"]):
        row(f"$\\pi^\\infty_{{{k+1}}}$", b["stationary"][k], "{:.4f}")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    pathlib.Path(path).write_text("\n".join(tex))
    print(f"[LaTeX] wrote {path}")


def _tex_num(x: float, fmt: str = ",.1f") -> str:
    """A number for a LaTeX table, with a typographic minus instead of a hyphen."""
    return ("$-$" if x < 0 else "") + format(abs(x), fmt)


def _write_table7(payload: dict, path) -> None:
    tex = [
        "% Generated by regime_engine.diagnostics -- do not edit by hand.",
        f"% All rows fitted on the same T = {payload['T']} hourly log-returns, in raw",
        "% return units. AIC = 2k - 2logL, BIC = k log T - 2logL; lower is better.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & $\\log\\mathcal{L}$ & \\# params & AIC & BIC \\\\",
        "\\midrule",
    ]
    for row in payload["rows"]:
        name = row["model"].replace("&", "\\&")
        tex.append(f"{name} & {_tex_num(row['logL'])} & {row['k']} & "
                   f"{_tex_num(row['AIC'])} & {_tex_num(row['BIC'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    pathlib.Path(path).write_text("\n".join(tex))
    print(f"[LaTeX] wrote {path}")


# ==========================================================================
#  CLI
# ==========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m regime_engine.diagnostics",
        description="Parametric bootstrap, residual diagnostics and out-of-family "
                    "benchmarks for the regime-switching paper.")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--all", action="store_true", help="run all three analyses")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--benchmarks", action="store_true")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--power-reps", type=int, default=200)
    ap.add_argument("--multistart-check", type=int, default=25)
    ap.add_argument("--msar-starts", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260819)
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out = pathlib.Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_any = args.all or args.bootstrap or args.diagnostics or args.benchmarks
    if not run_any:
        args.all = True

    if args.all or args.benchmarks:
        run_benchmarks(args.panel, out)
    if args.all or args.diagnostics:
        run_diagnostics(args.panel, out, msar_starts=args.msar_starts)
    if args.all or args.bootstrap:
        run_bootstrap(out, n_reps=args.reps, seed=args.seed, n_jobs=args.jobs,
                      multistart_check=args.multistart_check,
                      power_reps=args.power_reps)


if __name__ == "__main__":
    main()
