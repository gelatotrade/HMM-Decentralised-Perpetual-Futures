"""
Batched Baum--Welch for Gaussian hidden Markov models.

Many EM problems are run in lockstep -- independent starts on one series, or
one start on each of many simulated series -- so that the Python loop over
hours in the forward and backward recursions is paid once per iteration
rather than once per problem.  The E- and M-steps are those of
:class:`regime_engine.hmm.GaussianHMM`, written in scaled rather than
log-space form; ``tests/test_hmm_batch.py`` checks that the two agree.

On top of the plain EM the module provides

* a constrained M-step that holds one transition cell at a fixed value, which
  is what a profile likelihood of that cell needs;
* the boundary score ratio of a cell held at zero, which decides whether the
  maximum-likelihood estimate of the cell lies on the boundary;
* a multi-start fit that reproduces the starts and the selection rule of
  ``GaussianHMM.fit`` under an arbitrary iteration cap;
* :func:`fit_protocol`, the estimation protocol of ``GaussianHMM.fit``: every
  start run to the tolerance rule, the selected start then continued until
  both the log-likelihood and the parameters have stopped moving;
* :func:`wide_multistart`, the same protocol from a wider set of starting
  points (spread volatilities, and splits of the best fit with one state
  fewer), which model selection uses because the default starts can all end
  in one local optimum when K is large.

Arrays carry a leading batch axis ``R``: ``x`` is ``(R, T)`` (or ``(T,)``,
shared by every member), ``pi`` and ``mu`` and ``sigma`` are ``(R, K)``, ``A``
is ``(R, K, K)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hmm import GaussianHMM

__all__ = [
    "BatchParams",
    "EMResult",
    "boundary_score_ratio",
    "e_step",
    "fit_protocol",
    "loglik",
    "m_step",
    "max_param_step",
    "move_cell_mass",
    "multistart_fit",
    "run_em",
    "sort_by_sigma",
    "split_starts",
    "spread_starts",
    "wide_multistart",
]

LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
VAR_FLOOR = 1e-12          # the variance floor of GaussianHMM._m_step

#: Iteration cap of every EM run in the protocol. It is a safeguard, not a
#: stopping rule: on the paper's series the slowest start (K = 5) needs about
#: 5,000 iterations to meet the tolerance and its polish about 10,000 more.
FIT_CAP = 100_000

#: Tolerance of the per-start stopping rule (the rule of ``GaussianHMM.fit``:
#: stop once the log-likelihood rises by less than this).
FIT_TOL = 1e-6

#: The selected start is continued until |delta logL| falls below POLISH_TOL
#: and the largest parameter step below POLISH_STEP_TOL. The second condition
#: is the binding one: EM approaches the maximum linearly, slowly along flat
#: directions, and below about 1e-11 a change in a log-likelihood of order 1e4
#: is rounding noise, so |delta logL| alone cannot say that EM has arrived.
POLISH_TOL = 1e-9
POLISH_STEP_TOL = 1e-10


@dataclass
class BatchParams:
    """Parameters of ``R`` Gaussian HMMs with ``K`` states each."""

    pi: np.ndarray        # (R, K)
    A: np.ndarray         # (R, K, K)
    mu: np.ndarray        # (R, K)
    sigma: np.ndarray     # (R, K)

    @property
    def R(self) -> int:
        return int(self.mu.shape[0])

    @property
    def K(self) -> int:
        return int(self.mu.shape[1])

    def copy(self) -> BatchParams:
        return BatchParams(self.pi.copy(), self.A.copy(), self.mu.copy(), self.sigma.copy())

    def take(self, idx) -> BatchParams:
        idx = np.asarray(idx)
        return BatchParams(self.pi[idx].copy(), self.A[idx].copy(),
                           self.mu[idx].copy(), self.sigma[idx].copy())

    def put(self, idx, other: BatchParams) -> None:
        idx = np.asarray(idx)
        self.pi[idx] = other.pi
        self.A[idx] = other.A
        self.mu[idx] = other.mu
        self.sigma[idx] = other.sigma

    @classmethod
    def stack(cls, members) -> BatchParams:
        """Stack single-model parameter dicts (keys pi, A, mu, sigma) into a batch."""
        members = list(members)
        return cls(np.array([np.asarray(m["pi"], float) for m in members]),
                   np.array([np.asarray(m["A"], float) for m in members]),
                   np.array([np.asarray(m["mu"], float) for m in members]),
                   np.array([np.asarray(m["sigma"], float) for m in members]))

    def member(self, r: int) -> dict:
        return {"pi": self.pi[r].copy(), "A": self.A[r].copy(),
                "mu": self.mu[r].copy(), "sigma": self.sigma[r].copy()}


def _as_batch_x(x, R: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.broadcast_to(x, (R, x.size))
    if x.shape[0] != R:
        raise ValueError(f"x has {x.shape[0]} rows for {R} batch members")
    return x


def _emissions(x: np.ndarray, p: BatchParams):
    """Scaled emission densities b (R, T, K) and the per-hour log scale m (R, T)."""
    z = (x[:, :, None] - p.mu[:, None, :]) / p.sigma[:, None, :]
    logb = -0.5 * z * z - np.log(p.sigma)[:, None, :] - LOG_SQRT_2PI
    m = logb.max(axis=2)
    return np.exp(logb - m[:, :, None]), m


def _forward(b: np.ndarray, p: BatchParams):
    """Normalised forward variables (R, T, K) and scale factors (R, T)."""
    R, T, K = b.shape
    alpha = np.empty((R, T, K))
    c = np.empty((R, T))
    a = p.pi * b[:, 0]
    c[:, 0] = np.maximum(a.sum(axis=1), 1e-300)
    alpha[:, 0] = a / c[:, 0, None]
    A = p.A
    for t in range(1, T):
        a = np.matmul(alpha[:, t - 1, None, :], A)[:, 0, :] * b[:, t]
        s = np.maximum(a.sum(axis=1), 1e-300)
        c[:, t] = s
        alpha[:, t] = a / s[:, None]
    return alpha, c


def loglik(x, p: BatchParams) -> np.ndarray:
    """Log-likelihood of each member (forward pass only)."""
    x = _as_batch_x(x, p.R)
    b, m = _emissions(x, p)
    _, c = _forward(b, p)
    return np.log(c).sum(axis=1) + m.sum(axis=1)


def e_step(x, p: BatchParams):
    """Forward--backward for every member.

    Returns ``gamma`` (R, T, K), the expected transition counts ``xi_sum``
    (R, K, K), the transition score ``G`` (R, K, K) and the log-likelihood (R,).
    ``G[r, i, j]`` is the derivative of the log-likelihood with respect to
    ``A[r, i, j]``, so that ``xi_sum = A * G``; unlike ``xi_sum`` it stays
    informative where ``A[r, i, j] = 0``.
    """
    x = _as_batch_x(x, p.R)
    b, m = _emissions(x, p)
    alpha, c = _forward(b, p)
    R, T, K = b.shape
    beta = np.empty((R, T, K))
    beta[:, -1] = 1.0
    A = p.A
    for t in range(T - 2, -1, -1):
        v = b[:, t + 1] * beta[:, t + 1]
        beta[:, t] = np.matmul(A, v[:, :, None])[:, :, 0] / c[:, t + 1, None]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    w = b[:, 1:] * beta[:, 1:] / c[:, 1:, None]
    G = np.einsum("rti,rtj->rij", alpha[:, :-1], w)
    xi_sum = A * G
    ll = np.log(c).sum(axis=1) + m.sum(axis=1)
    return gamma, xi_sum, G, ll


def m_step(x, gamma, xi_sum, fixed_cell=None, fixed_value=None) -> BatchParams:
    """The M-step of ``GaussianHMM._m_step``, optionally with one cell held fixed.

    With ``fixed_cell = (i, j)`` the cell ``A[r, i, j]`` is set to
    ``fixed_value[r]`` and the rest of row ``i`` is proportional to the
    expected transition counts out of ``i``: that is the maximiser of the
    expected complete-data log-likelihood under the constraint.
    """
    R = gamma.shape[0]
    x = _as_batch_x(x, R)
    gsum = gamma.sum(axis=1)                                     # (R, K)
    pi = gamma[:, 0] / gamma[:, 0].sum(axis=1, keepdims=True)
    denom = xi_sum.sum(axis=2, keepdims=True)
    A = xi_sum / np.where(denom > 0, denom, 1.0)
    safe = np.where(gsum > 0, gsum, 1.0)
    mu = np.einsum("rtk,rt->rk", gamma, x) / safe
    dev = x[:, :, None] - mu[:, None, :]
    var = (gamma * dev * dev).sum(axis=1) / safe
    sigma = np.sqrt(np.maximum(var, VAR_FLOOR))
    if fixed_cell is not None:
        i, j = fixed_cell
        v = np.broadcast_to(np.asarray(fixed_value, dtype=float), (R,))
        row = xi_sum[:, i, :].copy()
        row[:, j] = 0.0
        s = row.sum(axis=1, keepdims=True)
        A[:, i, :] = (1.0 - v)[:, None] * row / np.where(s > 0, s, 1.0)
        A[:, i, j] = v
    return BatchParams(pi, A, mu, sigma)


def move_cell_mass(A, cell, value) -> np.ndarray:
    """Set ``A[..., i, j] = value`` and rescale the rest of row ``i`` to keep it stochastic."""
    A = np.array(A, dtype=float, copy=True)
    i, j = cell
    row = A[..., i, :].copy()
    row[..., j] = 0.0
    s = row.sum(axis=-1, keepdims=True)
    v = np.asarray(value, dtype=float)
    A[..., i, :] = (1.0 - v)[..., None] * row / np.where(s > 0, s, 1.0)
    A[..., i, j] = v
    return A


def max_param_step(old: BatchParams, new: BatchParams) -> np.ndarray:
    """Largest parameter change of each member, on a common scale.

    Means and volatilities are measured in units of the new volatility of
    their state, transition probabilities as they are; the step is the largest
    of the three.
    """
    s = new.sigma
    return np.maximum.reduce([
        np.max(np.abs(new.mu - old.mu) / s, axis=1),
        np.max(np.abs(new.sigma - old.sigma) / s, axis=1),
        np.max(np.abs(new.A - old.A), axis=(1, 2)),
    ])


@dataclass
class EMResult:
    params: BatchParams           # parameters after the last M-step
    ll: np.ndarray                # log-likelihood of the last E-step (R,)
    n_iter: np.ndarray            # E-steps run per member (R,)
    stopped: np.ndarray           # True where the tolerance rule stopped the member
    last_delta: np.ndarray        # ll[-1] - ll[-2] at the last iteration (R,)
    snapshots: dict               # iteration count -> (BatchParams, ll)
    last_step: np.ndarray = None  # max_param_step of the last M-step (R,)


def run_em(x, init: BatchParams, n_iter: int, tol: float, *, fixed_cell=None,
           fixed_value=None, rule: str = "fit", snapshot_iters=(),
           min_iter: int = 6, step_tol: float | None = None) -> EMResult:
    """EM for every member until its stopping rule fires or ``n_iter`` E-steps.

    ``rule = "fit"`` is the rule of ``GaussianHMM.fit``: stop after the M-step
    of iteration ``it`` once ``ll[it] - ll[it-1] < tol`` and ``it > 5``.
    ``rule = "abs"`` uses ``|ll[it] - ll[it-1]| < tol`` instead.  With
    ``step_tol`` the member must in addition have moved no parameter by more
    than ``step_tol`` in that M-step (:func:`max_param_step`).  Members that
    have stopped are frozen and dropped from the batch, so the cost of an
    iteration falls as members converge.  ``snapshot_iters`` records the state
    of every member after that many iterations, as a fit capped there would
    have returned it.
    """
    p = init.copy()
    R = p.R
    xb = _as_batch_x(x, R)
    fv = None if fixed_value is None else np.broadcast_to(
        np.asarray(fixed_value, dtype=float), (R,)).copy()
    ll = np.full(R, -np.inf)
    prev = np.full(R, -np.inf)
    n_used = np.zeros(R, dtype=int)
    stopped = np.zeros(R, dtype=bool)
    last_delta = np.full(R, np.nan)
    last_step = np.full(R, np.nan)
    active = np.arange(R)
    snaps = {}
    snap_set = sorted(int(s) for s in snapshot_iters)
    for it in range(int(n_iter)):
        if active.size == 0:
            break
        sub = p.take(active)
        xa = xb[active]
        gamma, xi_sum, _, ll_a = e_step(xa, sub)
        new = m_step(xa, gamma, xi_sum, fixed_cell,
                     None if fv is None else fv[active])
        step = max_param_step(sub, new)
        p.put(active, new)
        ll[active] = ll_a
        n_used[active] = it + 1
        d = ll_a - prev[active]
        last_delta[active] = d
        last_step[active] = step
        crit = np.abs(d) if rule == "abs" else d
        done = (crit < tol) & (it >= min_iter)
        if step_tol is not None:
            done &= step < step_tol
        stopped[active[done]] = True
        prev[active] = ll_a
        active = active[~done]
        if it + 1 in snap_set:
            snaps[it + 1] = (p.copy(), ll.copy())
    for s in snap_set:
        if s not in snaps:
            snaps[s] = (p.copy(), ll.copy())
    return EMResult(p, ll, n_used, stopped, last_delta, snaps, last_step)


def sort_by_sigma(p: BatchParams) -> BatchParams:
    """Relabel every member's states by sigma ascending, the paper's rule."""
    order = np.argsort(p.sigma, axis=1)
    r = np.arange(p.R)[:, None]
    A = p.A[r[:, :, None], order[:, :, None], order[:, None, :]]
    return BatchParams(p.pi[r, order], A, p.mu[r, order], p.sigma[r, order])


def boundary_score_ratio(x, p: BatchParams, cell) -> np.ndarray:
    """Score ratio of a transition cell held at zero.

    At a maximum of the likelihood constrained by ``A[i, j] = 0`` the
    directional derivative of the log-likelihood that moves mass into
    ``A[i, j]`` from the rest of row ``i`` (proportionally) is
    ``G[i, j] - n_i``, with ``n_i`` the expected number of transitions out of
    ``i``.  The ratio ``G[i, j] / n_i`` is therefore below one exactly when the
    constrained maximum satisfies the Kuhn--Tucker condition of the
    unconstrained problem, i.e. when the maximum-likelihood estimate of the
    cell is zero.  It is also the factor by which one EM step multiplies a
    small positive value of the cell, which is why EM approaches a boundary
    optimum geometrically, and slowly when the ratio is near one.
    """
    i, j = cell
    _, xi_sum, G, _ = e_step(x, p)
    return G[:, i, j] / xi_sum[:, i, :].sum(axis=1)


def _start_params(x: np.ndarray, K: int, n_starts: int, random_state: int) -> BatchParams:
    """The starting points ``GaussianHMM.fit`` draws, in its order."""
    master = np.random.default_rng(random_state)
    members = []
    h = GaussianHMM(K=K)
    for _ in range(n_starts):
        seed = master.integers(0, 10**9)
        h._init_random(x, np.random.default_rng(seed))
        members.append({"pi": h.pi, "A": h.A, "mu": h.mu, "sigma": h.sigma})
    return BatchParams.stack(members)


def multistart_fit(x, K: int, n_starts: int, random_state: int, n_iter: int,
                   tol: float = 1e-6, snapshot_iters=(), starts=None) -> dict:
    """``GaussianHMM(K, n_starts, n_iter, tol, random_state).fit(x)`` in batch form.

    The starts, the stopping rule, the selection (the first start with the
    largest final log-likelihood) and the sigma-ascending relabelling are those
    of ``GaussianHMM.fit``.  ``starts`` restricts the run to a subset of the
    start indices (to spread one fit over several processes); the selection is
    then over that subset.  Per-start convergence information is returned with
    the selected fit, and ``snapshot_iters`` gives the fit a smaller cap would
    have returned.
    """
    x = np.asarray(x, dtype=float)
    init = _start_params(x, K, n_starts, random_state)
    idx = np.arange(n_starts) if starts is None else np.asarray(starts, dtype=int)
    init = init.take(idx)
    res = run_em(x, init, n_iter, tol, snapshot_iters=snapshot_iters)

    def _select(params: BatchParams, ll: np.ndarray) -> dict:
        best = int(np.argmax(ll))            # first maximum, as fit() keeps it
        p = sort_by_sigma(params.take([best]))
        return {"best_start": int(idx[best]), "ll": float(ll[best]), **p.member(0)}

    out = {
        "K": int(K), "n_starts": int(n_starts), "random_state": int(random_state),
        "n_iter_cap": int(n_iter), "tol": float(tol),
        "starts": [int(s) for s in idx],
        "per_start": {
            "ll": res.ll.tolist(),
            "n_iter": res.n_iter.tolist(),
            "stopped_by_tol": res.stopped.tolist(),
            "last_delta": res.last_delta.tolist(),
        },
        "best": _select(res.params, res.ll),
        "snapshots": {int(k): _select(v[0], v[1]) for k, v in res.snapshots.items()},
        "_final_params": res.params,
    }
    return out


def merge_multistart(parts) -> dict:
    """Combine ``multistart_fit`` runs over disjoint start subsets of one fit."""
    parts = sorted(parts, key=lambda q: q["starts"][0])
    starts = [s for q in parts for s in q["starts"]]
    per = {k: [v for q in parts for v in q["per_start"][k]] for k in parts[0]["per_start"]}

    def _pick(blocks):
        best = None
        for b in blocks:                      # strict '>' keeps the first maximum
            if best is None or b["ll"] > best["ll"]:
                best = b
        return best

    out = {k: parts[0][k] for k in ("K", "n_starts", "random_state", "n_iter_cap", "tol")}
    out["starts"] = starts
    out["per_start"] = per
    out["best"] = _pick([q["best"] for q in parts])
    out["snapshots"] = {s: _pick([q["snapshots"][s] for q in parts])
                        for s in parts[0]["snapshots"]}
    return out


def fit_protocol(x, K: int, n_starts: int, random_state: int, n_iter: int = FIT_CAP,
                 tol: float = FIT_TOL, polish_tol: float | None = POLISH_TOL,
                 polish_step_tol: float | None = POLISH_STEP_TOL,
                 polish_cap: int = FIT_CAP, snapshot_iters=()) -> dict:
    """The estimation protocol of ``GaussianHMM.fit``.

    1. Every start of ``GaussianHMM.fit`` (the same seeds and starting points)
       runs EM until the tolerance rule fires (``ll[it] - ll[it-1] < tol``) or
       ``n_iter`` iterations have run.
    2. The selected start is the first with the largest final log-likelihood.
    3. That start is continued until ``|delta logL| < polish_tol`` and no
       parameter moves by more than ``polish_step_tol`` in an iteration, or
       ``polish_cap`` further iterations have run. ``polish_tol=None`` skips
       this step and returns the start as the tolerance rule left it.

    Returns the selected parameters in sigma-ascending order, the
    log-likelihood evaluated at those parameters, and a convergence record:
    per start the iterations run, whether the tolerance rule stopped it and the
    last change of the log-likelihood; for the polish the same plus the last
    parameter step. ``converged`` is true when every start met the tolerance
    rule within the cap and the polish met both of its conditions.
    ``snapshot_iters`` passes through to :func:`multistart_fit`.
    """
    x = np.asarray(x, dtype=float)
    ms = multistart_fit(x, K, n_starts, random_state, n_iter, tol,
                        snapshot_iters=snapshot_iters)
    final = ms.pop("_final_params")
    pos = ms["starts"].index(ms["best"]["best_start"])
    start = final.take([pos])
    per = ms["per_start"]
    record = {
        "protocol": ("all starts to the tolerance rule; the first start with the "
                     "largest final logL continued until |dlogL| < polish_tol and "
                     "the largest parameter step < polish_step_tol"),
        "n_starts": int(n_starts), "random_state": int(random_state),
        "iteration_cap": int(n_iter), "tol": float(tol),
        "selected_start": int(ms["best"]["best_start"]),
        "per_start_iterations": [int(v) for v in per["n_iter"]],
        "per_start_converged": [bool(v) for v in per["stopped_by_tol"]],
        "per_start_final_dlogL": [float(v) for v in per["last_delta"]],
        "all_starts_converged": bool(all(per["stopped_by_tol"])),
        "n_starts_converged": int(sum(per["stopped_by_tol"])),
    }
    if polish_tol is None:
        params, polish = start, None
        record["polish"] = None
        converged = bool(per["stopped_by_tol"][pos])
    else:
        res = run_em(x, start, polish_cap, polish_tol, rule="abs",
                     step_tol=polish_step_tol)
        params = res.params
        polish = res
        record["polish"] = {
            "tol": float(polish_tol),
            "step_tol": None if polish_step_tol is None else float(polish_step_tol),
            "cap": int(polish_cap),
            "iterations": int(res.n_iter[0]),
            "converged": bool(res.stopped[0]),
            "final_dlogL": float(res.last_delta[0]),
            "final_max_param_step": float(res.last_step[0]),
        }
        converged = bool(per["stopped_by_tol"][pos]) and bool(res.stopped[0])
    record["converged"] = bool(converged and record["all_starts_converged"])
    ll = float(loglik(x, params)[0])
    record["logL"] = ll
    out = {"K": int(K), "params": sort_by_sigma(params).member(0), "ll": ll,
           "record": record, "multistart": ms}
    if polish is not None:
        out["polish_result"] = polish
    return out


# --------------------------------------------------------------------------
#  Wide multi-start (model selection)
# --------------------------------------------------------------------------
#: Range of the starting volatilities of :func:`spread_starts`, as multiples of
#: the sample standard deviation; drawn log-uniformly.
SPREAD_VOL_RANGE = (0.15, 3.5)

#: Factors by which :func:`split_starts` divides and multiplies the volatility
#: of the state it splits.
SPLIT_FACTORS = (1.25, 1.6, 2.0)

#: Two terminal log-likelihoods closer than this count as the same optimum when
#: the starts that reach the best one are counted. The per-start rule stops at a
#: gain below 1e-6 per iteration, which on the slowest K = 5 starts can leave a
#: start a few hundredths below the polished optimum; distinct optima on the
#: paper's series lie at least about one log-likelihood point apart.
REACH_TOL = 0.1


def spread_starts(x, K: int, n: int, random_state: int,
                  vol_range=SPREAD_VOL_RANGE) -> BatchParams:
    """Starting points with volatilities spread log-uniformly over ``vol_range`` x s.d.

    The default starts of ``GaussianHMM._init_random`` draw every volatility
    from 0.5 to 1.5 times the sample standard deviation, so no start begins
    near a state much calmer or much wilder than the average hour. Here each
    state's volatility is drawn log-uniformly over ``vol_range`` times the
    standard deviation; means start at the sample mean with a jitter of 0.1
    s.d.; each row of the transition matrix puts a persistence drawn from
    U(0.5, 0.99) on its diagonal and spreads the rest by a flat Dirichlet;
    the initial distribution is a flat Dirichlet draw.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(random_state)
    sd, m = float(x.std()), float(x.mean())
    lo, hi = np.log(vol_range[0]), np.log(vol_range[1])
    sigma = sd * np.exp(rng.uniform(lo, hi, size=(n, K)))
    mu = m + rng.normal(0.0, 0.1 * sd, size=(n, K))
    stay = rng.uniform(0.5, 0.99, size=(n, K))
    if K > 1:
        off = rng.dirichlet(np.ones(K - 1), size=(n, K))
    A = np.empty((n, K, K))
    for i in range(K):
        A[:, i, i] = stay[:, i]
        others = [j for j in range(K) if j != i]
        if others:
            A[:, i, others] = (1.0 - stay[:, i])[:, None] * off[:, i, :]
    pi = rng.dirichlet(np.ones(K), size=n)
    return BatchParams(pi, A, mu, sigma)


def split_starts(parent: dict, factors=SPLIT_FACTORS) -> BatchParams:
    """Starting points for K + 1 states made by splitting one state of a K-state fit.

    For every state k and factor f the state is replaced by two copies with
    volatilities sigma_k / f and sigma_k * f and the parent's mean. Both copies
    inherit row k of the transition matrix, the probability of entering k is
    shared equally between them, and so is the initial probability. With
    f = 1 the result is the parent model written with one more state, and its
    likelihood is the parent's. The new state is appended as the last index.
    """
    pi0 = np.asarray(parent["pi"], dtype=float)
    A0 = np.asarray(parent["A"], dtype=float)
    mu0 = np.asarray(parent["mu"], dtype=float)
    s0 = np.asarray(parent["sigma"], dtype=float)
    K = mu0.size
    members = []
    for k in range(K):
        for f in factors:
            A = np.zeros((K + 1, K + 1))
            A[:K, :K] = A0
            A[K, :K] = A0[k]
            A[:, K] = A[:, k] / 2.0
            A[:, k] = A[:, k] / 2.0
            pi = np.append(pi0, pi0[k] / 2.0)
            pi[k] /= 2.0
            mu = np.append(mu0, mu0[k])
            sigma = np.append(s0, s0[k] * f)
            sigma[k] = s0[k] / f
            members.append({"pi": pi, "A": A, "mu": mu, "sigma": sigma})
    return BatchParams.stack(members)


def _concat(parts) -> BatchParams:
    parts = [p for p in parts if p is not None and p.R > 0]
    return BatchParams(*(np.concatenate([getattr(p, k) for p in parts])
                         for k in ("pi", "A", "mu", "sigma")))


def _em_chunk(args):
    """One chunk of starts to the tolerance rule (a picklable worker)."""
    x, init, n_iter, tol = args
    res = run_em(x, init, n_iter, tol)
    return res.params, res.ll, res.n_iter, res.stopped, res.last_delta


def wide_multistart(x, K: int, *, n_default: int, default_seed: int, n_spread: int,
                    spread_seed: int, parent: dict | None = None,
                    split_factors=SPLIT_FACTORS, n_iter: int = FIT_CAP,
                    tol: float = FIT_TOL, polish_tol: float = POLISH_TOL,
                    polish_step_tol: float = POLISH_STEP_TOL, polish_cap: int = FIT_CAP,
                    reach_tol: float = REACH_TOL, executor=None, chunk: int = 25) -> dict:
    """:func:`fit_protocol` run from a wider set of starting points.

    The starts are, in this order: the ``n_default`` starts of
    ``GaussianHMM.fit`` with ``random_state = default_seed`` (the protocol's
    own); ``n_spread`` starts from :func:`spread_starts`; and, when the best
    fit with K - 1 states is given as ``parent``, the starts of
    :func:`split_starts`. Every start runs to the tolerance rule, the first
    with the largest final log-likelihood is continued by the polish rule of
    :func:`fit_protocol`, and the returned parameters are sorted by sigma.
    The record counts the starts whose terminal log-likelihood lies within
    ``reach_tol`` of the best start's, overall and per family of starts.
    ``default_protocol`` is the fit the protocol's own starts alone select,
    polished the same way, so the gain from the wider search can be read off.
    ``executor`` (with a ``map`` method) spreads chunks of ``chunk`` starts
    over processes; the result does not depend on it.
    """
    x = np.asarray(x, dtype=float)
    fams = {"default": _start_params(x, K, n_default, default_seed) if n_default else None,
            "spread": spread_starts(x, K, n_spread, spread_seed) if n_spread else None,
            "split": split_starts(parent, split_factors) if parent is not None else None}
    counts = {k: (0 if v is None else v.R) for k, v in fams.items()}
    family = np.array([k for k, v in counts.items() for _ in range(v)])
    init = _concat(fams.values())
    bounds = [(i, min(i + chunk, init.R)) for i in range(0, init.R, chunk)]
    jobs = [(x, init.take(np.arange(a, b)), n_iter, tol) for a, b in bounds]
    mapper = map if executor is None else executor.map
    parts = list(mapper(_em_chunk, jobs))
    final = _concat([q[0] for q in parts])
    ll = np.concatenate([q[1] for q in parts])
    n_it = np.concatenate([q[2] for q in parts])
    stopped = np.concatenate([q[3] for q in parts])
    last = np.concatenate([q[4] for q in parts])

    def _polish(pos: int):
        res = run_em(x, final.take([pos]), polish_cap, polish_tol, rule="abs",
                     step_tol=polish_step_tol)
        rec = {"tol": float(polish_tol), "step_tol": float(polish_step_tol),
               "cap": int(polish_cap), "iterations": int(res.n_iter[0]),
               "converged": bool(res.stopped[0]),
               "final_dlogL": float(res.last_delta[0]),
               "final_max_param_step": float(res.last_step[0])}
        p = res.params
        return p, float(loglik(x, p)[0]), rec

    best = int(np.argmax(ll))
    params, ll_best, polish = _polish(best)
    reach = np.abs(ll - ll[best]) < reach_tol
    # distinct terminal optima: starts grouped by log-likelihood within reach_tol
    order = np.argsort(-ll)
    optima = []
    for i in order:
        if optima and abs(ll[i] - optima[-1]["logL_first"]) < reach_tol:
            optima[-1]["n_starts"] += 1
        else:
            optima.append({"logL_first": float(ll[i]), "n_starts": 1})
    record = {
        "protocol": ("fit_protocol from a wider set of starts: the protocol's own "
                     "starts, starts with volatilities spread log-uniformly over "
                     f"{SPREAD_VOL_RANGE[0]}-{SPREAD_VOL_RANGE[1]} x the sample s.d., "
                     "and starts made by splitting each state of the best K-1 fit; "
                     "every start to the tolerance rule, the first start with the "
                     "largest final logL continued until |dlogL| < polish_tol and "
                     "the largest parameter step < polish_step_tol"),
        "n_starts": int(init.R),
        "start_families": counts,
        "default_seed": int(default_seed), "spread_seed": int(spread_seed),
        "spread_vol_range": list(SPREAD_VOL_RANGE),
        "split_factors": [float(f) for f in split_factors] if parent is not None else [],
        "iteration_cap": int(n_iter), "tol": float(tol),
        "selected_start": best, "selected_family": str(family[best]),
        "per_start_logL": [float(v) for v in ll],
        "per_start_iterations": [int(v) for v in n_it],
        "per_start_converged": [bool(v) for v in stopped],
        "per_start_final_dlogL": [float(v) for v in last],
        "all_starts_converged": bool(stopped.all()),
        "n_starts_converged": int(stopped.sum()),
        "reach_tol": float(reach_tol),
        "n_starts_reaching_best": int(reach.sum()),
        "n_reaching_best_by_family": {k: int(reach[family == k].sum()) for k in counts},
        "terminal_optima": optima[:10],
        "polish": polish,
    }
    record["converged"] = bool(stopped[best] and polish["converged"]
                               and record["all_starts_converged"])
    record["logL"] = ll_best
    out = {"K": int(K), "params": sort_by_sigma(params).member(0), "ll": ll_best,
           "record": record}
    if counts["default"]:
        dsel = int(np.argmax(ll[:counts["default"]]))
        dp, dll, dpol = _polish(dsel)
        out["default_protocol"] = {
            "selected_start": dsel, "ll": dll, "polish": dpol,
            "params": sort_by_sigma(dp).member(0),
            "converged": bool(stopped[:counts["default"]].all() and dpol["converged"])}
    return out
