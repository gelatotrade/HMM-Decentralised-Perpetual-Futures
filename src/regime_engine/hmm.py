"""
Gaussian HMM for financial regime detection.

Implements the Rabiner (1989) Forward-Backward + Baum-Welch EM
from scratch in NumPy, with log-space stabilisation to avoid underflow.
Supports K regimes with state-dependent (mu_k, sigma_k).

Also provides deterministic single- and multi-start wrappers around
statsmodels' MarkovAutoregression for the Markov-switching AR(1) model of
the funding premium.

Author: Gregor Albiez, 2026
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

EPS = 1e-300


# --------------------------------------------------------------------------
#  From-scratch Gaussian HMM
# --------------------------------------------------------------------------
@dataclass
class GaussianHMM:
    K: int = 2
    n_iter: int = 200
    tol: float = 1e-6
    n_starts: int = 10
    random_state: int = 42

    # learned parameters (filled after .fit)
    pi: np.ndarray = field(default=None)        # (K,)
    A: np.ndarray = field(default=None)         # (K, K)
    mu: np.ndarray = field(default=None)        # (K,)
    sigma: np.ndarray = field(default=None)     # (K,)
    log_likelihood_: float = field(default=None)
    history_: list = field(default_factory=list)

    # ----- core algorithms -----
    def _emission_logprob(self, x: np.ndarray) -> np.ndarray:
        """Return T x K matrix of log p(x_t | S_t = k)."""
        return norm.logpdf(x[:, None], loc=self.mu[None, :], scale=self.sigma[None, :])

    def _forward_log(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        """Log-space forward pass. Returns log_alpha (T,K) and log-likelihood."""
        T, K = log_b.shape
        log_a = np.empty((T, K))
        log_pi = np.log(self.pi + EPS)
        log_A = np.log(self.A + EPS)

        log_a[0] = log_pi + log_b[0]
        for t in range(1, T):
            # log_a[t, j] = logsumexp_i(log_a[t-1, i] + log_A[i, j]) + log_b[t, j]
            # vectorised across j: shape (K, K) → reduce along i
            tmp = log_a[t - 1, :, None] + log_A           # (K, K)
            m = tmp.max(axis=0)                           # (K,)
            log_a[t] = m + np.log(np.exp(tmp - m).sum(axis=0)) + log_b[t]

        m = log_a[-1].max()
        ll = m + np.log(np.exp(log_a[-1] - m).sum())
        return log_a, ll

    def _backward_log(self, log_b: np.ndarray) -> np.ndarray:
        """Log-space backward pass — vectorised over states."""
        T, K = log_b.shape
        log_beta = np.empty((T, K))
        log_beta[-1] = 0.0
        log_A = np.log(self.A + EPS)
        for t in range(T - 2, -1, -1):
            # for each i: logsumexp_j(log_A[i,j] + log_b[t+1,j] + log_beta[t+1,j])
            tmp = log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :]   # (K, K)
            m = tmp.max(axis=1)
            log_beta[t] = m + np.log(np.exp(tmp - m[:, None]).sum(axis=1))
        return log_beta

    def _e_step(self, x: np.ndarray):
        log_b = self._emission_logprob(x)
        log_a, ll = self._forward_log(log_b)
        log_beta = self._backward_log(log_b)

        # gamma_t(i) = P(S_t = i | x_{1:T})
        log_gamma = log_a + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # xi_t(i,j) = P(S_t = i, S_{t+1} = j | x_{1:T})
        T, K = log_b.shape
        log_A = np.log(self.A + EPS)
        # broadcast: (T-1, K, K)
        terms = (
            log_a[:-1, :, None]                      # (T-1, K, 1)
            + log_A[None, :, :]                      # (1,   K, K)
            + log_b[1:, None, :]                     # (T-1, 1, K)
            + log_beta[1:, None, :]                  # (T-1, 1, K)
        ) - ll
        m = terms.max(axis=(0,))                     # (K, K) per-pair
        xi_sum = (np.exp(terms - m[None, :, :]).sum(axis=0))
        log_xi_sum = m + np.log(xi_sum + EPS)
        xi_sum = np.exp(log_xi_sum)

        return gamma, xi_sum, ll

    def _m_step(self, x: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray):
        gamma_sum = gamma.sum(axis=0)  # (K,)

        self.pi = gamma[0] / gamma[0].sum()

        denom = xi_sum.sum(axis=1, keepdims=True)
        self.A = xi_sum / np.where(denom > 0, denom, 1.0)

        self.mu = (gamma * x[:, None]).sum(axis=0) / np.where(gamma_sum > 0, gamma_sum, 1.0)
        var = (gamma * (x[:, None] - self.mu[None, :]) ** 2).sum(axis=0) / np.where(
            gamma_sum > 0, gamma_sum, 1.0
        )
        self.sigma = np.sqrt(np.maximum(var, 1e-12))

    # ----- public API -----
    def _init_random(self, x: np.ndarray, rng: np.random.Generator):
        self.pi = rng.dirichlet(np.ones(self.K))
        self.A = rng.dirichlet(np.ones(self.K) * 5.0, size=self.K)
        # spread initial means across observed range, jitter scales
        q = np.quantile(x, np.linspace(0.1, 0.9, self.K))
        self.mu = q + rng.normal(0, x.std() * 0.1, size=self.K)
        self.sigma = np.full(self.K, x.std()) * rng.uniform(0.5, 1.5, size=self.K)

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        best_ll = -np.inf
        best_params = None
        master_rng = np.random.default_rng(self.random_state)

        for _start in range(self.n_starts):
            seed = master_rng.integers(0, 10**9)
            rng = np.random.default_rng(seed)
            self._init_random(x, rng)

            history = []
            prev_ll = -np.inf
            for it in range(self.n_iter):
                gamma, xi_sum, ll = self._e_step(x)
                history.append(ll)
                self._m_step(x, gamma, xi_sum)
                if ll - prev_ll < self.tol and it > 5:
                    break
                prev_ll = ll

            if ll > best_ll:
                best_ll = ll
                best_params = (self.pi.copy(), self.A.copy(), self.mu.copy(), self.sigma.copy())
                self.history_ = history

        self.pi, self.A, self.mu, self.sigma = best_params
        self.log_likelihood_ = best_ll

        # Resolve label switching by sigma ascending, which is the rule the
        # paper states for the Gaussian HMM and the only one that is stable here:
        # two of the three fitted means sit within their sampling variability of
        # one another, so a mu-ascending sort permutes states from replication
        # to replication and the label noise swamps the estimator's own.
        # Sorting on sigma keeps index k meaning the same state in every table,
        # figure and JSON artifact this package writes.
        order = np.argsort(self.sigma)
        self.pi = self.pi[order]
        self.A = self.A[np.ix_(order, order)]
        self.mu = self.mu[order]
        self.sigma = self.sigma[order]
        return self

    def filtered_posterior(self, x: np.ndarray) -> np.ndarray:
        """Real-time (causal) posterior P(S_t = k | x_{1:t}). T x K."""
        x = np.asarray(x, dtype=float)
        log_b = self._emission_logprob(x)
        log_a, _ = self._forward_log(log_b)
        log_a -= np.logaddexp.reduce(log_a, axis=1, keepdims=True)
        return np.exp(log_a)

    def smoothed_posterior(self, x: np.ndarray) -> np.ndarray:
        """Two-sided posterior P(S_t = k | x_{1:T}). T x K."""
        x = np.asarray(x, dtype=float)
        gamma, _, _ = self._e_step(x)
        return gamma

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """Most likely state sequence via Viterbi."""
        x = np.asarray(x, dtype=float)
        log_b = self._emission_logprob(x)
        T, K = log_b.shape
        log_pi = np.log(self.pi + EPS)
        log_A = np.log(self.A + EPS)

        delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_A
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(K)] + log_b[t]
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    # ----- diagnostics -----
    def expected_durations(self) -> np.ndarray:
        """E[duration in regime k] = 1 / (1 - A_kk)."""
        return 1.0 / np.maximum(1.0 - np.diag(self.A), 1e-12)

    def stationary_distribution(self) -> np.ndarray:
        """Left eigenvector of A with eigenvalue 1."""
        evals, evecs = np.linalg.eig(self.A.T)
        idx = np.argmin(np.abs(evals - 1.0))
        v = np.real(evecs[:, idx])
        v = v / v.sum()
        return v

    def n_params(self) -> int:
        # transition: K(K-1), means: K, vars: K, initial: K-1
        return self.K * (self.K - 1) + 2 * self.K + (self.K - 1)

    def aic(self) -> float:
        return 2 * self.n_params() - 2 * self.log_likelihood_

    def bic(self, T: int) -> float:
        return self.n_params() * np.log(T) - 2 * self.log_likelihood_


# --------------------------------------------------------------------------
#  Markov-Switching AR(1) for funding rates (statsmodels wrapper)
# --------------------------------------------------------------------------
MSAR_SEED = 20260802
"""Default seed for the MS-AR start-parameter search.

statsmodels' ``_start_params_search`` draws its random perturbations from the
*legacy global* NumPy RNG (``np.random.uniform``).  Left unseeded the fit is not
reproducible: the log-likelihood varies across runs in the sixth decimal, and a
run can raise ``LinAlgError: SVD did not converge``.  We therefore seed the
global RNG for the duration of the fit and restore the caller's state
afterwards, and we record the seed in ``output/summary.json``.
"""


def fit_ms_ar1(y: np.ndarray, k_regimes: int = 2,
               switching_variance: bool = True,
               switching_ar: bool = True,
               seed: int = MSAR_SEED,
               search_reps: int = 20,
               em_iter: int = 50,
               seed_retries: int = 5):
    """Fit an MS-AR(1) with regime-dependent (mu, phi, sigma), deterministically.

    The funding-rate dynamics
        y_t - mu_{S_t} = phi_{S_t} (y_{t-1} - mu_{S_{t-1}}) + sigma_{S_t} eps_t
    capture asymmetric mean-reversion across regimes; the likelihood conditions
    on the pair (S_t, S_{t-1}).  Note that statsmodels parameterises this
    *mean-adjusted* form: ``const[k]`` is the regime MEAN mu_k, not an
    intercept, so within regime k (S_{t-1} = S_t = k) the conditional mean is
    mu_k (1 - phi_k) + phi_k y_{t-1}.

    Returns ``(model, res, seed_used)``.  ``seed_used`` equals ``seed`` unless
    that seed produced a numerical failure, in which case the next seed in the
    deterministic sequence ``seed, seed+1, ...`` that succeeded is returned.
    """
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

    y = np.asarray(y, dtype=float)
    model = MarkovAutoregression(
        y,
        k_regimes=k_regimes,
        order=1,
        switching_ar=switching_ar,
        switching_variance=switching_variance,
        switching_trend=True,
    )

    state = np.random.get_state()          # do not disturb the caller's RNG
    try:
        last_error = None
        for attempt in range(max(1, seed_retries)):
            seed_used = int(seed) + attempt
            np.random.seed(seed_used)
            try:
                # statsmodels' `start_params` calls np.linalg.pinv on the lagged
                # design matrix; on this data that raises benign divide/overflow
                # RuntimeWarnings from BLAS inside pinv's SVD reconstruction.
                # The returned start values are finite and the fit converges, so
                # silence the FP warnings around the call rather than the module.
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    res = model.fit(em_iter=em_iter, search_reps=search_reps, disp=False)
                return model, res, seed_used
            except Exception as exc:       # numerical failure in the start search
                last_error = exc
                print(f"[MS-AR] seed {seed_used} failed ({type(exc).__name__}: {exc}); "
                      f"retrying with seed {seed_used + 1}")
        raise RuntimeError(
            f"MS-AR fit failed for seeds {seed}..{seed + seed_retries - 1}"
        ) from last_error
    finally:
        np.random.set_state(state)


def fit_ms_ar1_multistart(y: np.ndarray, n_starts: int = 40,
                          seed: int = MSAR_SEED, search_reps: int = 40,
                          verbose: bool = True, **kwargs):
    """Fit an MS-AR(1) from many independent starts and keep the best.

    This likelihood is multi-modal, and on the Hyperliquid BTC premium its
    *dominant basin of attraction is not its maximum*.  On the 2 Oct 2025 --
    28 Apr 2026 window (T = 5,001; nobs = 5,000), running this function at its
    defaults finds exactly two optima: 24 of 40 starts converge to
    logL = -4828.67 and the remaining 16 reach -4808.53, higher by 20.14
    points.  Because the reported half-life is a steep transform of phi, that
    difference moves the calm / stressed half-life ratio from 11.2 (calm
    35.55 h, stressed 3.18 h, the maximum) to 24.6 (42.89 h, 1.74 h, the
    inferior optimum).  A single-start fit lands on the inferior optimum about
    three times in five, and lands there stably enough to look converged --
    which optimum a given seed reaches is not predictable from the fit itself.

    The counts were obtained with statsmodels 0.14.6 / NumPy 2.0.2 /
    CPython 3.9.6; they are deterministic given the seed base, ``n_starts``
    and ``search_reps``, and they do move with ``search_reps`` (26 / 14 at
    ``search_reps=20``).  The two optima themselves do not move.

    Returns ``(model, res, seed_used, diagnostics)`` where ``diagnostics``
    records every optimum found and how often, so the multi-modality is
    visible in the written output.
    """
    from collections import Counter

    best = None
    optima: Counter = Counter()
    failures = 0
    for i in range(max(1, n_starts)):
        s = int(seed) + i
        try:
            model, res, seed_used = fit_ms_ar1(
                y, seed=s, search_reps=search_reps, seed_retries=1, **kwargs)
        except Exception:
            failures += 1
            continue
        optima[round(float(res.llf), 2)] += 1
        if best is None or res.llf > best[1].llf:
            best = (model, res, seed_used)

    if best is None:
        raise RuntimeError(f"MS-AR multi-start failed on all {n_starts} starts")

    ranked = optima.most_common()
    diagnostics = {
        "n_starts": int(n_starts),
        "search_reps": int(search_reps),
        "seed_base": int(seed),
        "failures": int(failures),
        "optima_found": [{"llf": llf, "count": c} for llf, c in ranked],
        "best_llf": float(best[1].llf),
        "best_seed": int(best[2]),
        "modal_llf": ranked[0][0],
        "modal_is_best": bool(abs(ranked[0][0] - round(float(best[1].llf), 2)) < 1e-9),
    }
    if verbose:
        print(f"[MS-AR] multi-start: {len(ranked)} distinct optima over "
              f"{n_starts} starts ({failures} failures)")
        for llf, c in ranked:
            mark = "  <-- reported (max)" if llf == round(float(best[1].llf), 2) else ""
            print(f"         logL {llf:12.2f}  x{c:3d}{mark}")
        if not diagnostics["modal_is_best"]:
            print("[MS-AR] WARNING: the most frequent optimum is NOT the maximum. "
                  "A single-start fit would have reported the inferior one.")
    return best[0], best[1], best[2], diagnostics


if __name__ == "__main__":
    # Sanity check: simulate from a known 2-state HMM, recover parameters.
    rng = np.random.default_rng(0)
    A_true = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu_true = np.array([0.0005, -0.001])
    sigma_true = np.array([0.005, 0.020])

    T = 5000
    states = np.zeros(T, dtype=int)
    states[0] = 0
    for t in range(1, T):
        states[t] = rng.choice(2, p=A_true[states[t - 1]])
    x = rng.normal(mu_true[states], sigma_true[states])

    hmm = GaussianHMM(K=2, n_starts=8, random_state=1).fit(x)
    print("True mu:    ", mu_true)
    print("Fitted mu:  ", hmm.mu)
    print("True sigma: ", sigma_true)
    print("Fitted sig: ", hmm.sigma)
    print("True A:\n", A_true)
    print("Fitted A:\n", hmm.A)
    print("Expected durations:", hmm.expected_durations())
    print("Log-likelihood:    ", hmm.log_likelihood_)
    print("BIC:               ", hmm.bic(T))
