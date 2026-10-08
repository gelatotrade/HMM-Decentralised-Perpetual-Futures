"""Ground-truth tests for the Gaussian HMM.

These give an immediately-green baseline: the from-scratch Forward-Backward + EM
must recover the parameters of a simulated 2-state chain. Run with `pytest -q`.
"""
import numpy as np
import pytest

from regime_engine.hmm import GaussianHMM


@pytest.fixture(scope="module")
def simulated():
    """Simulate T obs from a known 2-state Gaussian HMM."""
    rng = np.random.default_rng(0)
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu = np.array([0.0005, -0.001])
    sigma = np.array([0.005, 0.020])
    T = 5000
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(2, p=A[states[t - 1]])
    x = rng.normal(mu[states], sigma[states])
    return dict(x=x, A=A, mu=mu, sigma=sigma, T=T)


def test_sigma_recovery(simulated):
    """Volatilities recovered to within 5% (the discriminating parameter)."""
    hmm = GaussianHMM(K=2, n_starts=8, random_state=1).fit(simulated["x"])
    recovered = np.sort(hmm.sigma)
    truth = np.sort(simulated["sigma"])
    assert np.allclose(recovered, truth, rtol=0.05), (recovered, truth)


def test_states_come_back_ordered_by_sigma_ascending(simulated):
    """The paper's labelling rule, enforced where the labels are assigned.

    Section 2.1 orders states by sigma ascending and Appendix B gives the case
    against mu-ascending: on this data two of three means sit inside their
    sampling variability, so that rule reorders states between replications.
    The estimator has to be the thing that guarantees the order, otherwise
    every downstream artifact (tables, figures, JSON) carries a different
    permutation from the manuscript.
    """
    hmm = GaussianHMM(K=2, n_starts=8, random_state=1).fit(simulated["x"])
    assert np.all(np.diff(hmm.sigma) > 0), hmm.sigma


def test_reordering_permutes_every_parameter_together(simulated):
    """Relabelling must not decouple a state's mean, variance and kernel row."""
    hmm = GaussianHMM(K=2, n_starts=8, random_state=1).fit(simulated["x"])
    # The simulated chain pairs the high-sigma state with the negative mean and
    # the lower self-transition probability; that pairing has to survive.
    loud = int(np.argmax(hmm.sigma))
    quiet = int(np.argmin(hmm.sigma))
    assert hmm.mu[loud] < hmm.mu[quiet], (hmm.mu, hmm.sigma)
    assert hmm.A[loud, loud] < hmm.A[quiet, quiet], hmm.A
    assert np.allclose(hmm.A.sum(axis=1), 1.0)


def test_transition_persistence(simulated):
    """Both regimes are persistent: diagonal of A well above 0.5."""
    hmm = GaussianHMM(K=2, n_starts=8, random_state=1).fit(simulated["x"])
    assert np.all(np.diag(hmm.A) > 0.5)


def test_expected_durations_positive(simulated):
    hmm = GaussianHMM(K=2, n_starts=8, random_state=1).fit(simulated["x"])
    d = hmm.expected_durations()
    assert np.all(d > 1.0) and np.all(np.isfinite(d))


def test_filtered_posterior_is_causal_distribution(simulated):
    """Filtered posterior rows are valid probability distributions."""
    hmm = GaussianHMM(K=2, n_starts=6, random_state=1).fit(simulated["x"])
    post = hmm.filtered_posterior(simulated["x"])
    assert post.shape == (simulated["T"], 2)
    assert np.allclose(post.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(post >= -1e-9)


def test_stationary_distribution_sums_to_one(simulated):
    hmm = GaussianHMM(K=2, n_starts=6, random_state=1).fit(simulated["x"])
    pi = hmm.stationary_distribution()
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(pi >= -1e-9)
