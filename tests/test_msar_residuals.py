"""The MS-AR one-step forecast must condition on the state *pair*.

The paper's MS-AR equation (``eq:msar``, Section 2.2) is

    y_t - mu_{S_t} = phi_{S_t} (y_{t-1} - mu_{S_{t-1}}) + sigma_{S_t} eps_t

and the paper says so explicitly: "the likelihood conditions on the pair
(S_t, S_{t-1})", so the filter runs over K^2 state combinations. The lagged
level subtracted is the level of the PREVIOUS regime. A forecast built from
P(S_t | F_{t-1}) alone can only use mu_{S_t} on both sides, which is the
same-state case; it drops the term phi_j (mu_j - mu_i) on every
transition. On the paper's own fit that term reaches 0.43 bps against a calm
regime sigma of 0.53 bps, with 4.7% of the predicted mass off the diagonal.
"""
import numpy as np
import pytest

from regime_engine.diagnostics import msar_standardised_residuals


class _Result:
    """The minimum of a statsmodels results object that the routine touches."""

    def __init__(self, filtered):
        self.filtered_marginal_probabilities = filtered
        self.predicted_marginal_probabilities = filtered


def _fit(mu, phi, sigma2, P, filtered):
    return {
        "result": _Result(np.asarray(filtered, float)),
        "order": np.array([0, 1]),
        "mu_const": np.asarray(mu, float),
        "phi": np.asarray(phi, float),
        "sigma2": np.asarray(sigma2, float),
        "filtered": np.asarray(filtered, float),
        "transition_matrix": np.asarray(P, float),
    }


def _reference(fit, y):
    """The K^2 mixture, written straight from the MS-AR equation (eq:msar)."""
    filt = np.asarray(fit["filtered"], float)
    P = np.asarray(fit["transition_matrix"], float)
    mu, phi = fit["mu_const"], fit["phi"]
    s2 = fit["sigma2"]
    n, K = filt.shape
    y = np.asarray(y, float)[-n:]
    out = []
    for t in range(1, n):
        mean = 0.0
        second = 0.0
        for i in range(K):
            for j in range(K):
                q = filt[t - 1, i] * P[i, j]
                m = mu[j] + phi[j] * (y[t - 1] - mu[i])
                mean += q * m
                second += q * (m * m + s2[j])
        var = second - mean * mean
        out.append((y[t] - mean) / np.sqrt(var))
    return np.array(out)


@pytest.fixture
def scenario():
    """Two regimes with clearly different levels and real off-diagonal mass."""
    rng = np.random.default_rng(11)
    n = 400
    filtered = rng.dirichlet([2.0, 2.0], size=n)
    P = np.array([[0.80, 0.20], [0.35, 0.65]])
    y = np.concatenate([[0.0], rng.normal(-3.0, 1.0, n)])
    return _fit(mu=[-2.0, -5.0], phi=[0.95, 0.60], sigma2=[0.25, 4.0],
                P=P, filtered=filtered), y


def test_residuals_match_the_state_pair_mixture(scenario):
    fit, y = scenario
    got = msar_standardised_residuals(fit, y)
    want = _reference(fit, y)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(got, want, atol=1e-10), np.abs(got - want).max()


def test_the_pair_term_actually_bites(scenario):
    """Guard the guard: the scenario must distinguish the two formulas."""
    fit, y = scenario
    filt = fit["filtered"]
    P = fit["transition_matrix"]
    mu, phi = fit["mu_const"], fit["phi"]
    n = filt.shape[0]
    yy = np.asarray(y, float)[-n:]
    same, pair = [], []
    for t in range(1, n):
        pred = filt[t - 1] @ P
        same.append(sum(pred[j] * (mu[j] + phi[j] * (yy[t - 1] - mu[j])) for j in range(2)))
        pair.append(sum(filt[t - 1, i] * P[i, j] * (mu[j] + phi[j] * (yy[t - 1] - mu[i]))
                        for i in range(2) for j in range(2)))
    gap = np.abs(np.array(same) - np.array(pair))
    assert gap.max() > 0.2, gap.max()


def test_regimes_with_equal_levels_make_the_two_formulas_agree(scenario):
    """Sanity: the dropped term is phi_j (mu_j - mu_i), so equal mu removes it."""
    fit, y = scenario
    fit["mu_const"] = np.array([-3.0, -3.0])
    got = msar_standardised_residuals(fit, y)
    want = _reference(fit, y)
    assert np.allclose(got, want, atol=1e-10)
