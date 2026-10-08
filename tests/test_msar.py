"""MS-AR(1) half-life, delta method, regime ordering and determinism.

Exercises ``regime_engine.msar``, which keeps the MS-AR fit separate from
``analysis.py`` so these properties are testable.  The fits here run on short
simulated series to keep the suite fast; the full-panel fit lives in the
analysis pipeline.
"""
import numpy as np
import pytest

from regime_engine.msar import (
    REGIME_LABELS,
    Z95,
    fit_msar_premium,
    half_life,
    half_life_derivative,
)


def test_half_life_formula():
    # tau = ln(0.5)/ln|phi|; phi=0.5 -> tau=1
    phi = 0.5
    tau = np.log(0.5) / np.log(abs(phi))
    assert np.isclose(tau, 1.0)
    assert np.isclose(half_life(0.5), 1.0)
    assert np.isclose(half_life(0.25), 0.5)


def test_half_life_derivative_matches_finite_difference():
    """dtau/dphi = -ln0.5 / (phi (ln|phi|)^2) — the delta-method gradient."""
    for phi in (0.5, 0.9, 0.97, 0.9839611690205514):
        h = 1e-7
        numeric = (half_life(phi + h) - half_life(phi - h)) / (2 * h)
        analytic = half_life_derivative(phi)
        assert analytic == pytest.approx(numeric, rel=1e-5), phi
        assert analytic > 0                       # tau increases with persistence


def test_half_life_derivative_is_large_near_unity():
    """The sensitivity that makes a bare point estimate of tau untrustworthy."""
    assert half_life_derivative(0.98396) == pytest.approx(2695, rel=0.01)


@pytest.fixture(scope="module")
def simulated_msar():
    """Two-regime AR(1): a calm, weakly persistent state and a volatile one."""
    rng = np.random.default_rng(7)
    P = np.array([[0.98, 0.02], [0.06, 0.94]])   # rows: from -> to
    mu = np.array([-4.0, 1.0])
    phi = np.array([0.60, 0.90])
    sigma = np.array([0.4, 2.0])
    T = 2500
    s = np.zeros(T, dtype=int)
    y = np.zeros(T)
    y[0] = mu[0]
    for t in range(1, T):
        s[t] = rng.choice(2, p=P[s[t - 1]])
        y[t] = mu[s[t]] + phi[s[t]] * (y[t - 1] - mu[s[t]]) + sigma[s[t]] * rng.normal()
    return dict(y=y, mu=mu, phi=phi, sigma=sigma, T=T)


def test_regime_ordering_by_variance(simulated_msar):
    """Regimes are relabelled sigma2 ascending -> index 0 = calm."""
    res = fit_msar_premium(simulated_msar["y"], n_starts=3, search_reps=10)
    assert res["sigma2"][0] < res["sigma2"][1]
    assert [b["regime"] for b in res["half_lives"]] == list(REGIME_LABELS)
    # the ordering must permute every per-regime vector consistently
    assert res["phi"][0] == pytest.approx(res["half_lives"][0]["phi"])
    assert res["phi"][1] == pytest.approx(res["half_lives"][1]["phi"])


def test_fit_recovers_simulated_parameters(simulated_msar):
    res = fit_msar_premium(simulated_msar["y"], n_starts=3, search_reps=10)
    assert res["sigma"][0] == pytest.approx(simulated_msar["sigma"][0], rel=0.25)
    assert res["sigma"][1] == pytest.approx(simulated_msar["sigma"][1], rel=0.25)
    assert res["phi"][0] == pytest.approx(simulated_msar["phi"][0], abs=0.10)
    assert res["phi"][1] == pytest.approx(simulated_msar["phi"][1], abs=0.10)
    # const[k] is the regime MEAN, not an intercept; the dominant regime's mean
    # is the one this short simulation identifies sharply.
    assert res["mu_const"][0] == pytest.approx(simulated_msar["mu"][0], abs=0.6)


def test_const_is_the_regime_mean_not_an_intercept(simulated_msar):
    """The fitted line in each regime of the phase plot (fig3_funding_dynamics).

    statsmodels estimates the mean-adjusted form
        y_t = a_{S_t} + phi_{S_t} (y_{t-1} - a_{S_{t-1}}) + eps,
    so within a regime the fitted line is ``a_k (1 - phi_k) + phi_k y_{t-1}``.
    Plotting ``a_k + phi_k y_{t-1}`` shifts the line by ``a_k phi_k``.
    """
    y = simulated_msar["y"]
    res = fit_msar_premium(y, n_starts=3, search_reps=10)
    conditional = np.asarray(
        res["model"].predict_conditional(np.asarray(res["result"].params))
    )                                    # axes: (S_t, S_{t-1}, obs)
    for k in range(2):
        raw = res["order"][k]            # index in statsmodels' own labelling
        mu_k, phi_k = res["mu_const"][k], res["phi"][k]
        correct = mu_k * (1.0 - phi_k) + phi_k * y[:-1]
        wrong = mu_k + phi_k * y[:-1]
        assert np.max(np.abs(conditional[raw, raw] - correct)) < 1e-10
        # and the wrong form is off by exactly a_k * phi_k, a vertical offset
        assert np.allclose(wrong - correct, mu_k * phi_k)


def test_fit_is_deterministic_given_the_seed(simulated_msar):
    """statsmodels' start search draws from the unseeded global RNG unless we
    seed it; two runs with the same seed must agree to the last bit."""
    a = fit_msar_premium(simulated_msar["y"], seed=123, n_starts=3, search_reps=10)
    np.random.seed(999)                        # deliberately disturb global state
    b = fit_msar_premium(simulated_msar["y"], seed=123, n_starts=3, search_reps=10)
    assert a["llf"] == b["llf"]
    assert np.array_equal(a["phi"], b["phi"])
    # Under the multi-start protocol the reported seed is that of the winning
    # start, which is deterministic but need not be the base seed.
    assert a["seed"] == b["seed"]
    assert a["seed"] in range(123, 123 + 3)
    assert a["multistart"]["seed_base"] == b["multistart"]["seed_base"] == 123

    # Single-start mode still reports exactly the seed it was given, and can
    # never beat the multi-start maximum.
    c = fit_msar_premium(simulated_msar["y"], seed=123, n_starts=1)
    assert c["seed"] == 123
    assert c["llf"] <= a["llf"] + 1e-9


def test_inference_block_is_complete(simulated_msar):
    res = fit_msar_premium(simulated_msar["y"], n_starts=3, search_reps=10)
    assert res["nobs"] == simulated_msar["T"] - 1        # AR(1) costs one obs
    for key in ("llf", "aic", "bic"):
        assert res[key] is not None and np.isfinite(res[key])
    assert len(res["param_table"]) == len(res["param_names"])
    for row in res["param_table"]:
        assert row["se"] is not None and row["se"] > 0
        assert row["z"] is not None
        assert row["ci95_low"] < row["coef"] < row["ci95_high"]
        assert row["ci95_high"] - row["ci95_low"] == pytest.approx(2 * Z95 * row["se"])


def test_half_life_confidence_intervals(simulated_msar):
    res = fit_msar_premium(simulated_msar["y"], n_starts=3, search_reps=10)
    for block in res["half_lives"]:
        lo, hi = block["half_life_ci95_delta"]
        assert lo < block["half_life_hours"] < hi
        assert block["half_life_se_delta"] == pytest.approx(
            abs(half_life_derivative(block["phi"])) * block["phi_se"]
        )
    ratio = res["half_life_ratio"]
    assert ratio["value"] == pytest.approx(
        res["half_life_hours"][0] / res["half_life_hours"][1]
    )
    assert ratio["se_delta"] is not None and ratio["se_delta"] > 0
    assert ratio["ci95_delta"][0] < ratio["value"] < ratio["ci95_delta"][1]


def test_transition_matrix_is_row_stochastic(simulated_msar):
    res = fit_msar_premium(simulated_msar["y"], n_starts=3, search_reps=10)
    P = np.asarray(res["transition_matrix"], dtype=float)
    assert P.shape == (2, 2)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)
