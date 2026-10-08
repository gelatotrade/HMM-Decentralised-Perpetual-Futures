"""Tests for regime_engine.diagnostics.

Three groups of properties:

* the parametric bootstrap must recover known parameters and must cover them at
  roughly the nominal rate;
* the diagnostic statistics must not reject on a series built to satisfy the
  null and must reject on one built to violate it;
* the information criteria must be on one consistent scale across every model
  in the benchmark table -- an inconsistency that raises no error and only
  shows up as a wrong ranking.
"""
import json
import pathlib

import numpy as np
import pytest

from regime_engine.diagnostics import (
    ANNUALISATION,
    _em_from,
    _ic,
    _sigma_rank_signature,
    _stationary,
    arch_lm,
    bootstrap_hmm,
    fit_garch11,
    fit_single_gaussian,
    garch11_variance,
    hmm_predictive_moments,
    hmm_standardised_residuals,
    jarque_bera,
    ljung_box,
    reconstruct_transition_matrix,
    residual_battery,
    simulate_gaussian_hmm,
)
from regime_engine.hmm import GaussianHMM
from regime_engine.paths import DEFAULT_OUT

# --------------------------------------------------------------------------
#  A well-separated 3-state chain, close in shape to the paper's fit but with
#  enough separation that a 500-observation-per-state sample identifies it.
# --------------------------------------------------------------------------
A_TRUE = np.array([
    [0.75, 0.00, 0.25],
    [0.01, 0.92, 0.07],
    [0.09, 0.03, 0.88],
])
MU_TRUE = np.array([-5.0e-4, 0.2e-4, 0.4e-4])
SIGMA_TRUE = np.array([90e-4, 15e-4, 35e-4])


# ==========================================================================
#  0.  Shared plumbing
# ==========================================================================
def test_paper_fit_hyperparameters_match_analysis():
    """diagnostics duplicates analysis.py's HMM settings; they must not drift."""
    from regime_engine import analysis, diagnostics
    assert diagnostics.HMM_SEED_BASE == analysis.HMM_SEED_BASE
    assert diagnostics.HMM_N_STARTS == analysis.HMM_N_STARTS
    assert diagnostics.HMM_N_ITER == analysis.HMM_N_ITER


def test_annualisation_constant():
    assert np.isclose(ANNUALISATION, np.sqrt(24 * 365))


# ==========================================================================
#  1.  Transition-matrix reconstruction
# ==========================================================================
def test_reconstruction_round_trips_a_known_kernel():
    A = reconstruct_transition_matrix(np.diag(A_TRUE), _stationary(A_TRUE), (0, 1))
    assert np.allclose(A, A_TRUE, atol=1e-12)
    assert np.allclose(A.sum(axis=1), 1.0)


def test_reconstruction_recovers_archived_subsample_kernel():
    """The load-bearing test for the full-sample bootstrap.

    The full-sample archive kept only diag(A) and pi.  The sub-sample archive
    kept the whole kernel, so it is the one place the reconstruction can be
    checked against ground truth -- and it must be exact, not close.
    """
    path = pathlib.Path(DEFAULT_OUT) / "summary.json"
    A_archived = np.array(json.loads(path.read_text())["return_K3"]["A"])
    # Locate the boundary cell rather than assuming where the labelling rule
    # puts it: sigma-ascending and mu-ascending orderings place the same
    # economic transition at different (i, j), and the reconstruction is a
    # statement about the kernel, not about the sort key.
    zero_cell = np.unravel_index(np.argmin(A_archived), A_archived.shape)
    assert A_archived[zero_cell] < 1e-12, A_archived
    A = reconstruct_transition_matrix(
        np.diag(A_archived), _stationary(A_archived), zero_cell)
    assert np.abs(A - A_archived).max() < 1e-12


def test_reconstruction_rejects_non_K3():
    with pytest.raises(ValueError):
        reconstruct_transition_matrix([0.9, 0.9], [0.5, 0.5], (0, 1))


# ==========================================================================
#  2.  Parametric bootstrap
# ==========================================================================
@pytest.fixture(scope="module")
def boot():
    """A small but real bootstrap on the synthetic chain above."""
    return bootstrap_hmm({"mu": MU_TRUE, "sigma": SIGMA_TRUE, "A": A_TRUE},
                         T=4000, n_reps=60, seed=7, n_jobs=4, progress=False)


def test_simulation_respects_the_transition_kernel():
    """A_12 = 0 in the kernel means zero 1->2 transitions in the simulated path."""
    rng = np.random.default_rng(3)
    _, states = simulate_gaussian_hmm(A_TRUE, MU_TRUE, SIGMA_TRUE, 40000, rng)
    counts = np.zeros((3, 3))
    np.add.at(counts, (states[:-1], states[1:]), 1)
    assert counts[0, 1] == 0
    empirical = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(empirical - A_TRUE).max() < 0.02


def test_simulation_state_shares_match_the_stationary_distribution():
    rng = np.random.default_rng(11)
    _, states = simulate_gaussian_hmm(A_TRUE, MU_TRUE, SIGMA_TRUE, 60000, rng)
    share = np.bincount(states, minlength=3) / states.size
    assert np.abs(share - _stationary(A_TRUE)).max() < 0.02


def test_em_from_truth_recovers_the_generating_parameters():
    rng = np.random.default_rng(5)
    x, _ = simulate_gaussian_hmm(A_TRUE, MU_TRUE, SIGMA_TRUE, 20000, rng)
    hmm, _ = _em_from(x, {"pi": _stationary(A_TRUE), "A": A_TRUE,
                          "mu": MU_TRUE, "sigma": SIGMA_TRUE})
    order = np.argsort(hmm.sigma)
    assert np.allclose(hmm.sigma[order], np.sort(SIGMA_TRUE), rtol=0.05)
    assert np.abs(np.diag(hmm.A[np.ix_(order, order)])
                  - np.diag(A_TRUE)[np.argsort(SIGMA_TRUE)]).max() < 0.05


def test_bootstrap_is_approximately_unbiased_for_sigma(boot):
    """sigma is the well-identified parameter; the bootstrap must centre on it."""
    for k, block in enumerate(boot["sigma_ascending"]["sigma_bps_per_hour"]):
        truth = np.sort(SIGMA_TRUE)[k] * 1e4
        assert abs(block["bias"]) < 0.1 * block["se"] + 0.02 * truth, (k, block)


def test_bootstrap_intervals_cover_the_truth(boot):
    """Every 95% percentile interval must contain the data-generating value."""
    order = np.argsort(SIGMA_TRUE)
    for k in range(3):
        s = boot["sigma_ascending"]["sigma_bps_per_hour"][k]
        lo, hi = s["ci95"]
        assert lo <= SIGMA_TRUE[order][k] * 1e4 <= hi
        d = boot["sigma_ascending"]["expected_duration_hours"][k]
        tau = 1.0 / (1.0 - np.diag(A_TRUE)[order][k])
        assert d["ci95"][0] <= tau <= d["ci95"][1]


def test_bootstrap_standard_errors_are_positive_and_finite(boot):
    for block in ("mu_bps_per_hour", "sigma_bps_per_hour", "annualised_sigma_pct",
                  "expected_duration_hours", "stationary"):
        for cell in boot["mu_ascending"][block]:
            assert cell["se"] is not None and cell["se"] > 0
            assert np.isfinite(cell["ci95"][0]) and np.isfinite(cell["ci95"][1])
            assert cell["ci95"][0] <= cell["boot_mean"] <= cell["ci95"][1]


def test_bootstrap_annualised_sigma_is_the_transform_of_sigma(boot):
    """The annualised column must be sqrt(24*365)*sigma applied rep by rep."""
    for k in range(3):
        s = boot["mu_ascending"]["sigma_bps_per_hour"][k]
        a = boot["mu_ascending"]["annualised_sigma_pct"][k]
        factor = ANNUALISATION * 100 / 1e4
        assert np.isclose(a["boot_mean"], s["boot_mean"] * factor, rtol=1e-10)
        assert np.isclose(a["ci95"][0], s["ci95"][0] * factor, rtol=1e-10)


def test_bootstrap_rows_of_A_sum_to_one_in_expectation(boot):
    for i in range(3):
        row = sum(boot["mu_ascending"]["A"][i][j]["boot_mean"] for j in range(3))
        assert np.isclose(row, 1.0, atol=1e-6)


def test_boundary_cell_is_degenerate_under_the_null(boot):
    """Simulating from A_13 = 0 can only ever return A_13 = 0: report, do not infer."""
    cell = boot["mu_ascending"]["A"][0][1]
    frac = boot["boundary"]["frac_reps_at_zero_sigma_labelling"]
    zero_i, zero_j = np.argwhere(A_TRUE == 0)[0]
    order = np.argsort(SIGMA_TRUE)
    inv = np.argsort(order)
    assert frac[inv[zero_i]][inv[zero_j]] == 1.0
    assert cell is not None


def test_rule_of_three_bound_is_reported(boot):
    b = boot["boundary"]
    assert b["rule_of_three_bound_per_hour"] > 0
    expected = 3.0 / (_stationary(A_TRUE)[0] * 4000)
    assert abs(b["rule_of_three_bound_per_hour"] - expected) < 0.25 * expected


def test_label_switching_signature_is_computed_from_both_orderings():
    mu = np.array([-5.0, 0.2, 0.4])
    sigma = np.array([90.0, 15.0, 35.0])
    assert _sigma_rank_signature(mu, sigma) == (2, 0, 1)
    # Swap the two upper means: the mu-ascending labelling now permutes them.
    assert _sigma_rank_signature(np.array([-5.0, 0.4, 0.2]), sigma) == (2, 1, 0)


def test_label_switching_block_is_present_and_consistent(boot):
    lab = boot["label_switching"]
    assert tuple(lab["point_signature"]) == _sigma_rank_signature(MU_TRUE, SIGMA_TRUE)
    assert 0.0 <= lab["disagreement_rate"] <= 1.0
    assert sum(d["count"] for d in lab["signature_counts"]) == boot["n_reps"]


def test_mu_sorted_and_sigma_sorted_blocks_differ_when_labels_switch(boot):
    """The whole point of reporting both: they are not the same numbers."""
    if boot["label_switching"]["disagreement_rate"] == 0:
        pytest.skip("no label switching in this small run")
    se_mu = [c["se"] for c in boot["mu_ascending"]["mu_bps_per_hour"]]
    se_sg = [c["se"] for c in boot["sigma_ascending"]["mu_bps_per_hour"]]
    assert se_mu != se_sg


# ==========================================================================
#  3.  Residual diagnostics
# ==========================================================================
def _rejection_rate(stat_p, n_samples=40, seed=0, alpha=0.05):
    """Share of independent white-noise draws on which a test rejects at ``alpha``.

    Asserting ``p > 0.05`` on ONE white-noise sample is a coin flip with a 5%
    failure rate by construction -- a flaky test that says nothing.  What we
    actually want to know is that the test has approximately its nominal size,
    so we check the rejection rate over many draws.
    """
    rng = np.random.default_rng(seed)
    return np.mean([stat_p(rng.standard_normal(2000)) < alpha
                    for _ in range(n_samples)])


def test_ljung_box_has_approximately_nominal_size_on_white_noise():
    for lag in (10, 24):
        rate = _rejection_rate(lambda z, lag=lag: ljung_box(z, (lag,))[0]["p"], seed=100 + lag)
        assert rate < 0.20, (lag, rate)


def test_ljung_box_rejects_on_an_ar1():
    rng = np.random.default_rng(1)
    z = np.empty(4000)
    z[0] = 0.0
    for t in range(1, 4000):
        z[t] = 0.5 * z[t - 1] + rng.standard_normal()
    for d in ljung_box(z, (10, 24)):
        assert d["p"] < 1e-10, d


def test_arch_lm_has_approximately_nominal_size_on_homoskedastic_noise():
    rate = _rejection_rate(lambda z: arch_lm(z, 12)["p"], seed=202)
    assert rate < 0.20, rate


def test_arch_lm_rejects_on_a_garch_series():
    x, _ = _simulate_garch(4000, 1.0, 0.10, 0.85, seed=3)
    assert arch_lm(x, 12)["p"] < 1e-6
    for d in ljung_box(x ** 2, (10, 24)):
        assert d["p"] < 1e-6, d


def test_jarque_bera_separates_normal_from_student_t():
    rate = _rejection_rate(lambda z: jarque_bera(z)["p"], seed=404)
    assert rate < 0.20, rate
    heavy = jarque_bera(np.random.default_rng(4).standard_t(3, size=4000))
    assert heavy["p"] < 1e-10 and heavy["excess_kurtosis"] > 2


def test_residual_battery_reports_every_statistic():
    b = residual_battery(np.random.default_rng(5).standard_normal(2000), "wn")
    for key in ("ljung_box_z", "ljung_box_z2", "ljung_box_abs_z", "ljung_box_log_z2",
                "arch_lm", "arch_lm_trimmed5", "ljung_box_z2_trimmed5",
                "jarque_bera"):
        assert key in b
    assert b["n"] == 2000
    assert abs(b["sd"] - 1.0) < 0.1


def test_hmm_residuals_are_white_and_unit_variance_when_the_model_is_true():
    """The end-to-end check: on data actually generated by the HMM, the causal
    standardisation must produce residuals with unit variance and no structure."""
    rng = np.random.default_rng(6)
    x, _ = simulate_gaussian_hmm(A_TRUE, MU_TRUE, SIGMA_TRUE, 8000, rng)
    hmm = GaussianHMM(K=3, n_starts=6, n_iter=200, random_state=1).fit(x)
    z = hmm_standardised_residuals(hmm, x, use="filtered")
    assert abs(z.std(ddof=1) - 1.0) < 0.05
    for d in ljung_box(z, (10, 24)):
        assert d["p"] > 0.01, d
    assert arch_lm(z, 12)["p"] > 0.01


def test_predictive_moments_use_only_the_past():
    """Perturbing r_T must not move the predictive moments for t < T.

    This is the property that makes the residuals admissible inputs to a
    Ljung--Box test, and it is exactly what the smoothed variant violates.
    """
    rng = np.random.default_rng(7)
    x, _ = simulate_gaussian_hmm(A_TRUE, MU_TRUE, SIGMA_TRUE, 600, rng)
    hmm = GaussianHMM(K=3, n_starts=4, n_iter=120, random_state=2).fit(x)
    m1, v1 = hmm_predictive_moments(hmm, x, use="filtered")
    x2 = x.copy()
    x2[-1] += 0.05
    m2, v2 = hmm_predictive_moments(hmm, x2, use="filtered")
    assert np.allclose(m1[:-1], m2[:-1]) and np.allclose(v1[:-1], v2[:-1])

    s1, w1 = hmm_predictive_moments(hmm, x, use="smoothed")
    s2, w2 = hmm_predictive_moments(hmm, x2, use="smoothed")
    assert not np.allclose(s1[:-1], s2[:-1])


def test_predictive_variance_includes_the_between_regime_term():
    """Var must exceed the within-regime mixture whenever the means differ."""
    hmm = GaussianHMM(K=3)
    hmm.pi = _stationary(A_TRUE)
    hmm.A, hmm.mu, hmm.sigma = A_TRUE, MU_TRUE, SIGMA_TRUE
    rng = np.random.default_rng(8)
    x, _ = simulate_gaussian_hmm(A_TRUE, MU_TRUE, SIGMA_TRUE, 2000, rng)
    with np.errstate(all="ignore"):
        post = hmm.filtered_posterior(x)
        pred = post[:-1] @ A_TRUE
        within = pred @ (SIGMA_TRUE ** 2)
    _, var = hmm_predictive_moments(hmm, x, use="filtered")
    assert np.all(var >= within - 1e-18)
    assert np.any(var > within * (1 + 1e-9))


# ==========================================================================
#  4.  Benchmarks
# ==========================================================================
def _simulate_garch(T, omega, alpha, beta, seed=0, nu=None):
    rng = np.random.default_rng(seed)
    h = np.empty(T)
    e = np.empty(T)
    h[0] = omega / (1 - alpha - beta)
    for t in range(T):
        if t:
            h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
        if nu is None:
            z = rng.standard_normal()
        else:
            z = rng.standard_t(nu) / np.sqrt(nu / (nu - 2))
        e[t] = np.sqrt(h[t]) * z
    return e, h


def test_single_gaussian_is_the_closed_form_mle():
    r = np.random.default_rng(0).standard_normal(3000) * 0.01 + 0.002
    fit = fit_single_gaussian(r)
    assert np.isclose(fit["params"]["mu"], r.mean())
    assert np.isclose(fit["params"]["sigma"], r.std(ddof=0))
    ll = -0.5 * r.size * (np.log(2 * np.pi) + np.log(r.var(ddof=0)) + 1)
    assert np.isclose(fit["logL"], ll)
    assert fit["k"] == 2


def test_single_gaussian_matches_a_one_state_hmm():
    """The K=1 HMM and the i.i.d. Gaussian are the same model; so are their ICs."""
    r = np.random.default_rng(1).standard_normal(2000) * 0.01
    fit = _ic(fit_single_gaussian(r))
    hmm = GaussianHMM(K=1, n_starts=2, n_iter=50, random_state=0).fit(r)
    assert np.isclose(hmm.log_likelihood_, fit["logL"], rtol=1e-9)
    assert hmm.n_params() == fit["k"]
    assert np.isclose(hmm.bic(len(r)), fit["BIC"], rtol=1e-9)
    assert np.isclose(hmm.aic(), fit["AIC"], rtol=1e-9)


def test_garch_variance_recursion_matches_the_definition():
    r = np.array([0.1, -0.2, 0.3, 0.05])
    h, e = garch11_variance(r, mu=0.0, omega=0.01, alpha=0.2, beta=0.7, h1=1.0)
    assert np.isclose(h[0], 1.0)
    for t in range(1, 4):
        assert np.isclose(h[t], 0.01 + 0.2 * r[t - 1] ** 2 + 0.7 * h[t - 1])
    assert np.allclose(e, r)


def test_garch_mle_recovers_known_parameters():
    e, _ = _simulate_garch(20000, omega=0.05, alpha=0.09, beta=0.88, seed=42)
    fit = fit_garch11(e * 1e-4, "normal", scale=1e4)
    assert abs(fit["params"]["alpha"] - 0.09) < 0.025, fit["params"]
    assert abs(fit["params"]["beta"] - 0.88) < 0.03, fit["params"]
    assert abs(fit["persistence"] - 0.97) < 0.02


def test_garch_t_recovers_the_tail_index():
    e, _ = _simulate_garch(20000, omega=0.05, alpha=0.08, beta=0.90, seed=43, nu=5.0)
    fit = fit_garch11(e * 1e-4, "t", scale=1e4)
    assert 3.5 < fit["params"]["nu"] < 8.0, fit["params"]
    assert abs(fit["persistence"] - 0.98) < 0.02


def test_garch_loglikelihood_is_scale_consistent():
    """logL(c*r) = logL(r) - T log c.  Getting this wrong is worth 32,300 nats."""
    e, _ = _simulate_garch(3000, omega=0.05, alpha=0.09, beta=0.88, seed=44)
    fit = fit_garch11(e * 1e-4, "normal", scale=1e4)
    assert abs(fit["jacobian_check_gap"]) < 1e-6
    T = fit["T"]
    assert np.isclose(fit["logL"], fit["logL_in_bps_units"] + T * np.log(1e4))


def test_information_criteria_share_one_convention_across_models():
    """Every benchmark row must use AIC = 2k - 2logL, BIC = k logT - 2logL."""
    r = np.random.default_rng(2).standard_normal(1500) * 0.005
    rows = [_ic(fit_single_gaussian(r)),
            _ic(fit_garch11(r, "normal")),
            _ic(fit_garch11(r, "t"))]
    hmm = GaussianHMM(K=2, n_starts=4, n_iter=100, random_state=3).fit(r)
    rows.append(_ic({"model": "HMM K=2", "logL": float(hmm.log_likelihood_),
                     "k": int(hmm.n_params()), "T": len(r)}))
    for row in rows:
        assert np.isclose(row["AIC"], 2 * row["k"] - 2 * row["logL"])
        assert np.isclose(row["BIC"], row["k"] * np.log(row["T"]) - 2 * row["logL"])
        assert row["T"] == len(r)
    # lower-is-better, and the HMM's own methods agree with the shared helper
    assert np.isclose(rows[-1]["BIC"], hmm.bic(len(r)))
    assert np.isclose(rows[-1]["AIC"], hmm.aic())


def test_garch_beats_a_single_gaussian_on_garch_data():
    """Sanity: the benchmark table must be able to detect a real ordering."""
    e, _ = _simulate_garch(5000, omega=0.05, alpha=0.10, beta=0.85, seed=45)
    r = e * 1e-4
    g = _ic(fit_garch11(r, "normal"))
    n = _ic(fit_single_gaussian(r))
    assert g["BIC"] < n["BIC"]
    assert g["logL"] > n["logL"]


# ==========================================================================
#  5.  MS-AR residuals and the table writers
# ==========================================================================
def _simulate_ms_ar1(T, mu, phi, sigma, P, seed=0):
    """Two-regime mean-adjusted AR(1): y_t = mu_k + phi_k (y_{t-1} - mu_k) + sigma_k e."""
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=int)
    y = np.zeros(T)
    y[0] = mu[0]
    for t in range(1, T):
        s[t] = rng.choice(2, p=P[s[t - 1]])
        y[t] = mu[s[t]] + phi[s[t]] * (y[t - 1] - mu[s[t]]) + sigma[s[t]] * rng.standard_normal()
    return y


def test_msar_residuals_are_standardised_and_white_when_the_model_is_true():
    """On data generated by an MS-AR(1), the residuals must be white and unit scale."""
    from regime_engine.diagnostics import msar_standardised_residuals
    from regime_engine.msar import fit_msar_premium

    y = _simulate_ms_ar1(
        3000, mu=np.array([-4.0, 2.0]), phi=np.array([0.90, 0.60]),
        sigma=np.array([0.4, 1.6]), P=np.array([[0.97, 0.03], [0.06, 0.94]]), seed=9)
    fit = fit_msar_premium(y)
    z = msar_standardised_residuals(fit, y)
    # One shorter than nobs: the one-step forecast marginalises over the state
    # *pair* (S_{t-1}, S_t), so the earliest usable t needs a filtered
    # predecessor and the first observation of the model's own sample has none.
    assert z.size == fit["nobs"] - 1
    assert abs(z.mean()) < 0.15
    assert 0.85 < z.std(ddof=1) < 1.15
    for d in ljung_box(z, (1, 2, 10)):
        assert d["p"] > 0.001, d


def test_msar_residuals_reject_on_a_series_with_an_added_ma1():
    """Temporal averaging leaves an MA(1) component that an AR(1) cannot absorb."""
    from regime_engine.diagnostics import msar_standardised_residuals
    from regime_engine.msar import fit_msar_premium

    y = _simulate_ms_ar1(
        3000, mu=np.array([-4.0, 2.0]), phi=np.array([0.90, 0.60]),
        sigma=np.array([0.4, 1.6]), P=np.array([[0.97, 0.03], [0.06, 0.94]]), seed=10)
    averaged = 0.5 * (y[1:] + y[:-1])          # non-overlapping-style time average
    fit = fit_msar_premium(averaged)
    z = msar_standardised_residuals(fit, averaged)
    qs = ljung_box(z, (1, 2, 10))
    assert min(d["p"] for d in qs) < 0.05, qs


def test_table_writers_emit_valid_booktabs(tmp_path):
    from regime_engine.diagnostics import _write_table6, _write_table7, _write_table8

    rng = np.random.default_rng(11)
    payload6 = {"tests": [residual_battery(rng.standard_normal(500), "raw log-returns"),
                          residual_battery(rng.standard_normal(500), "HMM z_t (a & b)",
                                           lb_lags=(1, 2, 10))]}
    p6 = tmp_path / "t6.tex"
    _write_table6(payload6, p6)
    body6 = p6.read_text()
    assert body6.count("\\toprule") == 1 and body6.count("\\bottomrule") == 1
    assert "\\&" in body6 and "\\begin{tabular}" in body6

    r = rng.standard_normal(600) * 0.004
    payload7 = {"T": 600, "rows": [_ic(fit_single_gaussian(r)),
                                   _ic(fit_garch11(r, "normal"))]}
    p7 = tmp_path / "t7.tex"
    _write_table7(payload7, p7)
    body7 = p7.read_text()
    assert "\\bottomrule" in body7
    # negative criteria print with a typographic minus, not a hyphen
    assert "$-$" in body7 and "& -" not in body7

    boot = bootstrap_hmm({"mu": MU_TRUE, "sigma": SIGMA_TRUE, "A": A_TRUE},
                         T=1500, n_reps=8, seed=12, n_jobs=2, progress=False)
    for rule in ("mu_ascending", "sigma_ascending"):
        p8 = tmp_path / f"t8_{rule}.tex"
        _write_table8(boot, p8, labelling=rule)
        body = p8.read_text()
        assert "\\toprule" in body and "\\bottomrule" in body
        # 3 mu + 3 sigma + 3 annualised + 9 A + 3 tau + 3 pi = 24 parameter rows
        assert body.count(" \\\\") >= 24


@pytest.mark.parametrize("dist", ["normal", "t"])
def test_garch_matches_the_arch_package_when_it_is_installed(dist):
    """External cross-check.  Skipped on a clean environment -- `arch` is not a
    dependency of this package and the shipped implementation is self-contained."""
    arch = pytest.importorskip("arch")
    from regime_engine.panel import load_panel
    from regime_engine.paths import DEFAULT_PANEL

    r = load_panel(DEFAULT_PANEL)["log_return"].values
    mine = fit_garch11(r, dist)
    ref = arch.arch_model(r * 1e4, mean="Constant", vol="GARCH", p=1, q=1,
                          dist=dist).fit(disp="off")
    assert abs(mine["params"]["alpha"] - ref.params["alpha[1]"]) < 5e-3
    assert abs(mine["params"]["beta"] - ref.params["beta[1]"]) < 5e-3
    # The two differ only in how h_1 is backcast, so the log-likelihoods agree to
    # a fraction of a nat -- three orders of magnitude below any model spacing.
    assert abs(mine["logL_in_bps_units"] - ref.loglikelihood) < 1.0
