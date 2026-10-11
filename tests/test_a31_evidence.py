"""Profile likelihood, boundary decision and GARCH simulation for A_31, and the
numbers of output/a31_evidence.json."""
import json

import numpy as np
import pytest

from regime_engine import a31 as A31
from regime_engine import diagnostics as dg
from regime_engine import hmm_batch as hb
from regime_engine.paths import DEFAULT_OUT

MU = np.array([0.0001, 0.0, -0.0003])
SIG = np.array([0.0012, 0.0030, 0.0090])


def _kernel(a31):
    A = np.array([[0.94, 0.05, 0.01],
                  [0.03, 0.92, 0.05],
                  [0.0, 0.20, 0.80]])
    return hb.move_cell_mass(A, (2, 0), a31)


def _sample(a31, T, seed):
    A = _kernel(a31)
    x, s = dg.simulate_gaussian_hmm(A, MU, SIG, T, np.random.default_rng(seed))
    return x, s, {"pi": dg._stationary(A), "A": A, "mu": MU, "sigma": SIG}


def _mle(x, truth):
    unc = A31.fit_converged(x, truth, tol=1e-9)
    m = A31.cell_mle(x, hb.sort_by_sigma(unc.params), tol=1e-9)
    return m["params"].member(0), float(m["a_hat"][0]), float(m["ll_max"][0]), m


# ---------------------------------------------------------------------------
#  synthetic chains with a known A_31
# ---------------------------------------------------------------------------
def test_profile_interval_covers_a_known_interior_a31():
    x, s, truth = _sample(0.04, 4000, seed=7)
    mle, a_hat, ll_max, bd = _mle(x, truth)
    assert not bd["at_boundary"][0]
    assert bd["ratio"][0] > 1.0
    # the profile is maximised at the MLE and the LR at the truth is small
    lr_true = 2 * (ll_max - A31.profile_likelihood(x, mle, [0.04], tol=1e-9)["ll"][0])
    assert 0.0 <= lr_true < A31.CHI2_1_95
    pb = A31.profile_upper_bound(x, mle, ll_max, [A31.CHIBAR_95, A31.CHI2_1_95],
                                 a_hat=a_hat, rounds=2, tol=1e-9)
    b1, b2 = pb["bound"]
    assert abs(a_hat - 0.04) < 0.02
    assert a_hat < b1 < b2
    assert 0.04 < b2 or 0.04 <= a_hat
    # the bound is where the LR statistic reaches the critical value
    lr_b1 = 2 * (ll_max - A31.profile_likelihood(x, mle, [b1], tol=1e-9)["ll"][0])
    assert lr_b1 == pytest.approx(A31.CHIBAR_95, abs=0.02)


def test_profile_bound_for_a_chain_without_the_transition():
    # a chain whose estimate lies on the boundary (seed 14) and one whose
    # estimate does not (seed 11): EM started at the true zero cannot tell them
    # apart, the score ratio can
    x, s, truth = _sample(0.0, 4000, seed=11)
    assert not np.any((s[:-1] == 2) & (s[1:] == 0))
    _, a_hat, _, bd = _mle(x, truth)
    assert bd["ratio"][0] > 1.0 and 0.0 < a_hat < 0.01
    stuck = A31.fit_converged(x, truth, tol=1e-9)
    assert stuck.params.A[0, 2, 0] == 0.0
    x, s, truth = _sample(0.0, 4000, seed=14)
    assert not np.any((s[:-1] == 2) & (s[1:] == 0))
    mle, a_hat, ll_max, bd = _mle(x, truth)
    assert bd["at_boundary"][0] and a_hat == 0.0
    pb = A31.profile_upper_bound(x, mle, ll_max, [A31.CHIBAR_95], rounds=2, tol=1e-9)
    b = pb["bound"][0]
    assert 0.0 < b < 0.05
    lr = np.array(pb["grid_LR"])
    assert np.all(np.diff(lr) > -1e-6)        # LR rises monotonically above a_hat = 0


def test_boundary_decision_flags_a_large_cell_as_interior():
    x, _, truth = _sample(0.10, 3000, seed=2)
    bd = A31.boundary_decision(x, hb.BatchParams.stack([truth]))
    assert not bd["at_boundary"][0]
    assert bd["params0"].A[0, 2, 0] == 0.0


def test_garch_simulation_matches_its_moments():
    rng = np.random.default_rng(0)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    r = A31.simulate_garch11(0.0, omega, alpha, beta, 200_000, rng, nu=5.0)
    assert np.var(r) == pytest.approx(omega / (1 - alpha - beta), rel=0.1)
    # the variance recursion is the estimator's
    h, _ = dg.garch11_variance(r[:50], 0.0, omega, alpha, beta, h1=omega / (1 - alpha - beta))
    rng = np.random.default_rng(1)
    r2 = A31.simulate_garch11(0.0, omega, alpha, beta, 50, rng)
    z = np.random.default_rng(1).standard_normal(50)
    h2, _ = dg.garch11_variance(r2, 0.0, omega, alpha, beta, h1=omega / (1 - alpha - beta))
    np.testing.assert_allclose(r2, np.sqrt(h2) * z, rtol=1e-12)
    assert h.shape == (50,)


def test_binomial_share():
    s = A31.binomial_share([True] * 30 + [False] * 70)
    assert s == {"k": 30, "n": 100, "share": 0.3, "se": pytest.approx(np.sqrt(0.21 / 100))}


def test_critical_values():
    from scipy.stats import chi2, norm
    assert A31.CHIBAR_95 == pytest.approx(norm.ppf(0.95) ** 2, rel=1e-12)
    assert A31.CHIBAR_95 == pytest.approx(chi2.ppf(0.90, 1), rel=1e-12)
    assert A31.CHI2_1_95 == pytest.approx(chi2.ppf(0.95, 1), rel=1e-12)


# ---------------------------------------------------------------------------
#  pins on output/a31_evidence.json
# ---------------------------------------------------------------------------
ARTIFACT = DEFAULT_OUT / "a31_evidence.json"


@pytest.fixture(scope="module")
def ev():
    if not ARTIFACT.exists():
        pytest.skip("run `PYTHONPATH=src python3 scripts/a31_evidence.py` first")
    return json.loads(ARTIFACT.read_text())


def test_profile_bounds_are_pinned(ev):
    p = ev["profile_likelihood"]
    assert p["T"] == 3507
    assert p["a_hat"] == 0.0 and p["boundary"]["mle_at_boundary"]
    assert p["boundary"]["score_ratio_at_zero"] == pytest.approx(0.7068, abs=5e-4)
    assert p["upper_bound_one_sided_95"] == pytest.approx(0.00641, abs=5e-5)
    assert p["upper_bound_two_sided_95"] == pytest.approx(0.00922, abs=5e-5)
    assert p["rule_of_three_same_sample"]["bound"] == pytest.approx(0.00403, abs=5e-5)
    # the LR statistic at the bounds is the critical value
    grid = dict(zip(p["profile_grid"]["a"], p["profile_grid"]["LR"]))
    a = np.array(sorted(grid))
    lr = np.array([grid[v] for v in a])
    assert np.interp(p["upper_bound_one_sided_95"], a, lr) == pytest.approx(A31.CHIBAR_95,
                                                                             abs=1e-3)
    assert np.all(np.diff(lr) > 0)


def test_profile_is_global_among_order_preserving_starts(ev):
    m = ev["profile_likelihood"]["multistart_at_bound"]
    assert m["n_starts"] == 48
    assert m["n_starts_order_kept"] == m["n_starts_at_profile_value"] == 28
    assert m["max_gain_over_profile"] < 1e-6


def test_lr_test_at_the_bound_is_conservative(ev):
    c = ev["profile_likelihood"]["calibration_at_bound"]
    assert c["n_reps"] == 400
    assert c["rejection_rate"]["k"] == 4
    assert c["q95_one_sided_statistic"] == pytest.approx(1.754, abs=1e-3)


POWER_PINS = {
    "full_sample": {"0": 547, "0.003": 390, "0.01": 188, "0.02": 53},
    "sub_period": {"0": 536, "0.003": 418, "0.01": 197, "0.02": 63},
}


def test_power_shares_are_pinned(ev):
    for sample, cells in POWER_PINS.items():
        prev = 1.0
        for a, k in cells.items():
            c = ev["power"][sample][a]
            assert c["n_reps"] == 1000
            assert c["share_mle_zero"]["k"] == k
            assert c["share_mle_zero"]["se"] <= 0.016
            assert c["n_decision_fits_stopped_by_tol"] == 1000
            assert c["share_mle_zero"]["share"] < prev
            prev = c["share_mle_zero"]["share"]


def test_garch_paths_put_a31_on_the_boundary(ev):
    g = ev["garch_generic"]
    assert g["garch_t_fitted"]["persistence"] == pytest.approx(0.99998, abs=1e-5)
    assert g["paths_per_specification"] == {"garch_t_fitted": 60, "garch_t_conventional": 60,
                                            "garch_normal_fitted": 240}
    for name, n, k_mle, k_150 in (("garch_t_fitted", 60, 60, 60),
                                  ("garch_t_conventional", 60, 60, 49),
                                  ("garch_normal_fitted", 240, 97, 28)):
        c = g[name]
        assert c["n_paths"] == n
        assert c["share_mle_zero"]["k"] == k_mle
        assert c["share_A31_zero_paper_protocol_150_iter"]["k"] == k_150
    # at the best of the 8 seeded starts the low-persistence Gaussian GARCH gives a
    # zero on about four paths in ten (a local-optimum share; see the next test);
    # its first 60 paths are the earlier 60-path sample (21 zeros)
    normal = g["garch_normal_fitted"]
    assert normal["share_mle_zero"]["share"] == pytest.approx(0.404, abs=1e-3)
    assert normal["share_mle_zero"]["se"] == pytest.approx(0.032, abs=1e-3)
    assert sum(q["at_boundary"] for q in normal["paths"][:60]) == 21
    seeds = [q["seed"] for q in g["garch_t_fitted"]["paths"]]
    assert seeds == [q["seed"] for q in normal["paths"][:60]]


def test_the_gaussian_garch_share_is_taken_at_the_best_optimum(ev):
    """The 8 seeded starts stop in a lower local optimum on 71 of 240 Gaussian-GARCH
    paths; with 24 spread starts more the zero share at the best optimum is 53.8%."""
    g = ev["garch_generic"]
    assert g["wide_paths_per_specification"] == {"garch_t_fitted": 30,
                                                 "garch_t_conventional": 30,
                                                 "garch_normal_fitted": 240}
    w = g["garch_normal_fitted"]["wide_multistart"]
    assert (w["n_paths"], w["n_starts_per_path"]) == (240, 32)
    assert w["share_mle_zero_8_starts"]["k"] == 97
    assert w["share_mle_zero_best_optimum"]["k"] == 129
    assert w["share_mle_zero_best_optimum"]["share"] == pytest.approx(129 / 240)
    assert round(w["share_mle_zero_best_optimum"]["se"], 3) == 0.032
    assert w["n_paths_higher_optimum"] == 71
    assert (w["n_interior_to_zero"], w["n_zero_to_interior"]) == (34, 2)
    assert round(w["logL_gain_on_those_paths"]["q500"], 1) == 3.6
    # the first 60 paths: 31 zeros at the best optimum, 20 with a higher optimum
    first = g["garch_normal_fitted"]["paths"][:60]
    assert sum(q["wide"]["at_boundary"] for q in first) == 31
    assert sum(q["wide"]["higher_optimum"] for q in first) == 20
    for q in g["garch_normal_fitted"]["paths"]:
        assert q["wide"]["logL_gain"] >= -1e-6        # the 8 starts are among the 32


def test_the_student_t_zero_survives_the_wider_search(ev):
    g = ev["garch_generic"]
    for name in ("garch_t_fitted", "garch_t_conventional"):
        w = g[name]["wide_multistart"]
        assert w["n_paths"] == 30
        assert w["share_mle_zero_best_optimum"]["k"] == 30
        assert w["n_paths_higher_optimum"] == 0 and w["n_decisions_changed"] == 0
        assert all("wide" not in q for q in g[name]["paths"][30:])


def test_garch_state_structure_differs_from_the_data_s(ev):
    """The GARCH-t zero arises with a rarer, more extreme high state than the data's."""
    g = ev["garch_generic"]
    fitted = g["garch_t_fitted"]
    assert [round(v, 1) for v in fitted["median_ann_sigma_pct"]] == [20.9, 53.7, 201.9]
    assert fitted["median_high_state_mass"] == pytest.approx(0.059, abs=1e-3)
    data_mass = ev["convergence"]["fits"]["analysis_K3"]["converged"]["stationary_pct"][2]
    assert round(data_mass, 1) == 21.2
    # the data's score ratio is within the range of the fitted GARCH-t (90th
    # percentile) and typical of the conventional one
    assert g["observed_score_ratio_at_zero"] == pytest.approx(
        ev["profile_likelihood"]["boundary"]["score_ratio_at_zero"], abs=1e-6)
    assert fitted["share_paths_ratio_at_most_observed"]["k"] == 54
    assert g["garch_t_conventional"]["share_paths_ratio_at_most_observed"]["k"] == 39
    assert [g[n]["largest_hourly_variance_fall_factor_beta"] for n in
            ("garch_normal_fitted", "garch_t_fitted", "garch_t_conventional")] == sorted(
        g[n]["largest_hourly_variance_fall_factor_beta"] for n in
        ("garch_normal_fitted", "garch_t_fitted", "garch_t_conventional"))


def test_published_power_figures_are_an_early_stopping_artifact(ev):
    e = ev["power_em"]
    # the published rule reproduces output/bootstrap_hmm.json exactly on the same paths
    for a in ("0.003", "0.01"):
        r = e["full_sample"][a]["published_rule_reproduction"]
        assert r["n"] == 200
        assert r["share_zero_here"] == r["share_zero_in_bootstrap_hmm_json"]
    # (published rule, continued EM, MLE at the maximum), zeros out of 200
    pins = {"full_sample": {"0.003": (12, 79, 80), "0.01": (0, 33, 35), "0.02": (0, 10, 10)},
            "sub_period": {"0.003": (34, 82, 88), "0.01": (10, 36, 37), "0.02": (4, 9, 10)}}
    for sample, cells in pins.items():
        for a, (k_pub, k_em, k_mle) in cells.items():
            c = e[sample][a]
            assert c["n_reps"] == 200
            assert c["share_zero_published_rule"]["k"] == k_pub
            assert c["share_zero_em_continued"]["k"] == k_em
            assert c["share_mle_zero"]["k"] == k_mle
            assert c["agreement_with_decision_from_truth"] == 1.0
    # at a true zero, EM started at the truth cannot leave zero
    assert e["full_sample"]["0"]["share_zero_published_rule"]["share"] == 1.0
    assert e["full_sample"]["0"]["share_mle_zero"]["k"] == 115


def test_old_cap_reproduces_every_printed_number(ev):
    rows = ev["convergence"]["printed_vs_converged"]
    assert len(rows) == 80
    assert all(r["capped_reproduces_printed"] for r in rows)


def test_converged_refits_change_these_printed_numbers(ev):
    c = ev["convergence"]
    changed = {(r["where"], r["quantity"]): r["converged_at_printed_precision"]
               for r in c["changed_at_printed_precision"]}
    raw3 = ("v1.2.0 Table A1 raw row; Sec. 4.3 sub-period fit", "annualised sigma state 3 (%)")
    # "where" is the place in version 1.2.0 of the paper, which printed these
    legends = "v1.2.0 Figures 4(a) and 5(a) legends (now Figures C1(a), C2(a))"
    expected = {
        ("v1.2.0 Table A1 raw row; Sec. 4.3 sub-period fit",
         "annualised sigma state 1 (%)"): 12.8,
        raw3: 87.6,
        ("v1.2.0 Sec. 4.3 robustness paragraph ('from 87.7% to 85.4%')",
         "annualised sigma state 3, raw (%)"): 87.6,
        ("v1.2.0 Sec. 4.3 ('87.7% lies inside the full sample's [83.0, 94.0]')",
         "annualised sigma state 3, raw (%)"): 87.6,
        (legends, "sigma state 3 (bps/h)"): 93.6,
        ("v1.2.0 Sec. 4.3 more than three states", "BIC K=5"): -28482.8,
        ("v1.2.0 Sec. 4.3", "logL gain of K=5 over K=4"): 7.8,
        ("v1.2.0 Sec. 4.3 two-hourly refit (T = 2,499, whole window)",
         "annualised sigma state 2 (%)"): 31.7,
        ("v1.2.0 Sec. 4.3 two-hourly refit (T = 2,499, whole window)",
         "annualised sigma state 3 (%)"): 88.9,
        ("v1.2.0 Sec. 4.3 two-hourly refit", "stationary mass high (%)"): 20.9,
        ("v1.2.0 Table A2 HMM K=3 filtered row", "Q_z_24"): 18.2,
        ("v1.2.0 Table A2 HMM K=3 filtered row", "JB"): 31194.0,
        ("v1.2.0 Table A2 HMM K=3 filtered row", "excess_kurtosis"): 14.47,
        ("v1.2.0 Table A2 HMM K=3 smoothed row", "Q_z2_10"): 53.4,
        ("v1.2.0 Table A2 HMM K=3 smoothed row", "Q_z2_24"): 92.5,
        ("v1.2.0 Table A2 HMM K=3 smoothed row", "ARCH_LM_12"): 46.2,
        ("v1.2.0 Table A2 HMM K=3 smoothed row", "JB_p"): 0.0016,
        ("v1.2.0 Sec. 4.5 text ('an excess kurtosis of 14.4 against 6.9')",
         "filtered excess kurtosis"): 14.5,
        ("v1.2.0 Sec. 4.5 text ('Q(z^2)(10) = 53.3' for the smoothed posterior)",
         "smoothed Q_z2_10"): 53.4,
    }
    assert changed == expected
    # the polished optimum, not a stop at |dlogL| < 1e-9 (which gives 31,193)
    jb = next(r for r in c["changed_at_printed_precision"]
              if r["quantity"] == "JB" and "filtered" in r["where"])
    assert jb["converged_unrounded"] == pytest.approx(31193.99, abs=0.02)
    # these stay: the filtered ARCH-LM p-value and the K=2..4 criteria among them
    stays = {(r["where"], r["quantity"]) for r in c["printed_vs_converged"] if not r["changes"]}
    assert ("v1.2.0 Table A2 HMM K=3 filtered row", "ARCH_LM_12_p") in stays
    assert ("v1.2.0 Sec. 4.3 more than three states", "BIC K=4") in stays


def test_convergence_bookkeeping(ev):
    c = ev["convergence"]
    f = c["fits"]
    assert c["all_converged"] and c["shipped_artifacts_are_the_converged_fits"]
    assert f["analysis_K3"]["n_starts_at_old_cap"] == 8
    assert f["selection_K5"]["n_starts_at_old_cap"] == 40
    assert f["selection_K4"]["n_starts_at_old_cap"] == 32
    assert f["analysis_K2"]["n_starts_at_old_cap"] == 0
    for name, fit in f.items():
        rec = fit["record"]
        assert rec["n_starts_converged"] == fit["n_starts"], name
        assert rec["polish"]["converged"], name
        assert abs(rec["polish"]["final_dlogL"]) < 1e-9, name
        assert rec["polish"]["final_max_param_step"] < 1e-10, name
        assert fit["shipped_agreement"]["matches"], name
    # the converged sub-period fit keeps A_31 on the boundary and A_32 at 0.26
    t = f["analysis_K3"]["converged"]
    assert t["A"][2][0] < 1e-8
    assert round(t["A"][2][1], 2) == 0.26
    assert f["analysis_K3"]["capped_best_start"] == 6 and f["analysis_K3"]["selected_start"] == 1
    assert f["selection_K5"]["converged"]["BIC"] == pytest.approx(-28482.78, abs=0.01)
    eff = c["sub_period_cap_effect"]
    assert 0 < eff["logL_converged_minus_capped"] < 2e-4
    assert eff["max_abs_change_annualised_sigma_pct"] == pytest.approx(0.040, abs=1e-3)


def test_model_selection_is_printed_at_the_best_optimum(ev):
    """The 40-start refit here stops in a lower K = 5 optimum; the paper prints the
    best optimum of the wide multi-start in model_selection_extended.json."""
    b = ev["convergence"]["selection_best_optimum"]
    rows = {r["quantity"]: r for r in b["rows"]}
    changed = {q for q, r in rows.items() if r["changes_at_one_decimal"]}
    assert changed == {"BIC K=5", "logL gain of K=5 over K=4", "BIC K=5 minus BIC K=3"}
    assert round(rows["BIC K=5"]["best_optimum_wide_multistart"], 1) == -28500.6
    assert round(rows["logL gain of K=5 over K=4"]["best_optimum_wide_multistart"], 1) == 16.7
    assert round(rows["BIC K=5 minus BIC K=3"]["best_optimum_wide_multistart"], 1) == 32.8
    assert round(rows["BIC margin of K=4 over K=3"]["best_optimum_wide_multistart"], 1) == 23.6
    assert (b["n_starts_reaching_best"]["5"], b["n_starts"]["5"]) == (19, 212)


def test_calibrated_bound_and_why_it_is_not_the_headline(ev):
    p = ev["profile_likelihood"]
    c = p["calibrated"]
    assert [round(r["a"], 4) for r in c["grid"]] == [0.0005, 0.001, 0.003, 0.004, 0.005, 0.0064]
    assert [r["p_value"]["k"] for r in c["grid"]] == [9, 4, 7, 4, 4, 4]
    # every A_31 > 0 on the grid is rejected: a misspecification signal, not a bound
    assert c["rejects_every_grid_value"] and c["upper_bound_bootstrap_95"] is None
    assert c["smallest_grid_value"] == 0.0005
    assert "not a bound" in c["note"] and "does not rest" not in c["note"]
    # within the model the chi2 bound is conservative: simulated size 1% at the bound
    assert p["calibration_at_bound"]["rejection_rate"]["share"] == 0.01
    # the observed score ratio sits in the lower tail of the Gaussian-HMM simulations
    # and in the body of the GARCH-t simulations
    ratio = p["boundary"]["score_ratio_at_zero"]
    assert ratio < ev["power"]["sub_period"]["0"]["score_ratio"]["q025"]
    assert ratio < ev["power"]["sub_period"]["0.003"]["score_ratio"]["q025"]
    g = np.array([q["ratio"] for q in ev["garch_generic"]["garch_t_conventional"]["paths"]])
    assert 0.1 < np.mean(g <= ratio) < 0.9


def _a31_script():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "a31_evidence.py"
    spec = importlib.util.spec_from_file_location("a31_evidence_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rebuilding_the_profile_keeps_the_calibration_while_its_inputs_hold(capsys):
    keep = _a31_script()._keep_calibrated
    old = {"upper_bound_one_sided_95": 0.0064, "calibration_at_bound": {"rejection_rate": 0.01},
           "calibrated": {"grid": [1, 2]}}
    def same():
        return {"upper_bound_one_sided_95": np.float64(0.0064),
                "calibration_at_bound": {"rejection_rate": 0.01}}

    assert keep(old, same())["calibrated"] == {"grid": [1, 2]}
    moved = {"upper_bound_one_sided_95": 0.0065, "calibration_at_bound": {"rejection_rate": 0.01}}
    assert "calibrated" not in keep(old, moved)
    assert "rerun --part profile_calibrated" in capsys.readouterr().out
    assert "calibrated" not in keep({}, same())
