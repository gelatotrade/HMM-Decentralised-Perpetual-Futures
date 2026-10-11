"""Robust covariance and the switching-aware inference block on simulated MS-AR fits.

Short simulated series keep these fits to a second or two each. The data
generating process is the paper's mean-adjusted MS-AR(1) with a calm,
persistent regime and a volatile, fast-reverting one; the innovations are
either Gaussian (the model is correctly specified, so the information-matrix
equality holds and the three covariance estimators agree) or Laplace (excess
kurtosis 3, so the equality fails for the variance parameters and only the
sandwich is valid).
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from regime_engine.hmm import fit_ms_ar1_multistart
from regime_engine.msar import (
    SIM_DRAWS,
    expected_durations,
    fit_msar_multistart,
    fit_msar_premium,
    msar_covariances,
    msar_robust_inference,
    ratio_sandwich_influence,
    ratio_wald_test,
    row_stochastic_transition,
    summarise_msar_fit,
    switching_half_life,
)

P_TRUE = np.array([[0.98, 0.02], [0.10, 0.90]])     # calm, stressed; from -> to
MU = np.array([-3.0, -3.0])
PHI = np.array([0.95, 0.70])
SIGMA = np.array([0.5, 2.5])


def simulate(T, seed, innovations="normal", P=P_TRUE):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=int)
    y = np.zeros(T)
    y[0] = MU[0]
    u = rng.random(T)
    if innovations == "normal":
        e = rng.standard_normal(T)
    else:                                   # unit-variance Laplace
        e = rng.laplace(scale=1 / np.sqrt(2), size=T)
    for t in range(1, T):
        s[t] = 1 if u[t] < P[s[t - 1], 1] else 0
        y[t] = MU[s[t]] + PHI[s[t]] * (y[t - 1] - MU[s[t - 1]]) + SIGMA[s[t]] * e[t]
    return y


def multistart(y, n_starts=2, search_reps=10):
    return fit_ms_ar1_multistart(y, n_starts=n_starts, search_reps=search_reps,
                                 verbose=False, k_regimes=2, switching_variance=True,
                                 switching_ar=True)


@pytest.fixture(scope="module")
def gaussian():
    y = simulate(4000, 11, "normal")
    return y, summarise_msar_fit(y, *multistart(y))


@pytest.fixture(scope="module")
def laplace():
    y = simulate(4000, 11, "laplace")
    return y, summarise_msar_fit(y, *multistart(y))


def se_ratio(fit, prefix, regime):
    """se_robust / se_hessian for ``prefix[raw index of regime]`` (0 calm, 1 stressed)."""
    names = fit["param_names"]
    i = names.index(f"{prefix}[{int(fit['order'][regime])}]")
    cov = msar_covariances(fit["result"])
    return float(np.sqrt(cov["robust"][i, i] / cov["hessian"][i, i]))


# ---------------------------------------------------------------------------
# The covariance estimators
# ---------------------------------------------------------------------------
class TestCovariances:
    def test_they_are_the_matrices_statsmodels_reports_for_each_cov_type(self, gaussian):
        y, fit = gaussian
        model, params = fit["model"], np.asarray(fit["result"].params)
        cov = msar_covariances(fit["result"])
        for name, cov_type in (("hessian", "approx"), ("opg", "opg"), ("robust", "robust")):
            ref = np.asarray(model.smooth(params, cov_type=cov_type).cov_params())
            assert cov[name] == pytest.approx(ref, rel=1e-8, abs=1e-14), name

    def test_the_sandwich_is_hessian_inverse_opg_hessian_inverse(self, gaussian):
        y, fit = gaussian
        model, params = fit["model"], np.asarray(fit["result"].params)
        with np.errstate(all="ignore"):
            H = model.hessian(params, transformed=True)
            S = model.score_obs(params, transformed=True)
        Hinv = np.linalg.inv(H)
        manual = Hinv @ (S.T @ S) @ Hinv
        assert msar_covariances(fit["result"])["robust"] == pytest.approx(manual, rel=1e-6)

    def test_under_gaussian_innovations_the_three_estimators_agree(self, gaussian):
        """Correct specification: the information-matrix equality holds."""
        _, fit = gaussian
        for prefix in ("ar.L1", "sigma2"):
            for regime in (0, 1):
                assert 0.85 < se_ratio(fit, prefix, regime) < 1.18, (prefix, regime)

    def test_under_laplace_innovations_the_sandwich_widens_the_variance_se(self, laplace):
        """Excess kurtosis 3: Var(sigma2_hat) is (kappa - 1) / 2 = 2.5 times the
        Gaussian one when the regime is known, a ratio of standard errors of
        1.58; regime uncertainty dilutes it. The autoregressive coefficient's
        standard error does not depend on the fourth moment and stays put."""
        _, fit = laplace
        assert se_ratio(fit, "sigma2", 0) > 1.25
        assert 0.8 < se_ratio(fit, "ar.L1", 0) < 1.2


# ---------------------------------------------------------------------------
# The statsmodels transition convention and the regime order
# ---------------------------------------------------------------------------
class TestTransitionConvention:
    def test_left_stochastic_raw_matrix_becomes_row_stochastic_calm_first(self):
        # statsmodels: regime_transition[i, j] = P(S_t = i | S_{t-1} = j).
        # Raw regime 0 is the stressed one here (it stays with 0.6).
        raw = np.array([[0.6, 0.03], [0.4, 0.97]])[:, :, None]
        P = row_stochastic_transition(raw, order=np.array([1, 0]))
        assert P == pytest.approx(np.array([[0.97, 0.03], [0.4, 0.6]]))

    def test_identity_order_only_transposes(self):
        raw = np.array([[0.9, 0.2], [0.1, 0.8]])
        P = row_stochastic_transition(raw, order=np.array([0, 1]))
        assert P == pytest.approx(raw.T)

    def test_staying_probabilities_agree_with_the_named_parameters(self, gaussian):
        """p[0->0] is P(S_t=0 | S_{t-1}=0), p[1->0] is P(S_t=0 | S_{t-1}=1)."""
        _, fit = gaussian
        names, params = fit["param_names"], np.asarray(fit["result"].params)
        p00, p10 = params[names.index("p[0->0]")], params[names.index("p[1->0]")]
        raw_stay = {0: p00, 1: 1.0 - p10}
        block = msar_robust_inference(fit)
        for regime in (0, 1):
            raw = int(fit["order"][regime])
            assert block["p_stay"][regime] == pytest.approx(raw_stay[raw], rel=1e-10)

    def test_the_calm_regime_gets_the_calm_chain(self):
        """Very different staying probabilities expose a swapped mapping."""
        P = np.array([[0.99, 0.01], [0.20, 0.80]])
        y = simulate(4000, 3, "normal", P=P)
        block = msar_robust_inference(summarise_msar_fit(y, *multistart(y)))
        assert block["p_stay"][0] == pytest.approx(0.99, abs=0.01)
        assert block["p_stay"][1] == pytest.approx(0.80, abs=0.08)
        d = block["expected_duration_hours"]
        assert d[0] > 40 and d[1] < 10


# ---------------------------------------------------------------------------
# The inference block
# ---------------------------------------------------------------------------
class TestInferenceBlock:
    @pytest.fixture(scope="class")
    def block(self, gaussian):
        return msar_robust_inference(gaussian[1])

    def test_it_carries_every_quantity(self, block):
        for key in ("p_stay", "expected_duration_hours", "stationary_share",
                    "chain_eigenvalue", "half_life_switching_hours",
                    "half_life_switching_interp_hours", "ratio_switching",
                    "ratio_switching_interp", "inference"):
            assert key in block, key
        assert sorted(block["inference"]) == ["hessian", "opg", "robust"]
        for cov in block["inference"].values():
            for key in ("phi_se", "phi_t_unit", "phi_ci95", "half_life_ci95_from_phi",
                        "p_stay_se", "expected_duration_se", "ratio_se", "ratio_ci95",
                        "ratio_wald", "chain_eigenvalue_se",
                        "half_life_switching_interp_se", "ratio_switching_interp_se",
                        "ratio_switching_interp_ci95", "ratio_switching_side",
                        "ratio_switching_sim"):
                assert key in cov, key
            assert sorted(cov["ratio_switching_interp_ci95"]) == ["linear", "log"]
            assert sorted(cov["ratio_ci95"]) == ["fieller", "linear", "log"]
            for form in cov["ratio_ci95"].values():
                assert form["significance"] in {"above", "below", "ns", "undetermined"}

    def test_durations_follow_the_staying_probabilities(self, block):
        P = np.diag(block["p_stay"])
        d = 1.0 / (1.0 - np.diag(P))
        assert block["expected_duration_hours"] == pytest.approx(d.tolist())

    def test_switching_half_lives_use_the_fitted_chain(self, block, gaussian):
        _, fit = gaussian
        P = np.asarray(fit["transition_matrix"], dtype=float)
        for k in range(2):
            ref = switching_half_life(P, fit["phi"], start=k)
            assert block["half_life_switching_hours"][k] == ref["hours"]
            assert block["half_life_switching_interp_hours"][k] == pytest.approx(
                ref["interpolated"])
        h = block["half_life_switching_hours"]
        assert block["ratio_switching"] == pytest.approx(h[0] / h[1])

    def test_hessian_ratio_inference_matches_the_existing_delta_block(self, block, gaussian):
        _, fit = gaussian
        old = fit["half_life_ratio"]
        hess = block["inference"]["hessian"]
        assert hess["ratio_se"] == pytest.approx(old["se_delta"], rel=1e-10)
        assert hess["ratio_ci95"]["linear"]["bounds"] == pytest.approx(old["ci95_delta"],
                                                                         rel=1e-10)
        assert hess["phi_se"] == pytest.approx([h["phi_se"] for h in fit["half_lives"]],
                                               rel=1e-10)

    def test_t_ratios_against_unity(self, block, gaussian):
        _, fit = gaussian
        for cov in block["inference"].values():
            for k in range(2):
                assert cov["phi_t_unit"][k] == pytest.approx(
                    (fit["phi"][k] - 1.0) / cov["phi_se"][k])

    def test_the_wald_test_uses_the_phi_block_of_each_covariance(self, block, gaussian):
        _, fit = gaussian
        names = fit["param_names"]
        i_phi = [names.index(f"ar.L1[{int(fit['order'][k])}]") for k in range(2)]
        for name, cov in msar_covariances(fit["result"]).items():
            ref = ratio_wald_test(fit["phi"][0], fit["phi"][1], cov[np.ix_(i_phi, i_phi)])
            assert block["inference"][name]["ratio_wald"] == ref, name

    def test_the_chain_eigenvalue_and_its_exact_standard_error(self, block, gaussian):
        """lambda = p_cc + p_ss - 1 = p[0->0] - p[1->0], linear in the parameters."""
        _, fit = gaussian
        assert block["chain_eigenvalue"] == pytest.approx(sum(block["p_stay"]) - 1.0)
        names = fit["param_names"]
        i = [names.index("p[0->0]"), names.index("p[1->0]")]
        for name, cov in msar_covariances(fit["result"]).items():
            V = cov[np.ix_(i, i)]
            se = math.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
            assert block["inference"][name]["chain_eigenvalue_se"] == pytest.approx(se), name

    def test_the_switching_ratio_takes_the_wald_verdict(self, block):
        """The simulated chain is persistent (lambda = 0.88), so R_sw lies on
        the side of unity of phi_calm - phi_stressed."""
        assert block["chain_eigenvalue"] > 0.5
        for cov in block["inference"].values():
            assert cov["ratio_switching_side"] == cov["ratio_wald"]["significance"]

    def test_the_switching_ratio_has_a_simulated_interval(self, block):
        for cov in block["inference"].values():
            sim = cov["ratio_switching_sim"]
            lo, hi = sim["ci95"]
            assert lo < block["ratio_switching_interp"] < hi
            assert sim["draws"] == SIM_DRAWS and sim["transform"] == "logit"
            assert sim["undefined"] == 0.0

    def test_simulated_durations_are_recovered(self, block):
        truth = expected_durations(P_TRUE)
        assert block["expected_duration_hours"][0] == pytest.approx(truth[0], rel=0.35)
        assert block["expected_duration_hours"][1] == pytest.approx(truth[1], rel=0.35)

    def test_the_block_is_json_native(self, block):
        import json
        json.dumps(block, allow_nan=False)


# ---------------------------------------------------------------------------
# One multi-start scan, not two
# ---------------------------------------------------------------------------
class TestSingleScan:
    def test_summarise_reproduces_fit_msar_premium(self, gaussian):
        y, fit = gaussian
        ref = fit_msar_premium(y, n_starts=2, search_reps=10)
        assert fit["llf"] == ref["llf"]
        assert np.array_equal(fit["phi"], ref["phi"])
        assert fit["half_life_ratio"] == ref["half_life_ratio"]
        assert fit["param_table"] == ref["param_table"]
        assert fit["seed"] == ref["seed"]

    def test_diagnostics_runs_each_start_exactly_once(self, monkeypatch):
        from regime_engine import diagnostics, hmm

        calls = []
        real = hmm.fit_ms_ar1

        def counting(*args, **kwargs):
            calls.append(kwargs.get("seed"))
            return real(*args, **kwargs)

        monkeypatch.setattr(hmm, "fit_ms_ar1", counting)
        y = simulate(1500, 5, "normal")
        fit = diagnostics._fit_msar_multistart(y, n_starts=3)
        assert len(calls) == 3
        assert fit["multistart_diagnostics"]["refit_logL_gap"] == 0.0
        assert fit["multistart_diagnostics"]["best_seed"] == fit["seed"]
        assert fit["multistart_diagnostics"]["excluded"] == {
            "non_finite_llf": 0, "variance_floor": 0, "singular_hessian": 0}

    def test_diagnostics_block_keeps_the_old_keys_and_adds_the_new(self, gaussian):
        from regime_engine import diagnostics

        _, fit = gaussian
        fit = dict(fit, multistart_diagnostics=dict(fit["multistart"], refit_logL_gap=0.0))
        block = diagnostics._msar_block(fit, "simulated")
        old = ("window", "regime_order", "parameterisation", "n_obs", "logL", "AIC", "BIC",
               "mu_bps", "sigma_bps", "phi", "phi_se", "half_life_hours", "half_life_se",
               "half_life_ci95", "half_life_ratio", "half_life_ratio_se",
               "half_life_ratio_ci95", "params", "multistart")
        assert list(block)[:len(old)] == list(old)
        for key in ("p_stay", "expected_duration_hours", "half_life_switching_hours",
                    "ratio_switching", "inference"):
            assert key in block
        top = block["ratio_sandwich_top_hours"]
        assert len(top) == 3 and top[0]["share"] >= top[1]["share"] >= top[2]["share"]
        assert sorted(top[0]) == ["index", "share", "y", "y_prev"]


# ---------------------------------------------------------------------------
# The out-of-sample records
# ---------------------------------------------------------------------------
class TestFitWindowRecord:
    @pytest.fixture(scope="class")
    def rec(self):
        from regime_engine.oos import fit_window
        return fit_window(simulate(1500, 7, "normal"), n_starts=3, search_reps=5,
                          label="simulated")

    def test_the_multistart_protocol_is_recorded(self, rec):
        assert "n_distinct_optima" not in rec
        assert rec["seed_base"] == 20260802
        assert rec["best_seed"] in range(20260802, 20260802 + 3)
        assert rec["failures"] + sum(o["count"] for o in rec["optima_found"]) == 3

    def test_a_non_finite_optimum_is_written_as_null(self):
        from regime_engine.oos import json_safe_optima
        optima = [{"llf": -10.5, "count": 3}, {"llf": float("nan"), "count": 1}]
        assert json_safe_optima(optima) == [{"llf": -10.5, "count": 3},
                                            {"llf": None, "count": 1}]
        assert json_safe_optima(None) is None

    def test_the_old_keys_are_still_there(self, rec):
        for key in ("label", "T", "tau_calm", "tau_stressed", "ratio", "ratio_se", "ci95",
                    "significance", "sigma2", "phi", "llf"):
            assert key in rec

    def test_the_excluded_starts_are_recorded(self, rec):
        assert rec["excluded"] == {"non_finite_llf": 0, "variance_floor": 0,
                                   "singular_hessian": 0}

    def test_the_wald_verdicts_are_attached(self, rec):
        for cov in ("hessian", "opg", "robust"):
            wald = rec["inference"][cov]["ratio_wald"]
            assert wald["significance"] in {"above", "below", "ns", "undetermined"}
            assert wald["restriction"] == "phi_calm = phi_stressed"

    def test_the_robust_inference_is_attached(self, rec):
        assert rec["inference"]["hessian"]["ratio_se"] == pytest.approx(rec["ratio_se"])
        assert rec["inference"]["hessian"]["ratio_ci95"]["linear"]["bounds"] == \
            pytest.approx(rec["ci95"])
        assert rec["inference"]["hessian"]["ratio_ci95"]["linear"]["significance"] == \
            rec["significance"]
        assert len(rec["p_stay"]) == 2 and len(rec["half_life_switching_hours"]) == 2


# ---------------------------------------------------------------------------
# Which hours carry the sandwich variance of R
# ---------------------------------------------------------------------------
class TestSandwichInfluence:
    def test_contributions_add_up_to_the_sandwich_variance(self, gaussian):
        _, fit = gaussian
        out = ratio_sandwich_influence(fit)
        se_rob = msar_robust_inference(fit)["inference"]["robust"]["ratio_se"]
        assert out["variance"] == pytest.approx(se_rob ** 2, rel=1e-6)
        assert sum(out["shares"]) == pytest.approx(1.0)

    def test_a_correctly_specified_series_has_no_dominant_hour(self, gaussian):
        """With some 600 stressed hours, the largest single share is a few per cent."""
        _, fit = gaussian
        out = ratio_sandwich_influence(fit)
        assert out["top"][0]["share"] < 0.10

    def test_one_extreme_reversal_dominates(self):
        """A spike that reverts within the hour is the leverage point of the
        stressed-regime autoregression, as 10 October 2025 is on the paper window."""
        y = simulate(3000, 21, "normal")
        y[1500] = 40.0                       # one huge positive hour, then back
        fit = summarise_msar_fit(y, *multistart(y))
        out = ratio_sandwich_influence(fit)
        assert out["top"][0]["index"] in (1500, 1501)
        assert out["top"][0]["share"] > 0.3
        assert out["top"][0]["y"] == y[out["top"][0]["index"]]


# ---------------------------------------------------------------------------
# The guarded multi-start scan
# ---------------------------------------------------------------------------
NAMES = ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "sigma2[0]", "sigma2[1]",
         "ar.L1[0]", "ar.L1[1]"]
BASE = 20260802


def _start(llf, sigma2=(0.3, 4.0), hessian_ok=True):
    """What one start of hmm.fit_ms_ar1 returns, reduced to what the scan reads."""
    params = np.array([0.9, 0.1, -3.0, -3.0, sigma2[0], sigma2[1], 0.9, 0.7])
    cov = np.eye(len(NAMES)) * 1e-4 if hessian_ok else np.full((len(NAMES),) * 2, np.nan)
    return (SimpleNamespace(param_names=list(NAMES)),
            SimpleNamespace(llf=llf, params=params, cov_params_approx=cov))


@pytest.fixture()
def scripted(monkeypatch):
    """Replace hmm.fit_ms_ar1 by a script: start i returns outcomes[i] or raises it."""
    from regime_engine import hmm

    def install(outcomes):
        def fit(y, seed, search_reps, seed_retries, **kwargs):
            out = outcomes[seed - BASE]
            if isinstance(out, Exception):
                raise out
            return out[0], out[1], seed
        monkeypatch.setattr(hmm, "fit_ms_ar1", fit)
    return install


class TestGuardedMultistart:
    Y = np.random.default_rng(0).normal(size=500)

    def scan(self, n):
        return fit_msar_multistart(self.Y, n_starts=n, seed=BASE, verbose=False)

    def test_a_non_finite_likelihood_is_never_the_optimum(self, scripted):
        """Not even from the first start, where `llf > nan` would keep it."""
        scripted([_start(float("nan")), _start(-10.0), _start(-12.0)])
        _, res, seed, diag = self.scan(3)
        assert (seed, res.llf) == (BASE + 1, -10.0)
        assert diag["excluded"] == {"non_finite_llf": 1, "variance_floor": 0,
                                    "singular_hessian": 0}

    def test_a_collapsed_regime_is_excluded_however_high_its_likelihood(self, scripted):
        """One regime's variance on a single hour sends the likelihood upward."""
        scripted([_start(-10.0), _start(-3.0, sigma2=(1e-232, 4.0)), _start(-12.0)])
        _, res, seed, diag = self.scan(3)
        assert (seed, res.llf) == (BASE, -10.0)
        assert diag["excluded"]["variance_floor"] == 1
        assert diag["sigma2_floor"] == pytest.approx(1e-6 * np.var(self.Y))

    def test_a_singular_hessian_is_excluded(self, scripted):
        scripted([_start(-10.0), _start(-9.0, hessian_ok=False)])
        _, res, seed, diag = self.scan(2)
        assert (seed, res.llf) == (BASE, -10.0)
        assert diag["excluded"]["singular_hessian"] == 1

    def test_failures_and_optima_are_counted_as_before(self, scripted):
        scripted([RuntimeError("no start"), _start(-10.0), _start(-10.001), _start(-12.0),
                  _start(float("nan"))])
        _, _, seed, diag = self.scan(5)
        assert diag["failures"] == 1
        found = [(o["llf"], o["count"]) for o in diag["optima_found"]]
        assert found[:2] == [(-10.0, 2), (-12.0, 1)]
        assert math.isnan(found[2][0]) and found[2][1] == 1
        assert (diag["best_llf"], diag["best_seed"], seed) == (-10.0, BASE + 1, BASE + 1)
        assert diag["modal_llf"] == -10.0 and diag["modal_is_best"] is True
        assert (diag["n_starts"], diag["seed_base"]) == (5, BASE)

    def test_no_admissible_start_is_an_error(self, scripted):
        scripted([_start(float("nan")), _start(-1.0, sigma2=(0.0, 1.0))])
        with pytest.raises(RuntimeError, match="admissible"):
            self.scan(2)

    def test_on_a_clean_series_it_is_the_unguarded_scan(self):
        y = simulate(1500, 7, "normal")
        model, res, seed, diag = fit_msar_multistart(y, n_starts=3, search_reps=5,
                                                     verbose=False)
        _, ref, ref_seed, ref_diag = fit_ms_ar1_multistart(y, n_starts=3, search_reps=5,
                                                           verbose=False)
        assert (seed, res.llf) == (ref_seed, ref.llf)
        assert {k: diag[k] for k in ref_diag} == ref_diag
        assert diag["excluded"] == {"non_finite_llf": 0, "variance_floor": 0,
                                    "singular_hessian": 0}

    def test_the_paper_protocol_runs_the_guarded_scan(self, gaussian):
        y, _ = gaussian
        assert "excluded" in fit_msar_premium(y, n_starts=2, search_reps=10)["multistart"]
