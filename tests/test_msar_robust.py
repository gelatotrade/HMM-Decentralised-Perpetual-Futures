"""Switching-aware half-lives, interval forms and the Wald test for the MS-AR ratio.

The functions under test are pure: they take a transition matrix, the two
autoregressive coefficients or a covariance block, and return numbers whose
ground truth is known in closed form or by simulation. Nothing here fits a
model; ``test_msar_robust_fits.py`` covers the estimation side.

Conventions: ``P`` is row-stochastic, ``P[i, j] = P(S_{t+1} = j | S_t = i)``,
and index 0 is the calm regime, index 1 the stressed one.
"""
import math

import numpy as np
import pytest

from regime_engine.msar import (
    Z95,
    chain_eigenvalue,
    expected_durations,
    fieller_interval,
    half_life,
    ratio_intervals,
    ratio_wald_test,
    side_of_unity,
    stationary_distribution,
    switching_decay_closed_form,
    switching_decay_path,
    switching_delta,
    switching_half_life,
    switching_half_lives_batch,
    switching_half_lives_from_raw,
    switching_ratio_side,
    switching_ratio_simulation,
)

# The paper-window optimum (output/diagnostics.json, raw statsmodels order:
# regime 0 is the stressed one). Rounded to the stored precision.
P_SS = 0.6162978791129136            # p[0->0]
P_CS = 0.02582452518519525           # p[1->0], calm -> stressed
PHI_CALM = 0.9806916785026165        # ar.L1[1]
PHI_STRESSED = 0.8040203091201261    # ar.L1[0]
PAPER_P = np.array([[1.0 - P_CS, P_CS], [1.0 - P_SS, P_SS]])
PAPER_PHI = np.array([PHI_CALM, PHI_STRESSED])


# ---------------------------------------------------------------------------
# Chain summaries
# ---------------------------------------------------------------------------
class TestChain:
    def test_expected_durations_are_geometric_means(self):
        P = np.array([[0.9, 0.1], [0.4, 0.6]])
        assert expected_durations(P) == pytest.approx([10.0, 2.5])

    def test_paper_window_durations(self):
        d = expected_durations(PAPER_P)
        assert round(d[0], 1) == 38.7
        assert round(d[1], 2) == 2.61

    def test_an_absorbing_regime_has_no_finite_duration(self):
        d = expected_durations(np.array([[1.0, 0.0], [0.5, 0.5]]))
        assert np.isinf(d[0]) and d[1] == pytest.approx(2.0)

    def test_stationary_distribution_solves_pi_p_equals_pi(self):
        P = np.array([[0.97, 0.03], [0.30, 0.70]])
        pi = stationary_distribution(P)
        assert pi.sum() == pytest.approx(1.0)
        assert pi @ P == pytest.approx(pi)
        # two-state closed form: pi_1 = p_01 / (p_01 + p_10)
        assert pi[1] == pytest.approx(0.03 / 0.33)

    def test_paper_window_stressed_share_is_six_per_cent(self):
        assert round(stationary_distribution(PAPER_P)[1], 3) == 0.063


# ---------------------------------------------------------------------------
# Switching-adjusted decay
# ---------------------------------------------------------------------------
class TestSwitchingDecay:
    def test_path_starts_at_one_and_follows_the_recursion(self):
        path = switching_decay_path(PAPER_P, PAPER_PHI, start=1, horizon=3)
        M = PAPER_P * PAPER_PHI[None, :]
        assert path[0] == 1.0
        for h in range(1, 4):
            expected = np.linalg.matrix_power(M, h)[1].sum()
            assert path[h] == pytest.approx(expected, rel=1e-12)

    def test_without_switching_the_decay_is_the_within_regime_decay(self):
        """P = I: the chain never leaves the regime, so m_k(h) = phi_k^h."""
        phi = np.array([0.97, 0.80])
        for k in range(2):
            path = switching_decay_path(np.eye(2), phi, start=k, horizon=60)
            assert path == pytest.approx(phi[k] ** np.arange(61), rel=1e-12)

    def test_without_switching_the_half_life_is_the_within_regime_one(self):
        phi = np.array([0.97, 0.80])
        tau = half_life(phi)
        for k in range(2):
            hl = switching_half_life(np.eye(2), phi, start=k)
            # log-linear interpolation is exact on a geometric path
            assert hl["interpolated"] == pytest.approx(tau[k], rel=1e-10)
            assert hl["hours"] == int(np.ceil(tau[k]))

    def test_equal_persistence_makes_switching_irrelevant(self):
        """phi_calm = phi_stressed: every row of P o Phi sums to phi."""
        phi = np.array([0.9, 0.9])
        P = np.array([[0.95, 0.05], [0.4, 0.6]])
        for k in range(2):
            hl = switching_half_life(P, phi, start=k)
            assert hl["interpolated"] == pytest.approx(float(half_life(0.9)), rel=1e-10)

    def test_closed_form_agrees_with_the_iteration(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            stay = rng.uniform(0.5, 0.99, size=2)
            P = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
            phi = rng.uniform(0.1, 0.99, size=2)
            hours = np.arange(0, 150)
            for k in range(2):
                path = switching_decay_path(P, phi, start=k, horizon=149)
                closed = switching_decay_closed_form(P, phi, start=k, hours=hours)
                assert closed == pytest.approx(path, rel=1e-8, abs=1e-14)

    def test_closed_form_handles_a_negative_second_eigenvalue(self):
        """p_cc + p_ss < 1 makes det(P o Phi) negative: the path oscillates."""
        P = np.array([[0.3, 0.7], [0.8, 0.2]])
        phi = np.array([0.9, 0.6])
        hours = np.arange(0, 40)
        for k in range(2):
            path = switching_decay_path(P, phi, start=k, horizon=39)
            closed = switching_decay_closed_form(P, phi, start=k, hours=hours)
            assert closed == pytest.approx(path, rel=1e-9, abs=1e-14)

    def test_the_decay_matches_simulated_paths(self):
        """E[x_{t+h} | x_t = 1, S_t = k] by Monte Carlo over regime paths.

        Noise-free deviations decay as x_{t+h} = phi_{S_{t+h}} x_{t+h-1}, with
        the regime drawn from the chain; their average must reproduce
        e_k' (P o Phi)^h 1 within Monte Carlo error.
        """
        rng = np.random.default_rng(20261008)
        n, horizon = 40_000, 40
        for k in range(2):
            s = np.full(n, k)
            x = np.ones(n)
            means = [1.0]
            for _ in range(horizon):
                s = (rng.random(n) < PAPER_P[s, 1]).astype(int)
                x = x * PAPER_PHI[s]
                means.append(x.mean())
            path = switching_decay_path(PAPER_P, PAPER_PHI, start=k, horizon=horizon)
            se = x.std() / np.sqrt(n)
            assert np.max(np.abs(np.array(means) - path)) < 5 * max(se, 1e-3)

    def test_paper_window_half_lives_are_25_and_17_hours(self):
        calm = switching_half_life(PAPER_P, PAPER_PHI, start=0)
        stressed = switching_half_life(PAPER_P, PAPER_PHI, start=1)
        assert (calm["hours"], stressed["hours"]) == (25, 17)
        assert round(calm["hours"] / stressed["hours"], 2) == 1.47
        # the integer hour is the first one at or below one half
        path = switching_decay_path(PAPER_P, PAPER_PHI, start=0, horizon=30)
        assert path[24] > 0.5 >= path[25]
        assert 24.0 < calm["interpolated"] <= 25.0
        assert 16.0 < stressed["interpolated"] <= 17.0

    def test_the_stressed_regime_ends_before_its_own_half_life(self):
        """The reason the within-regime ratio overstates the realised one."""
        assert expected_durations(PAPER_P)[1] < half_life(PHI_STRESSED)

    def test_a_unit_root_never_halves(self):
        hl = switching_half_life(np.eye(2), np.array([1.0, 1.0]), start=0, max_hours=500)
        assert hl == {"hours": None, "interpolated": None}

    def test_start_must_be_a_regime_index(self):
        with pytest.raises(ValueError):
            switching_decay_path(PAPER_P, PAPER_PHI, start=2, horizon=5)


# ---------------------------------------------------------------------------
# Fieller's interval for a ratio of two normal estimates
# ---------------------------------------------------------------------------
class TestFieller:
    def test_precise_denominator_gives_the_delta_interval(self):
        a, b = -2.0, -0.5
        var_a, var_b, cov_ab = 1e-6, 1e-8, 0.0
        f = fieller_interval(a, b, var_a, var_b, cov_ab)
        r = a / b
        se = np.sqrt(var_a / b**2 + a**2 * var_b / b**4)
        assert f["shape"] == "bounded"
        assert f["bounds"] == pytest.approx([r - Z95 * se, r + Z95 * se], rel=1e-4)

    def test_the_point_estimate_lies_inside_a_bounded_set(self):
        f = fieller_interval(-1.5, -0.2, 0.04, 0.002, 0.001)
        lo, hi = f["bounds"]
        assert f["shape"] == "bounded" and lo < -1.5 / -0.2 < hi

    def test_an_insignificant_denominator_gives_two_rays(self):
        # b^2 < z^2 var_b, but the numerator is precise
        f = fieller_interval(-1.0, -0.1, 0.0001, 0.01, 0.0)
        assert f["shape"] == "two_rays"
        lo, hi = f["bounds"]
        assert lo < hi
        assert -1.0 / -0.1 >= hi or -1.0 / -0.1 <= lo

    def test_an_insignificant_numerator_and_denominator_give_the_whole_line(self):
        f = fieller_interval(-0.01, -0.01, 1.0, 1.0, 0.0)
        assert f == {"shape": "whole_line", "bounds": [None, None]}

    def test_endpoints_solve_the_fieller_quadratic(self):
        a, b, va, vb, cab = -0.22, -0.0195, 0.0033, 1.39e-5, 1.2e-5
        f = fieller_interval(a, b, va, vb, cab)
        for r in f["bounds"]:
            lhs = (a - r * b) ** 2
            rhs = Z95**2 * (va - 2 * r * cab + r**2 * vb)
            assert lhs == pytest.approx(rhs, rel=1e-9)

    def test_coverage_matches_the_nominal_level_in_simulation(self):
        """With exactly normal (a, b) the pivot is exactly normal, so coverage is 95%.

        In the MS-AR application a and b are ln|phi_k|, whose variances come
        from the delta method, so there the set is first-order too.
        """
        rng = np.random.default_rng(42)
        mu_a, mu_b = -0.22, -0.02
        W = np.array([[0.0030, 0.00002], [0.00002, 0.000045]])   # |mu_b| / se_b = 3
        # Accelerate's BLAS raises spurious FP flags inside this matmul.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            draws = rng.multivariate_normal([mu_a, mu_b], W, size=20_000)
        assert np.all(np.isfinite(draws))
        truth = mu_a / mu_b
        covered = 0
        for a, b in draws:
            f = fieller_interval(a, b, W[0, 0], W[1, 1], W[0, 1])
            pivot = (a - truth * b) / np.sqrt(W[0, 0] - 2 * truth * W[0, 1] + truth**2 * W[1, 1])
            inside = _contains(f, truth)
            assert inside == (abs(pivot) <= Z95)
            covered += inside
        assert abs(covered / len(draws) - 0.95) < 0.006


def _contains(f, r) -> bool:
    lo, hi = f["bounds"]
    if f["shape"] == "whole_line":
        return True
    if f["shape"] == "bounded":
        return lo <= r <= hi
    return r <= lo or r >= hi


# ---------------------------------------------------------------------------
# Which side of unity, on the admissible (positive) part of a set
# ---------------------------------------------------------------------------
class TestSideOfUnity:
    @pytest.mark.parametrize("shape, bounds, side", [
        ("bounded", [1.2, 5.0], "above"),
        ("bounded", [0.1, 0.9], "below"),
        ("bounded", [0.5, 2.0], "ns"),
        ("bounded", [-3.0, 0.8], "below"),       # negative values are not ratios
        ("bounded", [-13.65, 36.03], "ns"),
        ("bounded", [-2.0, -1.0], "undetermined"),
        ("bounded", [None, 4.0], "undetermined"),
        ("two_rays", [-5.0, 3.0], "above"),      # positive part [3, inf)
        ("two_rays", [-5.0, 0.7], "ns"),
        ("two_rays", [0.5, 3.0], "undetermined"),  # (0, 0.5] U [3, inf)
        ("two_rays", [2.0, 3.0], "ns"),
        ("whole_line", [None, None], "ns"),
    ])
    def test_cases(self, shape, bounds, side):
        assert side_of_unity(shape, bounds) == side


# ---------------------------------------------------------------------------
# The three interval forms for R = tau_calm / tau_stressed
# ---------------------------------------------------------------------------
class TestRatioIntervals:
    # A covariance close to the paper window's Hessian block for (phi_c, phi_s).
    COV = np.array([[0.0036596**2, 1.0e-5], [1.0e-5, 0.0461898**2]])

    def test_point_estimate_and_linear_interval_are_the_delta_method(self):
        out = ratio_intervals(PHI_CALM, PHI_STRESSED, self.COV)
        r = half_life(PHI_CALM) / half_life(PHI_STRESSED)
        lc, ls = np.log(PHI_CALM), np.log(PHI_STRESSED)
        g = np.array([-ls / (PHI_CALM * lc**2), 1.0 / (PHI_STRESSED * lc)])
        se = float(np.sqrt(g @ self.COV @ g))
        assert out["value"] == pytest.approx(r)
        assert out["se"] == pytest.approx(se)
        assert out["linear"]["bounds"] == pytest.approx([r - Z95 * se, r + Z95 * se])
        assert out["linear"]["shape"] == "bounded"

    def test_log_interval_is_the_delta_method_on_log_r(self):
        out = ratio_intervals(PHI_CALM, PHI_STRESSED, self.COV)
        lo, hi = out["log"]["bounds"]
        r, se = out["value"], out["se"]
        assert lo * hi == pytest.approx(r * r)
        assert hi / lo == pytest.approx(np.exp(2 * Z95 * se / r))
        assert lo > 0

    def test_fieller_is_computed_on_the_log_coefficients(self):
        out = ratio_intervals(PHI_CALM, PHI_STRESSED, self.COV)
        a, b = np.log(PHI_STRESSED), np.log(PHI_CALM)
        J = np.diag([1 / PHI_STRESSED, 1 / PHI_CALM])
        W = J @ self.COV[::-1, ::-1] @ J
        ref = fieller_interval(a, b, W[0, 0], W[1, 1], W[0, 1])
        assert out["fieller"]["shape"] == ref["shape"]
        assert out["fieller"]["bounds"] == pytest.approx(ref["bounds"])

    def test_every_form_carries_a_side_of_unity(self):
        out = ratio_intervals(PHI_CALM, PHI_STRESSED, self.COV)
        for form in ("linear", "log", "fieller"):
            assert out[form]["significance"] in {"above", "below", "ns", "undetermined"}
        assert out["linear"]["significance"] == "above"

    def test_a_wide_covariance_separates_the_forms(self):
        """Linear covers unity where the log and Fieller forms do not.

        This is the replace-max situation of the paper window: the delta
        interval of a positive, right-skewed ratio is symmetric and runs
        through unity, while the log-scale and Fieller sets stay above it.
        """
        phi_c, phi_s = 0.9797605665800738, 0.9397031821210978   # replace-max fit
        cov = np.array([[0.0035**2, 0.0], [0.0, 0.021**2]])
        out = ratio_intervals(phi_c, phi_s, cov)
        assert out["linear"]["significance"] == "ns"
        assert out["log"]["significance"] == "above"

    def test_non_finite_covariance_is_undetermined(self):
        out = ratio_intervals(PHI_CALM, PHI_STRESSED, np.full((2, 2), np.nan))
        assert out["se"] is None
        for form in ("linear", "log", "fieller"):
            assert out[form]["significance"] == "undetermined"


# ---------------------------------------------------------------------------
# Delta method for the switching-adjusted half-lives
# ---------------------------------------------------------------------------
class TestSwitchingDelta:
    # raw statsmodels order of the paper window: regime 0 stressed, 1 calm
    THETA = np.array([P_SS, P_CS, PHI_STRESSED, PHI_CALM])   # p[0->0], p[1->0], ar.L1[0], ar.L1[1]
    ORDER = np.array([1, 0])

    def test_raw_parameters_map_to_the_calm_first_chain(self):
        h = switching_half_lives_from_raw(self.THETA, self.ORDER)
        ref = [switching_half_life(PAPER_P, PAPER_PHI, k)["interpolated"] for k in range(2)]
        assert h == pytest.approx(ref)

    def test_standard_errors_match_a_simulation_of_the_parameters(self):
        """With a small covariance the linearisation is accurate: the delta
        standard errors must equal the spread of the function over draws."""
        sd = np.array([0.004, 0.0008, 0.004, 0.0004])
        corr = np.eye(4)
        corr[2, 3] = corr[3, 2] = 0.3
        cov = corr * np.outer(sd, sd)
        out = switching_delta(self.THETA, self.ORDER, cov)
        rng = np.random.default_rng(3)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            draws = rng.multivariate_normal(self.THETA, cov, size=4000)
        sims = np.array([switching_half_lives_from_raw(t, self.ORDER) for t in draws])
        assert out["half_life_se"][0] == pytest.approx(sims[:, 0].std(), rel=0.06)
        assert out["half_life_se"][1] == pytest.approx(sims[:, 1].std(), rel=0.06)
        ratio = sims[:, 0] / sims[:, 1]
        assert out["ratio_se"] == pytest.approx(ratio.std(), rel=0.06)

    def test_no_covariance_means_no_uncertainty(self):
        out = switching_delta(self.THETA, self.ORDER, np.zeros((4, 4)))
        assert out["ratio_se"] == 0.0
        assert out["ratio_ci95"]["log"]["bounds"] == pytest.approx([out["ratio"]] * 2)

    def test_intervals_carry_a_side_of_unity(self):
        sd = np.array([0.04, 0.004, 0.17, 0.007])          # sandwich-sized
        out = switching_delta(self.THETA, self.ORDER, np.diag(sd ** 2))
        for form in ("linear", "log"):
            assert out["ratio_ci95"][form]["significance"] in {"above", "below", "ns",
                                                               "undetermined"}
        assert round(out["ratio"], 3) == round(
            switching_half_life(PAPER_P, PAPER_PHI, 0)["interpolated"]
            / switching_half_life(PAPER_P, PAPER_PHI, 1)["interpolated"], 3)


# ---------------------------------------------------------------------------
# The Wald test of R = 1
# ---------------------------------------------------------------------------
def _two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


class TestRatioWald:
    # A covariance close to the paper window's Hessian block for (phi_c, phi_s).
    COV = np.array([[0.0036596**2, 1.0e-5], [1.0e-5, 0.0461898**2]])

    def test_the_statistic_is_the_linear_restriction(self):
        out = ratio_wald_test(PHI_CALM, PHI_STRESSED, self.COV)
        se = math.sqrt(self.COV[0, 0] + self.COV[1, 1] - 2 * self.COV[0, 1])
        z = (PHI_CALM - PHI_STRESSED) / se
        assert out["restriction"] == "phi_calm = phi_stressed"
        assert out["difference"] == pytest.approx(PHI_CALM - PHI_STRESSED)
        assert out["se"] == pytest.approx(se)
        assert out["z"] == pytest.approx(z)
        assert out["p"] == pytest.approx(_two_sided_p(z))
        assert out["significance"] == "above"

    def test_a_unit_ratio_means_equal_coefficients(self):
        """R = ln|phi_s| / ln|phi_c| is one exactly when |phi_c| = |phi_s|, and
        above one exactly when the calm coefficient is the larger."""
        cov = np.diag([1e-4, 1e-4])
        for pc, ps in ((0.9, 0.9), (0.95, 0.80), (0.70, 0.90), (0.98, 0.979)):
            r = float(half_life(pc) / half_life(ps))
            z = ratio_wald_test(pc, ps, cov)["z"]
            assert np.sign(z) == np.sign(round(r - 1.0, 12)), (pc, ps)

    @pytest.mark.parametrize("z, side", [(1.95, "ns"), (1.97, "above"),
                                         (-1.95, "ns"), (-1.97, "below")])
    def test_the_verdict_is_two_sided_at_five_per_cent(self, z, side):
        cov = np.diag([0.005, 0.005])            # s.e. of the difference: 0.1
        d = 0.1 * z
        out = ratio_wald_test(0.6 + d / 2, 0.6 - d / 2, cov)
        assert out["z"] == pytest.approx(z)
        assert out["significance"] == side
        assert (out["p"] < 0.05) == (side != "ns")

    def test_negative_coefficients_compare_absolute_values(self):
        cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        both = ratio_wald_test(-0.9, -0.5, cov)
        assert both["restriction"] == "phi_calm = phi_stressed"
        assert both["difference"] == pytest.approx(0.4)          # |phi_c| - |phi_s|
        assert both["se"] == pytest.approx(math.sqrt(0.0004 + 0.0009 - 0.0002))
        mixed = ratio_wald_test(0.9, -0.5, cov)
        assert mixed["restriction"] == "phi_calm = -phi_stressed"
        assert mixed["difference"] == pytest.approx(0.4)
        assert mixed["se"] == pytest.approx(math.sqrt(0.0004 + 0.0009 + 0.0002))
        assert both["significance"] == mixed["significance"] == "above"

    def test_it_holds_its_size_where_the_log_interval_does_not(self):
        """Under phi_calm = phi_stressed, with phi_stressed's interval reaching unity.

        The statistic is exactly normal here, so the test rejects 5% of the
        time. The log-scale delta interval linearises ln R, which is singular
        at phi_stressed = 1, and rejects R = 1 far more often.
        """
        rng = np.random.default_rng(20261008)
        cov = np.diag([0.004**2, 0.05**2])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            draws = rng.multivariate_normal([0.95, 0.95], cov, size=6000)
        wald = log = 0
        for pc, ps in draws:
            wald += ratio_wald_test(pc, ps, cov)["significance"] != "ns"
            if 0 < ps < 1 and 0 < pc < 1:
                log += ratio_intervals(pc, ps, cov)["log"]["significance"] in ("above", "below")
        assert abs(wald / len(draws) - 0.05) < 0.009
        assert log / len(draws) > 0.07

    def test_a_stressed_interval_reaching_unity_misleads_only_the_log_form(self):
        """The paper window under the sandwich: phi_stressed 0.80 with s.e. 0.17."""
        se_c, se_s, se_d = 0.006937287382478705, 0.17273909245655528, 0.1746331603294749
        cov_cs = (se_c**2 + se_s**2 - se_d**2) / 2
        cov = np.array([[se_c**2, cov_cs], [cov_cs, se_s**2]])
        wald = ratio_wald_test(PHI_CALM, PHI_STRESSED, cov)
        forms = ratio_intervals(PHI_CALM, PHI_STRESSED, cov)
        assert round(wald["z"], 2) == 1.01 and round(wald["p"], 2) == 0.31
        assert wald["significance"] == "ns"
        assert [round(v, 2) for v in forms["log"]["bounds"]] == [1.21, 103.07]
        assert forms["log"]["significance"] == "above"
        assert forms["linear"]["significance"] == forms["fieller"]["significance"] == "ns"

    def test_a_non_finite_covariance_is_undetermined(self):
        out = ratio_wald_test(PHI_CALM, PHI_STRESSED, np.full((2, 2), np.nan))
        assert out["significance"] == "undetermined"
        assert out["z"] is None and out["p"] is None


# ---------------------------------------------------------------------------
# Which side of unity the switching-adjusted ratio falls on
# ---------------------------------------------------------------------------
class TestChainEigenvalue:
    def test_it_is_the_second_eigenvalue_of_p(self):
        P = np.array([[0.97, 0.03], [0.30, 0.70]])
        lam = chain_eigenvalue(P)
        assert lam == pytest.approx(0.67)
        assert sorted(np.linalg.eigvals(P).real) == pytest.approx([lam, 1.0])

    def test_the_paper_window_chain_is_persistent(self):
        assert round(chain_eigenvalue(PAPER_P), 3) == 0.590


class TestSwitchingOrder:
    def test_a_persistent_chain_orders_it_like_the_coefficients(self):
        """p_cc + p_ss > 1: a chain started in the calm regime is calm at least
        as often at every later hour (a monotone coupling), so with
        0 < phi_s < phi_c the calm path stays above the stressed one and halves
        later. R_sw > 1 exactly when phi_c > phi_s, as for R."""
        rng = np.random.default_rng(11)
        checked = 0
        while checked < 300:
            stay = rng.uniform(0.02, 0.999, size=2)
            phi = rng.uniform(0.05, 0.995, size=2)
            if stay.sum() <= 1.0 or abs(phi[0] - phi[1]) < 0.01:
                continue
            P = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
            hc, hs = (switching_half_life(P, phi, k)["interpolated"] for k in range(2))
            assert np.sign(hc / hs - 1.0) == np.sign(phi[0] - phi[1]), (stay, phi)
            checked += 1

    def test_an_alternating_chain_can_reverse_it(self):
        """p_cc + p_ss < 1: from the calm regime the next hour is usually stressed."""
        P = np.array([[0.2, 0.8], [0.9, 0.1]])
        phi = np.array([0.9, 0.5])
        hc, hs = (switching_half_life(P, phi, k)["interpolated"] for k in range(2))
        assert phi[0] > phi[1] and hc < hs

    @pytest.mark.parametrize("wald", ["above", "below", "ns"])
    def test_a_significantly_persistent_chain_takes_the_wald_verdict(self, wald):
        assert switching_ratio_side(wald, 0.59, 0.10, [0.98, 0.80]) == wald

    @pytest.mark.parametrize("lam, lam_se, phi", [
        (0.15, 0.10, [0.98, 0.80]),          # persistence not significant
        (-0.50, 0.05, [0.98, 0.80]),         # an alternating chain
        (0.59, 0.10, [0.98, -0.20]),         # a negative coefficient
        (0.59, None, [0.98, 0.80]),          # no standard error
    ])
    def test_otherwise_the_side_is_undetermined(self, lam, lam_se, phi):
        assert switching_ratio_side("above", lam, lam_se, phi) == "undetermined"

    def test_an_undetermined_wald_test_stays_undetermined(self):
        assert switching_ratio_side("undetermined", 0.59, 0.10, [0.98, 0.80]) == "undetermined"

    def test_equal_coefficients_not_rejected_leave_unity_in_any_chain(self):
        """phi_calm = phi_stressed makes R_sw = 1 whatever the chain."""
        assert switching_ratio_side("ns", -0.50, 0.05, [0.98, 0.80]) == "ns"


# ---------------------------------------------------------------------------
# Switching-adjusted half-lives for many chains at once, and a simulated
# interval for their ratio
# ---------------------------------------------------------------------------
class TestSwitchingBatch:
    def test_it_reproduces_the_scalar_half_life(self):
        rng = np.random.default_rng(12)
        n = 400
        stay = rng.uniform(0.01, 0.999, size=(n, 2))
        phi = rng.uniform(-0.6, 0.999, size=(n, 2))
        hours, interp = switching_half_lives_batch(stay, phi)
        for i in range(n):
            P = np.array([[stay[i, 0], 1 - stay[i, 0]], [1 - stay[i, 1], stay[i, 1]]])
            for k in range(2):
                ref = switching_half_life(P, phi[i], k)
                assert hours[i, k] == ref["hours"]
                assert interp[i, k] == pytest.approx(ref["interpolated"], rel=1e-12)

    def test_a_path_that_never_halves_is_infinite(self):
        hours, interp = switching_half_lives_batch(np.array([[0.9, 0.8]]),
                                                   np.array([[1.0, 1.0]]), max_hours=200)
        assert np.all(np.isinf(hours)) and np.all(np.isinf(interp))

    def test_it_can_return_where_an_unfinished_path_ends(self):
        stay = np.array([[0.9, 0.8], [0.9, 0.8]])
        phi = np.array([[1.0, 1.0], [0.5, 0.5]])
        hours, _, end = switching_half_lives_batch(stay, phi, max_hours=200, return_end=True)
        assert end[0] == pytest.approx([1.0, 1.0])          # never halves: m(200) = 1
        assert np.all(np.isnan(end[1])) and np.all(hours[1] == 1.0)


class TestSwitchingSimulation:
    # raw statsmodels order of the paper window: regime 0 stressed, 1 calm
    THETA = np.array([P_SS, P_CS, PHI_STRESSED, PHI_CALM])
    ORDER = np.array([1, 0])

    def test_no_uncertainty_gives_the_point_estimate(self):
        out = switching_ratio_simulation(self.THETA, self.ORDER, np.zeros((4, 4)), draws=200)
        h = switching_half_lives_from_raw(self.THETA, self.ORDER)
        assert out["ci95"] == pytest.approx([h[0] / h[1]] * 2)
        assert out["ci95_hours"] == pytest.approx([25 / 17] * 2)
        assert out["infinite"] == 0.0 and out["undefined"] == 0.0

    def test_a_small_covariance_reproduces_the_delta_interval(self):
        cov = np.diag(np.array([0.004, 0.0008, 0.004, 0.0004]) ** 2)
        sim = switching_ratio_simulation(self.THETA, self.ORDER, cov, draws=40_000)
        delta = switching_delta(self.THETA, self.ORDER, cov)["ratio_ci95"]["linear"]["bounds"]
        assert sim["ci95"] == pytest.approx(delta, rel=0.02)

    def test_transition_probabilities_are_drawn_on_the_logit_scale(self):
        """No draw leaves (0, 1), however wide the interval of a probability is.

        Discarding the draws a normal puts above one would condition the other
        parameters on the probability, through their correlation, and could
        move the interval across unity although the Wald test does not.
        """
        theta = np.array([0.98, 0.02, 0.80, 0.98])
        cov = np.diag([0.02**2, 1e-8, 1e-8, 1e-8])        # a normal draw > 1 with P = 0.16
        out = switching_ratio_simulation(theta, self.ORDER, cov, draws=20_000)
        assert out["transform"] == "logit"
        assert out["undefined"] == 0.0 and None not in out["ci95"]
        # the varied probability is the stressed regime's: its sojourn moves R_sw
        assert out["ci95"][0] < out["ci95"][1]

    def test_explosive_draws_keep_the_order_of_their_paths(self):
        """phi_calm at unity: one draw in six exceeds 1.02 and neither path halves.

        Such a draw has the calm path above the stressed one at every hour, so
        its ratio counts as unbounded above rather than as missing.
        """
        theta = np.array([P_SS, P_CS, PHI_STRESSED, 1.0])
        cov = np.diag([1e-8, 1e-8, 1e-6, 0.02**2])
        out = switching_ratio_simulation(theta, self.ORDER, cov, draws=20_000)
        assert out["infinite"] > 0.1 and out["undefined"] == 0.0
        assert out["ci95"][0] > 1.0 and out["ci95"][1] is None

    def test_it_is_reproducible(self):
        cov = np.diag(np.array([0.04, 0.004, 0.17, 0.007]) ** 2)
        a = switching_ratio_simulation(self.THETA, self.ORDER, cov)
        b = switching_ratio_simulation(self.THETA, self.ORDER, cov)
        assert a == b and a["draws"] >= 20_000

    @pytest.mark.parametrize("se_s, covers", [(0.05, False), (0.17, True)])
    def test_its_side_of_unity_is_the_coefficients(self, se_s, covers):
        """With a persistent chain a draw has R_sw > 1 exactly when its
        phi_c > phi_s, so the interval clears unity when the difference is
        significant (z = 3.5 here) and covers it when it is not (z = 1.0)."""
        cov = np.diag([1e-6, 1e-6, se_s**2, 0.004**2])  # p00, p10, phi_stressed, phi_calm
        lo, hi = switching_ratio_simulation(self.THETA, self.ORDER, cov, draws=20_000)["ci95"]
        assert (lo <= 1.0 <= hi) == covers
        assert hi > lo > (0.0 if covers else 1.0)
