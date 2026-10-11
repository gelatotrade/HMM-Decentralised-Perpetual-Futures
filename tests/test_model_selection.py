"""Model selection beyond K=3.

On the reproducible sub-sample every K is fitted from a wide multi-start (the
40 default starts, 160 with spread volatilities and splits of the best K-1
fit); K=4 attains a *better* BIC than K=3, so the paper reports the comparison
for K>=4 and keeps K=3 on stated grounds. These tests guard the helper and the
artifact.
"""
import numpy as np
import pytest

from regime_engine.robustness import model_selection, preferred_k


class TestPreferredK:
    def test_lowest_bic_wins(self):
        assert preferred_k([{"K": 2, "BIC": -10.0}, {"K": 3, "BIC": -12.0}]) == 3

    def test_ties_resolve_to_the_more_parsimonious_model(self):
        assert preferred_k([{"K": 2, "BIC": -12.0}, {"K": 3, "BIC": -12.0}]) == 2

    def test_a_non_finite_bic_is_ignored(self):
        got = preferred_k([{"K": 2, "BIC": -10.0}, {"K": 3, "BIC": float("nan")}])
        assert got == 2

    def test_no_usable_record_returns_none(self):
        assert preferred_k([{"K": 3, "BIC": float("nan")}]) is None


class TestModelSelection:
    @pytest.fixture(scope="class")
    def r(self):
        rng = np.random.default_rng(0)
        return np.where(rng.random(600) < 0.5, rng.normal(0, 1e-4, 600),
                        rng.normal(0, 5e-4, 600))

    def test_one_record_per_requested_k_in_order(self, r):
        got = model_selection(r, ks=(2, 3), n_starts=2, n_iter=20)
        assert [rec["K"] for rec in got] == [2, 3]

    def test_free_parameter_count_follows_the_paper_s_formula(self, r):
        """|theta_K| = K^2 + 2K - 1, the parameter count of Table C7."""
        got = model_selection(r, ks=(2, 3, 4), n_starts=2, n_iter=20)
        assert [rec["n_params"] for rec in got] == [7, 14, 23]

    def test_delta_bic_is_measured_against_the_reference(self, r):
        got = model_selection(r, ks=(2, 3), n_starts=2, n_iter=20, reference=2)
        by_k = {rec["K"]: rec for rec in got}
        assert by_k[2]["delta_BIC"] == 0.0
        assert by_k[3]["delta_BIC"] == pytest.approx(by_k[3]["BIC"] - by_k[2]["BIC"])


class TestManuscriptAgreement:
    """The BIC figures Section 6.2 prints, pinned against the artifact.

    The paper reports the fits for K>=4 alongside K=3, so these four numbers
    appear in the text and have to track the file that produced them.
    """

    @pytest.fixture(scope="class")
    def recs(self):
        import json
        import pathlib

        from regime_engine.paths import DEFAULT_OUT

        path = pathlib.Path(DEFAULT_OUT) / "model_selection_extended.json"
        if not path.exists():                   # pragma: no cover
            pytest.skip("run scripts/model_selection_extended.py first")
        return {r["K"]: r for r in json.loads(path.read_text())["records"]}

    @pytest.mark.parametrize("K,bic", [
        (2, -28356.8), (3, -28533.4), (4, -28557.0), (5, -28500.6),
    ])
    def test_bic_matches_the_printed_figure(self, recs, K, bic):
        assert round(recs[K]["BIC"], 1) == bic

    def test_k4_wins_by_the_margin_the_text_quotes(self, recs):
        assert round(recs[4]["BIC"] - recs[3]["BIC"], 1) == -23.6

    def test_k3_beats_k2_by_seven_and_a_half_times_that(self, recs):
        gain_k3 = recs[3]["BIC"] - recs[2]["BIC"]
        gain_k4 = recs[4]["BIC"] - recs[3]["BIC"]
        assert round(gain_k3, 1) == -176.6
        assert round(gain_k3 / gain_k4, 1) == 7.5

    def test_every_k_converges_under_the_wide_protocol(self, recs):
        for K in (2, 3, 4, 5):
            c = recs[K]["convergence"]
            assert recs[K]["converged"] and c["all_starts_converged"], K
            assert c["polish"]["converged"], K
            assert c["start_families"]["default"] == 40, K
            assert c["start_families"]["spread"] == 160, K
            assert c["start_families"]["split"] == 3 * (K - 1), K

    def test_how_many_starts_reach_the_best_optimum(self, recs):
        reach = {K: (recs[K]["n_starts_reaching_best"], recs[K]["n_starts"]) for K in recs}
        assert reach == {2: (203, 203), 3: (206, 206), 4: (143, 209), 5: (19, 212)}
        # the 40 default starts find the best optimum up to K = 4, not for K = 5
        by_family = recs[5]["convergence"]["n_reaching_best_by_family"]
        assert by_family == {"default": 0, "spread": 19, "split": 0}
        for K in (2, 3, 4):
            assert recs[K]["default_protocol"]["is_best_optimum"], K
        d5 = recs[5]["default_protocol"]
        assert not d5["is_best_optimum"]
        assert round(d5["logL"], 2) == 14380.15 and round(d5["BIC"], 1) == -28482.8

    def test_k5_gains_the_log_likelihood_the_text_quotes(self, recs):
        assert round(recs[5]["logL"], 4) == 14389.0743
        assert round(recs[5]["logL"] - recs[4]["logL"], 1) == 16.7

    def test_k5_falls_behind_k3_so_there_is_no_plateau(self, recs):
        assert round(recs[5]["BIC"] - recs[3]["BIC"], 1) == 32.8

    def test_the_fifth_state_captures_one_episode(self, recs):
        """K=5 repeats the K=4 ladder and adds a persistent state at the third rung."""
        k4 = [round(v, 1) for v in recs[4]["sigma_bps_per_hour"]]
        s5 = recs[5]["states"]
        sig = [round(st["sigma_bps_per_hour"], 1) for st in s5]
        assert sig == [11.1, 24.5, 56.8, 59.5, 143.5]
        assert k4 == [11.1, 24.8, 55.8, 134.9]
        extra = s5[3]
        assert round(extra["expected_duration_h"]) == 34
        assert round(s5[2]["expected_duration_h"], 1) == 4.6
        assert extra["first_hour_held"].startswith("2026-01-31")
        assert extra["last_hour_held"].startswith("2026-03-11")
        # every other state holds hours across the whole sub-period
        for st in s5[:3] + s5[4:]:
            assert st["first_hour_held"] < "2025-12-15" and st["last_hour_held"] > "2026-04-01"

    def test_the_k4_volatility_ladder_is_the_one_the_text_describes(self, recs):
        got = [round(v, 1) for v in recs[4]["sigma_bps_per_hour"]]
        assert got == [11.1, 24.8, 55.8, 134.9]
        # The text states the ladder to one decimal; check at that precision.
        ratios = [round(b / a, 1) for a, b in zip(got, got[1:])]
        assert all(2.2 <= x <= 2.4 for x in ratios), ratios

    def test_k2_and_k3_reproduce_the_archived_sub_sample_fit(self, recs):
        """The text claims agreement to the second decimal."""
        import json
        import pathlib

        from regime_engine.paths import DEFAULT_OUT

        prev = json.loads((pathlib.Path(DEFAULT_OUT) / "summary.json").read_text())
        for K in (2, 3):
            assert round(recs[K]["BIC"], 2) == round(prev["model_selection"][str(K)]["BIC"], 2)
