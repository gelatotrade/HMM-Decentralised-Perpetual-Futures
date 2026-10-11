"""Pin the switching-aware MS-AR inference written to the shipped artifacts.

output/diagnostics.json (paper window, hourly sub-period) and
output/oos_robustness.json (14 rolling windows, the whole series, four
upper-tail variants of the paper window, the later window and three
injections into it) carry, for every MS-AR fit, the fitted chain, the
switching-adjusted half-lives and the half-life ratio's standard error and
linear, log-scale and Fieller intervals under the Hessian, OPG and sandwich
covariances, and the Wald test of phi_calm = phi_stressed that decides the
ratio's side of unity. These tests pin the values a revised manuscript would
quote; an interval form's own ``significance`` is pinned as what that form
says, which is not the verdict where it differs from the Wald test.
"""
import json
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT

FORMS = ("linear", "log", "fieller")


def _rounded(values, digits):
    return [None if v is None else round(v, digits) for v in values]


@pytest.fixture(scope="module")
def diag():
    path = pathlib.Path(DEFAULT_OUT) / "diagnostics.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `make diagnostics` first")
    return json.loads(path.read_text())


def _block(diag, which):
    hits = [b["msar"] for b in diag["tests"] if "msar" in b and which in b["msar"]["window"]]
    assert len(hits) == 1
    return hits[0]


@pytest.fixture(scope="module")
def paper(diag):
    return _block(diag, "paper window")


@pytest.fixture(scope="module")
def sub(diag):
    return _block(diag, "sub-sample")


class TestPaperWindowChain:
    def test_staying_probabilities(self, paper):
        assert _rounded(paper["p_stay"], 4) == [0.9742, 0.6163]

    def test_expected_durations(self, paper):
        assert _rounded(paper["expected_duration_hours"], 2) == [38.72, 2.61]

    def test_the_stressed_regime_holds_six_per_cent_of_hours(self, paper):
        assert round(paper["stationary_share"][1], 3) == 0.063

    def test_the_stressed_regime_ends_before_its_within_regime_half_life(self, paper):
        assert paper["expected_duration_hours"][1] < paper["half_life_hours"][1]


class TestPaperWindowSwitchingHalfLives:
    def test_25_hours_from_calm_and_17_from_stressed(self, paper):
        assert paper["half_life_switching_hours"] == [25, 17]
        assert round(paper["ratio_switching"], 2) == 1.47

    def test_interpolated_crossings(self, paper):
        assert _rounded(paper["half_life_switching_interp_hours"], 2) == [24.51, 16.29]
        assert round(paper["ratio_switching_interp"], 2) == 1.51

    def test_the_switching_ratio_covers_unity_under_the_sandwich(self, paper):
        hess = paper["inference"]["hessian"]["ratio_switching_interp_ci95"]["log"]
        rob = paper["inference"]["robust"]["ratio_switching_interp_ci95"]["log"]
        assert _rounded(hess["bounds"], 2) == [1.15, 1.97] and hess["significance"] == "above"
        assert _rounded(rob["bounds"], 2) == [0.74, 3.06] and rob["significance"] == "ns"


class TestPaperWindowCovariances:
    def test_standard_errors_of_phi(self, paper):
        inf = paper["inference"]
        assert _rounded(inf["hessian"]["phi_se"], 5) == [0.00366, 0.04619]
        assert _rounded(inf["opg"]["phi_se"], 5) == [0.00262, 0.01594]
        assert _rounded(inf["robust"]["phi_se"], 5) == [0.00694, 0.17274]

    def test_the_hessian_block_is_the_one_table_3_prints(self, paper):
        assert paper["inference"]["hessian"]["phi_se"] == paper["phi_se"]
        assert paper["inference"]["hessian"]["ratio_se"] == paper["half_life_ratio_se"]
        assert paper["inference"]["hessian"]["ratio_ci95"]["linear"]["bounds"] == \
            paper["half_life_ratio_ci95"]

    def test_standard_errors_of_r(self, paper):
        se = {k: round(v["ratio_se"], 2) for k, v in paper["inference"].items()}
        assert se == {"hessian": 4.20, "opg": 1.84, "robust": 12.68}

    def test_opg_hessian_sandwich_ordering(self, paper):
        """OPG < Hessian < sandwich, consistent with a failure of the
        information-matrix equality (which implies no particular ordering)."""
        se = {k: v["ratio_se"] for k, v in paper["inference"].items()}
        assert se["opg"] < se["hessian"] < se["robust"]

    def test_t_ratio_of_phi_calm_against_unity(self, paper):
        assert round(paper["inference"]["hessian"]["phi_t_unit"][0], 2) == -5.28
        assert round(paper["inference"]["robust"]["phi_t_unit"][0], 2) == -2.78

    def test_half_life_intervals_from_phi(self, paper):
        hess = paper["inference"]["hessian"]["half_life_ci95_from_phi"]
        rob = paper["inference"]["robust"]["half_life_ci95_from_phi"]
        assert _rounded(hess[0], 1) == [25.8, 56.8]
        assert _rounded(rob[0], 1) == [20.7, 121.0]
        assert round(rob[1][0], 2) == 0.91 and rob[1][1] is None   # phi_str reaches 1


class TestPaperWindowRatioIntervals:
    EXPECTED = {
        "hessian": {"linear": ([2.95, 19.42], "above"), "log": ([5.36, 23.36], "above"),
                    "fieller": ([4.61, 23.19], "above")},
        "robust": {"linear": ([-13.66, 36.03], "ns"), "log": ([1.21, 103.07], "above"),
                   "fieller": ([-9.75, 70.82], "ns")},
    }

    @pytest.mark.parametrize("cov", ["hessian", "robust"])
    @pytest.mark.parametrize("form", FORMS)
    def test_bounds_and_side(self, paper, cov, form):
        got = paper["inference"][cov]["ratio_ci95"][form]
        bounds, side = self.EXPECTED[cov][form]
        assert got["shape"] == "bounded"
        assert _rounded(got["bounds"], 2) == bounds
        assert got["significance"] == side


class TestSandwichInfluence:
    def test_one_hour_carries_three_quarters_of_the_sandwich_variance(self, paper):
        top = paper["ratio_sandwich_top_hours"]
        assert top[0]["index"] == 215              # 10 Oct 2025 23:00 UTC
        assert (round(top[0]["y_prev"], 2), round(top[0]["y"], 2)) == (23.61, -3.13)
        assert round(top[0]["share"], 3) == 0.767
        assert round(top[1]["share"], 3) == 0.012

    def test_the_variance_is_the_sandwich_variance(self, paper):
        se = paper["inference"]["robust"]["ratio_se"]
        assert paper["ratio_sandwich_variance"] == pytest.approx(se ** 2, rel=1e-6)

    def test_no_hour_dominates_the_sub_period(self, sub):
        assert sub["ratio_sandwich_top_hours"][0]["share"] < 0.10


class TestSubPeriod:
    def test_switching_half_lives_shrink_the_reversal(self, sub):
        assert sub["half_life_switching_hours"] == [9, 13]
        assert round(sub["ratio_switching"], 3) == 0.692
        assert round(sub["half_life_ratio"], 2) == 0.29

    def test_the_chain(self, sub):
        assert _rounded(sub["p_stay"], 3) == [0.941, 0.866]
        assert _rounded(sub["expected_duration_hours"], 1) == [16.8, 7.5]

    def test_the_reversal_is_significant_only_under_the_hessian(self, sub):
        hess = sub["inference"]["hessian"]["ratio_ci95"]
        rob = sub["inference"]["robust"]["ratio_ci95"]
        assert [hess[f]["significance"] for f in FORMS] == ["below"] * 3
        assert [rob[f]["significance"] for f in FORMS] == ["below", "ns", "ns"]
        assert _rounded(rob["log"]["bounds"], 2) == [0.07, 1.09]


# ---------------------------------------------------------------------------
# oos_robustness.json: the 23 robustness fits
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def oos():
    path = pathlib.Path(DEFAULT_OUT) / "oos_robustness.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `python -m regime_engine.oos` first")
    return json.loads(path.read_text())


def _records(oos):
    return (oos["rolling"] + [oos["extended_window"]] + oos["sensitivity"]
            + oos["injection"])


class TestEveryRobustnessRecord:
    def test_there_are_23(self, oos):
        assert len(_records(oos)) == 23

    def test_the_artifact_is_strict_json(self):
        """No NaN or Infinity tokens: a non-finite optimum is written as null."""
        text = (pathlib.Path(DEFAULT_OUT) / "oos_robustness.json").read_text()

        def refuse(token):
            raise AssertionError(f"non-standard JSON constant {token}")

        json.loads(text, parse_constant=refuse)

    def test_the_multistart_protocol_is_recorded(self, oos):
        for rec in _records(oos):
            assert "n_distinct_optima" not in rec
            assert rec["seed_base"] == oos["protocol"]["seed_base"] == 20260802
            assert 20260802 <= rec["best_seed"] < 20260802 + 40
            assert rec["failures"] + sum(o["count"] for o in rec["optima_found"]) == 40
            finite = [o["llf"] for o in rec["optima_found"] if o["llf"] is not None]
            assert max(finite) == pytest.approx(rec["llf"], abs=0.006)

    def test_non_finite_optima_are_kept_as_null(self, oos):
        nulls = [(i, o["count"]) for i, rec in enumerate(_records(oos))
                 for o in rec["optima_found"] if o["llf"] is None]
        assert len(nulls) == 3 and all(count == 1 for _, count in nulls)

    def test_the_hessian_linear_interval_is_the_printed_one(self, oos):
        for rec in _records(oos):
            lin = rec["inference"]["hessian"]["ratio_ci95"]["linear"]
            assert lin["bounds"] == pytest.approx(rec["ci95"], rel=1e-12)
            assert lin["significance"] == rec["significance"]
            assert rec["inference"]["hessian"]["ratio_se"] == pytest.approx(rec["ratio_se"],
                                                                            rel=1e-12)

    def test_switching_ratio_is_the_ratio_of_the_hours(self, oos):
        for rec in _records(oos):
            hc, hs = rec["half_life_switching_hours"]
            assert rec["ratio_switching"] == pytest.approx(hc / hs)

    def test_every_interval_form_has_a_verdict(self, oos):
        for rec in _records(oos):
            for cov in ("hessian", "opg", "robust"):
                for form in FORMS:
                    got = rec["inference"][cov]["ratio_ci95"][form]
                    assert got["significance"] in {"above", "below", "ns", "undetermined"}


def _sides(rec, cov):
    return [rec["inference"][cov]["ratio_ci95"][f]["significance"] for f in FORMS]


def _bounds(rec, cov, form, digits=2):
    return _rounded(rec["inference"][cov]["ratio_ci95"][form]["bounds"], digits)


@pytest.fixture(scope="module")
def variant(oos):
    return {r["variant"]: r for r in oos["sensitivity"]}


class TestSwitchingAcrossTheRefits:
    def test_the_switching_ratio_runs_from_036_to_175(self, oos):
        ratios = [r["ratio_switching"] for r in _records(oos)]
        assert (round(min(ratios), 3), round(max(ratios), 2)) == (0.357, 1.75)
        assert oos["rolling"][11]["half_life_switching_hours"] == [5, 14]
        assert oos["rolling"][0]["half_life_switching_hours"] == [21, 12]

    def test_later_window_and_whole_series(self, oos):
        later, whole = oos["injection"][0], oos["extended_window"]
        assert later["half_life_switching_hours"] == [12, 22]
        assert round(later["ratio_switching"], 3) == 0.545
        assert whole["half_life_switching_hours"] == [24, 19]

    def test_the_stressed_regime_is_short_lived_where_the_spike_is(self, oos):
        assert round(oos["rolling"][0]["expected_duration_hours"][1], 2) == 2.07
        assert _rounded([r["expected_duration_hours"][1] for r in oos["injection"][1:]], 2) \
            == [1.69, 1.71, 1.98]


class TestPaperWindowVariants:
    def test_replacing_the_maximum_depends_on_the_interval_form(self, variant):
        rec = variant["replace_max"]
        assert _sides(rec, "hessian") == ["ns", "above", "above"]
        assert _bounds(rec, "hessian", "log") == [1.34, 6.89]
        assert _bounds(rec, "hessian", "fieller") == [1.09, 6.78]
        assert _sides(rec, "robust") == ["ns", "ns", "ns"]
        assert _bounds(rec, "robust", "log") == [0.60, 15.50]

    def test_deleting_the_maximum_is_not_significant_under_any_form(self, variant):
        rec = variant["drop_max"]
        assert _sides(rec, "hessian") == ["ns"] * 3
        assert _sides(rec, "robust") == ["ns"] * 3

    def test_winsorising(self, variant):
        rec = variant["winsorise_top1"]
        assert _sides(rec, "hessian") == ["above"] * 3
        assert _sides(rec, "robust") == ["ns", "above", "above"]
        assert _bounds(rec, "robust", "log") == [1.75, 11.53]

    def test_the_baseline_is_the_diagnostics_fit(self, variant, paper):
        rec = variant["baseline"]
        assert rec["llf"] == pytest.approx(paper["logL"], abs=1e-6)
        assert rec["inference"]["robust"]["ratio_ci95"] == \
            paper["inference"]["robust"]["ratio_ci95"]

    def test_drop_max_reaches_three_optima(self, variant):
        assert [o["count"] for o in variant["drop_max"]["optima_found"]] == [36, 3, 1]


class TestRollingVerdicts:
    def test_windows_above_unity_by_covariance_and_form(self, oos):
        counts = {cov: [sum(_sides(r, cov)[i] == "above" for r in oos["rolling"])
                        for i in range(3)] for cov in ("hessian", "robust")}
        assert counts == {"hessian": [1, 2, 2], "robust": [0, 1, 0]}

    def test_windows_below_unity_by_covariance_and_form(self, oos):
        counts = {cov: [sum(_sides(r, cov)[i] == "below" for r in oos["rolling"])
                        for i in range(3)] for cov in ("hessian", "robust")}
        assert counts == {"hessian": [9, 8, 9], "robust": [8, 5, 6]}

    def test_the_second_window_excludes_unity_under_the_hessian_log_form(self, oos):
        rec = oos["rolling"][1]
        assert str(rec["mid"]).startswith("2025-12-11")
        assert rec["start"] > "2025-10-11"          # it does not contain 10 October
        assert _bounds(rec, "hessian", "log") == [1.18, 7.34]
        assert _sides(rec, "hessian") == ["ns", "above", "above"]
        assert _sides(rec, "robust") == ["ns", "ns", "ns"]

    def test_the_october_window_under_the_sandwich(self, oos):
        rec = oos["rolling"][0]
        assert _bounds(rec, "robust", "log") == [7.15, 214.58]
        assert rec["inference"]["robust"]["ratio_ci95"]["fieller"]["shape"] == "whole_line"


class TestOtherWindows:
    def test_the_whole_series_loses_significance_under_the_sandwich(self, oos):
        rec = oos["extended_window"]
        assert _sides(rec, "hessian") == ["above"] * 3
        assert _sides(rec, "robust") == ["ns"] * 3
        assert _bounds(rec, "robust", "log") == [0.65, 31.68]

    def test_the_later_window_stays_below_unity_under_every_form(self, oos):
        rec = oos["injection"][0]
        assert _sides(rec, "hessian") == ["below"] * 3
        assert _sides(rec, "robust") == ["below"] * 3
        assert round(rec["inference"]["robust"]["ratio_ci95"]["log"]["bounds"][1], 3) == 0.999

    def test_injected_hours(self, oos):
        inj = oos["injection"][1:]
        assert [_sides(r, "hessian") for r in inj] == [["above"] * 3, ["ns", "above", "above"],
                                                      ["ns", "above", "ns"]]
        assert [_sides(r, "robust")[1] for r in inj] == ["above", "above", "ns"]


# ---------------------------------------------------------------------------
# The verdict: the Wald test of phi_calm = phi_stressed
# ---------------------------------------------------------------------------
def _wald(rec, cov):
    w = rec["inference"][cov]["ratio_wald"]
    return round(w["z"], 2), w["significance"]


def _fits(oos, paper, sub):
    """The 24 distinct fits. The sensitivity baseline is the paper-window fit and is
    not listed again, so there are 48 fit-covariance cases under Hessian and sandwich."""
    named = [("paper", paper), ("sub", sub), ("whole", oos["extended_window"]),
             ("later", oos["injection"][0])]
    named += [(f"rolling {r['index']}", r) for r in oos["rolling"]]
    named += [(r["variant"], r) for r in oos["sensitivity"][1:]]
    named += [(r["variant"], r) for r in oos["injection"][1:]]
    return named


def test_the_distinct_fits_number_24(oos, paper, sub):
    fits = _fits(oos, paper, sub)
    assert len(fits) == 24 and len({name for name, _ in fits}) == 24
    assert oos["sensitivity"][0]["variant"] == "baseline"


class TestWaldVerdicts:
    def test_the_paper_window(self, paper):
        assert _wald(paper, "hessian") == (3.71, "above")
        assert _wald(paper, "opg") == (10.92, "above")
        assert _wald(paper, "robust") == (1.01, "ns")
        assert round(paper["inference"]["robust"]["ratio_wald"]["p"], 2) == 0.31
        assert paper["inference"]["hessian"]["ratio_wald"]["p"] < 0.001

    def test_under_the_sandwich_only_the_log_scale_set_excludes_unity(self, paper):
        forms = paper["inference"]["robust"]["ratio_ci95"]
        assert forms["log"]["significance"] == "above"
        assert forms["linear"]["significance"] == forms["fieller"]["significance"] == "ns"

    def test_the_sub_period_reversal_is_not_significant_under_the_sandwich(self, sub):
        assert _wald(sub, "hessian") == (-3.74, "below")
        assert _wald(sub, "robust") == (-1.68, "ns")
        assert round(sub["inference"]["robust"]["ratio_wald"]["p"], 3) == 0.093
        # the linear delta interval excludes unity there; the test does not
        assert sub["inference"]["robust"]["ratio_ci95"]["linear"]["significance"] == "below"

    def test_fieller_ends_to_the_printed_precision(self, sub, variant):
        assert _rounded(sub["inference"]["robust"]["ratio_ci95"]["fieller"]["bounds"], 3) \
            == [0.008, 1.505]
        assert round(sub["inference"]["robust"]["ratio_ci95"]["fieller"]["bounds"][1], 2) == 1.5
        rob = variant["replace_max"]["inference"]["robust"]["ratio_ci95"]["fieller"]
        assert round(rob["bounds"][0], 2) == -0.31

    def test_the_variants_of_the_paper_window(self, variant):
        assert [_wald(variant[k], "hessian") for k in ("replace_max", "drop_max",
                                                        "winsorise_top1")] == \
            [(2.10, "above"), (1.48, "ns"), (3.20, "above")]
        assert [_wald(variant[k], "robust") for k in ("replace_max", "drop_max",
                                                       "winsorise_top1")] == \
            [(1.04, "ns"), (1.07, "ns"), (2.14, "above")]

    def test_the_whole_series_and_the_later_window(self, oos):
        whole, later = oos["extended_window"], oos["injection"][0]
        assert (_wald(whole, "hessian"), _wald(whole, "robust")) == ((3.22, "above"),
                                                                     (0.86, "ns"))
        assert (_wald(later, "hessian"), _wald(later, "robust")) == ((-4.31, "below"),
                                                                     (-3.06, "below"))

    def test_the_injected_hours(self, oos):
        inj = oos["injection"][1:]
        assert [_wald(r, "hessian") for r in inj] == [(2.90, "above"), (2.37, "above"),
                                                      (1.48, "ns")]
        assert [_wald(r, "robust") for r in inj] == [(1.30, "ns"), (1.01, "ns"), (0.54, "ns")]

    def test_the_rolling_windows(self, oos):
        def where(cov, side):
            return [r["index"] for r in oos["rolling"]
                    if r["inference"][cov]["ratio_wald"]["significance"] == side]
        assert where("hessian", "above") == [0, 1]
        assert where("hessian", "below") == [4, 5, 8, 9, 10, 11, 12, 13]
        assert where("robust", "above") == []
        assert where("robust", "below") == [5, 8, 10, 11, 12, 13]
        assert round(oos["rolling"][6]["inference"]["hessian"]["ratio_wald"]["p"], 4) == 0.0506

    def test_where_each_interval_form_disagrees_with_the_test(self, oos, paper, sub):
        """Over 24 fits and two covariances (48 cases): the log-scale set errs in 6
        of 48, five of them a false 'above'; the linear set in 8; Fieller in 1."""
        wrong = {"linear": [], "log": [], "fieller": []}
        for name, rec in _fits(oos, paper, sub):
            for cov in ("hessian", "robust"):
                verdict = rec["inference"][cov]["ratio_wald"]["significance"]
                for form in wrong:
                    if rec["inference"][cov]["ratio_ci95"][form]["significance"] != verdict:
                        wrong[form].append(f"{name}/{cov}")
        assert wrong["log"] == ["paper/robust", "rolling 0/robust", "rolling 12/robust",
                                "inject@934/robust", "inject@1868/robust",
                                "inject@2802/hessian"]
        assert len(wrong["linear"]) == 8
        assert wrong["fieller"] == ["rolling 6/hessian"]


# ---------------------------------------------------------------------------
# The switching-adjusted ratio
# ---------------------------------------------------------------------------
class TestSwitchingVerdicts:
    def test_the_chain_is_significantly_persistent_in_every_fit(self, oos, paper, sub):
        lam = {name: rec["chain_eigenvalue"] for name, rec in _fits(oos, paper, sub)}
        z_s = {name: rec["chain_eigenvalue"] / rec["inference"]["robust"]["chain_eigenvalue_se"]
               for name, rec in _fits(oos, paper, sub)}
        assert round(min(lam.values()), 3) == 0.400 and min(lam, key=lam.get) == "inject@934"
        assert round(min(z_s.values()), 2) == 2.83 and min(z_s, key=z_s.get) == "inject@2802"
        assert round(paper["chain_eigenvalue"], 3) == 0.590

    def test_the_switching_ratio_takes_the_wald_verdict_everywhere(self, oos, paper, sub):
        for _, rec in _fits(oos, paper, sub):
            for cov in ("hessian", "opg", "robust"):
                inf = rec["inference"][cov]
                assert inf["ratio_switching_side"] == inf["ratio_wald"]["significance"]

    def test_interpolated_and_whole_hour_ratios(self, oos, paper, sub):
        """The whole-hour ratio moves in steps; the interpolated one is the one
        the intervals are built on."""
        fits = [rec for _, rec in _fits(oos, paper, sub)]
        whole = [r["ratio_switching"] for r in fits]
        interp = [r["ratio_switching_interp"] for r in fits]
        assert (round(min(whole), 3), round(max(whole), 2)) == (0.357, 1.75)
        assert (round(min(interp), 3), round(max(interp), 3)) == (0.318, 1.898)
        assert (round(paper["ratio_switching_interp"], 3), round(paper["ratio_switching"], 3)) \
            == (1.505, 1.471)


class TestSimulatedSwitchingIntervals:
    """50,000 draws per fit and covariance, the transition probabilities on the
    logit scale; an end at zero or infinity comes from draws whose expected
    path does not halve within 10,000 hours."""

    def test_the_paper_window(self, paper):
        hess = paper["inference"]["hessian"]["ratio_switching_sim"]
        rob = paper["inference"]["robust"]["ratio_switching_sim"]
        assert (hess["draws"], hess["seed"], hess["transform"]) == (50_000, 20261009, "logit")
        assert _rounded(hess["ci95"], 2) == [1.20, 2.11]
        assert _rounded(rob["ci95"], 2) == [0.00, 3.14]
        assert round(rob["infinite"], 3) == 0.042 and round(rob["undefined"], 4) == 0.0043
        assert _rounded(rob["ci95_hours"], 2) == [0.00, 3.00]

    def test_the_sub_period_covers_unity_under_the_sandwich(self, sub):
        assert _rounded(sub["inference"]["robust"]["ratio_switching_sim"]["ci95"], 2) \
            == [0.24, 1.01]
        assert _rounded(sub["inference"]["hessian"]["ratio_switching_sim"]["ci95"], 2) \
            == [0.49, 0.88]

    def test_unbounded_ends(self, oos):
        """The October window's stressed and the fifth window's calm coefficient
        reach above one under the sandwich."""
        assert _rounded(oos["rolling"][0]["inference"]["robust"]["ratio_switching_sim"]["ci95"],
                        2) == [0.00, 5.60]
        lo, hi = oos["rolling"][4]["inference"]["robust"]["ratio_switching_sim"]["ci95"]
        assert round(lo, 3) == 0.025 and hi is None

    def test_the_interval_falls_where_the_wald_test_says(self, oos, paper, sub):
        """Up to one knife-edge case: the Hessian test on the 15 Mar 2026 window
        has p = 0.0506 and the simulated interval ends at 0.998."""
        off = []
        for name, rec in _fits(oos, paper, sub):
            for cov in ("hessian", "opg", "robust"):
                inf = rec["inference"][cov]
                lo, hi = inf["ratio_switching_sim"]["ci95"]
                side = "above" if lo > 1 else "below" if hi is not None and hi < 1 else "ns"
                if side != inf["ratio_wald"]["significance"]:
                    off.append((name, cov, round(hi, 3)))
        assert off == [("rolling 6", "hessian", 0.998)]


# ---------------------------------------------------------------------------
# The guarded multi-start scan
# ---------------------------------------------------------------------------
class TestGuardedScan:
    def test_excluded_starts(self, oos, paper, sub):
        recs = _records(oos)
        totals = {k: sum(r["excluded"][k] for r in recs) for k in recs[0]["excluded"]}
        assert totals == {"non_finite_llf": 3, "variance_floor": 29, "singular_hessian": 6}
        for block in (paper, sub):
            assert block["multistart"]["excluded"] == {"non_finite_llf": 0,
                                                       "variance_floor": 0,
                                                       "singular_hessian": 0}

    def test_the_non_finite_starts_are_the_null_optima(self, oos):
        for rec in _records(oos):
            nulls = sum(o["count"] for o in rec["optima_found"] if o["llf"] is None)
            assert rec["excluded"]["non_finite_llf"] == nulls

    def test_the_15_july_fit_is_reached_by_two_starts(self, oos):
        """18 starts fail, 17 collapse a regime variance, and 2 reach the optimum."""
        rec = oos["injection"][2]
        assert rec["variant"] == "inject@1868"
        assert rec["failures"] == 18
        assert rec["excluded"] == {"non_finite_llf": 0, "variance_floor": 17,
                                   "singular_hessian": 1}
        best = max(o["llf"] for o in rec["optima_found"] if o["llf"] is not None)
        assert [o["count"] for o in rec["optima_found"] if o["llf"] == best] == [2]
        assert round(rec["llf"], 2) == best == -3329.18

    def test_the_oos_artifact_says_where_the_verdict_is(self, oos):
        assert "inference.<cov>.ratio_wald" in oos["_verdicts"]
        assert "descriptive" in oos["_verdicts"]
