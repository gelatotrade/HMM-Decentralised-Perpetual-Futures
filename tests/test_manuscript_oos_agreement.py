"""Pin the later-window and sensitivity fits of Section 6.1 against the artifact.

The same guard as ``test_manuscript_bootstrap_agreement``, for the same reason:
regenerating an artifact can move a printed number without the prose following,
so that the text contradicts a table shipped beside it. Section 6.1 is more
exposed than most, because its point is that these numbers move when the
window moves.

The values below are those the robust tables and figures draw on
(table8b_rolling_robust.tex, table9b_sensitivity_robust.tex, fig7b and fig8b)
and those the output-only linear-interval versions keep; the sentences of
Section 6.1 that quote them are checked in test_manuscript_numbers.
"""
import json
import math
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT


@pytest.fixture(scope="module")
def oos():
    path = pathlib.Path(DEFAULT_OUT) / "oos_robustness.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `python -m regime_engine.oos` first")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def by_variant(oos):
    return {r["variant"]: r for r in oos["sensitivity"]}


def sd_ratio(rec):
    """Ratio of the conditional standard deviations, stressed over calm."""
    return math.sqrt(max(rec["sigma2"]) / min(rec["sigma2"]))


class TestRollingWindows:
    def test_there_are_fourteen_windows(self, oos):
        assert len(oos["rolling"]) == 14

    def test_exactly_one_window_puts_the_ratio_above_unity(self, oos):
        above = [r for r in oos["rolling"] if r["significance"] == "above"]
        assert len(above) == 1
        assert str(above[0]["mid"]).startswith("2025-11")

    def test_that_window_is_the_one_the_text_quotes(self, oos):
        rec = oos["rolling"][0]
        assert round(rec["ratio"], 2) == 39.17
        assert [round(v, 1) for v in rec["ci95"]] == [7.6, 70.8]

    def test_the_other_thirteen_span_the_range_the_text_quotes(self, oos):
        rest = [r["ratio"] for r in oos["rolling"][1:]]
        assert round(min(rest), 2) == 0.22
        assert round(max(rest), 2) == 2.94

    def test_eleven_fall_below_one_and_nine_exclude_unity(self, oos):
        rest = oos["rolling"][1:]
        assert sum(r["ratio"] < 1 for r in rest) == 11
        assert sum(r["significance"] == "below" for r in rest) == 9

    def test_the_drawdown_deepens_across_the_sample(self, oos):
        first, last = oos["rolling"][0], oos["rolling"][-1]
        assert round(100 * first["drawdown_mean"]) == -21
        assert round(100 * last["drawdown_mean"]) == -47


class TestSingleObservationSensitivity:
    def test_the_baseline_reproduces_table_seven(self, by_variant):
        rec = by_variant["baseline"]
        assert round(rec["ratio"], 2) == 11.19
        assert [round(v, 2) for v in rec["ci95"]] == [2.95, 19.42]
        assert rec["significance"] == "above"

    def test_replacing_the_maximum_costs_the_significance(self, by_variant):
        rec = by_variant["replace_max"]
        assert round(rec["ratio"], 2) == 3.04
        assert [round(v, 2) for v in rec["ci95"]] == [0.55, 5.53]
        assert rec["significance"] == "ns", "the interval must cover unity"

    def test_deleting_the_maximum_costs_it_too(self, by_variant):
        rec = by_variant["drop_max"]
        assert round(rec["ratio"], 2) == 1.83
        assert rec["T"] == 5000
        assert rec["significance"] == "ns"

    def test_it_is_the_stressed_half_life_that_moves(self, by_variant):
        """Section 6.1 (Figure 5, Tables C4, C5): the calm half-life barely shifts,
        tau_stressed triples."""
        base = by_variant["baseline"]
        assert round(base["tau_stressed"], 2) == 3.18
        assert round(by_variant["replace_max"]["tau_stressed"], 2) == 11.15
        assert round(by_variant["drop_max"]["tau_stressed"], 2) == 18.12
        for key in ("replace_max", "drop_max"):
            assert abs(by_variant[key]["tau_calm"] - base["tau_calm"]) < 3.0

    def test_winsorising_the_upper_tail_leaves_the_ordering_intact(self, by_variant):
        rec = by_variant["winsorise_top1"]
        assert round(rec["ratio"], 2) == 4.49
        assert rec["significance"] == "above", "the tail as a whole still carries some"


class TestPlaceboInjection:
    def test_the_post_april_window_shows_the_reversed_ordering(self, oos):
        rec = oos["injection"][0]
        assert rec["T"] == 3736
        assert round(rec["ratio"], 3) == 0.226
        assert rec["significance"] == "below"
        assert round(rec["ci95"][1], 2) == 0.49
        assert rec["ci95"][0] < 0, "the text leaves the lower end open"

    def test_one_injected_hour_multiplies_the_ratio_by_19_to_79(self, oos):
        base, injected = oos["injection"][0], oos["injection"][1:]
        assert [round(r["ratio"], 1) for r in injected] == [18.0, 11.4, 4.3]
        factors = [r["ratio"] / base["ratio"] for r in injected]
        assert (round(min(factors)), round(max(factors))) == (19, 79)

    def test_only_the_first_injected_fit_is_significant(self, oos):
        """One hour can manufacture significance; the paper says so."""
        injected = oos["injection"][1:]
        assert [r["significance"] for r in injected] == ["above", "ns", "ns"]
        assert [round(v, 2) for v in injected[0]["ci95"]] == [1.97, 33.98]

    def test_the_variance_ratio_collapses_after_the_window(self, oos, by_variant):
        assert round(sd_ratio(by_variant["baseline"]), 1) == 5.1
        assert round(sd_ratio(oos["injection"][0]), 1) == 2.1


class TestExtendedWindow:
    def test_the_extended_window_matches_the_text(self, oos):
        rec = oos["extended_window"]
        assert rec["T"] == 8737
        assert round(rec["ratio"], 2) == 4.55
        assert [round(v, 2) for v in rec["ci95"]] == [1.95, 7.15]

    def test_the_extended_interval_is_disjoint_from_the_post_april_one(self, oos):
        """Why Section 6.1 (Figure 5, Tables C4, C5) reports these as distinct
        estimates rather than a range."""
        ext, post = oos["extended_window"], oos["injection"][0]
        assert post["ci95"][1] < ext["ci95"][0]


class TestUpperTail:
    """What the winsorised variant caps, and where its robust significance comes from."""

    def test_the_cap_sets_fifty_hours_forty_one_of_them_in_october(self, oos):
        t = oos["upper_tail"]
        assert round(t["cap_bps"], 3) == 6.747
        assert (t["n_capped"], t["n_capped_in_month"], t["n_capped_outside_month"]) == (50, 41, 9)
        assert t["maximum_in_month"] and t["month"] == "2025-10"

    def test_capping_october_alone_keeps_the_asymmetry_under_both_covariances(self, oos):
        r = oos["upper_tail"]["october_tail_capped"]
        w = {c: r["inference"][c]["ratio_wald"] for c in ("hessian", "robust")}
        assert r["T"] == 5001 and round(r["ratio"], 2) == 3.93
        assert [w[c]["significance"] for c in w] == ["above", "above"]
        assert (round(w["hessian"]["p"], 4), round(w["robust"]["p"], 3)) == (0.0028, 0.036)

    def test_from_11_october_the_winsorised_series_shows_no_asymmetry(self, oos):
        r = oos["upper_tail"]["winsorised_from_11_oct"]
        w = {c: r["inference"][c]["ratio_wald"] for c in ("hessian", "robust")}
        assert r["T"] == oos["upper_tail"]["T_from"] == 4785
        assert round(r["ratio"], 2) == 1.11
        assert [w[c]["significance"] for c in w] == ["ns", "ns"]
        assert (round(w["hessian"]["p"], 2), round(w["robust"]["p"], 2)) == (0.79, 0.87)
