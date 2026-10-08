"""Pin the Section 5.3 figures the manuscript prints against the artifact.

The same guard as ``test_manuscript_bootstrap_agreement``, for the same reason:
regenerating an artifact can move a printed interval without the prose
following, so that the text contradicts a table shipped beside it. Section 5.3
is more exposed than most, because its entire point is that these numbers move
when the window moves.

Every value below appears in the manuscript text, in the figures
fig7_october_hour and fig8_rolling, or in the tables table8_rolling.tex and
table9_sensitivity.tex.
"""
import json
import math
import pathlib

import pytest

from regime_engine.paths import DEFAULT_OUT, repo_root


@pytest.fixture(scope="module")
def oos():
    path = pathlib.Path(DEFAULT_OUT) / "oos_robustness.json"
    if not path.exists():                       # pragma: no cover
        pytest.skip("run `python -m regime_engine.oos` first")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def by_variant(oos):
    return {r["variant"]: r for r in oos["sensitivity"]}


@pytest.fixture(scope="module")
def tex():
    return (repo_root() / "paper" / "paper.tex").read_text()


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
        """Section 5.3: the calm half-life barely shifts, tau_stressed triples."""
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
        """Why Section 5.3 reports these as distinct estimates rather than a range."""
        ext, post = oos["extended_window"], oos["injection"][0]
        assert post["ci95"][1] < ext["ci95"][0]


class TestProse:
    """The sentences of Sections 1 and 5.3 that quote the later windows."""

    def test_the_extended_window(self, oos, tex):
        rec = oos["extended_window"]
        lo, hi = rec["ci95"]
        assert f"($T = {rec['T']:,}$)".replace(",", "{,}") in tex
        assert f"at ${rec['ratio']:.2f}$, $[{lo:.2f}, {hi:.2f}]$" in tex

    def test_the_post_april_window(self, oos, tex):
        rec = oos["injection"][0]
        assert f"the ratio is ${rec['ratio']:.3f}$" in tex
        assert f"ending at ${rec['ci95'][1]:.2f}$" in tex
        assert (f"at ${rec['ratio']:.2f}$, its $95\\%$ interval ending at "
                f"${rec['ci95'][1]:.2f}$") in tex
        assert f"one observation in ${rec['T']:,}$".replace(",", "{,}") in tex
        assert (f"${rec['tau_calm']:.1f}$ hours calm against "
                f"${rec['tau_stressed']:.1f}$ stressed") in tex
        # Section 6 quotes the stressed half-life of the same fit.
        assert f"is ${rec['tau_stressed']:.0f}$ hours on the window after April 2026" in tex

    def test_the_injected_ratios(self, oos, tex):
        base = oos["injection"][0]["ratio"]
        a, b, c = (r["ratio"] for r in oos["injection"][1:])
        assert f"raises it to ${a:.1f}$, ${b:.1f}$ and ${c:.1f}$" in tex
        assert f"a factor of ${c / base:.0f}$ to ${a / base:.0f}$" in tex
        lo, hi = oos["injection"][1]["ci95"]
        assert f"$[{lo:.2f}, {hi:.2f}]$" in tex

    def test_the_hours_no_rolling_window_reaches(self, oos, tex):
        p = oos["protocol"]
        covered = (len(oos["rolling"]) - 1) * p["step"] + p["window"]
        tail = oos["premium_series"]["T"] - covered
        assert f"the final ${tail}$ hours of the series fill no further window" in tex

    def test_the_october_hour_against_the_next_largest(self, tex):
        from regime_engine.panel import load_paper_window_premium
        y = sorted(load_paper_window_premium(), reverse=True)
        assert (round(y[0], 2), round(y[1], 2)) == (23.61, 9.52)
        assert round(2 * y[0] / y[1]) / 2 == 2.5
        assert "this one is two and a half times the next largest" in tex

    def test_the_window_counts(self, oos, tex):
        words = {1: "one", 9: "nine", 11: "eleven", 14: "fourteen"}
        rolling = oos["rolling"]
        n, rest = len(rolling), rolling[1:]
        below = sum(r["ratio"] < 1 for r in rest)
        excluding = sum(r["significance"] == "below" for r in rest)
        above = sum(r["significance"] == "above" for r in rolling)
        assert f"{words[n].capitalize()} overlapping windows of" in tex
        assert (f"{words[below]} of the {words[n]} fall below one outright and "
                f"{words[excluding]} of those exclude unity") in tex
        assert (f"only {words[above]} of {words[n]} rolling windows lies significantly "
                f"above unity") in tex
