"""The explanatory figures: what each one is drawn from.

The drawing code only plots; everything a figure asserts about the data is
assembled by a pure function, and those functions are pinned here against the
shipped data and artifacts. A figure can then be redrawn freely without its
content drifting from the numbers the paper prints beside it.
"""
import numpy as np
import pytest

from regime_engine import censoring as c
from regime_engine import paper_figures as pf


class TestFundingRule:
    def test_rule_pays_the_floor_inside_the_band(self):
        assert c.funding_rule([-3.99, 0.0, 5.99]).tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_outside_the_band_the_premium_is_shifted_by_the_clamp(self):
        assert c.funding_rule([-4.75, -10.0, 10.0]).tolist() == pytest.approx([0.25, -5.0, 5.0])

    def test_band_edges_are_where_the_rule_leaves_the_floor(self):
        lo, hi = c.BAND_BPS
        assert c.funding_rule([lo, hi]).tolist() == pytest.approx([1.0, 1.0])
        assert c.funding_rule([lo - 0.01, hi + 0.01]).tolist() == pytest.approx([0.99, 1.01])

    def test_below_the_band_the_rule_inverts_to_the_recovered_premium(self):
        # recovered_premium_bps reads a settlement below the floor as F - 5 bps
        p = -4.75
        rate = c.funding_rule([p])[0] / 1e4
        assert c.recovered_premium_bps([f"{rate:.8f}"])[0] == pytest.approx(p)


class TestCensoringFigure:
    def test_the_rule_sits_above_the_premiums_on_a_shared_axis(self):
        fig, axes = c.figure_layout()
        assert sorted(axes) == ["bars", "density", "rule"]
        assert axes["rule"].get_shared_x_axes().joined(axes["rule"], axes["density"])

    def test_the_figure_builds_from_the_shipped_data(self, tmp_path):
        pdf = c.figure(c.build_payload(), out_dir=tmp_path)
        assert pdf.exists() and pdf.stat().st_size > 10_000


class TestWindows:
    @pytest.fixture(scope="class")
    def spans(self):
        return {s["label"]: s for s in pf.window_spans()}

    def test_estimation_window_follows_the_constant(self, spans):
        w = spans["estimation window"]
        assert (str(w["start"]), str(w["end"])) == ("2025-10-02 00:00:00+00:00",
                                                     "2026-04-28 08:00:00+00:00")
        assert w["hours"] == 5001

    def test_later_window_follows_the_censoring_module(self, spans):
        w = spans["later window"]
        assert str(w["end"]) == "2026-10-01 00:00:00+00:00"
        assert w["hours"] == 3736

    def test_hourly_sub_period_lies_inside_the_estimation_window(self, spans):
        sub, est = spans["hourly sub-period"], spans["estimation window"]
        assert est["start"] < sub["start"] < sub["end"] <= est["end"]
        assert sub["hours"] == 3507

    def test_the_common_rule_span_is_the_long_history_s(self, spans):
        w = spans["common rule"]
        assert (str(w["start"]), str(w["end"])) == ("2023-12-11 23:00:00+00:00",
                                                     "2026-10-01 00:00:00+00:00")
        assert w["hours"] == 24577

    def test_availability_rows_cover_the_series_actually_shipped(self):
        rows = {r["label"]: r for r in pf.availability_rows()}
        assert str(rows["two-hourly prices"]["end"]).startswith("2026-09-30 22:00")
        assert str(rows["hourly prices"]["start"]).startswith("2025-12-03 05:00")
        assert str(rows["hourly prices"]["end"]).startswith("2026-08-27 08:00")
        hl = rows["hyperliquid funding and premium"]
        assert str(hl["start"]).startswith("2023-05-12 00:00")
        assert str(hl["end"]).startswith("2026-10-01 00:00")
        assert str(hl["common_rule_from"]).startswith("2023-12-11 23:00")
        assert str(rows["exchange settlements"]["start"]) < "2023-05-12"
        assert str(rows["exchange settlements"]["end"]).startswith("2026-10-01 00:00")
        prem = rows["exchange premium index"]
        assert str(prem["start"]).startswith("2023-12-11 17:00")
        assert str(prem["end"]).startswith("2026-10-01 00:00")

    def test_the_price_line_is_the_binance_perpetual_over_the_long_history(self):
        px = pf.price_history()
        assert str(px["time"].iloc[0]).startswith("2023-12-11 17:00")
        assert str(px["time"].iloc[-1]).startswith("2026-10-01 00:00")
        assert 15_000 < px["close"].min() < px["close"].max() < 200_000

    def test_the_windows_figure_prints_its_text_at_seven_points(self, tmp_path, monkeypatch):
        from test_figure_print_sizes import printed_min_pt
        monkeypatch.setattr(pf, "FIG_OUT", tmp_path)
        monkeypatch.setattr(pf, "FIG_PAPER", tmp_path / "paper")
        pdf = pf.figure_windows()
        assert printed_min_pt(pdf) >= 7.0

    def test_price_extremes_are_the_ones_the_abstract_quotes(self):
        hi, lo = pf.price_extremes()
        assert (round(hi["price"]), round(lo["price"])) == (126101, 62883)


class TestTransitionGeometry:
    @pytest.fixture(scope="class")
    def geo(self):
        return pf.transition_geometry()

    def test_rows_are_probability_distributions(self, geo):
        for row in geo["A"]:
            assert sum(row) == pytest.approx(1.0, abs=1e-9)

    def test_the_reconstruction_carries_the_boundary_estimate(self, geo):
        """The archive kept diag(A) and pi only; the zero is imposed, and the caption says so."""
        assert geo["A"][2][0] == 0.0
        assert "reconstruct" in geo["provenance"]

    def test_some_270_entries_into_the_high_state(self, geo):
        assert round(geo["entries_high"]) == 270

    def test_rule_of_three_bound_is_three_in_a_thousand(self, geo):
        assert round(geo["hours_high"], -1) == 1000
        assert round(100 * geo["bound_high_low"], 2) == 0.30

    def test_the_high_to_low_label_does_not_present_the_imposed_zero_as_a_count(self):
        label = pf.high_low_label()
        assert "entries" not in label and "rule of three" not in label
        assert "imposed" in label and "profile likelihood, sub-period" in label

    def test_the_high_to_low_label_quotes_the_profile_likelihood_bound(self, tmp_path):
        import json
        ev = json.loads(pf.A31_EVIDENCE.read_text())
        b = ev["profile_likelihood"]["upper_bound_one_sided_95"]
        assert f"bound {100 * b:.2f}%/h" in pf.high_low_label()
        assert "0.64%/h" in pf.high_low_label()
        fake = tmp_path / "a31.json"
        fake.write_text(json.dumps({"profile_likelihood": {"upper_bound_one_sided_95": 0.0123}}))
        assert "bound 1.23%/h" in pf.high_low_label(fake)

    def test_state_volatilities_are_the_abstract_s(self, geo):
        assert [round(v, 1) for v in geo["vol_pct"]] == [14.6, 33.3, 88.4]


class TestTransitionsLayout:
    def test_no_arrow_runs_through_a_label(self):
        """Every probability label stays at least a point clear of every arrow."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        fig = pf.draw_transitions()
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax = fig.axes[0]
            arrows = [a for a in ax.patches if isinstance(a, FancyArrowPatch)]
            labels = [t for t in ax.texts if "[" in t.get_text() or "\u2192" in t.get_text()]
            assert len(arrows) == 6 and len(labels) == 6
            pad = fig.dpi / 72
            for text in labels:
                box = text.get_window_extent(renderer).padded(pad)
                for arrow in arrows:
                    display = arrow.get_transform().transform_path(arrow.get_path())
                    for line in display.to_polygons(closed_only=False):
                        steps = np.linspace(0.0, 1.0, 25)[:, None]
                        for a, b in zip(line[:-1], line[1:]):
                            x, y = (a + steps * (b - a)).T
                            inside = (x >= box.x0) & (x <= box.x1) & (y >= box.y0) & (y <= box.y1)
                            assert not inside.any(), text.get_text()
        finally:
            plt.close(fig)


class TestOctoberHour:
    def test_the_slice_peaks_at_the_crash_hour(self):
        d = pf.october_slice()
        top = d.loc[d["bps"].idxmax()]
        assert str(top["time"]) == "2025-10-10 22:00:00+00:00"
        assert round(top["bps"], 2) == 23.61

    def test_the_crash_hour_is_the_maximum_of_the_whole_series(self):
        peak = pf.october_peak()
        assert (str(peak["time"]), round(peak["bps"], 2)) == ("2025-10-10 22:00:00+00:00", 23.61)

    def test_the_overwritten_hours_are_the_quartiles_of_the_later_window(self):
        times = [str(t) for t in pf.injection_times()]
        assert times == ["2026-06-06 07:00:00+00:00", "2026-07-15 05:00:00+00:00",
                         "2026-08-23 03:00:00+00:00"]


class TestForest:
    @pytest.fixture(scope="class")
    def rows(self):
        return pf.forest_rows()

    def test_ten_rows_in_three_groups(self, rows):
        assert len(rows) == 10
        assert [r["group"] for r in rows] == ["paper"] * 4 + ["windows"] * 3 + ["overwritten"] * 3

    def test_the_first_row_is_the_headline(self, rows):
        assert round(rows[0]["ratio"], 2) == 11.19
        assert [round(v, 2) for v in (rows[0]["lo"], rows[0]["hi"])] == [2.95, 19.42]

    def test_the_sub_period_reverses_the_ordering(self, rows):
        sub = next(r for r in rows if "sub-period" in r["label"])
        assert round(sub["ratio"], 2) == 0.29

    def test_open_lower_ends_are_exactly_where_the_delta_method_goes_negative(self, rows):
        assert [r["label"] for r in rows if r["open_lo"]] == [
            "after the window, 28 Apr – 30 Sep 2026", "at 23 Aug 2026"]
        assert all(r["lo"] is None for r in rows if r["open_lo"])

    def test_overwritten_rows_name_their_dates(self, rows):
        assert [r["label"] for r in rows if r["group"] == "overwritten"] == [
            "at 6 Jun 2026", "at 15 Jul 2026", "at 23 Aug 2026"]


class TestRollingWindows:
    @pytest.fixture(scope="class")
    def rows(self):
        return pf.rolling_rows()

    def test_fourteen_windows(self, rows):
        assert len(rows["ratio"]) == 14

    def test_only_the_october_window_lies_above_unity(self, rows):
        assert [i for i, s in enumerate(rows["significance"]) if s == "above"] == [0]

    def test_the_last_two_lower_ends_are_open(self, rows):
        assert rows["open_lo"] == [False] * 12 + [True, True]

    def test_the_drawdown_deepens_from_21_to_47_per_cent(self, rows):
        assert (round(100 * rows["drawdown"][0]), round(100 * rows["drawdown"][-1])) == (-21, -47)


class TestTwoOptima:
    def test_the_record_holds_both_optima_and_what_each_implies(self):
        rec = pf.optima_record()
        assert [o["llf"] for o in rec["optima"]] == [-4808.53, -4828.67]
        assert round(rec["optima"][0]["ratio"], 2) == 11.19
        assert round(rec["optima"][1]["ratio"], 1) == 24.6
        assert rec["logL_gap"] == 20.14

    def test_every_block_splits_its_forty_starts_between_the_two(self):
        blocks = pf.optima_by_block()
        assert [b["inferior"] for b in blocks] == [24, 25, 24, 20, 25, 30]
        assert all(b["best"] + b["inferior"] == 40 and b["failures"] == 0 for b in blocks)


class TestWhatReachesThePaper:
    """Only the figures the paper prints are copied to paper/figures/."""

    @pytest.mark.parametrize("draw,stem", [("figure_october", "fig7_october_hour"),
                                           ("figure_rolling", "fig8_rolling")])
    def test_the_linear_delta_figures_stay_in_output(self, tmp_path, monkeypatch, draw, stem):
        monkeypatch.setattr(pf, "FIG_OUT", tmp_path / "out")
        monkeypatch.setattr(pf, "FIG_PAPER", tmp_path / "paper")
        pdf = getattr(pf, draw)()
        assert pdf == tmp_path / "out" / f"{stem}.pdf" and pdf.exists()
        assert not (tmp_path / "paper" / f"{stem}.pdf").exists()

    def test_a_printed_figure_is_copied(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pf, "FIG_OUT", tmp_path / "out")
        monkeypatch.setattr(pf, "FIG_PAPER", tmp_path / "paper")
        pf.figure_optima()
        assert (tmp_path / "paper" / "fig9_optima.pdf").exists()
