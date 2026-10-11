"""fig7b_october_hour_robust and fig8b_rolling_robust: what each one is drawn from.

The robust counterparts of fig7_october_hour and fig8_rolling show, for every
estimate of R = tau_calm / tau_stressed, its 95% Fieller set under the Hessian
and under the sandwich covariance, coloured and marked by the Wald test of
phi_calm = phi_stressed, and the switching-adjusted ratio beside the
within-regime one. The data functions are pinned here against the shipped
artifacts, as in test_paper_figures.py, and the drawings are checked for
legibility at the paper's text width.
"""
import re

import matplotlib
import pytest

from regime_engine import paper_figures as pf

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def forest():
    return pf.forest_rows_robust()


@pytest.fixture(scope="module")
def rolling():
    return pf.rolling_rows_robust()


def _verdict(z):
    return "above" if z > 1.959963984540054 else "below" if z < -1.959963984540054 else "ns"


class TestForestRobust:
    def test_the_rows_are_fig7s_estimates_in_fig7s_order(self, forest):
        old = pf.forest_rows()
        assert [r["group"] for r in forest] == [r["group"] for r in old]
        for new, ref in zip(forest, old):
            assert new["ratio"] == pytest.approx(ref["ratio"], rel=1e-12)

    def test_the_headline_row(self, forest):
        base = forest[0]
        assert base["label"] == "as estimated" and round(base["ratio"], 2) == 11.19
        assert base["side"] == {"hessian": "above", "robust": "ns"}
        assert round(base["wald"]["robust"]["z"], 2) == 1.01
        assert round(base["wald"]["robust"]["p"], 2) == 0.31
        assert [round(v, 2) for v in base["set"]["hessian"]["bounds"]] == [4.61, 23.19]
        assert [round(v, 2) for v in base["set"]["robust"]["bounds"]] == [-9.75, 70.82]
        assert round(base["ratio_switching"], 2) == 1.51          # interpolated hours
        assert round(base["ratio_switching_hours"], 2) == 1.47    # whole hours, 25/17

    def test_every_side_is_the_wald_verdict(self, forest):
        for r in forest:
            for cov in pf.COVS_ROBUST:
                assert r["side"][cov] == r["wald"][cov]["side"] == _verdict(r["wald"][cov]["z"])

    def test_the_log_scale_false_positives_are_gone(self, forest):
        """The log-scale interval put these rows above unity; the Wald test does not."""
        later = [r for r in forest if r["group"] == "overwritten"]
        assert [r["label"] for r in later] == ["at 6 Jun 2026", "at 15 Jul 2026",
                                               "at 23 Aug 2026"]
        assert [r["side"]["robust"] for r in later] == ["ns", "ns", "ns"]
        assert [r["side"]["hessian"] for r in later] == ["above", "above", "ns"]
        assert forest[0]["side"]["robust"] == "ns"

    def test_the_sub_period_comes_from_the_diagnostics_fit(self, forest):
        sub = next(r for r in forest if r["label"] == "hourly sub-period")
        assert round(sub["ratio"], 2) == 0.29
        assert round(sub["ratio_switching"], 2) == 0.69
        assert sub["side"] == {"hessian": "below", "robust": "ns"}


class TestRollingRobust:
    def test_fourteen_windows(self, rolling):
        assert len(rolling["ratio"]) == len(rolling["ratio_switching"]) == 14

    def test_the_october_window_is_first(self, rolling):
        assert round(rolling["ratio"][0], 2) == 39.17
        assert round(rolling["ratio_switching"][0], 2) == 1.90
        assert round(rolling["ratio_switching_hours"][0], 2) == 1.75
        assert rolling["set"]["robust"][0]["shape"] == "whole_line"

    def test_sides_by_covariance(self, rolling):
        def where(cov, side):
            return [i for i, s in enumerate(rolling["side"][cov]) if s == side]
        assert where("hessian", "above") == [0, 1]
        assert where("hessian", "below") == [4, 5, 8, 9, 10, 11, 12, 13]
        assert where("robust", "above") == []
        assert where("robust", "below") == [5, 8, 10, 11, 12, 13]

    def test_every_side_is_the_wald_verdict(self, rolling):
        for cov in pf.COVS_ROBUST:
            for side, wald in zip(rolling["side"][cov], rolling["wald"][cov]):
                assert side == wald["side"] == _verdict(wald["z"])


class TestPositivePieces:
    LIM = (0.01, 300.0)

    def pieces(self, shape, bounds):
        return pf.positive_pieces({"shape": shape, "bounds": bounds}, *self.LIM)

    def test_a_bounded_set_inside_the_axis(self):
        assert self.pieces("bounded", [4.61, 23.19]) == [
            {"lo": 4.61, "hi": 23.19, "open_lo": False, "open_hi": False}]

    def test_a_negative_lower_end_is_open(self):
        assert self.pieces("bounded", [-9.75, 70.82]) == [
            {"lo": 0.01, "hi": 70.82, "open_lo": True, "open_hi": False}]

    def test_an_end_beyond_the_axis_is_open(self):
        assert self.pieces("bounded", [0.5, 400.0]) == [
            {"lo": 0.5, "hi": 300.0, "open_lo": False, "open_hi": True}]

    def test_the_whole_line(self):
        assert self.pieces("whole_line", [None, None]) == [
            {"lo": 0.01, "hi": 300.0, "open_lo": True, "open_hi": True}]

    def test_two_rays_with_a_negative_lower_ray(self):
        assert self.pieces("two_rays", [-5.0, 3.0]) == [
            {"lo": 3.0, "hi": 300.0, "open_lo": False, "open_hi": True}]
        # (-inf, -0.52] U [-0.02, inf): every positive value
        assert self.pieces("two_rays", [-0.523, -0.0177]) == [
            {"lo": 0.01, "hi": 300.0, "open_lo": True, "open_hi": True}]

    def test_two_positive_rays(self):
        assert self.pieces("two_rays", [0.5, 3.0]) == [
            {"lo": 0.01, "hi": 0.5, "open_lo": True, "open_hi": False},
            {"lo": 3.0, "hi": 300.0, "open_lo": False, "open_hi": True}]

    def test_nothing_to_draw(self):
        assert self.pieces("bounded", [-2.0, -1.0]) == []
        assert self.pieces("undefined", [None, None]) == []


class TestLogAxis:
    def test_limits_snap_to_the_tick_ladder_with_a_margin(self):
        assert pf.log_axis_limits([0.0512, 2.0], [5.0, 123.46]) == (0.03, 300)
        assert pf.log_axis_limits([0.0307], [214.58]) == (0.01, 300)
        assert pf.log_axis_limits([0.5], [2.0]) == (0.3, 3)

    def test_undefined_interval_ends_are_ignored(self):
        assert pf.log_axis_limits([None, 0.5], [2.0, None]) == (0.3, 3)

    def test_the_axis_stops_at_its_floor(self, forest, rolling):
        """The sub-period's sandwich set reaches 0.008: it is drawn open at 0.01."""
        assert pf.set_axis_limits(forest) == (0.01, 100)
        entries = [{"ratio": r, "ratio_switching": s,
                    "set": {c: rolling["set"][c][i] for c in pf.COVS_ROBUST}}
                   for i, (r, s) in enumerate(zip(rolling["ratio"], rolling["ratio_switching"]))]
        assert pf.set_axis_limits(entries) == (0.01, 300)


def _pdf_width_in(path) -> float:
    box = re.search(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+[\d.\-]+\s+([\d.\-]+)", path.read_bytes())
    return (float(box.group(2)) - float(box.group(1))) / 72.0


class TestDrawing:
    @pytest.fixture()
    def redirect(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pf, "FIG_OUT", tmp_path / "out")
        monkeypatch.setattr(pf, "FIG_PAPER", tmp_path / "paper")
        return tmp_path

    @pytest.mark.parametrize("name, stem", [("october_robust", "fig7b_october_hour_robust"),
                                            ("rolling_robust", "fig8b_rolling_robust")])
    def test_the_figure_is_drawn(self, redirect, name, stem):
        pdf = pf.FIGURES[name]()
        assert pdf == redirect / "out" / f"{stem}.pdf"
        assert pdf.stat().st_size > 10_000
        assert (redirect / "out" / f"{stem}.png").exists()
        assert (redirect / "paper" / f"{stem}.pdf").read_bytes() == pdf.read_bytes()

    def test_the_october_figure_names_the_estimation_window(self):
        fig = pf.build_october_robust()
        try:
            labels = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
            assert "estimation window" in labels and "paper window" not in labels
        finally:
            matplotlib.pyplot.close(fig)

    def test_the_rolling_panels_sit_side_by_side(self):
        """The two covariances share one row, so the figure takes a third of a page."""
        fig = pf.build_rolling_robust()
        try:
            a, b = (ax.get_position() for ax in fig.axes[:2])
            assert a.y0 == pytest.approx(b.y0) and a.x1 < b.x0
            assert fig.get_size_inches()[1] <= 3.0
        finally:
            matplotlib.pyplot.close(fig)

    @pytest.mark.parametrize("build", ["build_october_robust", "build_rolling_robust"])
    def test_no_text_is_below_seven_points_at_text_width(self, tmp_path, build):
        """Scaled to the 15.8 cm text width, every label stays at 7 pt or more.

        Mathtext sets sub- and superscripts at 70% of the font size, so a
        label with one would print its index below 7 pt; the figures use
        none.
        """
        from matplotlib.text import Text

        fig = getattr(pf, build)()
        pdf = tmp_path / "f.pdf"
        fig.savefig(pdf)
        scale = pf.TEXT_WIDTH_IN / _pdf_width_in(pdf)
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        matplotlib.pyplot.close(fig)
        sizes = [t.get_fontsize() for t in texts]
        assert min(sizes) * scale >= pf.MIN_FONT_PT, (min(sizes), scale)
        indexed = [t.get_text() for t in texts
                   if re.search(r"\$[^$]*[_^][^$]*\$", t.get_text())]
        assert indexed == []
