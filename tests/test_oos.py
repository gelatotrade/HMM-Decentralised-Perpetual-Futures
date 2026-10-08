"""Out-of-sample robustness: rolling windows and single-observation sensitivity.

Two claims of Section 5.3 rest on this module, and both are easy to get wrong
without any visible error:

* The rolling estimates must be labelled by ``sigma2`` ascending, not by
  half-life. Sorting the half-lives instead makes every window look like the
  paper window, because it assumes the very ordering that is under test.
* The sensitivity variants must not mutate the caller's series. The baseline is
  re-fitted alongside the variants; if a variant edits in place, the baseline
  silently becomes the variant and the comparison reports no effect at all.
"""
import numpy as np
import pytest

from regime_engine.oos import (
    injection_variants,
    label_regimes,
    rolling_windows,
    sensitivity_variants,
    span_label,
    write_rolling_table,
    write_sensitivity_table,
)


class TestRollingWindows:
    def test_windows_have_the_requested_length(self):
        assert all(b - a == 100 for a, b in rolling_windows(500, window=100, step=50))

    def test_windows_advance_by_the_step(self):
        starts = [a for a, _ in rolling_windows(500, window=100, step=50)]
        assert starts == list(range(0, 401, 50))

    def test_no_window_runs_past_the_end_of_the_series(self):
        assert all(b <= 500 for _, b in rolling_windows(500, window=100, step=50))

    def test_a_series_shorter_than_one_window_yields_nothing(self):
        assert rolling_windows(80, window=100, step=50) == []

    def test_a_series_of_exactly_one_window_yields_that_window(self):
        assert rolling_windows(100, window=100, step=50) == [(0, 100)]


class TestSpanLabel:
    def test_a_midnight_end_closes_the_day_before(self):
        import pandas as pd

        t = [pd.Timestamp(x, tz="UTC") for x in ("2025-10-02 00:00", "2026-10-01 00:00")]
        assert span_label(t) == "2 Oct 2025 - 30 Sep 2026"
        t[-1] = pd.Timestamp("2026-08-27 08:00", tz="UTC")
        assert span_label(t) == "2 Oct 2025 - 27 Aug 2026"


class TestSensitivityVariants:
    @pytest.fixture
    def y(self):
        return np.array([1.0, 2.0, 3.0, 50.0, 4.0, 5.0], dtype=float)

    def test_replace_max_substitutes_the_second_largest_value(self, y):
        got = sensitivity_variants(y)["replace_max"]
        assert got[3] == 5.0, "the 50.0 outlier must become the second largest, 5.0"
        assert got.max() == 5.0

    def test_drop_max_removes_exactly_one_observation(self, y):
        got = sensitivity_variants(y)["drop_max"]
        assert len(got) == len(y) - 1
        assert 50.0 not in got

    def test_winsorise_caps_the_upper_tail_without_shortening_the_series(self, y):
        got = sensitivity_variants(y)["winsorise_top1"]
        assert len(got) == len(y)
        assert got.max() < 50.0

    def test_baseline_is_the_untouched_series(self, y):
        assert np.array_equal(sensitivity_variants(y)["baseline"], y)

    def test_variants_do_not_mutate_the_caller_s_array(self, y):
        before = y.copy()
        sensitivity_variants(y)
        assert np.array_equal(y, before), "a variant edited the input in place"


class TestRegimeLabelling:
    """calm is the lower-variance regime -- never the longer-lived one.

    Ordering by half-life would make the ratio exceed unity in every window by
    construction, which is precisely the claim Section 5.3 puts under test.
    """

    def test_calm_is_the_low_variance_regime_when_it_is_also_the_persistent_one(self):
        rec = label_regimes(sigma2=[8.0, 0.3], half_lives=[3.0, 40.0])
        assert rec["tau_calm"] == 40.0
        assert rec["tau_stressed"] == 3.0
        assert rec["ratio"] == pytest.approx(40.0 / 3.0)

    def test_calm_is_still_the_low_variance_regime_when_it_is_the_transient_one(self):
        """The reversed case. A half-life sort would report 7.2 here, not 0.14."""
        rec = label_regimes(sigma2=[0.3, 8.0], half_lives=[5.0, 36.0])
        assert rec["tau_calm"] == 5.0
        assert rec["tau_stressed"] == 36.0
        assert rec["ratio"] == pytest.approx(5.0 / 36.0)
        assert rec["ratio"] < 1.0

    def test_ratio_is_calm_over_stressed_by_definition(self):
        rec = label_regimes(sigma2=[1.0, 2.0], half_lives=[10.0, 4.0])
        assert rec["ratio"] == pytest.approx(rec["tau_calm"] / rec["tau_stressed"])

    def test_a_degenerate_stressed_half_life_does_not_raise(self):
        rec = label_regimes(sigma2=[1.0, 2.0], half_lives=[10.0, 0.0])
        assert rec["ratio"] is None


class TestInjectionVariants:
    """The placebo: can one hour manufacture the paper's asymmetry?

    Overwriting a single hour with a synthetic +23.61 bps in a window that shows
    the reversed ordering lifts the point estimate by more than an order of magnitude.
    That is the constructive half of the single-observation argument, so the
    insertion itself has to be exact and has to leave the rest of the series alone.
    """

    @pytest.fixture
    def y(self):
        return np.arange(10, dtype=float)

    def test_each_position_receives_the_injected_value(self, y):
        got = injection_variants(y, value=99.0, positions=(2, 5))
        assert got["inject@2"][2] == 99.0
        assert got["inject@5"][5] == 99.0

    def test_only_the_injected_position_changes(self, y):
        got = injection_variants(y, value=99.0, positions=(2,))["inject@2"]
        assert np.array_equal(np.delete(got, 2), np.delete(y, 2))

    def test_series_keeps_its_length(self, y):
        assert len(injection_variants(y, value=99.0, positions=(2,))["inject@2"]) == len(y)

    def test_input_is_not_mutated(self, y):
        before = y.copy()
        injection_variants(y, value=99.0, positions=(2, 5))
        assert np.array_equal(y, before)

    def test_a_position_outside_the_series_is_rejected(self, y):
        with pytest.raises(ValueError):
            injection_variants(y, value=99.0, positions=(99,))


class TestTableWriters:
    """The tables must be generated, never hand-edited.

    A hand-edited table keeps whatever regime order and values it was typed
    with when the fit changes. These writers keep the Section 5.3 tables in
    step with the artifact, which matters because their whole point is that
    the numbers move between windows.
    """

    @pytest.fixture
    def payload(self):
        return {
            "protocol": {"window": 2500, "step": 450, "n_starts": 40},
            "rolling": [
                {"index": 0, "mid": "2025-11-23 00:00:00+00:00", "tau_calm": 50.91,
                 "tau_stressed": 1.30, "ratio": 39.17, "ci95": [7.586, 70.755],
                 "significance": "above", "price_mean": 98691.0,
                 "drawdown_mean": -0.213},
                {"index": 1, "mid": "2026-07-06 00:00:00+00:00", "tau_calm": 9.36,
                 "tau_stressed": 41.82, "ratio": 0.224, "ci95": [None, 0.509],
                 "significance": "below", "price_mean": 67280.0,
                 "drawdown_mean": -0.464},
            ],
            "sensitivity": [
                {"variant": "baseline", "description": "unchanged", "T": 5001,
                 "tau_calm": 35.55, "tau_stressed": 3.18, "ratio": 11.188,
                 "ci95": [2.951, 19.424], "significance": "above"},
                {"variant": "drop_max", "description": "premium maximum deleted",
                 "T": 5000, "tau_calm": 33.10, "tau_stressed": 18.12,
                 "ratio": 1.827, "ci95": [0.503, 3.151], "significance": "ns"},
            ],
        }

    def test_rolling_table_has_one_row_per_window(self, payload, tmp_path):
        body = write_rolling_table(payload, tmp_path).read_text()
        assert body.count("\\\\\n") >= 2
        assert "39.17" in body and "0.22" in body

    def test_rolling_table_marks_which_side_of_unity_each_window_falls(self, payload, tmp_path):
        body = write_rolling_table(payload, tmp_path).read_text()
        assert "$>1$" in body and "$<1$" in body

    def test_an_open_interval_endpoint_renders_without_crashing(self, payload, tmp_path):
        """A delta-method lower bound can come back None or negative."""
        body = write_rolling_table(payload, tmp_path).read_text()
        assert "None" not in body

    def test_sensitivity_table_names_each_variant(self, payload, tmp_path):
        body = write_sensitivity_table(payload, tmp_path).read_text().lower()
        assert "unchanged" in body and "deleted" in body

    def test_tables_carry_a_do_not_edit_banner(self, payload, tmp_path):
        for w in (write_rolling_table, write_sensitivity_table):
            assert "do not edit by hand" in w(payload, tmp_path).read_text()


class TestTableTypesetting:
    """LaTeX details that render wrong rather than failing loudly."""

    @pytest.fixture
    def payload(self):
        return {
            "protocol": {"window": 2500, "step": 450, "n_starts": 40},
            "rolling": [{"index": 0, "mid": "2026-07-06 00:00:00+00:00",
                         "tau_calm": 9.36, "tau_stressed": 42.04, "ratio": 0.223,
                         "ci95": [-0.062, 0.507], "significance": "below",
                         "price_mean": 66633.0, "drawdown_mean": -0.47}],
            "sensitivity": [{"variant": "replace_max",
                             "description": "premium maximum replaced by the second largest value",
                             "T": 5001, "tau_calm": 33.90, "tau_stressed": 11.15,
                             "ratio": 3.042, "ci95": [0.554, 5.530],
                             "significance": "ns"}],
        }

    def test_a_negative_drawdown_is_set_in_math_mode(self, payload, tmp_path):
        """A bare hyphen typesets as a hyphen, not a minus sign."""
        body = write_rolling_table(payload, tmp_path).read_text()
        assert "$-47\\%$" in body
        assert " -47\\%" not in body

    def test_the_variant_column_uses_short_labels(self, payload, tmp_path):
        """The full description belongs in the artifact and the caption, not the cell."""
        body = write_sensitivity_table(payload, tmp_path).read_text()
        row = [line for line in body.splitlines() if "3.04" in line][0]
        assert len(row.split("&")[0].strip()) <= 32, row
