"""The premium gap between Hyperliquid and the exchanges: methods and shipped numbers.

The methods are checked on synthetic series whose answer is known: the closed
band, the period a record belongs to, the gap statistics, the half-life of an
AR(1) with and without a drifting level, negative spells, block-bootstrap
intervals, the level-shift comparison, the eight-hour average with linearly
rising weights, the settlement fit, the phases and the decomposition of the
floor-share difference, whose parts must add up. The second half pins the
numbers the artifact holds on the shipped data, group by group and quarter by
quarter, checks that Tables 1, 5, C1 and C2 and Figure 4 rebuild, and that the
figure's text is legible at the printed width and does not overlap.
"""
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from _json_close import assert_json_close

from regime_engine import censoring
from regime_engine import fetch_premium as fp
from regime_engine import premium_gap as pg


def _hours(start, n):
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


class TestBand:
    def test_the_band_is_closed_where_the_rule_pays_the_floor(self):
        x = pd.Series([-4.0001, -4.0, 0.0, 6.0, 6.0001, np.nan])
        assert pg.in_band(x).tolist()[:5] == [0.0, 1.0, 1.0, 1.0, 0.0]
        assert np.isnan(pg.in_band(x).iloc[5])
        floor = float(censoring.FLOOR_8H) * 1e4
        assert (censoring.funding_rule(x.dropna()) == floor).tolist() == [False, True, True, True,
                                                                           False]

    def test_premium_summary_counts_the_edges_inside(self):
        out = pg.premium_summary(pd.Series([-5.0, -4.0, 1.0, 6.0, 7.0]))
        assert (out["share_below"], out["share_in_band"], out["share_above"]) == pytest.approx(
            (0.2, 0.6, 0.2))


class TestPeriods:
    def test_a_record_belongs_to_the_period_its_hour_accrued_in(self):
        t = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00",
                                             "2024-04-01 00:00"], utc=True))
        assert list(pg.accrual_period(t, "Q")) == ["2023Q4", "2024Q1", "2024Q1"]
        assert list(pg.accrual_period(t, "M")) == ["2023-12", "2024-01", "2024-03"]
        assert list(pg.accrual_period(t, "D")) == ["2023-12-31", "2024-01-01", "2024-03-31"]


class TestGapStatistics:
    def test_location_spread_and_sign(self):
        out = pg.gap_stats(pd.Series([-1.0, 0.0, 1.0, 2.0, np.nan]))
        assert out["n"] == 4
        assert out["mean_bps"] == pytest.approx(0.5)
        assert out["median_bps"] == pytest.approx(0.5)
        assert (out["q25_bps"], out["q75_bps"]) == pytest.approx((-0.25, 1.25))
        assert out["share_positive"] == pytest.approx(0.5)

    def test_grouping_by_accrual_period(self):
        t = _hours("2024-03-31 22:00", 4)           # 22:00, 23:00, 00:00 (still Q1), 01:00
        gap = pd.Series([1.0, 2.0, 3.0, 10.0], index=t)
        out = pg.gap_by(gap, "Q")
        assert list(out) == ["2024Q1", "2024Q2"]
        assert out["2024Q1"]["median_bps"] == pytest.approx(2.0)
        assert out["2024Q2"]["n"] == 1


class TestPersistence:
    def test_daily_means_need_enough_hours(self):
        t = _hours("2024-01-01 01:00", 30)          # one full day, then six hours
        out = pg.daily_means(pd.Series(1.0, index=t), min_hours=20)
        assert list(out.index) == ["2024-01-01"]
        assert out.iloc[0] == pytest.approx(1.0)

    def test_ar1_half_life_recovers_the_truth(self):
        rng = np.random.default_rng(3)
        phi, x = 0.9, [0.0]
        for e in rng.normal(size=20_000):
            x.append(0.5 + phi * x[-1] + e)
        out = pg.ar1_half_life(pd.Series(x))
        assert out["phi"] == pytest.approx(phi, abs=0.01)
        assert out["half_life"] == pytest.approx(np.log(0.5) / np.log(phi), rel=0.08)

    def test_no_half_life_without_mean_reversion(self):
        assert pg.ar1_half_life(pd.Series([1.0, -1.0] * 50))["half_life"] is None

    def test_autocorrelation_of_a_known_series(self):
        x = pd.Series(np.tile([1.0, -1.0], 100))
        assert pg.autocorrelations(x, (1, 2)) == pytest.approx({1: -0.995, 2: 0.99}, abs=1e-9)

    def test_runs_of_negative_days_break_at_a_missing_day(self):
        days = pd.date_range("2024-01-01", periods=9, freq="D").strftime("%Y-%m-%d")
        x = pd.Series([1, -1, -2, 1, -1, -1, -1, 2, -3], index=days, dtype=float)
        out = pg.negative_runs(x)
        assert out["longest_days"] == 3
        assert (out["longest_start"], out["longest_end"]) == ("2024-01-05", "2024-01-07")
        assert out["negative_days"] == 6
        x = x.drop("2024-01-06")
        assert pg.negative_runs(x)["longest_days"] == 2

    def test_negative_spells_of_days_and_of_weeks(self):
        days = pd.date_range("2024-01-01", periods=12, freq="D").strftime("%Y-%m-%d")
        x = pd.Series([-1, -1, 1, -1, -1, -1, 2, -1, -1, -1, -1, 3], index=days, dtype=float)
        out = pg.negative_spells(x, days=1, min_length=3)
        assert out == [{"start": "2024-01-04", "end": "2024-01-06", "length": 3},
                       {"start": "2024-01-08", "end": "2024-01-11", "length": 4}]
        weeks = pd.Series([-1.0, -2.0, -0.5, 1.0],
                          index=["2025-03-31", "2025-04-07", "2025-04-14", "2025-04-21"])
        assert pg.negative_spells(weeks, days=7, min_length=2) == [
            {"start": "2025-03-31", "end": "2025-04-14", "length": 3}]
        assert pg.negative_spells(weeks.drop("2025-04-07"), days=7, min_length=2) == []

    def test_rolling_mean_over_calendar_days(self):
        days = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
        x = pd.Series([1.0, 1.0, -3.0, -3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=days)
        out = pg.rolling_mean_stats(x, days=3)
        assert out["windows"] == 8
        assert out["negative_windows"] == 4              # windows ending 3 to 6 Jan
        assert (out["min_bps"], out["min_end_day"]) == (pytest.approx(-5 / 3), "2024-01-04")

    def test_within_quarter_ar1_removes_a_drifting_level(self):
        rng = np.random.default_rng(8)
        days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        level = pd.Series(days.quarter + 4 * (days.year - 2024), index=days).astype(float)
        e, phi = [0.0], 0.5
        for v in rng.normal(0, 0.3, len(days) - 1):
            e.append(phi * e[-1] + v)
        x = pd.Series(level.to_numpy() + np.array(e), index=days.strftime("%Y-%m-%d"))
        raw, within = pg.ar1_half_life(x), pg.ar1_half_life(pg.within_period(x, "Q"))
        assert raw["phi"] > 0.9                        # the drifting level looks persistent
        assert within["phi"] == pytest.approx(phi, abs=0.08)


class TestMedianIntervals:
    def test_block_bootstrap_is_reproducible_and_covers_the_median(self):
        rng = np.random.default_rng(9)
        x = rng.normal(1.0, 1.0, 24 * 30)
        a = pg.block_bootstrap_median_ci(x, np.random.default_rng(1))
        b = pg.block_bootstrap_median_ci(x, np.random.default_rng(1))
        assert a == b
        assert a[0] < np.median(x) < a[1]
        assert a[0] < 1.0 < a[1]

    def test_dependence_widens_the_interval(self):
        rng = np.random.default_rng(10)
        iid = rng.normal(0.0, 1.0, 24 * 30)
        persistent = np.repeat(rng.normal(0.0, 1.0, 30), 24)    # one value per day
        w_iid = np.diff(pg.block_bootstrap_median_ci(iid, np.random.default_rng(2)))[0]
        w_day = np.diff(pg.block_bootstrap_median_ci(persistent, np.random.default_rng(2)))[0]
        assert w_day > 3 * w_iid

    def test_intervals_by_period_and_the_ones_covering_zero(self):
        t = _hours("2024-01-01 01:00", 24 * 60)
        rng = np.random.default_rng(11)
        gap = pd.Series(np.where(t < pd.Timestamp("2024-02-01 01:00", tz="UTC"), 3.0, 0.0)
                        + rng.normal(0, 1, len(t)), index=t)
        out = pg.median_intervals(gap, "M", np.random.default_rng(3))
        assert list(out) == ["2024-01", "2024-02"]      # the last stamp, 1 Mar 00:00, closes Feb
        assert out["2024-01"][0] > 2 and out["2024-02"][0] < 0 < out["2024-02"][1]


class TestMedianCis:
    @pytest.fixture(scope="class")
    def out(self):
        t = _hours("2024-01-01 01:00", 24 * 92)
        rng = np.random.default_rng(13)
        days = np.repeat(rng.normal(0.0, 1.0, 92), 24)          # day-long deviations
        gap = pd.Series(0.6 + days + rng.normal(0, 0.3, len(t)), index=t)
        return pg._median_cis(gap, (0, 0))

    def test_headline_block_is_a_week_with_shorter_blocks_as_sensitivity(self, out):
        assert pg.BOOTSTRAP["block_hours"] == 168
        assert pg.BOOTSTRAP["sensitivity_block_hours"] == [24, 72]
        assert out["block_hours"] == 168
        assert set(out["block_sensitivity"]) == {"24", "72"}
        for b in ("24", "72"):
            s = out["block_sensitivity"][b]
            assert set(s) >= {"monthly", "quarterly", "months_covering_zero",
                              "quarters_covering_zero"}

    def test_longer_blocks_widen_the_interval_of_a_persistent_gap(self, out):
        w = {b: np.diff(out["block_sensitivity"][b]["monthly"]["2024-02"])[0]
             for b in ("24", "72")}
        w["168"] = np.diff(out["monthly"]["2024-02"])[0]
        assert w["24"] < w["168"]

    def test_coverage_changes_against_the_24_hour_blocks_are_listed(self, out):
        ch = out["coverage_change_vs_24h"]
        for freq, key in (("monthly", "months_covering_zero"),
                          ("quarterly", "quarters_covering_zero")):
            old = set(out["block_sensitivity"]["24"][key])
            new = set(out[key])
            assert set(ch[freq]["now_covering_zero"]) == new - old
            assert set(ch[freq]["no_longer_covering_zero"]) == old - new

    def test_partial_quarters_are_kept_apart(self):
        t = _hours("2023-12-11 23:00", 24 * 40)
        gap = pd.Series(np.random.default_rng(1).normal(0, 1, len(t)), index=t)
        out = pg._median_cis(gap, (0, 0))
        assert "2023Q4" not in out["full_quarters_covering_zero"]
        assert set(out["partial_quarters_covering_zero"]) <= set(pg.PARTIAL_QUARTERS)
        assert "2023-12" not in out["full_months_covering_zero"]
        assert set(out["partial_months_covering_zero"]) <= set(pg.PARTIAL_MONTHS)
        assert (sorted(out["full_months_covering_zero"] + out["partial_months_covering_zero"])
                == sorted(out["months_covering_zero"]))


class TestLevelShift:
    def _daily(self, step_day, step):
        days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        rng = np.random.default_rng(12)
        x = rng.normal(0, 0.2, len(days)) + np.where(days >= pd.Timestamp(step_day), step, 0.0)
        return pd.Series(x, index=days.strftime("%Y-%m-%d"))

    def test_a_step_inside_the_interval_is_found_and_is_rare(self):
        out = pg.level_shift(self._daily("2025-03-01", -2.0), "2025-02-10", "2025-03-20", days=28)
        assert out["before_bps"] == pytest.approx(0.0, abs=0.1)
        assert out["after_bps"] == pytest.approx(-2.0, abs=0.1)
        assert out["shift_bps"] == pytest.approx(-2.0, abs=0.15)
        assert out["between_days"] == 37
        assert out["share_as_large"] < 0.1
        # a fall: the one-sided share counts falls at least as large, never more than both
        assert 0 < out["share_same_direction_as_large"] <= out["share_as_large"]

    def test_the_one_sided_share_follows_the_direction_of_the_shift(self):
        down = pg.level_shift(self._daily("2025-03-01", -2.0), "2025-02-10", "2025-03-20")
        up = pg.level_shift(self._daily("2025-03-01", 2.0), "2025-02-10", "2025-03-20")
        assert down["shift_bps"] < 0 < up["shift_bps"]
        for out in (down, up):
            assert 0 < out["share_same_direction_as_large"] <= out["share_as_large"]

    def test_a_constant_series_has_no_shift_and_every_placement_matches_it(self):
        days = pd.date_range("2024-01-01", "2025-12-31", freq="D").strftime("%Y-%m-%d")
        out = pg.level_shift(pd.Series(1.5, index=days), "2025-02-10", "2025-03-20", days=28)
        assert (out["before_bps"], out["after_bps"], out["shift_bps"]) == (1.5, 1.5, 0.0)
        assert out["share_as_large"] == 1.0
        assert out["placements"] == len(days) - 2 * 28 - 37 + 1


class TestEightHourAverages:
    def test_weights_are_the_minute_weights_summed_by_hour(self):
        assert pg.LINEAR_WEIGHTS.tolist() == [1830 + 3600 * h for h in range(8)]
        assert pg.LINEAR_WEIGHTS.sum() == 480 * 481 // 2

    def test_constant_premium_is_its_own_average(self):
        hourly = pd.Series(-4.5, index=_hours("2024-01-01 01:00", 24))
        stamps = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"],
                                                 utc=True))
        out = pg.eight_hour_average(hourly, stamps)
        assert out["value"].tolist() == pytest.approx([-4.5, -4.5])
        assert out["hours"].tolist() == [8, 8]

    def test_later_hours_weigh_more_and_a_missing_hour_is_left_out(self):
        hourly = pd.Series(np.arange(8, dtype=float), index=_hours("2024-01-01 01:00", 8))
        stamp = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 08:00"], utc=True))
        w = pg.LINEAR_WEIGHTS
        expected = (w * np.arange(8)).sum() / w.sum()
        assert pg.eight_hour_average(hourly, stamp)["value"].iloc[0] == pytest.approx(expected)
        out = pg.eight_hour_average(hourly.drop(hourly.index[3]), stamp)
        keep = np.arange(8) != 3
        assert out["value"].iloc[0] == pytest.approx((w * np.arange(8))[keep].sum()
                                                     / w[keep].sum())
        assert out["hours"].iloc[0] == 7

    def test_exchange_blocks_rebuild_the_minute_weighted_average(self):
        rng = np.random.default_rng(11)
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        minutes = pd.DataFrame({
            "open_ms": [int(start.value // 1_000_000) + 60_000 * i for i in range(16 * 60)],
            "close": [f"{v:.8f}" for v in rng.normal(-4.6e-4, 5e-5, 16 * 60)]})
        hourly = fp.hourly_from_minutes(minutes, "2024-01-01 01:00", "2024-01-01 16:00")
        frame = pg.exchange_hourly_frame(hourly)
        stamps = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"],
                                                 utc=True))
        out = pg.exchange_eight_hour(frame, stamps)
        x = minutes["close"].astype(float).to_numpy() * 1e4
        k = np.arange(1, 481)
        expected = [(k * x[:480]).sum() / k.sum(), (k * x[480:]).sum() / k.sum()]
        assert out.tolist() == pytest.approx(expected, abs=1e-5)


class TestWeightingSchedule:
    def test_bybit_switched_to_linear_weights_on_7_march_2024(self):
        stamps = pd.DatetimeIndex(pd.to_datetime(["2024-03-07 08:00", "2024-03-07 16:00",
                                                  "2025-01-01 00:00"], utc=True))
        assert pg.linear_blocks("bybit", stamps).tolist() == [False, True, True]
        assert pg.linear_blocks("binance", stamps).tolist() == [True, True, True]

    def test_plain_weights_give_the_plain_mean(self):
        hourly = pd.Series(np.arange(8, dtype=float), index=_hours("2024-01-01 01:00", 8))
        stamp = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 08:00"], utc=True))
        assert pg.eight_hour_average(hourly, stamp, False)["value"].iloc[0] == pytest.approx(3.5)

    def test_exchange_blocks_rebuild_the_plain_minute_average(self):
        rng = np.random.default_rng(12)
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        x = rng.normal(-4.6e-4, 5e-5, 16 * 60)
        minutes = pd.DataFrame({
            "open_ms": [int(start.value // 1_000_000) + 60_000 * i for i in range(16 * 60)],
            "close": [f"{v:.8f}" for v in x]})
        frame = pg.exchange_hourly_frame(
            fp.hourly_from_minutes(minutes, "2024-01-01 01:00", "2024-01-01 16:00"))
        stamps = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"],
                                                 utc=True))
        out = pg.exchange_eight_hour(frame, stamps, [False, True])
        k = np.arange(1, 481)
        bps = np.round(x, 8) * 1e4
        assert out.tolist() == pytest.approx([bps[:480].mean(),
                                              (k * bps[480:]).sum() / k.sum()], abs=1e-5)


class TestRobustnessBlocks:
    def test_the_first_block_draws_on_the_hours_before_the_rule(self):
        # six pre-rule hours at 10 bps, then zero; the exchange sits at zero throughout
        t = _hours("2024-01-01 17:00", 24)
        start = pd.Timestamp("2024-01-01 23:00", tz="UTC")
        hl_all = pd.Series(np.where(t < start, 10.0, 0.0), index=t)
        hl = pd.DataFrame({"premium": hl_all[hl_all.index >= start]})
        frame = pd.DataFrame({"n": 60, "mean": 0.0, "lwmean": 0.0, "close": 0.0}, index=t)
        stamps = pd.DatetimeIndex(pd.to_datetime(["2024-01-02 00:00", "2024-01-02 08:00"],
                                                 utc=True))
        v = {"frame": frame, "mean": frame["mean"], "close": frame["close"],
             "settlements": pd.DataFrame({"at_floor": [True, True]}, index=stamps)}
        out = pg._robustness(hl, hl_all, v, True)["common"]["eight_hour_blocks"]
        w = pg.LINEAR_WEIGHTS
        first = 10.0 * w[:6].sum() / w.sum()           # hours stamped 17:00 .. 22:00 of eight
        assert out["n"] == 2
        assert out["mean_bps"] == pytest.approx(first / 2)


class TestSettlementFit:
    def test_settlements_from_the_rule_are_reproduced(self):
        rng = np.random.default_rng(5)
        stamps = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00",
                                                  "2024-01-02 00:00"], utc=True))
        averaged = pd.Series([-4.8, 0.3, 7.2], index=stamps)
        rates = [f"{censoring.funding_rule([p])[0] / 1e4:.8f}" for p in averaged]
        settlements = pg.settlement_frame(pd.DataFrame({"time": stamps.astype(str),
                                                        "fundingRate": rates}))
        noisy = averaged + rng.normal(0, 0.01, 3)
        out = pg.settlement_fit(settlements, noisy)
        assert out["n_identified"] == 2 and out["n_compared"] == 2
        assert out["median_abs_error_bps"] < 0.05
        assert out["share_within_0.1bp"] == 1.0
        assert out["floor_agreement"] == 1.0

    def test_recovered_premium_and_floor_flags(self):
        s = pg.settlement_frame(pd.DataFrame({"time": ["2024-01-01 00:00:00+00:00"] * 3,
                                              "fundingRate": ["0.00002500", "0.0001",
                                                              "0.00030000"]}))
        assert s["at_floor"].tolist() == [False, True, False]
        assert s["premium"].iloc[0] == pytest.approx(-4.75)
        assert s["premium"].iloc[2] == pytest.approx(8.0)


class TestBins:
    def test_values_fall_in_right_closed_bins(self):
        x = pd.Series([-6.0, -5.5, -4.0, 0.0, 7.0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = pg.binned_medians(y, x, [-np.inf, -6, -4, 6, np.inf])
        assert [b["n"] for b in out] == [1, 2, 1, 1]
        assert out[1]["median_bps"] == pytest.approx(2.5)

    def test_daily_realised_volatility(self):
        t = _hours("2024-01-01 00:00", 49)                # closes ending 00:00 .. 00:00
        close = pd.Series(np.exp(0.01 * np.arange(49)), index=t)
        out = pg.daily_realised_vol(close)
        assert list(out.index) == ["2024-01-01", "2024-01-02"]
        assert out.tolist() == pytest.approx([100 * 0.01 * np.sqrt(24)] * 2)


class TestUsdcPricing:
    def test_a_book_priced_in_usdc_falls_one_for_one_with_the_basis(self):
        rng = np.random.default_rng(4)
        t = _hours("2024-01-01 01:00", 24 * 200)
        daily = np.repeat(rng.normal(0, 3, 200), 24)
        basis = pd.Series(daily + rng.normal(0, 0.5, len(t)), index=t)
        gap = 2.0 - basis + rng.normal(0, 0.5, len(t))
        out = pg.usdc_pricing_check(gap, basis)
        assert out["days"] == 200
        assert out["slope_daily"] == pytest.approx(-1.0, abs=0.05)
        assert out["intercept_daily_bps"] == pytest.approx(2.0, abs=0.1)
        assert out["pearson_daily"] < -0.95

    def test_a_book_priced_in_usdt_does_not_move_with_the_basis(self):
        rng = np.random.default_rng(6)
        t = _hours("2024-01-01 01:00", 24 * 200)
        basis = pd.Series(np.repeat(rng.normal(0, 3, 200), 24), index=t)
        gap = pd.Series(2.0 + rng.normal(0, 0.5, len(t)), index=t)
        out = pg.usdc_pricing_check(gap, basis)
        assert out["slope_daily"] == pytest.approx(0.0, abs=0.03)
        assert out["r2_daily"] < 0.05


    def test_the_slope_carries_a_hac_standard_error_and_a_within_quarter_slope(self):
        rng = np.random.default_rng(8)
        n = 400
        t = _hours("2024-01-01 01:00", 24 * n)
        # the basis drifts from quarter to quarter and so does the gap, the other way
        quarter = pd.Series(np.arange(n) // 91).repeat(24).to_numpy()
        basis = pd.Series(np.repeat(rng.normal(0, 1, n), 24) + 2.0 * quarter, index=t)
        gap = pd.Series(3.0 - 0.5 * quarter + rng.normal(0, 0.5, len(t)), index=t)
        out = pg.usdc_pricing_check(gap, basis)
        assert out["hac_lags"] == pg.USDC_HAC_LAGS == 20
        assert out["slope_daily"] < -0.15                      # driven by the drift
        assert out["slope_hac_se"] > 0
        lo, hi = out["slope_ci95"]
        assert lo == pytest.approx(out["slope_daily"] - 1.96 * out["slope_hac_se"], rel=1e-3)
        assert hi == pytest.approx(out["slope_daily"] + 1.96 * out["slope_hac_se"], rel=1e-3)
        w = out["within_quarter"]
        assert w["slope"] == pytest.approx(0.0, abs=0.05)      # no link within quarters
        assert w["slope_hac_se"] > 0


class TestDecomposition:
    @pytest.fixture
    def synthetic(self):
        rng = np.random.default_rng(1)
        t = _hours("2024-01-01 01:00", 24 * 60)
        x = pd.Series(rng.normal(-4.6, 0.6, len(t)), index=t)
        hl = x + 1.5 + rng.normal(0, 1.0, len(t))
        stamps = t[(t.hour % 8 == 0)]
        stamps = stamps[stamps >= t[0] + pd.Timedelta(hours=7)]
        x8 = pg.eight_hour_average(x, stamps)["value"]
        rates = [f"{censoring.funding_rule([p])[0] / 1e4:.8f}" for p in x8]
        settle = pg.settlement_frame(pd.DataFrame({"time": stamps.astype(str),
                                                   "fundingRate": rates}))
        return hl, x, x8, settle

    def test_shapley_parts_add_up_to_the_difference(self, synthetic):
        out = pg.decompose(*synthetic)
        parts = sum(out["shapley"][f] for f in pg.FACTORS)
        assert parts == pytest.approx(out["difference"], abs=1e-12)
        s = out["shares"]
        assert out["difference"] == pytest.approx(s["a_hl_hourly"] - s["d_exchange_settlements"])

    def test_shapley_of_an_additive_cube_is_each_factors_effect(self):
        effect = {"location": 0.3, "averaging": -0.05, "dispersion": 0.1}
        corners = {(i, j, k): 1.0 - 0.3 * i + 0.05 * j - 0.1 * k
                   for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        out = pg.shapley(corners)
        for f, v in effect.items():
            assert out[f] == pytest.approx(v)
            assert out["range"][f] == pytest.approx([v, v])

    def test_an_interaction_is_shared_and_shows_in_the_range(self):
        # only switching both location and dispersion changes the share
        corners = {(i, j, k): 1.0 - 0.4 * (i * k) for i in (0, 1) for j in (0, 1)
                   for k in (0, 1)}
        out = pg.shapley(corners)
        assert (out["location"], out["dispersion"], out["averaging"]) == pytest.approx(
            (0.2, 0.2, 0.0))
        assert out["range"]["location"] == pytest.approx([0.0, 0.4])

    def test_shift_by_the_gap_moves_hyperliquid_onto_the_exchange(self, synthetic):
        out = pg.decompose(*synthetic)
        assert out["gap_by_quarter"]["2024Q1"] == pytest.approx(1.5, abs=0.1)
        assert out["contrasts"]["shift_hl_hourly"] > 0.1     # the shift takes hours out of the band
        assert out["shapley"]["location"] > 0.1

    def test_the_path_is_one_order_of_the_split_and_adds_up(self, synthetic):
        out = pg.decompose(*synthetic)
        path, s = out["path"], out["shares"]
        assert sum(path[f] for f in pg.FACTORS) == pytest.approx(out["difference"], abs=1e-12)
        assert path["location"] == pytest.approx(s["a_hl_hourly"] - s["b_hl_hourly_shifted"])
        assert path["averaging"] == pytest.approx(s["b_hl_hourly_shifted"]
                                                  - s["e_hl_eight_hour_shifted"])
        for f in pg.FACTORS:
            lo, hi = out["shapley"]["range"][f]
            assert lo - 1e-12 <= path[f] <= hi + 1e-12

    def test_contrasts_are_single_comparisons_that_need_not_add_up(self, synthetic):
        out = pg.decompose(*synthetic)
        c, s = out["contrasts"], out["shares"]
        assert set(c) == {"shift_hl_hourly", "shift_exchange_eight_hour", "average_hl",
                          "remainder"}
        assert c["shift_exchange_eight_hour"] == pytest.approx(
            s["h_exchange_eight_hour_shifted"] - s["d_exchange_settlements"])
        assert c["average_hl"] == pytest.approx(s["a_hl_hourly"] - s["c_hl_eight_hour"])
        assert c["remainder"] == pytest.approx(s["e_hl_eight_hour_shifted"]
                                               - s["d_exchange_settlements"])
        # location measured at the exchange's narrow spread exceeds it at Hyperliquid's
        assert c["shift_exchange_eight_hour"] > c["shift_hl_hourly"]

    def test_the_rebuilt_exchange_average_sits_where_its_settlements_do(self, synthetic):
        s = pg.decompose(*synthetic)["shares"]
        assert s["exchange_eight_hour"] == pytest.approx(s["d_exchange_settlements"], abs=0.01)

    def test_without_a_gap_only_dispersion_and_averaging_remain(self, synthetic):
        _, x, x8, settle = synthetic
        out = pg.decompose(x, x, x8, settle)
        assert out["shapley"]["location"] == pytest.approx(0.0)
        assert out["shapley"]["dispersion"] == pytest.approx(0.0, abs=0.02)

    def test_a_span_restricts_the_shares_but_not_the_gaps(self, synthetic, monkeypatch):
        monkeypatch.setitem(pg.SPANS, "ten days", ("2024-01-10 01:00", "2024-01-20 00:00"))
        full = pg.decompose(*synthetic)
        part = pg.decompose(*synthetic, span="ten days")
        assert (part["n_hours"], part["n_blocks"]) == (240, 30)
        assert part["gap_by_quarter"] == full["gap_by_quarter"]
        hl = synthetic[0]
        inside = hl[(hl.index >= pd.Timestamp("2024-01-10 01:00", tz="UTC"))
                    & (hl.index <= pd.Timestamp("2024-01-20 00:00", tz="UTC"))]
        assert part["shares"]["a_hl_hourly"] == pytest.approx(pg.in_band(inside).mean())

    def test_phase_follows_where_the_exchange_median_sits(self):
        t = _hours("2024-01-01 01:00", 24 * 366)
        quarter = np.asarray(pg.accrual_period(t, "Q"))
        level = pd.Series(quarter, index=t).map({"2024Q1": 7.0, "2024Q2": -3.0,
                                                 "2024Q3": -4.5, "2024Q4": 0.0}).fillna(2.0)
        assert pg.quarter_phases(level) == {"2024Q1": "positive", "2024Q2": "inside_nonpositive",
                                            "2024Q3": "below", "2024Q4": "inside_nonpositive"}
        assert [pg.phase_of(m) for m in (-4.0, -3.99, 0.0, 0.01, 6.5)] == [
            "below", "inside_nonpositive", "inside_nonpositive", "positive", "positive"]

    def test_phase_groups_split_the_below_band_quarters_early_and_late(self):
        phases = {"2023Q4": "positive", "2024Q1": "positive", "2024Q2": "inside_nonpositive",
                  "2024Q3": "below", "2024Q4": "inside_nonpositive", "2025Q1": "below",
                  "2025Q4": "below", "2026Q1": "below", "2026Q3": "below"}
        out = pg.phase_groups(phases)
        assert out == {"below": ["2024Q3", "2025Q1", "2025Q4", "2026Q1", "2026Q3"],
                       "inside_nonpositive": ["2024Q2", "2024Q4"],
                       "positive": ["2023Q4", "2024Q1"],
                       "below_2024Q3_2025Q4": ["2024Q3", "2025Q1", "2025Q4"],
                       "below_2026": ["2026Q1", "2026Q3"]}


# ----------------------------------------------------------------------
#  Shipped data
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def payload():
    return pg.build_payload()


class TestShippedData:
    def test_hyperliquid_btc_equals_the_interim_series_on_the_overlap(self, payload):
        check = payload["data"]["BTC"]["hyperliquid"]["interim_overlap"]
        assert check["hours"] == check["premiums_equal"] == check["rates_equal"] == 8737

    def test_spans_follow_the_common_rule(self, payload):
        for coin in ("BTC", "ETH"):
            d = payload["data"][coin]["hyperliquid"]
            assert (d["hours"], d["missing_hours"]) == (24577, ["2024-08-15 13:00:00+00:00"])
            assert d["first"] == "2023-12-11 17:00:00+00:00"        # six hours of pre-roll
            for venue in ("binance", "bybit"):
                v = payload["data"][coin][venue]
                assert (v["settlements"], v["spacing_hours"]) == (3073, [8])
                assert v["hours"] - v["complete_hours"] == {"binance": 1, "bybit": 0}[venue]

    def test_hyperliquid_floor_hours_are_its_in_band_hours(self, payload):
        counts = {"BTC": 13578, "ETH": 12768}
        for coin, n in counts.items():
            d = payload["data"][coin]["hyperliquid"]
            assert d["floor_hours"] == d["in_band_hours"] == n

    def test_exchange_settlements_match_the_censoring_artifact(self, payload):
        shipped = censoring.build_payload()["venues"]
        for venue in ("binance", "bybit"):
            assert payload["data"]["BTC"][venue]["interim_overlap"] == {"settlements": 1096,
                                                                        "rates_equal": 1096}
            for window in ("paper", "later"):
                mine = payload["windows"]["BTC"][venue][window]
                assert mine["settlements_at_floor"] == shipped[venue][window]["at_floor"]
                assert mine["settlements"] == shipped[venue][window]["n"]
        assert payload["windows"]["BTC"]["binance"]["paper"]["hl_floor_share"] == (
            shipped["hyperliquid"]["paper"]["share_at_floor"])

    def test_hourly_closes_equal_the_shipped_hourly_klines(self, payload):
        for venue in ("binance", "bybit"):
            v = payload["validation"]["hourly_close_vs_interim"][venue]
            assert v["hours"] == v["equal"] == 5001

    def test_settlements_are_reproduced_from_the_minutes(self, payload):
        fits = payload["validation"]["settlements"]
        for coin in ("BTC", "ETH"):
            b, y = fits[coin]["binance"], fits[coin]["bybit"]
            assert b["median_abs_error_bps"] < 0.05 and b["floor_agreement"] > 0.98
            assert y["median_abs_error_bps"] < 0.005 and y["share_within_0.1bp"] == 1.0
            # before 7 Mar 2024 16:00 Bybit settled on the plain eight-hour average
            early = y["before_switch"]
            assert early["plain"]["share_within_0.1bp"] == 1.0
            assert early["linear"]["share_within_0.1bp"] < 0.25
        assert round(fits["BTC"]["binance"]["median_abs_error_bps"], 4) == 0.0285
        assert round(fits["BTC"]["binance"]["share_within_0.1bp"], 3) == 0.928

    def test_the_gap_is_positive_in_every_btc_quarter_and_month(self, payload):
        for venue in ("binance", "bybit"):
            g = payload["gap"]["BTC"][venue]
            assert all(q["median_bps"] > 0 for q in g["by_quarter"].values())
            assert all(m["median_bps"] > 0 for m in payload["monthly"]["BTC"][venue].values())

    def test_eth_turned_negative_in_april_2025(self, payload):
        for venue in ("binance", "bybit"):
            months = payload["monthly"]["ETH"][venue]
            assert [m for m, d in months.items() if d["median_bps"] < 0] == ["2025-04"]
            assert payload["persistence"]["ETH"][venue]["negative_days"]["longest_start"] == (
                "2025-04-03")

    def test_gap_levels(self, payload):
        medians = {(c, v): round(payload["gap"][c][v]["common"]["median_bps"], 2)
                   for c in ("BTC", "ETH") for v in ("binance", "bybit")}
        assert medians == {("BTC", "binance"): 2.19, ("BTC", "bybit"): 2.20,
                           ("ETH", "binance"): 1.69, ("ETH", "bybit"): 1.67}
        paper = {(c, v): round(payload["windows"][c][v]["paper"]["gap"]["median_bps"], 2)
                 for c in ("BTC", "ETH") for v in ("binance", "bybit")}
        assert paper == {("BTC", "binance"): 0.81, ("BTC", "bybit"): 0.75,
                         ("ETH", "binance"): 0.93, ("ETH", "bybit"): 0.85}

    def test_decomposition_adds_up_and_location_dominates_below_the_band(self, payload):
        for coin in ("BTC", "ETH"):
            for venue in ("binance", "bybit"):
                dec = payload["decomposition"][coin][venue]
                for sample in ("all", *pg.PHASES, *pg.SUBPERIODS, "paper", "later"):
                    r = dec[sample]
                    for split in (r["shapley"], r["path"]):
                        assert sum(split[f] for f in pg.FACTORS) == pytest.approx(
                            r["difference"], abs=1e-12)
                below = dec["below"]
                assert below["shapley"]["location"] > 0.8 * below["difference"] > 0.3
        btc = payload["decomposition"]["BTC"]["binance"]["all"]
        assert [round(100 * btc[k], 1) for k in ("difference",)] == [31.5]
        assert [round(100 * btc["shapley"][f], 1) for f in pg.FACTORS] == [28.7, 1.6, 1.2]

    def test_artifact_matches_the_shipped_file(self, payload):
        import json
        shipped = json.loads(pg.ARTIFACT.read_text())
        assert_json_close(json.loads(json.dumps(payload)), shipped)

    def test_floor_counts_compare_as_decimals(self):
        assert pg.FLOOR_8H == Decimal("0.0001") and pg.FLOOR_HOURLY == Decimal("0.0000125")


def _pp(x):
    return round(100 * x, 2)


def _group(payload, coin, venue, name):
    r = payload["decomposition"][coin][venue][name]
    sh = r["shapley"]
    return (_pp(r["shares"]["a_hl_hourly"]), _pp(r["shares"]["d_exchange_settlements"]),
            _pp(r["difference"]), _pp(sh["location"]), _pp(sh["averaging"]),
            _pp(sh["dispersion"]), [_pp(x) for x in sh["range"]["location"]])


PAIRS = [(c, v) for c in ("BTC", "ETH") for v in ("binance", "bybit")]


class TestShippedPhases:
    """Where the exchanges' median premium sat, and the split in each group of quarters."""

    def test_quarters_of_each_phase(self, payload):
        below_bn = ["2024Q3", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2",
                    "2026Q3"]
        expected = {
            "binance": {"below": below_bn, "inside_nonpositive": ["2024Q2", "2024Q4"],
                        "positive": ["2023Q4", "2024Q1"]},
            "bybit": {"below": [q for q in below_bn if q != "2025Q3"],
                      "inside_nonpositive": ["2024Q2", "2024Q4", "2025Q3"],
                      "positive": ["2023Q4", "2024Q1"]}}
        for coin, venue in PAIRS:
            dec = payload["decomposition"][coin][venue]
            assert {ph: dec[ph]["quarters"] for ph in pg.PHASES} == expected[venue]
            assert dec["below_2024Q3_2025Q4"]["quarters"] == [
                q for q in expected[venue]["below"] if q < "2026"]
            assert dec["below_2026"]["quarters"] == ["2026Q1", "2026Q2", "2026Q3"]
        # 2023Q4 (from 11 Dec) sat above the band, 2024Q1 inside it with a positive median
        med = payload["decomposition"]["BTC"]["binance"]["exchange_median_bps"]
        assert (round(med["2023Q4"], 2), round(med["2024Q1"], 2)) == (6.71, 5.27)
        assert payload["partial_quarters"] == {"2023Q4": pg.PARTIAL_QUARTERS["2023Q4"]}

    def test_positive_premium_quarters_reverse_the_ordering(self, payload):
        # (HL %, exchange %, difference, Shapley location, averaging, dispersion, location range)
        assert {pair: _group(payload, *pair, "positive") for pair in PAIRS} == {
            ("BTC", "binance"): (31.66, 44.91, -13.25, -8.78, 1.01, -5.48, [-11.44, -6.59]),
            ("BTC", "bybit"): (31.66, 54.49, -22.83, -12.37, -0.08, -10.38, [-13.77, -10.18]),
            ("ETH", "binance"): (31.66, 43.11, -11.46, -7.83, 1.32, -4.95, [-12.64, -3.29]),
            ("ETH", "bybit"): (31.66, 47.6, -15.95, -10.87, 0.37, -5.44, [-13.17, -8.68])}
        # location lowers Hyperliquid's share in every order of the split
        for pair in PAIRS:
            assert _group(payload, *pair, "positive")[6][1] < 0

    def test_the_reversal_holds_in_seven_of_eight_pair_quarters(self, payload):
        reversed_ = {pair: payload["decomposition"][pair[0]][pair[1]]["positive"]
                     ["reversed_quarters"] for pair in PAIRS}
        assert reversed_ == {("BTC", "binance"): ["2023Q4", "2024Q1"],
                             ("BTC", "bybit"): ["2024Q1"],
                             ("ETH", "binance"): ["2023Q4", "2024Q1"],
                             ("ETH", "bybit"): ["2023Q4", "2024Q1"]}
        q = payload["decomposition"]["BTC"]["bybit"]["by_quarter"]["2023Q4"]
        assert (_pp(q["difference"]), _pp(q["shapley"]["location"])) == (0.7, -16.22)

    def test_inside_band_quarters_with_a_nonpositive_median(self, payload):
        assert {pair: _group(payload, *pair, "inside_nonpositive")[:3] for pair in PAIRS} == {
            ("BTC", "binance"): (58.31, 53.01, 5.31), ("BTC", "bybit"): (62.61, 47.39, 15.21),
            ("ETH", "binance"): (51.32, 49.18, 2.14), ("ETH", "bybit"): (52.8, 48.73, 4.08)}

    def test_below_band_quarters_pooled_early_and_late(self, payload):
        expected = {
            "below": {("BTC", "binance"): (58.07, 13.2, 44.87, 41.61, 2.1, 1.17, [19.32, 65.2]),
                      ("BTC", "bybit"): (56.18, 13.9, 42.28, 39.52, 1.58, 1.19, [18.59, 61.34])},
            "below_2024Q3_2025Q4": {
                ("BTC", "binance"): (67.1, 17.07, 50.03, 51.81, 1.44, -3.22, [22.91, 82.64]),
                ("BTC", "bybit"): (66.08, 19.54, 46.54, 50.12, 1.2, -4.78, [21.99, 79.91])},
            "below_2026": {
                ("BTC", "binance"): (42.95, 6.72, 36.23, 24.53, 3.2, 8.51, [13.29, 36.02]),
                ("BTC", "bybit"): (42.95, 6.35, 36.6, 25.34, 2.09, 9.17, [14.04, 36.51])}}
        for name, rows in expected.items():
            for pair, row in rows.items():
                assert _group(payload, *pair, name) == row
        # ETH: location carries the 2024Q3-2025Q4 difference too, and most of 2026's
        for venue in ("binance", "bybit"):
            assert _group(payload, "ETH", venue, "below_2026")[3:4] == (
                {"binance": (34.89,), "bybit": (31.71,)}[venue])

    def test_location_dominates_until_2025_and_in_2026Q3_not_in_2026Q1_Q2(self, payload):
        def row(venue, q):
            r = payload["decomposition"]["BTC"][venue]["by_quarter"][q]
            return (_pp(r["difference"]), _pp(r["shapley"]["location"]),
                    _pp(r["shapley"]["averaging"]), _pp(r["shapley"]["dispersion"]))
        assert row("binance", "2025Q2") == (56.96, 56.51, 0.95, -0.5)
        assert row("binance", "2025Q4") == (52.85, 53.11, 2.84, -3.09)
        assert row("binance", "2026Q1") == (35.32, 15.36, 2.82, 17.15)
        assert row("binance", "2026Q2") == (22.99, 6.05, 3.51, 13.42)
        assert row("binance", "2026Q3") == (50.23, 51.77, 3.26, -4.8)
        assert row("bybit", "2026Q1") == (31.62, 11.84, 2.49, 17.28)
        assert row("bybit", "2026Q2") == (23.35, 7.38, 2.23, 13.74)
        # the quarter's median gap G was 0.3-0.5 bp in 2026Q1-Q2 and 1.3 bp in 2026Q3
        g = payload["decomposition"]["BTC"]["binance"]["all"]["gap_by_quarter"]
        assert [round(g[q], 2) for q in ("2026Q1", "2026Q2", "2026Q3")] == [0.52, 0.29, 1.3]

    def test_path_and_contrasts(self, payload):
        dec = payload["decomposition"]["BTC"]["binance"]
        path = {k: _pp(v) for k, v in dec["all"]["path"].items()}
        assert path == {"location": 10.32, "averaging": -0.14, "dispersion": 21.31}
        c = {k: _pp(v) for k, v in dec["all"]["contrasts"].items()}
        assert c == {"shift_hl_hourly": 10.32, "shift_exchange_eight_hour": 48.15,
                     "average_hl": -1.31, "remainder": 21.31}
        # the single comparisons a - b, a - c and e - d do not add up to the difference (31.49)
        assert c["shift_hl_hourly"] + c["average_hl"] + c["remainder"] == pytest.approx(30.32)
        below = {k: _pp(v) for k, v in dec["below"]["contrasts"].items()}
        assert (below["shift_hl_hourly"], below["shift_exchange_eight_hour"]) == (19.32, 65.2)

    def test_the_paper_window_split(self, payload):
        assert _group(payload, "BTC", "binance", "paper") == (
            45.99, 5.91, 40.08, 30.25, 2.6, 7.23, [14.7, 46.96])


class TestShippedPersistence:
    def test_monthly_medians_near_zero_are_not_distinguishable_from_it(self, payload):
        """One-week blocks (the headline); the daily gap has a half-life of 2-3 days."""
        ci = {pair: payload["median_ci"][pair[0]][pair[1]] for pair in PAIRS}
        assert all(c["block_hours"] == 168 for c in ci.values())
        assert {pair: c["months_covering_zero"] for pair, c in ci.items()} == {
            ("BTC", "binance"): ["2024-02", "2024-03", "2026-01", "2026-04", "2026-05"],
            ("BTC", "bybit"): ["2023-12", "2025-03", "2026-04", "2026-05"],
            ("ETH", "binance"): ["2024-03", "2024-10", "2025-03", "2026-04"],
            ("ETH", "bybit"): ["2024-03", "2025-02", "2025-03", "2026-03", "2026-04"]}
        # full months only: the partial December 2023 is listed apart
        assert {pair: (len(c["full_months_covering_zero"]), c["partial_months_covering_zero"])
                for pair, c in ci.items()} == {
            ("BTC", "binance"): (5, []), ("BTC", "bybit"): (3, ["2023-12"]),
            ("ETH", "binance"): (4, []), ("ETH", "bybit"): (5, [])}
        # every full quarter excludes zero; only the partial 2023Q4 (from 11 Dec) may not
        assert all(c["full_quarters_covering_zero"] == [] for c in ci.values())
        assert ci[("BTC", "bybit")]["partial_quarters_covering_zero"] == ["2023Q4"]
        low = {pair: round(c["lowest_full_quarter_lower_bound"], 3) for pair, c in ci.items()}
        assert low == {("BTC", "binance"): 0.13, ("BTC", "bybit"): 0.182,
                       ("ETH", "binance"): 0.086, ("ETH", "bybit"): 0.049}
        lo, hi = ci[("BTC", "binance")]["monthly"]["2026-04"]
        assert (round(lo, 3), round(hi, 3)) == (-0.079, 0.645)
        assert payload["bootstrap"] == pg.BOOTSTRAP

    def test_block_length_sensitivity(self, payload):
        ci = {pair: payload["median_ci"][pair[0]][pair[1]] for pair in PAIRS}
        # the 24-hour blocks reproduce the earlier headline intervals
        assert {pair: c["block_sensitivity"]["24"]["months_covering_zero"]
                for pair, c in ci.items()} == {
            ("BTC", "binance"): ["2026-04", "2026-05"],
            ("BTC", "bybit"): ["2026-02", "2026-03", "2026-04", "2026-05"],
            ("ETH", "binance"): ["2024-03", "2025-03", "2026-04"],
            ("ETH", "bybit"): ["2025-03", "2026-03"]}
        assert all(c["block_sensitivity"]["24"]["quarters_covering_zero"] == []
                   for c in ci.values())
        q72 = {pair: c["block_sensitivity"]["72"]["quarters_covering_zero"]
               for pair, c in ci.items()}
        assert q72 == {("BTC", "binance"): ["2023Q4"], ("BTC", "bybit"): ["2023Q4"],
                       ("ETH", "binance"): [], ("ETH", "bybit"): []}
        assert ci[("BTC", "bybit")]["coverage_change_vs_24h"] == {
            "monthly": {"now_covering_zero": ["2023-12", "2025-03"],
                        "no_longer_covering_zero": ["2026-02", "2026-03"]},
            "quarterly": {"now_covering_zero": ["2023Q4"], "no_longer_covering_zero": []}}

    def test_deviations_halve_within_two_to_three_days(self, payload):
        per = {pair: payload["persistence"][pair[0]][pair[1]] for pair in PAIRS}
        raw = {pair: round(p["ar1_daily"]["half_life"], 2) for pair, p in per.items()}
        within = {pair: round(p["ar1_daily_within_quarter"]["half_life"], 2)
                  for pair, p in per.items()}
        acf30 = {pair: round(p["acf_daily"]["30"], 3) for pair, p in per.items()}
        assert raw == {("BTC", "binance"): 2.8, ("BTC", "bybit"): 2.58,
                       ("ETH", "binance"): 2.56, ("ETH", "bybit"): 2.45}
        assert within == {("BTC", "binance"): 2.13, ("BTC", "bybit"): 1.96,
                          ("ETH", "binance"): 2.12, ("ETH", "bybit"): 2.0}
        assert acf30 == {("BTC", "binance"): 0.21, ("BTC", "bybit"): 0.22,
                         ("ETH", "binance"): 0.109, ("ETH", "bybit"): 0.135}

    def test_eth_was_negative_for_about_a_month_in_spring_2025(self, payload):
        bn, by = (payload["persistence"]["ETH"][v] for v in ("binance", "bybit"))
        april = {"start": "2025-03-31", "end": "2025-04-21", "length": 4}   # weeks to 27 Apr
        assert bn["negative_week_spells"] == [april]
        assert by["negative_week_spells"] == [
            {"start": "2025-02-17", "end": "2025-03-03", "length": 3}, april]
        assert (bn["negative_days"]["longest_days"], by["negative_days"]["longest_days"]) == (
            11, 16)
        assert {"start": "2025-02-21", "end": "2025-02-26", "length": 6} in (
            by["negative_day_spells"])
        roll = [(p["rolling_28d"]["negative_windows"], round(p["rolling_28d"]["min_bps"], 2),
                 p["rolling_28d"]["min_end_day"]) for p in (bn, by)]
        assert roll == [(26, -0.39, "2025-04-27"), (41, -0.55, "2025-04-27")]

    def test_btc_never_had_a_negative_four_week_mean(self, payload):
        for venue, low in (("binance", 0.185), ("bybit", 0.095)):
            r = payload["persistence"]["BTC"][venue]["rolling_28d"]
            assert (r["negative_windows"], round(r["min_bps"], 3)) == (0, low)
            assert payload["persistence"]["BTC"][venue]["negative_week_spells"] == []


class TestShippedPhaseByYear:
    def test_within_2024_the_gap_is_larger_below_the_band_than_inside_it(self, payload):
        """The pooled contrast (inside > below) reverses within 2024 for every pair."""
        med = {pair: {y: {k: v.get("median_bps") for k, v in pos.items()}
                      for y, pos in payload["phase"][pair[0]][pair[1]][
                          "by_band_position_by_year"].items()}
               for pair in PAIRS}
        for pair, m in med.items():
            assert m["2024"]["below"] > m["2024"]["inside"], pair
        btc = {v: (round(med[("BTC", v)]["2024"]["below"], 2),
                   round(med[("BTC", v)]["2024"]["inside"], 2)) for v in ("binance", "bybit")}
        assert btc == {"binance": (5.05, 3.96), "bybit": (5.14, 3.92)}

    def test_the_years_partition_the_pooled_hours(self, payload):
        for pair in PAIRS:
            ph = payload["phase"][pair[0]][pair[1]]
            for k, pooled in ph["by_band_position"].items():
                n = sum(y[k]["n"] for y in ph["by_band_position_by_year"].values())
                assert n == pooled["n"], (pair, k)


class TestHyperliquidPremiumAr1:
    def test_the_estimation_window_half_life_is_that_of_the_msar_series(self, payload):
        from regime_engine.panel import load_paper_window_premium
        ar1 = payload["hyperliquid_premium_ar1"]["BTC"]
        y = pd.Series(np.asarray(load_paper_window_premium(), dtype=float))
        assert ar1["paper"] == pg.ar1_half_life(y)
        assert ar1["paper"]["n"] == 5000
        assert round(ar1["paper"]["half_life"], 1) == 14.5
        others = [round(ar1[w]["half_life"], 1) for w in ("common", "before_paper", "later")]
        assert (min(others), max(others)) == (13.8, 22.5)


class TestShippedChecks:
    def test_traders_price_hyperliquid_in_usdt_terms(self, payload):
        slopes = {pair: round(payload["checks"][pair[0]][pair[1]]["usdc_pricing"]
                              ["slope_daily"], 3) for pair in PAIRS}
        assert slopes == {("BTC", "binance"): -0.125, ("BTC", "bybit"): -0.061,
                          ("ETH", "binance"): -0.068, ("ETH", "bybit"): -0.025}
        assert max(payload["checks"][c][v]["usdc_pricing"]["r2_daily"] for c, v in PAIRS) < 0.04
        # far from -1 everywhere; BTC-Binance's small slope differs from zero over the
        # full sample but vanishes within quarters
        u = {pair: payload["checks"][pair[0]][pair[1]]["usdc_pricing"] for pair in PAIRS}
        bn = u[("BTC", "binance")]
        assert round(bn["slope_hac_se"], 3) == 0.048
        assert [round(x, 2) for x in bn["slope_ci95"]] == [-0.22, -0.03]
        for pair, c in u.items():
            assert c["slope_ci95"][0] > -0.5, pair
            assert abs(c["within_quarter"]["slope"]) < 0.05, pair
            assert c["within_quarter"]["slope_ci95"][0] < 0 < c["within_quarter"]["slope_ci95"][1]

    def test_the_gap_across_the_oracle_wording_change(self, payload):
        out = {pair: payload["checks"][pair[0]][pair[1]]["oracle_wording_shift"]
               for pair in PAIRS}
        summary = {pair: (round(o["before_bps"], 2), round(o["after_bps"], 2),
                          round(o["shift_bps"], 2), round(o["share_as_large"], 3))
                   for pair, o in out.items()}
        assert summary == {("BTC", "binance"): (4.21, 0.99, -3.21, 0.118),
                           ("BTC", "bybit"): (4.22, 1.17, -3.06, 0.166),
                           ("ETH", "binance"): (4.58, 2.47, -2.11, 0.435),
                           ("ETH", "bybit"): (4.0, 2.66, -1.34, 0.618)}
        # share_as_large is two-sided; a fall at least as large is rarer
        falls = {pair: round(o["share_same_direction_as_large"], 3) for pair, o in out.items()}
        assert falls == {("BTC", "binance"): 0.086, ("BTC", "bybit"): 0.115,
                         ("ETH", "binance"): 0.214, ("ETH", "bybit"): 0.335}
        btc = out[("BTC", "binance")]
        assert (btc["between_days"], btc["placements"]) == (64, 905)

    def test_eight_hour_blocks_use_the_pre_roll(self, payload):
        r = {pair: payload["robustness"][pair[0]][pair[1]]["common"]["eight_hour_blocks"]
             for pair in PAIRS}
        assert {pair: round(x["mean_bps"], 4) for pair, x in r.items()} == {
            ("BTC", "binance"): 3.2899, ("BTC", "bybit"): 3.1831,
            ("ETH", "binance"): 2.806, ("ETH", "bybit"): 2.6615}
        assert {pair: x["n"] for pair, x in r.items()} == {
            ("BTC", "binance"): 3072, ("BTC", "bybit"): 3073,
            ("ETH", "binance"): 3072, ("ETH", "bybit"): 3073}


class TestTables:
    def test_table_12_regenerates_from_the_artifact(self, payload):
        assert pg.write_table(payload) == pg.PAPER_TABLE.read_text()
        assert pg.PAPER_TABLE.read_text() == pg.OUTPUT_TABLE.read_text()

    def test_table_12_names_the_estimation_window_as_the_paper_does(self, payload):
        tex = pg.write_table(payload) + pg.write_group_table(payload)
        assert " & Estim. & Hours $>0$" in tex
        # the caption explains the columns; Panel A carries no note row of its own
        assert "Estim.: " not in tex and "2023: from 11 Dec" not in tex
        assert "\nestimation window & 46.0 & 5.9 &" in tex
        assert "Paper" not in tex and "paper window" not in tex

    def test_table_12_prints_the_gap_and_the_groups_with_their_ranges(self, payload):
        """Two panels; the quarter-by-quarter split, with ranges, is the quarter table."""
        tex = pg.write_table(payload)
        # both panels at the line width, so that their rules have one length
        assert tex.count("\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}") == 2
        assert "\\begin{tabular}" not in tex
        assert "$\\tilde P_X\\le-4$ (8) & 58.1 & 13.2 & 44.9 & 41.6 & [19.3, 65.2]" in tex
        assert "$\\tilde P_X>0$ (2) & 31.7 & 44.9 & $-$13.3 & $-$8.8 & [$-$11.4, $-$6.6]" in tex
        assert "\\quad below band, 2026 (3) & 42.9 & 6.7 & 36.2 & 24.5 & [13.3, 36.0]" in tex
        assert "\\quad below band, 2024Q3--2025Q4 (5) & " in tex
        assert "2026Q1" not in tex and "C. BTC by quarter" not in tex
        quarters = pg.write_quarter_table(payload)
        assert "2026Q1 & $-$4.84 & 36.8 & 1.5 & 35.3 & 15.4 &" in quarters

    def test_appendix_tables_regenerate_from_the_artifact(self, payload):
        for stem, text in ((pg.ETH_GROUPS_STEM, pg.write_group_table(payload)),
                           (pg.QUARTERS_STEM, pg.write_quarter_table(payload))):
            for folder in (pg.OUTPUT_TABLE.parent, pg.PAPER_TABLE.parent):
                assert (folder / stem).read_text() == text, stem

    def test_eth_groups_table_is_panel_b_for_eth(self, payload):
        tex = pg.write_group_table(payload)
        assert pg.DETAIL_ASSETS == ("ETH",)
        assert "ETH, Hyperliquid against Binance" in tex and "BTC," not in tex
        assert "$\\tilde P_X>0$ (2) & 31.7 & 43.1 & $-$11.5 & $-$7.8 & [$-$12.6, $-$3.3]" in tex

    def test_quarter_table_gives_every_quarter_with_its_ranges(self, payload):
        tex = pg.write_quarter_table(payload)
        assert "ETH, Hyperliquid against Bybit" in tex and "BTC, Hyperliquid against Binance" in tex
        r = payload["decomposition"]["BTC"]["binance"]["by_quarter"]["2026Q1"]
        lo, hi = r["shapley"]["range"]["location"]
        dlo, dhi = r["shapley"]["range"]["dispersion"]
        row = (f"2026Q1 & $-$4.84 & 36.8 & 1.5 & 35.3 & 15.4 & [{100 * lo:.1f}, {100 * hi:.1f}] "
               f"& 2.8 & 17.1 & [{100 * dlo:.1f}, {100 * dhi:.1f}] \\\\")
        assert row in tex
        assert "[9.8, 22.3]" in row and "[7.8, 25.2]" in row
        n_quarters = len(payload["decomposition"]["BTC"]["binance"]["by_quarter"])
        assert sum(line.startswith(("2023Q", "2024Q", "2025Q", "2026Q"))
                   for line in tex.splitlines()) == 4 * n_quarters

    def test_table_13_bybit_weights_cover_all_three_snapshots(self):
        ref = dict((r[0], r[1:]) for r in pg.CONSTRUCTION)["Reference price"][2]
        for rng in ("Binance 0.339--0.348", "Bybit 0.220--0.234", "Gate 0.215--0.219",
                    "OKX 0.138--0.147", "KuCoin 0.070--0.071", "Coinbase 0.001"):
            assert rng in ref
        notes = " ".join(pg.CONSTRUCTION_NOTES)
        assert "three snapshots between 16:29 and 18:29 UTC" in notes

    def test_table_13_wording(self):
        rows = dict((r[0], r[1:]) for r in pg.CONSTRUCTION)
        assert rows["Changes since 2023"][1] == "none seen in the settlements from Dec 2023 on"
        cap = rows["Cap"][2]
        assert "\\min(" not in cap and "at most" in cap

    def test_table_13_regenerates(self):
        text = pg.write_construction_table()
        for folder in (pg.OUTPUT_TABLE.parent, pg.PAPER_TABLE.parent):
            assert (folder / pg.CONSTRUCTION_STEM).read_text() == text
        assert sum(pg.CONSTRUCTION_WIDTHS) == pytest.approx(1.0)

    def test_table_13_has_the_documented_rows(self):
        assert [r[0] for r in pg.CONSTRUCTION] == [
            "Premium", "Sampling", "Averaging", "Reference price", "Impact notional",
            "Interest $r$", "Clamp $c$", "Cap", "Settlement", "Margin, quote",
            "Changes since 2023"]
        assert all(len(r) == 4 for r in pg.CONSTRUCTION)

    def test_table_13_states_the_quanto_construction(self):
        hl, bn, by = dict((r[0], r[1:]) for r in pg.CONSTRUCTION)["Margin, quote"]
        assert "USDC margin" in hl and "oracle in USDT" in hl and "quanto" in hl
        assert "USDC" not in bn + by and "USDT" in bn and "USDT" in by

    def test_table_13_dates_agree_with_the_code(self):
        rows = dict((r[0], r[1:]) for r in pg.CONSTRUCTION)
        assert pg.FIRST_LINEAR_SETTLEMENT["bybit"] == "2024-03-07 16:00"
        assert "7 Mar 2024" in rows["Averaging"][2]
        assert "7 Mar 2024" in rows["Changes since 2023"][2]
        assert pg.RULE_START == "2023-12-11 23:00"
        assert "11 Dec 2023 23:00 UTC" in rows["Clamp $c$"][0]
        notes = " ".join(pg.CONSTRUCTION_NOTES)
        assert "11 December 2023 22:00 UTC" in notes and "16 June 21:00 to 15 July 02:00" in notes
        assert "8 June 2023" in notes
        assert pg.ORACLE_WORDING["last_spot_prices"] == "2025-10-06"
        assert pg.ORACLE_WORDING["first_spot_mid_prices"] == "2025-12-10"
        assert "6 October 2025" in notes and "10 December 2025" in notes
        assert "``spot mid prices''" in notes


class TestConstructionFootnotes:
    """Notes a and c of Table 1 state what the shipped data show."""

    def test_hyperliquid_rule_history_in_the_long_history(self):
        from regime_engine import longhistory as lh
        notes = " ".join(pg.CONSTRUCTION_NOTES)
        for coin, others in (("BTC", 1), ("ETH", 7)):
            d = lh.detect_switch(lh.load("hyperliquid", coin))
            assert (d["switch"], d["last_before_switch"]) == ("2023-12-11 23:00",
                                                              "2023-12-11 22:00")
            assert d["hourly_from"] == "2023-06-08 01:00"           # 8-hourly until 8 June
            free = d["regimes"]["unclamped"]
            assert (free["first"], free["last"], free["n"]) == ("2023-06-16 21:00",
                                                                "2023-07-15 02:00", 677)
            other = d["regimes"]["other"]
            assert other["n"] == others and other["first"].startswith("2023-07-16")
            assert other["last"].startswith("2023-07-16")
            fits = [r["fits_clamp_bps"] for era in d["history"] for r in era["fits_no_rule"]]
            assert len(fits) == others and set(fits) == {2}         # each fits +-0.02%
        assert "8 June 2023" in notes and "16 June 21:00 to 15 July 02:00 UTC 2023" in notes
        assert "on 16 July a few records fit a $\\pm0.02\\%$ clamp instead" in notes

    def test_bybit_switch_in_the_settlement_fits(self, payload):
        notes = " ".join(pg.CONSTRUCTION_NOTES)
        for coin in ("BTC", "ETH"):
            fit = payload["validation"]["settlements"][coin]["bybit"]
            assert fit["before_switch"]["last"] == "2024-03-07 08:00:00+00:00"
            assert fit["before_switch"]["plain"]["share_within_0.1bp"] == 1.0
        assert "7 March 2024 08:00 UTC and by rising weights from 16:00" in notes


def _drawn_text_boxes(fig, pad_pt: float = 0.0) -> list:
    """(text, padded window extent) of every text the figure draws.

    Tick labels of ticks outside the axis limits exist but are not drawn, so
    they are left out.
    """
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    undrawn = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    undrawn.update((id(tick.label1), id(tick.label2)))
    pad = pad_pt * fig.dpi / 72
    return [(t.get_text(), t.get_window_extent(renderer).padded(pad))
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and id(t) not in undrawn]


class TestFigure:
    @pytest.fixture(scope="class")
    def drawn(self, payload, tmp_path_factory):
        import matplotlib.pyplot as plt
        pdf = pg.figure(payload, out_dir=tmp_path_factory.mktemp("fig11"))
        fig = pg.draw_figure(payload)
        yield fig, pdf
        plt.close(fig)

    def test_figure_builds_from_the_shipped_data(self, drawn):
        _, pdf = drawn
        assert pdf.exists() and pdf.stat().st_size > 10_000

    def test_text_is_legible_at_the_printed_width(self, drawn):
        fig, pdf = drawn
        assert pg.smallest_font_pt(fig) >= 7.5
        assert pg.smallest_font_pt(fig) * pg.print_scale(pdf) >= 7.0

    def test_no_two_texts_overlap_within_a_panel(self, drawn):
        fig, _ = drawn
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            boxes = [t.get_window_extent(renderer) for t in ax.texts if t.get_text()]
            for i, a in enumerate(boxes):
                assert not any(a.overlaps(b) for b in boxes[i + 1:])

    def test_every_drawn_text_keeps_two_points_from_the_others(self, drawn):
        fig, _ = drawn
        boxes = _drawn_text_boxes(fig, pad_pt=1.0)
        touching = [(a, b) for i, (a, box) in enumerate(boxes) for b, other in boxes[i + 1:]
                    if box.overlaps(other)]
        assert touching == []

    def test_the_shaded_window_is_named_as_the_paper_names_it(self, drawn):
        fig, _ = drawn
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert texts.count("estimation\nwindow") == 2
        assert not any("paper" in t for t in texts)

    def test_monthly_panels_follow_the_artifact(self, payload):
        rows = pg.monthly_rows(payload, "BTC")
        assert len(rows["binance"]["mid"]) == 34                    # Dec 2023 .. Sep 2026
        assert rows["binance"]["months"][0] == "2023-12"            # drawn open: from 11 Dec

    def test_mechanism_panels_show_location_and_the_reversal(self, payload):
        assert pg.MECHANISM_QUARTERS == {"location": "2025Q4", "positive": "2024Q1"}
        # Hyperliquid's median sits inside the band in the location quarter, so the
        # title names the mechanism rather than calling every venue 'below the band'
        assert pg.MECHANISM_TITLE["location"] == "location"
        assert payload["mechanism"]["BTC"]["location"]["hyperliquid"]["median_bps"] > -4
        for name, q in pg.MECHANISM_QUARTERS.items():
            mech = pg.mechanism_series("BTC", q)
            assert float(pg.in_band(mech["hyperliquid"]).mean()) == pytest.approx(
                payload["mechanism"]["BTC"][name]["hyperliquid"]["share_in_band"])
        for venue in ("binance", "bybit"):
            quarters = payload["decomposition"]["BTC"][venue]["by_quarter"]
            loc = quarters["2025Q4"]
            assert loc["shapley"]["location"] >= 0.95 * loc["difference"] > 0.5
            assert quarters["2024Q1"]["difference"] < -0.1
        m = payload["mechanism"]["BTC"]
        assert {v: round(m["location"][v]["median_bps"], 2) for v in
                ("hyperliquid", "binance", "bybit")} == {
            "hyperliquid": -2.94, "binance": -4.51, "bybit": -4.63}
        assert {v: round(100 * m["positive"][v]["share_in_band"], 1) for v in
                ("hyperliquid", "binance", "bybit")} == {
            "hyperliquid": 33.4, "binance": 47.9, "bybit": 59.2}
