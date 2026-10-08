"""Funding-rate censoring across venues: exact classification and windows.

A settlement at the interest floor (0.01% per 8h; 0.125 bps/h on Hyperliquid)
says only that the premium sat inside the clamp band. These tests pin the
counting rules the cross-venue comparison rests on: decimals compared exactly,
window boundaries as the paper states them, and the 00/08/16 UTC stamps at
which the eight-hourly venues settle.
"""
from decimal import Decimal

import pandas as pd
import pytest

from regime_engine import censoring as c


def _times(*stamps):
    return pd.Series(pd.to_datetime(list(stamps), utc=True))


class TestClassify:
    def test_floor_strings_count_exactly_whatever_their_padding(self):
        out = c.classify(["0.0001", "0.00010000", "0.00009999", "0.00010001", "-0.0002"])
        assert (out["at_floor"], out["below"], out["above"], out["n"]) == (2, 2, 1, 5)

    def test_shares_add_up(self):
        out = c.classify(["0.0001", "0.00005", "0.0003", "0.0001"])
        assert out["share_at_floor"] == 0.5
        assert out["share_at_floor"] + out["share_below"] + out["share_above"] == pytest.approx(1)

    def test_hyperliquid_floor_in_scientific_notation(self):
        out = c.classify(["1.25e-05", "0.0000125", "1.2500001e-05", "1.1e-05"], c.FLOOR_HOURLY)
        assert (out["at_floor"], out["above"], out["below"]) == (2, 1, 1)

    def test_floors_are_one_basis_point_per_eight_hours(self):
        assert c.FLOOR_8H == Decimal("0.0001")
        assert c.FLOOR_HOURLY * 8 == c.FLOOR_8H

    def test_empty_input_is_an_error(self):
        with pytest.raises(ValueError):
            c.classify([])


class TestWindows:
    def test_paper_window_is_closed_at_both_ends(self):
        t = _times("2025-10-01 16:00", "2025-10-02 00:00", "2026-04-28 08:00", "2026-04-28 16:00")
        assert c.in_window(t, "paper").tolist() == [False, True, True, False]

    def test_later_window_starts_after_the_paper_window(self):
        t = _times("2026-04-28 08:00", "2026-04-28 16:00", "2026-10-01 00:00", "2026-10-01 08:00")
        assert c.in_window(t, "later").tolist() == [False, True, True, False]

    def test_labels_close_a_midnight_end_on_the_day_before(self):
        assert c.window_label("paper") == "2 Oct 2025 -- 28 Apr 2026"
        assert c.window_label("later") == "28 Apr -- 30 Sep 2026"

    def test_settlement_stamps(self):
        t = _times("2026-01-01 00:00", "2026-01-01 08:00", "2026-01-01 16:00",
                   "2026-01-01 01:00", "2026-01-01 09:00")
        assert c.at_settlement_stamps(t).tolist() == [True, True, True, False, False]

    def test_spacing_in_hours(self):
        t = _times("2026-01-01 00:00", "2026-01-01 08:00", "2026-01-01 16:00")
        assert c.spacing_hours(t) == [8]


class TestPremium:
    def test_location_counts_the_open_band(self):
        out = c.premium_location(pd.Series([-5.0, -4.0, -3.0, 0.0, 6.0, 7.0]))
        assert out["n"] == 6
        assert out["share_in_band"] == pytest.approx(2 / 6)
        assert out["mean_bps"] == pytest.approx(1 / 6)

    def test_eight_hour_means_align_to_settlement_blocks(self):
        times = pd.date_range("2026-01-01 00:00", periods=16, freq="1h", tz="UTC")
        df = pd.DataFrame({"time": times, "close": ["0.0001"] * 8 + ["-0.0003"] * 8})
        means = c.eight_hour_means(df)
        # Readings stamped 00:00-07:00 open the hours that the 08:00 settlement
        # averages, so each block carries the settlement stamp that closes it.
        assert [str(t) for t in means.index] == ["2026-01-01 08:00:00+00:00",
                                                 "2026-01-01 16:00:00+00:00"]
        assert means.tolist() == pytest.approx([1.0, -3.0])


class TestRecoveredPremium:
    def test_outside_the_band_the_settlement_identifies_the_premium(self):
        out = c.recovered_premium_bps(["0.00002500", "0.0001", "0.00030000"])
        assert out[0] == pytest.approx(-4.75)        # 0.25 bp - 5 bp
        assert out[1] != out[1]                      # at the floor: only "inside the band"
        assert out[2] == pytest.approx(8.0)          # 3 bp + 5 bp

    def test_quantiles_are_exact_when_censored_values_sit_above_them(self):
        # three identified below the band, one censored inside it
        x = pd.Series([-5.0, -4.8, -4.6, float("nan")])
        assert c.censored_quantile(x, 0.5) == pytest.approx(-4.7)
        assert c.censored_quantile(x, 0.25) == pytest.approx(-4.85)

    def test_a_quantile_that_falls_on_censored_values_is_not_reported(self):
        x = pd.Series([-5.0, float("nan"), float("nan"), float("nan")])
        assert c.censored_quantile(x, 0.5) is None


class TestShippedData:
    """The comparison on the shipped series; pins the counts the paper prints."""

    @pytest.fixture(scope="class")
    def payload(self):
        return c.build_payload()

    def test_hyperliquid_matches_the_paper(self, payload):
        hl = payload["venues"]["hyperliquid"]["paper"]
        assert (hl["n"], hl["at_floor"]) == (5001, 2300)

    def test_hyperliquid_at_the_eight_hourly_stamps(self, payload):
        assert payload["venues"]["hyperliquid_at_8h_stamps"]["paper"]["n"] == 626

    def test_both_exchanges_settle_every_eight_hours(self, payload):
        for venue in ("binance", "bybit"):
            assert payload["venues"][venue]["spacing_hours"] == [8]
            assert payload["venues"][venue]["paper"]["n"] == 626

    def test_no_venue_misses_a_settlement_after_the_window(self, payload):
        # 467 eight-hourly stamps after 28 Apr 08:00 up to 1 Oct 00:00; a venue
        # one short would be compared over a shorter window without saying so.
        for venue in ("hyperliquid_at_8h_stamps", "binance", "bybit"):
            assert payload["venues"][venue]["later"]["n"] == 467, venue
        assert payload["venues"]["hyperliquid"]["later"]["n"] == 3736

    def test_shares_add_up_everywhere(self, payload):
        for venue in payload["venues"].values():
            for window in ("paper", "later"):
                d = venue[window]
                assert d["at_floor"] + d["below"] + d["above"] == d["n"]

    def test_exchange_premium_quantiles_come_from_settlements(self, payload):
        for venue in ("binance", "bybit"):
            prem = payload["venues"][venue]["paper"]["premium"]
            assert prem["source"] == "settlements"
            assert prem["q25_bps"] < prem["median_bps"] < prem["q75_bps"] < -4

    def test_table_regenerates_from_the_artifact(self, payload):
        shipped = (c.PAPER_TABLE).read_text()
        assert c.write_table(payload) == shipped
