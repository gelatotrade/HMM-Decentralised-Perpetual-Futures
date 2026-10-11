"""Long history of funding-rate censoring: fetchers, rule detection, counts, artifact.

The fetchers are run against fake endpoints (no network). The rule detection,
period assignment, shares and block comparison are checked on synthetic series
with known answers. The shipped series then pin the numbers the paper quotes:
the common-rule shares, the paper-window counts of Table 3, the switch stamp
and the month-by-month ordering of the venues.
"""
import json
from decimal import Decimal
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from regime_engine import censoring as cz
from regime_engine import longhistory as lh

HOUR = 3_600_000
T0 = 1_759_363_200_000            # 2025-10-02 00:00 UTC


def _q(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


# ----------------------------------------------------------------------
#  Fetchers
# ----------------------------------------------------------------------
class FakeHyperliquid:
    """fundingHistory: ascending pages of at most 500 records from startTime."""

    def __init__(self, n=1_234):
        jitter = [48, 5, 0, 433]
        self.rows = [{"coin": "BTC", "time": T0 + HOUR * i + jitter[i % 4],
                      "fundingRate": "0.0000125" if i % 2 else f"0.00000{i % 9}1",
                      "premium": f"-0.000{i % 7}5"} for i in range(n)]
        # a late duplicate of hour 3, two minutes after the hour
        self.rows.insert(4, {"coin": "BTC", "time": T0 + 3 * HOUR + 120_000,
                             "fundingRate": "9", "premium": "9"})
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        lo, hi = payload["startTime"], payload["endTime"]
        return [r for r in self.rows if lo <= r["time"] <= hi][:lh.HL_PAGE]


class FakeExchanges:
    def __init__(self, n=700):
        self.binance = [(T0 + 8 * HOUR * i + (13 if i % 2 else 0), f"0.0000{i % 9}5")
                        for i in range(n)]
        self.bybit = [(T0 + 8 * HOUR * i, "0.0001" if i % 3 else "-0.00002")
                      for i in range(n)]
        self.symbols = set()

    def __call__(self, url):
        q = _q(url)
        self.symbols.add(q["symbol"])
        if "binance" in url:
            lo, hi, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
            return [{"symbol": q["symbol"], "fundingTime": t, "fundingRate": v,
                     "markPrice": "1.0"} for t, v in self.binance if lo <= t <= hi][:lim]
        lo, hi, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
        page = [r for r in self.bybit if lo <= r[0] <= hi][::-1][:lim]
        return {"result": {"list": [{"symbol": q["symbol"], "fundingRate": v,
                                     "fundingRateTimestamp": str(t)} for t, v in page]}}


@pytest.mark.filterwarnings("ignore:.*shares an hour")
class TestFetch:
    def test_hyperliquid_pages_through_and_floors_stamps_to_the_hour(self):
        api = FakeHyperliquid()
        rows = lh.hyperliquid_funding("BTC", "2025-10-02", "2025-12-31", post=api, pause=0)
        assert len(rows) == 1_234
        assert len(api.calls) >= 3
        assert rows[0] == {"time": "2025-10-02 00:00:00+00:00", "fundingRate": "0.0000001",
                           "premium": "-0.00005"}
        assert [r["time"] for r in rows] == sorted({r["time"] for r in rows})

    def test_hyperliquid_keeps_the_first_record_of_an_hour(self):
        rows = lh.hyperliquid_funding("BTC", "2025-10-02", "2025-10-03", post=FakeHyperliquid(),
                                      pause=0)
        hour3 = [r for r in rows if r["time"] == "2025-10-02 03:00:00+00:00"]
        assert len(hour3) == 1 and hour3[0]["fundingRate"] != "9"

    def test_hyperliquid_window_is_closed_at_both_ends(self):
        rows = lh.hyperliquid_funding("BTC", "2025-10-02 01:00", "2025-10-02 05:00",
                                      post=FakeHyperliquid(), pause=0)
        assert [r["time"][11:16] for r in rows] == ["01:00", "02:00", "03:00", "04:00", "05:00"]

    def test_exchanges_use_the_requested_symbol_and_keep_the_digits(self):
        api = FakeExchanges()
        b = lh.binance_funding("ETHUSDT", "2025-10-02", "2026-04-01", get=api, pause=0)
        y = lh.bybit_funding("ETHUSDT", "2025-10-02", "2026-04-01", get=api, pause=0)
        assert api.symbols == {"ETHUSDT"}
        assert b[1] == {"time": "2025-10-02 08:00:00+00:00", "fundingRate": "0.000015",
                        "markPrice": "1.0"}
        assert len(b) == len(y) == 181 * 3 + 1          # both ends included
        assert [r["fundingRate"] for r in y[:3]] == ["-0.00002", "0.0001", "0.0001"]

    def test_hyperliquid_pages_are_paced_within_the_weight_budget(self, monkeypatch):
        # a 500-record fundingHistory page weighs about 45 of 1,200 per minute
        assert lh.HL_PAUSE >= 60 / (1_200 / 45)
        slept = []
        monkeypatch.setattr(lh.time, "sleep", slept.append)
        lh.hyperliquid_funding("BTC", "2025-10-02", "2025-12-31", post=FakeHyperliquid())
        assert slept and set(slept) == {lh.HL_PAUSE}

    def test_a_late_duplicate_is_reported(self):
        with pytest.warns(UserWarning, match="1 record"):
            lh.hyperliquid_funding("BTC", "2025-10-02", "2025-10-03", post=FakeHyperliquid(),
                                   pause=0)

    def test_a_stamp_half_an_hour_or_more_from_its_hour_is_an_error(self):
        assert lh._hour(T0 + 29 * 60_000) == "2025-10-02 00:00:00+00:00"
        for late in (30 * 60_000, HOUR - 1):
            with pytest.raises(ValueError, match="past the hour"):
                lh._hour(T0 + late)

    def test_rate_limit_waits_and_retries(self, monkeypatch):
        import io
        import urllib.error

        calls, slept = [], []

        def urlopen(req, timeout):
            calls.append(req)
            if len(calls) < 3:
                headers = {"Retry-After": "7"} if len(calls) == 2 else {}
                raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests",
                                             headers, io.BytesIO(b""))
            return io.BytesIO(b"[1]")

        monkeypatch.setattr(lh.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(lh.time, "sleep", slept.append)
        assert lh.post_json({"type": "fundingHistory"}) == [1]
        assert slept == [lh.RATE_LIMIT_WAIT, 7.0]

    def test_other_http_errors_back_off_and_finally_raise(self, monkeypatch):
        import io
        import urllib.error

        def urlopen(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b""))

        slept = []
        monkeypatch.setattr(lh.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(lh.time, "sleep", slept.append)
        with pytest.raises(urllib.error.HTTPError):
            lh.post_json({}, retries=3)
        assert slept == [1, 2]

    def test_fetch_all_writes_six_files(self, tmp_path, monkeypatch):
        monkeypatch.setitem(lh.FIRST, "binance", "2025-10-02")
        monkeypatch.setitem(lh.FIRST, "bybit", "2025-10-02")
        monkeypatch.setitem(lh.FIRST, "hyperliquid", "2025-10-02")
        with pytest.warns(UserWarning):
            counts = lh.fetch_all(tmp_path, end="2025-10-20", get=FakeExchanges(),
                                  post=FakeHyperliquid(), pause=0, hl_pause=0)
        assert sorted(counts) == sorted(lh.file_name(v, a) for a in lh.ASSETS
                                        for v in lh.VENUES)
        df = lh.load("hyperliquid", "ETH", tmp_path)
        assert list(df.columns) == ["time", "fundingRate", "premium"]
        assert df["fundingRate"].map(type).eq(str).all()

    def test_fetch_all_paces_hyperliquid_pages_by_hl_pause(self, tmp_path, monkeypatch):
        for venue in lh.VENUES:
            monkeypatch.setitem(lh.FIRST, venue, "2025-10-02")
        slept = []
        monkeypatch.setattr(lh.time, "sleep", slept.append)
        with pytest.warns(UserWarning):
            lh.fetch_all(tmp_path, end="2025-12-31", get=FakeExchanges(),
                         post=FakeHyperliquid(), pause=0.25, hl_pause=0.75)
        # more than one 500-record Hyperliquid page per asset, so pauses between them
        assert slept.count(0.75) >= len(lh.ASSETS)
        assert set(slept) <= {0.25, 0.75}
        assert lh.HL_PAUSE not in slept


# ----------------------------------------------------------------------
#  The clamp rule
# ----------------------------------------------------------------------
def _hl(records):
    """records: (stamp, rate, premium) as strings."""
    df = pd.DataFrame(records, columns=["time", "fundingRate", "premium"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _hourly(start, n, rule, premia):
    """n hourly records from ``start`` under ``rule`` ('3', '5', 'free')."""
    out = []
    t = pd.Timestamp(start, tz="UTC")
    for i in range(n):
        p = Decimal(premia[i % len(premia)])
        c = {"3": lh.CLAMP_OLD, "5": lh.CLAMP, "free": None}[rule]
        f8 = p if c is None else p + min(max(lh.FLOOR_8H - p, -c), c)
        out.append((str(t + pd.Timedelta(hours=i)), str(f8 / 8), str(p)))
    return out


def _synthetic_history():
    prem = ["0.0009", "-0.0007", "0.00002", "-0.0001"]
    eight = [(str(pd.Timestamp("2023-06-01", tz="UTC") + pd.Timedelta(hours=8 * i)),
              str(Decimal(p) + min(max(lh.FLOOR_8H - Decimal(p), -lh.CLAMP_OLD), lh.CLAMP_OLD)),
              p) for i, p in enumerate(prem)]
    recs = (eight + _hourly("2023-06-02 01:00", 12, "3", prem)
            + _hourly("2023-06-02 13:00", 6, "free", prem)
            + _hourly("2023-06-02 19:00", 10, "3", prem)
            + _hourly("2023-06-03 05:00", 20, "5", prem + ["0.00055", "-0.00035"]))
    return _hl(recs)


class TestRule:
    def test_each_record_is_labelled_by_its_rule(self):
        reg = lh.clamp_regimes(_synthetic_history())
        assert (reg["schedule"] == "8h").sum() == 4
        assert reg.loc[reg["time"] < "2023-06-03 05:00", "regime"].isin(
            ["floor", "clamp_3bp", "unclamped"]).all()
        assert set(reg.loc[reg["time"] >= "2023-06-03 05:00", "regime"]) == {"floor",
                                                                          "clamp_5bp"}

    def test_the_switch_is_the_first_record_with_the_wide_clamp(self):
        out = lh.detect_switch(_synthetic_history())
        assert out["switch"] == "2023-06-03 05:00"
        assert out["last_before_switch"] == "2023-06-03 04:00"
        assert out["hourly_from"] == "2023-06-02 01:00"
        assert out["regimes"]["unclamped"]["first"] == "2023-06-02 13:00"
        assert out["common_rule_hours"] == 20
        assert out["common_rule_floor_not_in_band"] == 0
        assert out["common_rule_in_band_not_floor"] == 0

    def test_a_premium_in_the_band_but_off_the_floor_is_counted(self):
        df = _synthetic_history()
        i = df.index[-1]
        df.loc[i, "premium"] = "0.0002"                 # in the band, but rate not the floor
        df.loc[i, "fundingRate"] = str((Decimal("0.0002") - lh.CLAMP) / 8)
        assert lh.detect_switch(df)["common_rule_in_band_not_floor"] == 1

    def test_rule_history_lists_each_era(self):
        df = _synthetic_history()
        # one record in the second narrow era with 8F - P = +0.02%, which fits no rule
        i = df.index[df["time"] == pd.Timestamp("2023-06-02 21:00", tz="UTC")][0]
        df.loc[i, "premium"] = "0.00002"
        df.loc[i, "fundingRate"] = str((Decimal("0.00002") + Decimal("0.0002")) / 8)
        hist = lh.rule_history(df)
        assert [(h["schedule"], h["rule"], h["from"], h["to"], h["n_records"]) for h in hist] == [
            ("8h", "clamp_3bp", "2023-06-01 00:00", "2023-06-02 00:00", 4),
            ("1h", "clamp_3bp", "2023-06-02 01:00", "2023-06-02 12:00", 12),
            ("1h", "unclamped", "2023-06-02 13:00", "2023-06-02 18:00", 6),
            ("1h", "clamp_3bp", "2023-06-02 19:00", "2023-06-03 04:00", 10),
            ("1h", "clamp_5bp", "2023-06-03 05:00", "2023-06-04 00:00", 20),
        ]
        assert hist[3]["fits_no_rule"] == [{"time": "2023-06-02 21:00", "premium_bps": 0.2,
                                            "deviation_bps": 2.0, "fits_clamp_bps": 2}]
        assert all(h["floor_not_in_band"] == 0 for h in hist)
        assert [h["in_band_not_floor"] for h in hist] == [0, 0, None, 1, 0]
        assert hist[2]["n_floor"] == 0

    def test_floor_records_between_two_clamps_are_left_undetermined(self):
        # 3bp and 5bp eras separated only by floor records: the boundary is unknown
        # (the switch is the first record that reveals the new clamp, as in detect_switch)
        df = _hl(_hourly("2023-06-02 01:00", 4, "3", ["0.0009"])
                 + _hourly("2023-06-02 05:00", 3, "3", ["0.00002"])
                 + _hourly("2023-06-02 08:00", 3, "5", ["0.0009"]))
        hist = lh.rule_history(df)
        assert [(h["rule"], h["n_records"], h["undetermined_after"]) for h in hist] == [
            ("clamp_3bp", 7, 3), ("clamp_5bp", 3, 0)]

    def test_an_old_rule_after_the_switch_is_an_error(self):
        df = pd.concat([_synthetic_history(), _hl(_hourly("2023-06-04 01:00", 4, "3",
                                                         ["0.0009"]))], ignore_index=True)
        with pytest.raises(ValueError):
            lh.detect_switch(df)


# ----------------------------------------------------------------------
#  Periods, shares and the block comparison
# ----------------------------------------------------------------------
class TestPeriods:
    def test_a_midnight_settlement_belongs_to_the_period_it_closes(self):
        t = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00",
                                      "2024-07-01 00:00", "2024-07-01 08:00"], utc=True))
        assert lh.period_of(t, "M").tolist() == ["2023-12", "2024-01", "2024-06", "2024-07"]
        assert lh.period_of(t, "H").tolist() == ["2023H2", "2024H1", "2024H1", "2024H2"]
        assert lh.period_of(t, "Q").tolist() == ["2023Q4", "2024Q1", "2024Q2", "2024Q3"]
        assert lh.period_of(t, "Y").tolist() == ["2023", "2024", "2024", "2024"]


    def test_partial_periods_at_either_end(self):
        def t(x):
            return pd.Timestamp(x, tz="UTC")
        # an hourly span is complete from the stamp one hour after the period starts
        assert lh.partial_periods(t("2024-01-01 01:00"), t("2024-04-01 00:00"), "Q") == []
        assert lh.partial_periods(t("2024-01-01 00:00"), t("2024-04-01 00:00"), "Q") == \
            ["2023Q4"]
        assert lh.partial_periods(t("2023-12-11 23:00"), t("2026-09-30 12:00"), "M") == \
            ["2023-12", "2026-09"]
        # half-years, labelled as period_of labels them
        assert lh.partial_periods(t("2023-12-11 23:00"), t("2026-10-01 00:00"), "H") == \
            ["2023H2", "2026H2"]
        assert lh.partial_periods(t("2024-01-01 01:00"), t("2025-01-01 00:00"), "H") == []
        assert lh.partial_periods(t("2024-07-01 01:00"), t("2024-12-01 00:00"), "H") == \
            ["2024H2"]


class TestShares:
    def test_summary_counts_and_censored_median(self):
        rates = ["0.0001", "0.0001", "0.00002", "-0.0001", "0.0003"]
        prem = cz.recovered_premium_bps(rates)
        out = lh.summarise(rates, lh.FLOOR_8H, prem)
        assert (out["n"], out["at_floor"], out["below"], out["above"]) == (5, 2, 2, 1)
        assert out["median_bps"] is None                 # the middle value is censored
        out = lh.summarise(["0.00002", "-0.0001", "0.0001"], lh.FLOOR_8H,
                           cz.recovered_premium_bps(["0.00002", "-0.0001", "0.0001"]))
        assert out["median_bps"] == pytest.approx(-4.8)

    def test_ranking_against_the_exchanges(self):
        def row(h, b, y):
            return {"hyperliquid": {"share_at_floor": h}, "binance": {"share_at_floor": b},
                    "bybit": {"share_at_floor": y}}
        assert lh.rank_against_exchanges(row(0.5, 0.2, 0.3)) == "above_both"
        assert lh.rank_against_exchanges(row(0.1, 0.2, 0.3)) == "below_both"
        assert lh.rank_against_exchanges(row(0.25, 0.2, 0.3)) == "between"

    def test_month_majority_under_the_lower_bound(self):
        months = {"2024-01": {"share_hl_above_of_decided": 0.8, "share_hl_above_lower": 0.45},
                  "2024-02": {"share_hl_above_of_decided": 0.6, "share_hl_above_lower": 0.55},
                  "2024-03": {"share_hl_above_of_decided": 0.4, "share_hl_above_lower": 0.30}}
        m = lh.month_majority(months)
        assert (m["n_months"], m["hl_above_in_most_decided_blocks"],
                m["hl_above_in_most_blocks_lower_bound"]) == (3, 2, 1)
        assert m["months_not_lower_bound"] == ["2024-01", "2024-03"]
        assert m["lowest_lower_bound"] == {"month": "2024-03", "share": 0.30}
        # a month without decided blocks has no decided share and is not the lowest
        months["2024-04"] = {"share_hl_above_of_decided": None, "share_hl_above_lower": 0.0}
        m = lh.month_majority(months)
        assert m["lowest"] == {"month": "2024-03", "share": 0.4}
        assert "2024-04" in m["months_not"]

    def test_final_run_start(self):
        ranks = {"2024-01": "above_both", "2024-02": "between", "2024-03": "above_both",
                 "2024-04": "above_both"}
        assert lh._run_start(ranks) == "2024-03"
        assert lh._run_start({"2024-01": "between"}) is None


class TestBlocks:
    def test_block_means_cover_the_hours_a_settlement_averages(self):
        times = pd.date_range("2026-01-01 01:00", periods=16, freq="1h", tz="UTC")
        hl = pd.DataFrame({"time": times, "premium": ["0.0001"] * 7 + ["0.0009"]
                           + ["-0.0002"] * 8})
        m = lh.hl_block_means(hl)
        # records stamped 01:00-08:00 average [00:00, 08:00): the 08:00 settlement
        assert [str(t) for t in m.index] == ["2026-01-01 08:00:00+00:00",
                                             "2026-01-01 16:00:00+00:00"]
        assert m.iloc[0] == pytest.approx(2.0)
        assert m.iloc[1] == pytest.approx(-2.0)

    def test_incomplete_blocks_are_dropped(self):
        times = pd.date_range("2026-01-01 01:00", periods=16, freq="1h", tz="UTC").delete(3)
        hl = pd.DataFrame({"time": times, "premium": ["0.0001"] * 15})
        assert [str(t) for t in lh.hl_block_means(hl).index] == ["2026-01-01 16:00:00+00:00"]

    def test_order_is_decided_by_exact_values_or_by_the_band(self):
        stamps = pd.to_datetime(["2026-01-01 08:00", "2026-01-01 16:00", "2026-01-02 00:00",
                                 "2026-01-02 08:00", "2026-01-02 16:00"], utc=True)
        hl_means = pd.Series([-4.2, -6.0, 7.0, -5.0, 2.0], index=stamps)
        ex = pd.DataFrame({"time": stamps,
                           # -4.8, -4.8 bps identified; then three settlements at the floor
                           "fundingRate": ["0.00002", "0.00002", "0.0001", "0.0001", "0.0001"]})
        cmp = lh.block_comparison(hl_means, ex)
        assert cmp["order"].tolist() == ["above", "below", "above", "below", "undecided"]
        s = lh.comparison_summary(cmp)
        assert (s["hl_above"], s["hl_below"], s["undecided"], s["identified"]) == (2, 2, 1, 2)
        assert s["share_hl_above_of_decided"] == 0.5
        # bounds without assumptions: undecided blocks count against, then for, the claim
        assert (s["share_hl_above_lower"], s["share_hl_above_upper"]) == (0.4, 0.6)
        assert s["median_diff_given_exchange_identified_bps"] == pytest.approx((0.6 - 1.2) / 2)

    def _cmp(self, hl, ex):
        """A comparison frame from HL block means and exchange premia (None: at the floor)."""
        return pd.DataFrame({"hl_bps": hl, "exchange_bps": [np.nan if e is None else e
                                                            for e in ex]})

    def test_quantile_gaps_compare_the_two_distributions(self):
        hl = [-6.0, -5.0, -4.5, -1.0, 3.0]
        ex = [-7.0, -6.0, -5.5, -5.0, None]
        q = lh.quantile_gaps(self._cmp(hl, ex), probs=(0.25, 0.5, 0.9))
        assert q["n_blocks"] == 5
        assert q["q25"] == {"hyperliquid_bps": -5.0, "exchange_bps": -6.0, "gap_bps": 1.0}
        assert q["q50"]["gap_bps"] == pytest.approx(-4.5 + 5.5)
        # the 90% quantile of the exchange falls on a censored value
        assert q["q90"]["exchange_bps"] is None and q["q90"]["gap_bps"] is None

    def test_a_pure_shift_is_tested_by_moving_hyperliquid_down_by_the_median_gap(self):
        hl = [-5.0, -3.0, -2.0, 7.0]
        ex = [-6.0, -5.0, None, None]
        s = lh.shift_test(self._cmp(hl, ex))
        # exchange median: between -5.0 and the band, censored; no shift is identified
        assert s["median_gap_bps"] is None
        hl, ex = [-5.0, -4.5, -3.0, -2.0, 7.0], [-6.0, -5.5, -5.0, None, None]
        s = lh.shift_test(self._cmp(hl, ex))
        assert s["median_gap_bps"] == pytest.approx(-3.0 + 5.0)
        assert s["exchange_at_floor"] == pytest.approx(2 / 5)
        assert s["hyperliquid_in_band"] == pytest.approx(2 / 5)
        # moved down by 2 bps: -7, -6.5, -5, -4 (on the edge, in the band), 5
        assert s["hyperliquid_shifted_in_band"] == pytest.approx(2 / 5)
        # the per-block variant: the median of the identified block differences
        # (1.0, 1.0, 2.0) is 1.0; moved down by it: -6, -5.5, -4, -3, 6
        assert s["median_block_diff_bps"] == pytest.approx(1.0)
        assert s["hyperliquid_shifted_by_block_diff_in_band"] == pytest.approx(3 / 5)


# ----------------------------------------------------------------------
#  The shipped series
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def payload():
    return lh.build_payload()


class TestShippedData:
    def test_the_common_rule_starts_on_11_december_2023(self, payload):
        assert payload["common_rule"] == {"from": "2023-12-11 23:00", "to": "2026-10-01 00:00"}
        for asset in lh.ASSETS:
            rule = payload["assets"][asset]["rule"]
            assert rule["last_before_switch"] == "2023-12-11 22:00"
            assert rule["hourly_from"] == "2023-06-08 01:00"
            assert rule["regimes"]["unclamped"]["first"] == "2023-06-16 21:00"
            assert rule["regimes"]["unclamped"]["last"] == "2023-07-15 02:00"
            assert rule["regimes"]["unclamped"]["n"] == 677
            assert rule["regimes"]["clamp_3bp"]["last"] == "2023-12-11 22:00"
            assert rule["common_rule_hours"] == 24_577

    def test_rule_history(self, payload):
        eras = [("8h", "clamp_3bp", "2023-05-12 00:00", "2023-06-08 00:00", 82),
                ("1h", "clamp_3bp", "2023-06-08 01:00", "2023-06-16 20:00", 212),
                ("1h", "unclamped", "2023-06-16 21:00", "2023-07-15 02:00", 677),
                ("1h", "clamp_3bp", "2023-07-15 03:00", "2023-12-11 22:00", 3_595),
                ("1h", "clamp_5bp", "2023-12-11 23:00", "2026-10-01 00:00", 24_577)]
        for asset, n_odd in (("BTC", 1), ("ETH", 7)):
            hist = payload["assets"][asset]["rule"]["history"]
            assert [(h["schedule"], h["rule"], h["from"], h["to"], h["n_records"])
                    for h in hist] == eras
            assert all(h["undetermined_after"] == 0 and h["floor_not_in_band"] == 0
                       for h in hist)
            assert [h["n_other"] for h in hist] == [0, 0, 0, n_odd, 0]
            odd = hist[3]["fits_no_rule"]
            # the odd records all sit on 16 July 2023 with |8F - P| = 0.02%
            assert all(r["time"].startswith("2023-07-16") for r in odd)
            assert {abs(r["deviation_bps"]) for r in odd} == {2.0}
            # they fit a 0.02% clamp exactly
            assert {r["fits_clamp_bps"] for r in odd} == {2}
            assert hist[3]["in_band_not_floor"] == n_odd

    def test_floor_and_band_coincide_exactly_under_the_common_rule(self, payload):
        for asset in lh.ASSETS:
            rule = payload["assets"][asset]["rule"]
            assert rule["common_rule_floor_not_in_band"] == 0
            assert rule["common_rule_in_band_not_floor"] == 0
            assert rule["common_rule_non_floor_not_5bp"] == 0

    @pytest.mark.parametrize("asset, counts", [
        ("BTC", {"hyperliquid": (13_578, 24_577), "hyperliquid_at_8h_stamps": (1_673, 3_073),
                 "binance": (730, 3_073), "bybit": (839, 3_073)}),
        ("ETH", {"hyperliquid": (12_768, 24_577), "hyperliquid_at_8h_stamps": (1_578, 3_073),
                 "binance": (711, 3_073), "bybit": (869, 3_073)}),
    ])
    def test_common_rule_floor_counts(self, payload, asset, counts):
        w = payload["assets"][asset]["windows"]["common_rule"]
        assert {v: (w[v]["at_floor"], w[v]["n"]) for v in counts} == counts

    def test_paper_and_later_windows_equal_table_3(self, payload):
        w = payload["assets"]["BTC"]["windows"]
        got = {k: {v: (w[k][v]["at_floor"], w[k][v]["n"]) for v in lh.COLUMNS}
               for k in ("paper", "later")}
        assert got == {
            "paper": {"hyperliquid": (2_300, 5_001), "hyperliquid_at_8h_stamps": (303, 626),
                      "binance": (37, 626), "bybit": (48, 626)},
            "later": {"hyperliquid": (1_931, 3_736), "hyperliquid_at_8h_stamps": (246, 467),
                      "binance": (51, 467), "bybit": (37, 467)},
        }

    def test_paper_windows_agree_with_the_censoring_module(self, payload):
        ref = cz.build_payload()["venues"]
        mine = payload["assets"]["BTC"]["windows"]
        for w in ("paper", "later"):
            for v in lh.COLUMNS:
                for key in ("n", "at_floor", "below", "above"):
                    assert mine[w][v][key] == ref[v][w][key], (w, v, key)
            for v in ("hyperliquid", "binance", "bybit"):
                assert mine[w][v]["median_bps"] == pytest.approx(ref[v][w]["premium"]
                                                                 ["median_bps"])

    def test_windows_partition_the_common_rule(self, payload):
        for asset in lh.ASSETS:
            w = payload["assets"][asset]["windows"]
            for v in lh.COLUMNS:
                parts = sum(w[k][v]["n"] for k in ("before_paper", "paper", "later"))
                assert parts == w["common_rule"][v]["n"], (asset, v)

    def test_eth_paper_and_later_windows(self, payload):
        w = payload["assets"]["ETH"]["windows"]
        got = {k: {v: (w[k][v]["at_floor"], w[k][v]["n"]) for v in lh.COLUMNS}
               for k in ("paper", "later")}
        assert got == {
            "paper": {"hyperliquid": (2_355, 5_001), "hyperliquid_at_8h_stamps": (294, 626),
                      "binance": (35, 626), "bybit": (69, 626)},
            "later": {"hyperliquid": (2_354, 3_736), "hyperliquid_at_8h_stamps": (295, 467),
                      "binance": (31, 467), "bybit": (31, 467)},
        }

    def test_years_and_months_add_up_to_the_common_rule(self, payload):
        for asset in lh.ASSETS:
            a = payload["assets"][asset]
            for key in ("years", "half_years", "quarters", "months"):
                for v in lh.COLUMNS:
                    for count in ("n", "at_floor", "below", "above"):
                        total = sum(r[v][count] for r in a[key].values())
                        assert total == a["windows"]["common_rule"][v][count], \
                            (asset, key, v, count)
            for venue in lh.EXCHANGES:
                b = a["block_comparison"][venue]
                for key in ("years", "half_years", "quarters", "months"):
                    for count in ("n_blocks", "hl_above", "hl_below", "undecided"):
                        assert sum(r[count] for r in b[key].values()) == \
                            b["common_rule"][count], (asset, venue, key, count)

    def test_partial_periods(self, payload):
        assert payload["partial_periods"] == {"Y": ["2023", "2026"], "H": ["2023H2", "2026H2"],
                                              "Q": ["2023Q4"], "M": ["2023-12"]}

    def test_2024(self, payload):
        y = payload["assets"]["BTC"]["years"]["2024"]
        assert {v: (y[v]["at_floor"], y[v]["n"]) for v in ("hyperliquid", "binance", "bybit")} \
            == {"hyperliquid": (4_861, 8_783), "binance": (463, 1_098), "bybit": (494, 1_098)}

    def test_month_ranking(self, payload):
        for asset, below in (("BTC", ["2024-01", "2024-02", "2024-11", "2025-07"]),
                             ("ETH", ["2024-01", "2024-02", "2024-11"])):
            a = payload["assets"][asset]
            r = a["month_ranking"]
            assert (r["first_full_month"], r["last_month"], r["n_full_months"]) == \
                ("2024-01", "2026-09", 33)
            assert r["hl_above_both"] == 26
            assert r["hl_at_8h_stamps_above_both"] == 25
            assert r["above_both_every_month_from"] == "2025-08"
            assert [m["month"] for m in a["months_hl_below_both"]] == below
            assert r["hl_share_min"]["month"] == "2026-04"

    def test_reversal_months_have_positive_premia(self, payload):
        jan = payload["assets"]["BTC"]["months_hl_below_both"][0]
        assert round(100 * jan["hyperliquid"], 1) == 39.0
        assert round(100 * jan["binance"], 1) == 84.9
        assert round(100 * jan["bybit"], 1) == 87.1
        for asset in lh.ASSETS:
            for m in payload["assets"][asset]["months_hl_below_both"]:
                assert m["hyperliquid_share_above_band"] > 0.3
                assert m["hyperliquid_median_bps"] > 0

    def test_quarter_median_gaps_on_eight_hour_blocks(self, payload):
        for asset, rng in (("BTC", (0.22, 5.36)), ("ETH", (0.47, 5.21))):
            g = payload["assets"][asset]["quarter_median_gap_range"]
            assert g["all_positive"] and g["excludes"] == ["2023Q4"]
            assert (round(g["min_bps"], 2), round(g["max_bps"], 2)) == rng
        assert payload["assets"]["BTC"]["quarter_median_gap_range"]["n_identified"] == 16

    @pytest.mark.parametrize("asset, venue, counts, bounds", [
        ("BTC", "binance", (2_180, 443, 448), (71.0, 85.6)),
        ("BTC", "bybit", (2_124, 413, 534), (69.2, 86.6)),
        ("ETH", "binance", (2_105, 540, 426), (68.5, 82.4)),
        ("ETH", "bybit", (2_009, 530, 532), (65.4, 82.7)),
    ])
    def test_block_ordering_bounds(self, payload, asset, venue, counts, bounds):
        b = payload["assets"][asset]["block_comparison"][venue]["common_rule"]
        assert b["n_blocks"] == 3_071
        assert (b["hl_above"], b["hl_below"], b["undecided"]) == counts
        assert (round(100 * b["share_hl_above_lower"], 1),
                round(100 * b["share_hl_above_upper"], 1)) == bounds
        assert b["share_hl_above_lower"] <= b["share_hl_above_of_decided"] \
            <= b["share_hl_above_upper"]

    def test_month_majority_over_the_full_months(self, payload):
        for asset, decided, lower in (("BTC", (33, 33), (30, 31)),
                                      ("ETH", (32, 31), (29, 27))):
            b = payload["assets"][asset]["block_comparison"]
            mm = [b[v]["month_majority"] for v in lh.EXCHANGES]
            assert all(m["n_months"] == 33 for m in mm)
            assert tuple(m["hl_above_in_most_decided_blocks"] for m in mm) == decided
            assert tuple(m["hl_above_in_most_blocks_lower_bound"] for m in mm) == lower

    def test_the_gap_is_not_a_constant_offset(self, payload):
        q = payload["assets"]["BTC"]["block_comparison"]["binance"]["quantile_gaps"]
        paper = [round(q["paper"][k]["gap_bps"], 2)
                 for k in ("q05", "q10", "q25", "q50", "q75", "q90")]
        assert paper == [-0.20, -0.05, 0.23, 0.73, 2.22, 4.72]
        common = [q["common_rule"][k]["gap_bps"] for k in ("q05", "q25", "q50", "q75")]
        assert [round(x, 2) for x in common[:3]] == [-0.05, 0.83, 3.39]
        assert common[3] is None                         # the exchange's q75 is in the band
        assert q["paper"]["n_blocks"] == 626 and q["common_rule"]["n_blocks"] == 3_071
        for asset in lh.ASSETS:
            for venue in lh.EXCHANGES:
                s = payload["assets"][asset]["block_comparison"][venue]["shift_test"]["paper"]
                # moved down by the median gap, Hyperliquid stays in the band far more
                # often than the exchange prints the floor
                assert s["hyperliquid_shifted_in_band"] > 3 * s["exchange_at_floor"]
        s = payload["assets"]["BTC"]["block_comparison"]["binance"]["shift_test"]["paper"]
        assert (round(100 * s["hyperliquid_shifted_in_band"], 1),
                round(100 * s["exchange_at_floor"], 1)) == (34.0, 5.9)

    def test_exchanges_settle_every_eight_hours(self, payload):
        for asset in lh.ASSETS:
            d = payload["assets"][asset]["data"]
            assert d["binance"]["spacing_hours_common_rule"] == [8]
            assert d["bybit"]["spacing_hours_common_rule"] == [8]
            assert d["hyperliquid"]["spacing_hours_common_rule"] == [1, 2]
            assert all(d[v]["last"] == "2026-10-01 00:00" for v in lh.VENUES)

    def test_overlap_with_the_interim_series_is_exact(self):
        pairs = [("hyperliquid", "premium_BTC_full.csv", ["fundingRate", "premium"]),
                 ("binance", "binance_BTCUSDT_funding.csv", ["fundingRate"]),
                 ("bybit", "bybit_BTCUSDT_funding.csv", ["fundingRate"])]
        for venue, name, cols in pairs:
            old = pd.read_csv(cz.DATA / name, dtype=str)
            old["time"] = pd.to_datetime(old["time"], utc=True)
            new = lh.load(venue, "BTC")
            j = old.merge(new, on="time", how="left", suffixes=("", "_new"))
            assert j[cols[0] + "_new"].notna().all(), venue
            for c in cols:
                same = [Decimal(a) == Decimal(b) for a, b in zip(j[c], j[c + "_new"])]
                assert all(same), (venue, c)

    def test_shipped_artifact_and_table_are_current(self, payload):
        shipped = json.loads(lh.ARTIFACT.read_text())
        assert shipped == json.loads(json.dumps(payload))
        assert lh.write_table(payload) == lh.PAPER_TABLE.read_text()
        assert lh.OUTPUT_TABLE.read_text() == lh.PAPER_TABLE.read_text()

    def test_table_rows_by_year_and_half_year(self, payload):
        for asset in lh.ASSETS:
            labels = [label for label, _ in lh.table_rows(payload, asset)]
            assert labels == [
                "11 Dec 2023 -- 30 Sep 2026", "2023, from 11 Dec",
                "2024", "\\quad Jan--Jun", "\\quad Jul--Dec",
                "2025", "\\quad Jan--Jun", "\\quad Jul--Dec",
                "2026, Jan--Sep", "\\quad Jan--Jun", "\\quad Jul--Sep",
                cz.window_label("paper"), cz.window_label("later")]

    def test_table_has_one_panel_per_asset_and_eight_columns(self, payload):
        tex = lh.write_table(payload)
        assert "\\begin{tabular}{@{}lrrrrrrr@{}}" in tex
        assert "\\emph{A. BTC}" in tex and "\\emph{B. ETH}" in tex
        body = [ln for ln in tex.splitlines() if ln.endswith("\\\\") and "&" in ln]
        assert all(ln.count("&") == 7 for ln in body if "multicolumn" not in ln)
        # the calendar year 2024, BTC: Hyperliquid hourly, Binance, Bybit
        row = next(ln for ln in body if ln.startswith("2024 &"))
        cells = [c.strip() for c in row.rstrip("\\ ").split("&")]
        assert (cells[1], cells[3], cells[4]) == ("55.3", "42.2", "45.0")

    def test_figure_data_and_drawing(self, payload):
        data = lh.figure_data(payload)
        assert len(data["BTC"]["x"]) == 33 and data["BTC"]["months"][0] == "2024-01"
        assert "2023-12" not in data["ETH"]["months"]
        assert np.isclose(data["BTC"]["share"]["binance"][0],
                          payload["assets"]["BTC"]["months"]["2024-01"]["binance"]
                          ["share_at_floor"])
        for asset in lh.ASSETS:
            for v in lh.EXCHANGES:
                assert all(lo <= hi for lo, hi in zip(data[asset]["lower"][v],
                                                      data[asset]["upper"][v]))
        fig = lh.draw_figure(data)
        assert len(fig.axes) == 4
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert texts.count("estimation\nwindow") == 2 and not any("paper" in t for t in texts)
        smallest, width = lh.printed_font_sizes(fig)
        assert smallest >= lh.MIN_FONT_PT
        assert width <= lh.FIGURE_WIDTH_IN * 1.1
        # every month and exchange shows: a month whose bounds coincide still gets
        # its end marks, so it does not vanish as a zero-length bar
        for col, asset in enumerate(lh.ASSETS):
            ax = fig.axes[2 + col]
            for v in lh.EXCHANGES:
                marks = [ln for ln in ax.get_lines()
                         if ln.get_marker() == "_" and ln.get_color() == lh.PALETTE[v]]
                assert len(marks) == 2, (asset, v)
                for ln, key in zip(marks, ("lower", "upper")):
                    assert list(ln.get_ydata()) == data[asset][key][v]
                    assert ln.get_markersize() >= 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_printed_width_includes_the_saved_padding(self, payload, tmp_path):
        import re
        pdf = lh.figure(payload, out_dir=tmp_path)
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", pdf.read_bytes())
        saved_in = float(box.group(1)) / 72
        fig = lh.draw_figure(lh.figure_data(payload))
        smallest, width = lh.printed_font_sizes(fig)
        assert width == pytest.approx(saved_in, abs=0.02)
        assert smallest >= lh.MIN_FONT_PT
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_paper_window_shading_matches_figure_11(self):
        assert lh.WINDOW_SHADE == {"color": "#2E86AB", "alpha": 0.08}

    def test_venue_colours_follow_the_censoring_figure(self):
        import inspect
        source = inspect.getsource(cz.figure)
        for venue, colour in lh.PALETTE.items():
            assert f'"{venue}": "{colour}"' in source


def test_main_writes_artifact_table_and_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(lh, "ARTIFACT", tmp_path / "out" / "a.json")
    monkeypatch.setattr(lh, "OUTPUT_TABLE", tmp_path / "out" / "t.tex")
    monkeypatch.setattr(lh, "PAPER_TABLE", tmp_path / "paper" / "t.tex")
    monkeypatch.setattr(lh, "FIGURE_DIR", tmp_path / "out" / "figures")
    monkeypatch.setattr(lh, "PAPER_FIGURE_DIR", tmp_path / "paper" / "figures")
    assert lh.main([]) == 0
    payload = json.loads((tmp_path / "out" / "a.json").read_text())
    assert payload["common_rule"]["from"] == "2023-12-11 23:00"
    assert (tmp_path / "out" / "t.tex").read_text() == (tmp_path / "paper" / "t.tex").read_text()
    for name in ("out/figures/fig10_long_history.pdf", "out/figures/fig10_long_history.png",
                 "paper/figures/fig10_long_history.pdf"):
        assert (tmp_path / name).stat().st_size > 10_000
