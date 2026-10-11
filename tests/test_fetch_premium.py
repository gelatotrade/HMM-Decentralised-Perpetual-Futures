"""Premium-gap data: fetchers against a fake API, and the hourly aggregation.

The fake honours what the real endpoints do: Hyperliquid's fundingHistory pages
of at most 500 records with jittered stamps; Binance's premium-index klines in
ascending pages that include both ends of the requested span; Bybit's in
newest-first pages; funding settlements with a few milliseconds of jitter. The
aggregation is checked against values built by hand: the hour stamped t holds
the one-minute klines that open in [t - 1h, t).
"""
import gzip
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from regime_engine import fetch_premium as fp

MIN = 60_000
HOUR = 3_600_000
T0 = 1_702_332_000_000            # 2023-12-11 22:00 UTC


def _q(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


class FakeAPI:
    """Minute klines, settlements and hourly funding records served page by page."""

    def __init__(self, minutes=4_000, missing=(), hours=1_300, settlements=700):
        self.minutes = [(T0 + MIN * i, f"-0.000{i % 7}1", f"-0.000{(i + 3) % 7}1")
                        for i in range(minutes) if i not in set(missing)]
        jitter = [0, 13, 0, 6]
        self.settle = [(T0 + 2 * HOUR + 8 * HOUR * i + jitter[i % 4],
                        "0.00010000" if i % 3 == 0 else f"0.0000{i % 9}123")
                       for i in range(settlements)]
        hl_jitter = [48, 51, 50, 887_645]
        self.hl = [{"coin": "BTC", "fundingRate": "0.0000125", "premium": f"-0.000{i % 9}5",
                    "time": T0 + HOUR * i + hl_jitter[i % 4]} for i in range(hours)]
        self.prices = [(T0 + HOUR * i, f"{40_000 + i}.5") for i in range(hours)]
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        q = _q(url)
        if "fapi.binance.com" in url:
            lo, hi, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
            if "premiumIndexKlines" in url:
                assert q["interval"] == "1m"
                page = [r for r in self.minutes if lo <= r[0] <= hi][:lim]
                return [[t, o, o, c, c, "0", t + MIN - 1, "0", 12, "0", "0", "0"]
                        for t, o, c in page]
            if "fundingRate" in url:
                page = [r for r in self.settle if lo <= r[0] <= hi][:lim]
                return [{"symbol": q["symbol"], "fundingTime": t, "fundingRate": v,
                         "markPrice": "1"} for t, v in page]
            if "/klines" in url:
                assert q["interval"] == "1h"
                page = [r for r in self.prices if lo <= r[0] <= hi][:lim]
                return [[t, c, c, c, c, "1", t + HOUR - 1, "1", 1, "0", "0", "0"]
                        for t, c in page]
        if "api.binance.com/api/v3/klines" in url:
            assert q["symbol"] == "USDCUSDT" and q["interval"] == "1h"
            lo, hi, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
            page = [r for r in self.prices if lo <= r[0] <= hi][:lim]
            return [[t, "1.0001", "1.0002", "0.9999", "1.0001", "1", t + HOUR - 1, "1", 1, "0",
                     "0", "0"] for t, _ in page]
        if "api.bybit.com" in url:
            lim = int(q["limit"])
            if "premium-index-price-kline" in url:
                assert q["interval"] == "1"
                lo, hi = int(q["start"]), int(q["end"])
                page = sorted((r for r in self.minutes if lo <= r[0] <= hi), reverse=True)[:lim]
                return {"retCode": 0, "result": {"list": [[str(t), o, c, o, c]
                                                          for t, o, c in page]}}
            if "funding/history" in url:
                lo, hi = int(q["startTime"]), int(q["endTime"])
                page = sorted((r for r in self.settle if lo <= r[0] <= hi), reverse=True)[:lim]
                return {"retCode": 0, "result": {"list": [
                    {"symbol": q["symbol"], "fundingRate": v, "fundingRateTimestamp": str(t)}
                    for t, v in page]}}
        raise AssertionError(f"unexpected url {url}")

    def post(self, url, body):
        self.calls.append(url)
        assert body["type"] == "fundingHistory"
        lo, hi = body["startTime"], body.get("endTime", 10 ** 15)
        return [r for r in self.hl if lo <= r["time"] <= hi][:500]


@pytest.fixture
def api():
    return FakeAPI()


def _stamp(ms):
    return str(pd.Timestamp(ms, unit="ms", tz="UTC"))


class TestMinuteKlines:
    @pytest.mark.parametrize("fetch,page", [(fp.binance_premium_1m, 1500),
                                            (fp.bybit_premium_1m, 1000)])
    def test_every_minute_once_in_order(self, api, fetch, page):
        start, end = T0, T0 + MIN * 3_999
        rows = fetch("BTCUSDT", start, end, get=api.get)
        assert [r[0] for r in rows] == [r[0] for r in api.minutes]
        assert len(api.calls) == -(-4_000 // page)            # one request per page

    def test_values_are_kept_as_printed(self, api):
        rows = fp.bybit_premium_1m("BTCUSDT", T0, T0 + MIN * 10, get=api.get)
        assert rows[1] == (T0 + MIN, "-0.00011", "-0.00041", "-0.00011", "-0.00041", "")
        rows = fp.binance_premium_1m("BTCUSDT", T0, T0 + MIN * 10, get=api.get)
        assert rows[1] == (T0 + MIN, "-0.00011", "-0.00011", "-0.00041", "-0.00041", "12")

    def test_the_span_is_closed_and_nothing_outside_it_is_kept(self, api):
        rows = fp.binance_premium_1m("BTCUSDT", T0 + MIN, T0 + MIN * 3, get=api.get)
        assert [r[0] for r in rows] == [T0 + MIN, T0 + 2 * MIN, T0 + 3 * MIN]

    def test_a_missing_minute_stays_missing(self):
        api = FakeAPI(minutes=200, missing=(70,))
        rows = fp.bybit_premium_1m("BTCUSDT", T0, T0 + MIN * 199, get=api.get)
        assert len(rows) == 199
        assert T0 + 70 * MIN not in {r[0] for r in rows}


class TestSettlementsAndPrices:
    @pytest.mark.parametrize("fetch", [fp.binance_funding, fp.bybit_funding])
    def test_every_settlement_once_with_its_printed_rate(self, api, fetch):
        rows = fetch("ETHUSDT", "2023-12-12 00:00", "2024-07-20 00:00", get=api.get)
        expected = [s for s in api.settle if s[0] <= pd.Timestamp("2024-07-20", tz="UTC").value
                    // 1_000_000 + 60_000]
        assert len(rows) == len(expected)
        assert rows[0] == {"time": "2023-12-12 00:00:00+00:00", "fundingRate": "0.00010000"}
        assert rows[1]["time"] == "2023-12-12 08:00:00+00:00"       # +13 ms of jitter
        assert all("ETHUSDT" in u for u in api.calls)

    def test_prices_are_hourly_closes_at_open_times(self, api):
        rows = fp.binance_price_1h("BTCUSDT", "2023-12-11 22:00", "2024-01-10 00:00",
                                   get=api.get)
        assert rows[0] == {"open_time": "2023-12-11 22:00:00+00:00", "close": "40000.5"}
        assert len(rows) == len({r["open_time"] for r in rows})

    def test_the_stablecoin_pair_comes_from_the_spot_market_in_pages_of_1000(self, api):
        assert fp.STABLECOIN_PAIR == "USDCUSDT"
        rows = fp.binance_price_1h(fp.STABLECOIN_PAIR, "2023-12-11 22:00", "2024-01-10 00:00",
                                   get=api.get, spot=True)
        assert len(rows) == 699 and rows[0]["close"] == "1.0001"
        assert all("api.binance.com/api/v3/klines" in u and "limit=1000" in u
                   for u in api.calls)


class TestHyperliquid:
    def test_stamps_are_floored_to_the_hour_and_strings_kept(self, api):
        rows = fp.hyperliquid_funding("BTC", "2023-12-11 23:00", "2024-01-30 00:00",
                                      post=api.post)
        assert rows[0] == {"time": "2023-12-11 23:00:00+00:00", "fundingRate": "0.0000125",
                           "premium": "-0.00015"}
        # the fourth record is stamped almost fifteen minutes late; it is that hour's record
        assert rows[2]["time"] == "2023-12-12 01:00:00+00:00"
        times = [r["time"] for r in rows]
        assert times == sorted(times) and len(set(times)) == len(times)
        assert times[-1] == "2024-01-30 00:00:00+00:00"
        assert len(api.calls) > 1

    def test_a_record_before_the_start_is_not_kept(self, api):
        rows = fp.hyperliquid_funding("BTC", "2023-12-12 00:00", "2023-12-12 03:00",
                                      post=api.post)
        assert [r["time"][11:16] for r in rows] == ["00:00", "01:00", "02:00", "03:00"]


def _minutes(values, start=T0, missing=()):
    return pd.DataFrame({"open_ms": [start + MIN * i for i in range(len(values))
                                     if i not in missing],
                         "close": [f"{v:.8f}" for i, v in enumerate(values) if i not in missing]})


class TestHourlyAggregation:
    def test_the_hour_stamped_t_holds_the_minutes_opening_in_the_hour_before(self):
        vals = [1e-4] * 60 + [3e-4] * 60             # 22:00-22:59 and 23:00-23:59
        out = fp.hourly_from_minutes(_minutes(vals), "2023-12-11 23:00", "2023-12-12 00:00")
        assert list(out["time"]) == ["2023-12-11 23:00:00+00:00", "2023-12-12 00:00:00+00:00"]
        assert out["mean"].astype(float).tolist() == pytest.approx([1e-4, 3e-4])
        assert out["n"].tolist() == [60, 60]

    def test_linear_weights_rise_with_the_minute_and_close_is_the_last_minute(self):
        vals = list(np.arange(60) * 1e-6)              # x_j = (j - 1) * 1e-6, j = 1..60
        out = fp.hourly_from_minutes(_minutes(vals), "2023-12-11 23:00", "2023-12-11 23:00")
        j = np.arange(1, 61)
        assert float(out["lwmean"].iloc[0]) == pytest.approx((j * (j - 1) * 1e-6).sum()
                                                            / j.sum(), abs=1e-10)
        assert float(out["mean"].iloc[0]) == pytest.approx(29.5e-6, abs=1e-10)
        assert out["close"].iloc[0] == "0.00005900"

    def test_missing_minutes_are_counted_and_a_missing_last_minute_has_no_close(self):
        vals = [2e-4] * 60
        out = fp.hourly_from_minutes(_minutes(vals, missing=(5, 59)),
                                     "2023-12-11 23:00", "2023-12-12 00:00")
        assert out["n"].tolist() == [58, 0]
        assert out["close"].iloc[0] == ""
        assert out["mean"].iloc[1] == ""                  # an empty hour stays empty

    def test_eight_hour_linear_mean_from_hourly_aggregates_equals_the_minute_sum(self):
        rng = np.random.default_rng(7)
        vals = rng.normal(-4.5e-4, 1e-4, 8 * 60)
        hourly = fp.hourly_from_minutes(_minutes(vals, start=T0 + 2 * HOUR),
                                        "2023-12-12 01:00", "2023-12-12 08:00")
        direct = (np.arange(1, 481) * np.round(vals, 8)).sum() / np.arange(1, 481).sum()
        rebuilt = fp.eight_hour_linear_mean(hourly["mean"].astype(float).to_numpy(),
                                            hourly["lwmean"].astype(float).to_numpy())
        assert rebuilt == pytest.approx(direct, abs=1e-10)


class TestFiles:
    def test_fetch_all_writes_every_file_and_keeps_the_minutes_when_asked(self, api, tmp_path):
        raw = tmp_path / "raw"
        counts = fp.fetch_all("2023-12-11 23:00", "2023-12-13 00:00", tmp_path / "out",
                              assets=("BTC",), get=api.get, post=api.post, raw_dir=raw)
        names = sorted(p.name for p in (tmp_path / "out").iterdir())
        assert names == sorted(counts)
        assert names == ["binance_BTCUSDT_funding.csv", "binance_BTCUSDT_premium_hourly.csv",
                         "binance_BTCUSDT_price_1h.csv", "binance_USDCUSDT_spot_1h.csv",
                         "bybit_BTCUSDT_funding.csv", "bybit_BTCUSDT_premium_hourly.csv",
                         "hyperliquid_BTC_funding.csv"]
        hourly = pd.read_csv(tmp_path / "out" / "bybit_BTCUSDT_premium_hourly.csv")
        assert len(hourly) == 26                           # 23:00 on 11 Dec to 00:00 on 13 Dec
        with gzip.open(raw / "bybit_BTCUSDT_premium_1m.csv.gz", "rt") as fh:
            assert fh.readline().strip() == "open_ms,open,high,low,close,n"

    def test_aggregates_rebuild_from_kept_minutes(self, api, tmp_path):
        raw = tmp_path / "raw"
        fp.fetch_all("2023-12-11 23:00", "2023-12-13 00:00", tmp_path / "a", assets=("BTC",),
                     get=api.get, post=api.post, raw_dir=raw)
        fp.aggregate_raw(raw, "2023-12-11 23:00", "2023-12-13 00:00", tmp_path / "b",
                         assets=("BTC",))
        for venue in ("binance", "bybit"):
            name = f"{venue}_BTCUSDT_premium_hourly.csv"
            assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()

    def test_kept_minutes_that_do_not_cover_the_span_are_fetched_again(self, api, tmp_path):
        raw = tmp_path / "raw"
        fp.fetch_all("2023-12-12 00:00", "2023-12-12 06:00", tmp_path / "a", assets=("BTC",),
                     get=api.get, post=api.post, raw_dir=raw)
        calls = len(api.calls)
        # the same span again: the kept minutes are reused, no kline is requested
        fp.fetch_all("2023-12-12 00:00", "2023-12-12 06:00", tmp_path / "a", assets=("BTC",),
                     get=api.get, post=api.post, raw_dir=raw)
        assert not [u for u in api.calls[calls:] if "premium" in u.lower()]
        # an earlier start needs minutes the kept file lacks
        out = fp.fetch_all("2023-12-11 23:00", "2023-12-12 06:00", tmp_path / "b",
                           assets=("BTC",), get=api.get, post=api.post, raw_dir=raw)
        hourly = pd.read_csv(tmp_path / "b" / "binance_BTCUSDT_premium_hourly.csv")
        assert out["binance_BTCUSDT_premium_hourly.csv"] == 8
        assert (hourly["n"] == 60).all()
