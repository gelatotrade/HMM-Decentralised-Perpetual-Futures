"""Binance and Bybit history fetchers, against a fake API (no network).

The fake honours what the real endpoints do: a page limit, ascending pages for
Binance (``startTime``/``endTime``), newest-first pages for Bybit (``endTime`` or
``end`` moving backwards), and settlement stamps with a few milliseconds of
jitter. The fetchers must return every record once, in time order, with each
rate exactly as the venue printed it.
"""
from urllib.parse import parse_qs, urlparse

import pytest

from regime_engine import fetch_cex as f

HOUR = 3_600_000
T0 = 1_759_363_200_000            # 2025-10-02 00:00 UTC


def _q(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


class FakeAPI:
    def __init__(self, n_funding=2_300, n_premium=3_200):
        jitter = [0, 13, 0, 6]
        self.funding = [(T0 + 8 * HOUR * i + jitter[i % 4],
                         "0.00010000" if i % 3 == 0 else f"0.0000{i % 9}123")
                        for i in range(n_funding)]
        self.premium = [(T0 + HOUR * i, f"-0.000{i % 7}5") for i in range(n_premium)]
        self.calls = 0

    def __call__(self, url):
        self.calls += 1
        q = _q(url)
        if "fapi.binance.com" in url:
            rows = self.funding if "fundingRate" in url else self.premium
            lo, hi, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
            page = [r for r in rows if lo <= r[0] <= hi][:lim]
            if "fundingRate" in url:
                return [{"symbol": "BTCUSDT", "fundingTime": t, "fundingRate": v,
                         "markPrice": "100000.0"} for t, v in page]
            return [[t, v, v, v, v, "0", t + HOUR - 1, "0", 60, "0", "0", "0"] for t, v in page]
        if "api.bybit.com" in url:
            funding = "funding/history" in url
            rows = self.funding if funding else self.premium
            lo = int(q["startTime" if funding else "start"])
            hi = int(q["endTime" if funding else "end"])
            lim = int(q["limit"])
            page = sorted((r for r in rows if lo <= r[0] <= hi), reverse=True)[:lim]
            if funding:
                return {"retCode": 0, "result": {"list": [
                    {"symbol": "BTCUSDT", "fundingRate": v, "fundingRateTimestamp": str(t)}
                    for t, v in page]}}
            return {"retCode": 0, "result": {"list": [[str(t), v, v, v, v] for t, v in page]}}
        raise AssertionError(f"unexpected url {url}")


START, END = "2025-10-02 00:00", "2026-08-27 08:00"


@pytest.fixture
def api():
    return FakeAPI()


# The funding span runs past the paper window so that Binance's 1,000-record
# pages, not just Bybit's 200, have to be turned more than once.
LONG_END = "2027-12-31 00:00"


@pytest.mark.parametrize("fetch,kind,end", [
    (f.binance_funding, "funding", LONG_END), (f.bybit_funding, "funding", LONG_END),
    (f.binance_premium_1h, "premium", END), (f.bybit_premium_1h, "premium", END),
])
def test_every_record_once_in_order(api, fetch, kind, end):
    rows = fetch(START, end, get=api)
    source = api.funding if kind == "funding" else api.premium
    expected = [r for r in source if r[0] <= f._ms(end)]
    assert len(rows) == len(expected)
    assert [r["time"] for r in rows] == sorted(r["time"] for r in rows)
    assert len({r["time"] for r in rows}) == len(rows)
    assert api.calls > 1                       # more than one page was needed


def test_rates_keep_the_printed_string(api):
    rows = f.binance_funding(START, END, get=api)
    assert rows[0]["fundingRate"] == "0.00010000"
    assert rows[1]["fundingRate"] == "0.00001123"


def test_jitter_is_rounded_to_the_minute(api):
    rows = f.binance_funding(START, END, get=api)
    assert rows[1]["time"] == "2025-10-02 08:00:00+00:00"   # fundingTime carried +13 ms


@pytest.mark.parametrize("fetch", [f.binance_funding, f.bybit_funding])
def test_a_settlement_stamped_just_after_the_end_is_kept(api, fetch):
    # The 08:00 settlement carries +13 ms; asking for everything up to 08:00
    # must still return it, and nothing after it.
    rows = fetch(START, "2025-10-02 08:00", get=api)
    assert [r["time"] for r in rows] == ["2025-10-02 00:00:00+00:00",
                                         "2025-10-02 08:00:00+00:00"]


def test_bybit_pages_backwards_to_the_start(api):
    rows = f.bybit_funding(START, END, get=api)
    assert rows[0]["time"] == "2025-10-02 00:00:00+00:00"


def test_fetch_all_writes_four_files(api, tmp_path):
    counts = f.fetch_all(START, END, tmp_path, get=api)
    assert sorted(counts) == sorted(f.FILES)
    for name in f.FILES:
        text = (tmp_path / name).read_text().splitlines()
        assert text[0].startswith("time,")
        assert len(text) == counts[name] + 1
