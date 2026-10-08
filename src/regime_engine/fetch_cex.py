"""Funding and premium history of the BTCUSDT perpetuals on Binance and Bybit.

Both venues serve their history through free, unauthenticated REST endpoints.
This module pages through four of them and writes one CSV each to
``data/interim/``, keeping every rate exactly as the venue prints it, as a
decimal string, so that "exactly at the floor" is decided by the venue's own
digits and not by a float tolerance.

    python3 -m regime_engine.fetch_cex --start 2025-10-01 --end "2026-10-01 00:00"
    python3 -m regime_engine.fetch_cex --self-test      # offline

Network access happens only when the module is run; every function takes the
HTTP getter as an argument so the tests can serve pages from memory.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import urllib.request

import pandas as pd

from .paths import REPO_ROOT

BINANCE = "https://fapi.binance.com/fapi/v1"
BYBIT = "https://api.bybit.com/v5/market"
SYMBOL = "BTCUSDT"
HOUR_MS = 3_600_000
MINUTE_MS = 60_000


def get_json(url: str, retries: int = 4):
    """GET ``url`` and decode JSON, retrying with exponential back-off."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "regime-engine/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _ms(ts) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _iso(ms) -> str:
    # Settlement stamps carry a few milliseconds of jitter; round to the minute.
    return str(pd.Timestamp(int(ms), unit="ms", tz="UTC").round("min"))


def _span(start, end) -> tuple:
    # Ask for a minute more on each side: a settlement stamped a few
    # milliseconds after ``end`` still belongs to it. ``_dedup`` trims back.
    return _ms(start) - MINUTE_MS, _ms(end) + MINUTE_MS


def _dedup(rows: list, start, end) -> list:
    """One record per rounded stamp inside [start, end], in time order."""
    lo, hi = _iso(_ms(start)), _iso(_ms(end))
    by_time = {r["time"]: r for r in rows if lo <= r["time"] <= hi}
    return [by_time[t] for t in sorted(by_time)]


def binance_funding(start, end, get=get_json) -> list:
    rows, (cursor, stop) = [], _span(start, end)
    while cursor <= stop:
        page = get(f"{BINANCE}/fundingRate?symbol={SYMBOL}"
                   f"&startTime={cursor}&endTime={stop}&limit=1000")
        if not page:
            break
        rows += [{"time": _iso(x["fundingTime"]), "fundingRate": x["fundingRate"],
                  "markPrice": x.get("markPrice", "")} for x in page]
        if len(page) < 1000:
            break
        cursor = int(page[-1]["fundingTime"]) + 1
    return _dedup(rows, start, end)


def binance_premium_1h(start, end, get=get_json) -> list:
    rows, (cursor, stop) = [], _span(start, end)
    while cursor <= stop:
        page = get(f"{BINANCE}/premiumIndexKlines?symbol={SYMBOL}&interval=1h"
                   f"&startTime={cursor}&endTime={stop}&limit=1500")
        if not page:
            break
        rows += [{"time": _iso(k[0]), "close": k[4]} for k in page]
        if len(page) < 1500:
            break
        cursor = int(page[-1][0]) + HOUR_MS
    return _dedup(rows, start, end)


def bybit_funding(start, end, get=get_json) -> list:
    rows, (first, cursor) = [], _span(start, end)
    while cursor >= first:
        page = get(f"{BYBIT}/funding/history?category=linear&symbol={SYMBOL}"
                   f"&startTime={first}&endTime={cursor}&limit=200")["result"]["list"]
        if not page:
            break
        rows += [{"time": _iso(x["fundingRateTimestamp"]), "fundingRate": x["fundingRate"]}
                 for x in page]
        if len(page) < 200:
            break
        cursor = min(int(x["fundingRateTimestamp"]) for x in page) - 1
    return _dedup(rows, start, end)


def bybit_premium_1h(start, end, get=get_json) -> list:
    rows, (first, cursor) = [], _span(start, end)
    while cursor >= first:
        page = get(f"{BYBIT}/premium-index-price-kline?category=linear&symbol={SYMBOL}"
                   f"&interval=60&start={first}&end={cursor}&limit=1000")["result"]["list"]
        if not page:
            break
        rows += [{"time": _iso(k[0]), "close": k[4]} for k in page]
        if len(page) < 1000:
            break
        cursor = min(int(k[0]) for k in page) - 1
    return _dedup(rows, start, end)


FILES = {
    "binance_BTCUSDT_funding.csv": binance_funding,
    "binance_BTCUSDT_premium_1h.csv": binance_premium_1h,
    "bybit_BTCUSDT_funding.csv": bybit_funding,
    "bybit_BTCUSDT_premium_1h.csv": bybit_premium_1h,
}


def write_csv(rows: list, path) -> None:
    with pathlib.Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fetch_all(start, end, out_dir, get=get_json) -> dict:
    """Fetch all four series and write them to ``out_dir``; return the row counts."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, fetch in FILES.items():
        rows = fetch(start, end, get=get)
        if not rows:
            raise RuntimeError(f"{name}: the endpoint returned no records")
        write_csv(rows, out / name)
        counts[name] = len(rows)
    return counts


def run_self_test() -> int:
    """Offline check of parsing and pagination against two in-memory pages."""
    t0 = _ms("2025-10-02 00:00")
    pages = {
        "binance": [{"fundingTime": t0 + 13, "fundingRate": "0.00010000", "markPrice": "1"},
                    {"fundingTime": t0 + 8 * HOUR_MS, "fundingRate": "0.00002500",
                     "markPrice": "1"}],
        "bybit": {"result": {"list": [
            {"fundingRateTimestamp": str(t0 + 8 * HOUR_MS), "fundingRate": "0.0001"},
            {"fundingRateTimestamp": str(t0), "fundingRate": "0.00005"}]}},
    }

    def fake(url):
        return pages["binance"] if "binance" in url else pages["bybit"]

    b = binance_funding("2025-10-02", "2025-10-03", get=fake)
    y = bybit_funding("2025-10-02", "2025-10-03", get=fake)
    ok = (b[0] == {"time": "2025-10-02 00:00:00+00:00", "fundingRate": "0.00010000",
                   "markPrice": "1"}
          and [r["fundingRate"] for r in y] == ["0.00005", "0.0001"])
    print("fetch_cex self-test:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m regime_engine.fetch_cex",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-10-01 00:00")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "interim"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    for name, n in fetch_all(args.start, args.end, args.out).items():
        print(f"{name}: {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
