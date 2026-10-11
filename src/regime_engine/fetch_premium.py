"""Premium histories of Hyperliquid, Binance and Bybit, BTC and ETH, from December 2023.

    python3 -m regime_engine.fetch_premium                   # fetch and aggregate (about 1 h)
    python3 -m regime_engine.fetch_premium --raw-dir DIR     # also keep the one-minute klines
    python3 -m regime_engine.fetch_premium --from-raw DIR    # rebuild the hourly files offline

Every source is a free, unauthenticated REST endpoint:

    Hyperliquid  POST api.hyperliquid.xyz/info, type fundingHistory (hourly)
    Binance      GET  fapi/v1/premiumIndexKlines (1m), fapi/v1/fundingRate, fapi/v1/klines (1h),
                      api/v3/klines (spot USDCUSDT, 1h)
    Bybit        GET  v5/market/premium-index-price-kline (1), v5/market/funding/history

and the module writes ``data/premium_gap/``:

    hyperliquid_<COIN>_funding.csv          time, fundingRate, premium
    <venue>_<SYMBOL>_premium_hourly.csv     time, n, mean, lwmean, close
    <venue>_<SYMBOL>_funding.csv            time, fundingRate
    binance_<SYMBOL>_price_1h.csv           open_time, close
    binance_USDCUSDT_spot_1h.csv            open_time, close (USDC in USDT, spot)

Span. Hourly records are stamped 11 Dec 2023 17:00 to 1 Oct 2026 00:00 UTC.
Hyperliquid has settled under the +-0.05% clamp from the hour stamped 23:00 on
11 Dec 2023; the six hours before it complete the eight-hour block that closes at
the first exchange settlement of the comparison, 12 Dec 2023 00:00.

Alignment. Hyperliquid's funding record stamped t carries the premium averaged over
the hour [t - 1h, t); its stamps arrive up to a quarter of an hour late and are
floored to the hour. Kline stamps are open times. The exchanges' hourly record
stamped t is built from the one-minute klines that open in [t - 1h, t), so that
it covers the same hour as Hyperliquid's record t: ``n`` minutes, the mean of
their closes (``mean``), the closes weighted by the minute's position 1..60 in
the hour (``lwmean``), and the close of the minute that opens at t - 1 min
(``close``: the hour's last premium index, which an hourly kline prints as its
close). From ``mean`` and ``lwmean`` the eight-hour average with linearly rising
weights, on which the exchanges settle, is rebuilt exactly
(:func:`eight_hour_linear_mean`).

Rates, premiums and closes are kept as the venues print them, as decimal
strings; the hourly means carry ten decimals (1e-6 bp). Requests are spaced per
host (:data:`MIN_INTERVAL`) to stay well inside the venues' rate limits.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import fetch_cex
from .paths import REPO_ROOT

HL_URL = "https://api.hyperliquid.xyz/info"
BINANCE = "https://fapi.binance.com/fapi/v1"
BINANCE_SPOT = "https://api.binance.com/api/v3"
BYBIT = "https://api.bybit.com/v5/market"
MINUTE_MS = 60_000
HOUR_MS = 3_600_000
HL_PAGE = 500                       # fundingHistory returns at most 500 records per call

# 11 Dec 2023 23:00 UTC is the first Hyperliquid hour settled under the +-0.05%
# clamp (from 19 Jul 2023 up to that hour, +-0.03%). The files start six hours
# earlier, so that the eight-hour block closing at the first exchange settlement
# under the common rule (12 Dec 2023 00:00) is complete. 1 Oct 2026 00:00 UTC is
# the last stamp.
START = "2023-12-11 17:00"
RULE_START = "2023-12-11 23:00"
END = "2026-10-01 00:00"
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
EXCHANGES = ("binance", "bybit")
OUT_DIR = REPO_ROOT / "data" / "premium_gap"
MINUTE_COLUMNS = ("open_ms", "open", "high", "low", "close", "n")
HOURLY_COLUMNS = ("time", "n", "mean", "lwmean", "close")

# Seconds between two requests to the same host.
MIN_INTERVAL = {"fapi.binance.com": 0.4, "api.binance.com": 0.4, "api.bybit.com": 0.2,
                "api.hyperliquid.xyz": 0.3}
# USDC priced in USDT. Hyperliquid margins its BTC and ETH perpetuals in USDC but
# denominates them, and its oracle, in USDT, without conversion (quanto
# contracts); Binance and Bybit margin and denominate theirs in USDT. The rate
# lets the analysis check whether traders price Hyperliquid's book in USDC.
STABLECOIN_PAIR = "USDCUSDT"
_last_call: dict = {}


def _throttle(url: str) -> None:
    host = urllib.parse.urlparse(url).netloc
    wait = MIN_INTERVAL.get(host, 0.5) - (time.monotonic() - _last_call.get(host, -1e9))
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


def _request(req: urllib.request.Request, retries: int):
    for attempt in range(retries):
        _throttle(req.full_url)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if attempt == retries - 1 or exc.code not in (418, 429, 500, 502, 503, 504):
                raise
            time.sleep(float(exc.headers.get("Retry-After") or 2 ** (attempt + 1)))
        except (urllib.error.URLError, OSError, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def get_json(url: str, retries: int = 6):
    """GET ``url`` and decode JSON, spaced per host and retried with back-off."""
    return _request(urllib.request.Request(url, headers={"User-Agent": "regime-engine/1.0"}),
                    retries)


def post_json(url: str, body: dict, retries: int = 6):
    """POST ``body`` as JSON to ``url`` and decode the answer, as :func:`get_json`."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "regime-engine/1.0"})
    return _request(req, retries)


def _ms(ts) -> int:
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // 1_000_000)


def _iso(ms: int) -> str:
    return str(pd.Timestamp(int(ms), unit="ms", tz="UTC"))


# ----------------------------------------------------------------------
#  One-minute premium-index klines
# ----------------------------------------------------------------------
def _minute_pages(lo: int, hi: int, size: int):
    """Consecutive spans of ``size`` minutes, both ends included, covering [lo, hi]."""
    t = lo
    while t <= hi:
        yield t, min(hi, t + (size - 1) * MINUTE_MS)
        t += size * MINUTE_MS


def binance_premium_1m(symbol: str, start, end, get=get_json) -> list:
    """Binance's one-minute premium-index klines opening in [start, end], oldest first.

    Each row is (open_ms, open, high, low, close, n) with the values as printed;
    ``n`` is the number of premium-index samples in the minute (12, one every 5 s).
    """
    lo, hi = _ms(start), _ms(end)
    rows = {}
    for a, b in _minute_pages(lo, hi, 1500):
        page = get(f"{BINANCE}/premiumIndexKlines?symbol={symbol}&interval=1m"
                   f"&startTime={a}&endTime={b}&limit=1500")
        for k in page:
            t = int(k[0])
            if lo <= t <= hi:
                rows[t] = (t, k[1], k[2], k[3], k[4], str(k[8]))
    return [rows[t] for t in sorted(rows)]


def bybit_premium_1m(symbol: str, start, end, get=get_json) -> list:
    """Bybit's one-minute premium-index klines opening in [start, end], oldest first.

    The endpoint answers newest first; each page is asked for an explicit span, so
    the order of the answer does not matter. Bybit computes one premium index per
    minute: a kline's open repeats the previous close. ``n`` is left empty.
    """
    lo, hi = _ms(start), _ms(end)
    rows = {}
    for a, b in _minute_pages(lo, hi, 1000):
        res = get(f"{BYBIT}/premium-index-price-kline?category=linear&symbol={symbol}"
                  f"&interval=1&start={a}&end={b}&limit=1000")
        if res.get("retCode", 0) != 0:
            raise RuntimeError(f"Bybit {symbol}: {res.get('retMsg')}")
        for k in res["result"]["list"]:
            t = int(k[0])
            if lo <= t <= hi:
                rows[t] = (t, k[1], k[2], k[3], k[4], "")
    return [rows[t] for t in sorted(rows)]


MINUTE_FETCHERS = {"binance": binance_premium_1m, "bybit": bybit_premium_1m}


# ----------------------------------------------------------------------
#  Funding settlements, prices and Hyperliquid's funding history
# ----------------------------------------------------------------------
def binance_funding(symbol: str, start, end, get=get_json) -> list:
    """Binance settlements stamped in [start, end], rates as printed, stamps to the minute."""
    rows, (cursor, stop) = [], fetch_cex._span(start, end)
    while cursor <= stop:
        page = get(f"{BINANCE}/fundingRate?symbol={symbol}"
                   f"&startTime={cursor}&endTime={stop}&limit=1000")
        if not page:
            break
        rows += [{"time": fetch_cex._iso(x["fundingTime"]), "fundingRate": x["fundingRate"]}
                 for x in page]
        if len(page) < 1000:
            break
        cursor = int(page[-1]["fundingTime"]) + 1
    return fetch_cex._dedup(rows, start, end)


def bybit_funding(symbol: str, start, end, get=get_json) -> list:
    """Bybit settlements stamped in [start, end]; the endpoint pages newest first."""
    rows, (first, cursor) = [], fetch_cex._span(start, end)
    while cursor >= first:
        page = get(f"{BYBIT}/funding/history?category=linear&symbol={symbol}"
                   f"&startTime={first}&endTime={cursor}&limit=200")["result"]["list"]
        if not page:
            break
        rows += [{"time": fetch_cex._iso(x["fundingRateTimestamp"]),
                  "fundingRate": x["fundingRate"]} for x in page]
        if len(page) < 200:
            break
        cursor = min(int(x["fundingRateTimestamp"]) for x in page) - 1
    return fetch_cex._dedup(rows, start, end)


FUNDING_FETCHERS = {"binance": binance_funding, "bybit": bybit_funding}


def binance_price_1h(symbol: str, start, end, get=get_json, spot: bool = False) -> list:
    """Closes of Binance's hourly klines opening in [start, end]: perpetual, or ``spot``."""
    lo, hi = _ms(start), _ms(end)
    base, limit = (BINANCE_SPOT, 1000) if spot else (BINANCE, 1500)
    rows, cursor = {}, lo
    while cursor <= hi:
        page = get(f"{base}/klines?symbol={symbol}&interval=1h"
                   f"&startTime={cursor}&endTime={hi}&limit={limit}")
        if not page:
            break
        for k in page:
            if lo <= int(k[0]) <= hi:
                rows[int(k[0])] = {"open_time": _iso(k[0]), "close": k[4]}
        if len(page) < limit:
            break
        cursor = int(page[-1][0]) + HOUR_MS
    return [rows[t] for t in sorted(rows)]


def hyperliquid_funding(coin: str, start, end, post=post_json) -> list:
    """Hyperliquid's hourly records for the hours stamped in [start, end].

    A record is stamped when the hour it averages closes, sometimes minutes late;
    the stamp is floored to the hour, and the first record of an hour is kept.
    Funding rate and premium stay the strings the endpoint prints.
    """
    lo, hi = _ms(start), _ms(end)
    stop = hi + HOUR_MS - 1                       # a late record still belongs to its hour
    raw, cursor = [], lo
    while cursor <= stop:
        page = post(HL_URL, {"type": "fundingHistory", "coin": coin,
                             "startTime": cursor, "endTime": stop})
        if not page:
            break
        raw += page
        if len(page) < HL_PAGE:
            break
        cursor = max(int(x["time"]) for x in page) + 1
    rows = {}
    for x in sorted(raw, key=lambda r: int(r["time"])):
        hour = int(x["time"]) // HOUR_MS * HOUR_MS
        if lo <= hour <= hi and hour not in rows:
            rows[hour] = {"time": _iso(hour), "fundingRate": x["fundingRate"],
                          "premium": x["premium"]}
    return [rows[t] for t in sorted(rows)]


# ----------------------------------------------------------------------
#  Hourly aggregates
# ----------------------------------------------------------------------
def _fmt(x: float) -> str:
    return "" if x != x else f"{x:.10f}"


def hourly_from_minutes(minutes: pd.DataFrame, first, last) -> pd.DataFrame:
    """Hourly records stamped ``first`` .. ``last`` from one-minute klines, as CSV strings.

    The record stamped t holds the minutes that open in [t - 1h, t): their number
    ``n``, their mean close, their close weighted by the minute's position 1..60
    in the hour (``lwmean``), and the close of the minute opening at t - 1 min. An
    hour without that minute has no ``close``; an hour without minutes has no means.
    """
    t = minutes["open_ms"].astype("int64").to_numpy()
    x = minutes["close"].astype(float).to_numpy()
    stamp = (t // HOUR_MS + 1) * HOUR_MS
    j = (t % HOUR_MS) // MINUTE_MS + 1
    g = pd.DataFrame({"stamp": stamp, "x": x, "jx": j * x, "j": j}).groupby("stamp").sum()
    g["n"] = pd.Series(stamp).value_counts()
    last_minute = pd.Series(minutes["close"].astype(str).to_numpy()[j == 60],
                            index=stamp[j == 60])
    stamps = np.arange(_ms(first), _ms(last) + 1, HOUR_MS)
    g = g.reindex(stamps)
    n = g["n"].fillna(0).astype(int)
    return pd.DataFrame({
        "time": [_iso(s) for s in stamps],
        "n": n.to_numpy(),
        "mean": [_fmt(v) for v in (g["x"] / g["n"]).to_numpy()],
        "lwmean": [_fmt(v) for v in (g["jx"] / g["j"]).to_numpy()],
        "close": last_minute.reindex(stamps).fillna("").to_numpy(),
    })


def eight_hour_linear_mean(mean, lwmean, minutes_per_hour: int = 60) -> float:
    """The average with weights 1, 2, ... rising over consecutive full hours of minutes.

    ``mean`` and ``lwmean`` are the hours' plain and position-weighted means,
    earliest hour first. Minute j of hour h carries weight m*h + j, so the hour
    contributes m*m*h*mean_h + m(m+1)/2*lwmean_h to a total weight of N(N+1)/2,
    N = m * hours.
    """
    m = minutes_per_hour
    mean, lwmean = np.asarray(mean, dtype=float), np.asarray(lwmean, dtype=float)
    h = np.arange(len(mean))
    total = len(mean) * m
    return float((m * m * h * mean + m * (m + 1) / 2 * lwmean).sum() / (total * (total + 1) / 2))


# ----------------------------------------------------------------------
#  Files
# ----------------------------------------------------------------------
def write_csv(rows, path, columns=None) -> int:
    """Write dict rows (or a DataFrame) as CSV with "\\n" line ends; return the row count."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows), columns=columns)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(list(frame.columns))
        writer.writerows(frame.itertuples(index=False, name=None))
    return len(frame)


def _raw_path(raw_dir, venue: str, symbol: str) -> pathlib.Path:
    return pathlib.Path(raw_dir) / f"{venue}_{symbol}_premium_1m.csv.gz"


def write_minutes(rows: list, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(MINUTE_COLUMNS)
        writer.writerows(rows)


def read_minutes(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _minute_span(start, end) -> tuple:
    """The minutes behind the hours stamped start .. end: open times start - 1h .. end - 1 min."""
    return _ms(start) - HOUR_MS, _ms(end) - MINUTE_MS


def _covers(minutes: pd.DataFrame, start, end) -> bool:
    """Whether kept minutes reach from the first to the last minute the span needs."""
    lo, hi = _minute_span(start, end)
    t = minutes["open_ms"].astype("int64")
    return len(t) > 0 and int(t.min()) <= lo and int(t.max()) >= hi


def fetch_all(start=START, end=END, out_dir=OUT_DIR, assets=tuple(ASSETS), get=get_json,
              post=post_json, raw_dir=None, log=print) -> dict:
    """Fetch every series for ``assets`` and write the files; return their row counts.

    With ``raw_dir``, the one-minute klines are also written there (gzip CSV), and
    a download already there is reused instead of fetched again.
    """
    out = pathlib.Path(out_dir)
    counts = {}
    for coin in assets:
        symbol = ASSETS[coin]
        name = f"hyperliquid_{coin}_funding.csv"
        counts[name] = write_csv(hyperliquid_funding(coin, start, end, post=post), out / name,
                                 ["time", "fundingRate", "premium"])
        log(f"{name}: {counts[name]} rows")
        for venue in EXCHANGES:
            raw = _raw_path(raw_dir, venue, symbol) if raw_dir else None
            minutes = read_minutes(raw) if raw is not None and raw.exists() else None
            if minutes is not None and _covers(minutes, start, end):
                log(f"{raw.name}: reusing {len(minutes)} minutes")
            else:
                rows = MINUTE_FETCHERS[venue](symbol, *_minute_span(start, end), get=get)
                if raw is not None:
                    write_minutes(rows, raw)
                minutes = pd.DataFrame(rows, columns=MINUTE_COLUMNS)
            name = f"{venue}_{symbol}_premium_hourly.csv"
            hourly = hourly_from_minutes(minutes, start, end)
            counts[name] = write_csv(hourly, out / name)
            log(f"{name}: {counts[name]} hours, {int((hourly['n'] < 60).sum())} incomplete")
            name = f"{venue}_{symbol}_funding.csv"
            counts[name] = write_csv(FUNDING_FETCHERS[venue](symbol, start, end, get=get),
                                     out / name, ["time", "fundingRate"])
            log(f"{name}: {counts[name]} settlements")
        name = f"binance_{symbol}_price_1h.csv"
        counts[name] = write_csv(binance_price_1h(symbol, _ms(start) - HOUR_MS,
                                                  _ms(end) - HOUR_MS, get=get),
                                 out / name, ["open_time", "close"])
        log(f"{name}: {counts[name]} hours")
    name = f"binance_{STABLECOIN_PAIR}_spot_1h.csv"
    counts[name] = write_csv(binance_price_1h(STABLECOIN_PAIR, _ms(start) - HOUR_MS,
                                              _ms(end) - HOUR_MS, get=get, spot=True),
                             out / name, ["open_time", "close"])
    log(f"{name}: {counts[name]} hours")
    return counts


def aggregate_raw(raw_dir, start=START, end=END, out_dir=OUT_DIR, assets=tuple(ASSETS)) -> dict:
    """Rebuild the exchanges' hourly files from one-minute klines kept by :func:`fetch_all`."""
    counts = {}
    for coin in assets:
        for venue in EXCHANGES:
            symbol = ASSETS[coin]
            name = f"{venue}_{symbol}_premium_hourly.csv"
            minutes = read_minutes(_raw_path(raw_dir, venue, symbol))
            counts[name] = write_csv(hourly_from_minutes(minutes, start, end),
                                     pathlib.Path(out_dir) / name)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m regime_engine.fetch_premium",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--assets", nargs="+", default=list(ASSETS), choices=sorted(ASSETS))
    ap.add_argument("--raw-dir", help="keep the one-minute klines here (and reuse them)")
    ap.add_argument("--from-raw", metavar="DIR",
                    help="only rebuild the hourly premium files from klines kept in DIR")
    args = ap.parse_args(argv)
    if args.from_raw:
        for name, n in aggregate_raw(args.from_raw, args.start, args.end, args.out,
                                     args.assets).items():
            print(f"{name}: {n} hours")
    else:
        fetch_all(args.start, args.end, args.out, args.assets, raw_dir=args.raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
