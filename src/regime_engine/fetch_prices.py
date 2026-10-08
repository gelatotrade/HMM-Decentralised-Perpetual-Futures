"""Recover Hyperliquid BTC PRICE history for 2 Oct - 2 Dec 2025.

The companion to ``fetch_official.py``, which recovers the funding premium.
This module recovers *prices* for the segment the shipped hourly panel does not
cover.

RETENTION
---------
``candleSnapshot`` retention is a cap on the NUMBER OF RECORDS returned
(~5,000 per interval), not on the age of the data. Consequently the reach of
the endpoint is ``5000 * interval``, and coarser intervals reach much further
back. Verified live on 2026-08-19 (BTC):

    interval   records   earliest served        covers 2025-10-02?
    1h           5001    2026-01-22 23:00       no
    2h           5000    2025-06-28 16:00       YES
    4h           5000    2024-05-08 00:00       YES
    1d           1692    2022-01-01 00:00       YES

So the Oct-Nov 2025 window is still served at 2h and coarser, free, from the
same endpoint and with no credentials.

The 2h series is bit-identical to the shipped 1h panel where they overlap:
1,753 two-hour bars, every one of open/high/low/close/volume matching exactly
(``--validate``). It is the same object, not a close approximation.

DEADLINE
--------
The 2h window advances one day per day. 2 Oct 2025 leaves it on
**22 November 2026**. After that the finest interval covering the sample start
is 4h (good until 2028-01-13). Fetch and archive the data before then.

SOURCES
-------
``candles``        official ``candleSnapshot``. Free, no credentials, verified
                   bit-exact against the shipped panel. The primary source.
``tardis-sample``  Tardis.dev's free first-day-of-month tick archive. Gives
                   TRUE HOURLY closes for 2025-10-01, 11-01 and 12-01. Their
                   2h aggregates match the official candles on 36/36 closes,
                   which independently corroborates ``candles``.
``binance``        Binance BTCUSDT perp klines. A DIFFERENT INSTRUMENT, kept
                   only as a labelled robustness proxy (hourly log-return
                   correlation 0.9995 against the archived panel).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
INTERIM_DIR = REPO_ROOT / "data" / "interim"
RAW_DIR = REPO_ROOT / "data" / "raw" / "tardis"
PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel_BTC_subsample.csv"

REST_URL = "https://api.hyperliquid.xyz/info"
TARDIS_URL = "https://datasets.tardis.dev/v1/hyperliquid/{dtype}/{y}/{m:02d}/{d:02d}/{coin}.csv.gz"
BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"

#: candleSnapshot returns at most this many records per request, per interval.
RECORD_CAP = 5000

#: interval name -> seconds. Only intervals the endpoint actually accepts.
INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}

PRICE_COLUMNS = ["time", "o", "h", "l", "c", "v", "log_return"]

#: the segment of the paper window the shipped hourly panel does not cover
MISSING_START = "2025-10-02"
MISSING_END = "2025-12-02"


class PriceRecoveryError(RuntimeError):
    """Actionable failure: the message is meant to be read by the user."""


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def parse_day(s: str) -> datetime:
    """Parse YYYY-MM-DD or YYYYMMDD into a UTC midnight datetime."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise PriceRecoveryError(
        f"Could not parse date {s!r}. Use YYYY-MM-DD (e.g. 2025-10-02)."
    )


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def interval_seconds(interval: str) -> int:
    try:
        return INTERVAL_SECONDS[interval]
    except KeyError as exc:
        raise PriceRecoveryError(
            "Unknown interval {!r}. Known: {}".format(
                interval, ", ".join(INTERVAL_SECONDS)
            )
        ) from exc


def reach(interval: str, now: datetime | None = None) -> datetime:
    """Earliest timestamp candleSnapshot can serve for `interval`.

    The cap is on record COUNT, so the horizon is RECORD_CAP * interval.

    >>> reach("2h", datetime(2026, 8, 19, tzinfo=timezone.utc)).date().isoformat()
    '2025-06-28'
    """
    now = now or datetime.now(timezone.utc)
    return now - timedelta(seconds=RECORD_CAP * interval_seconds(interval))


def expiry(interval: str, target: datetime) -> datetime:
    """The date on which `target` falls out of `interval`'s retention window.

    >>> expiry("2h", datetime(2025, 10, 2, tzinfo=timezone.utc)).date().isoformat()
    '2026-11-22'
    """
    return target + timedelta(seconds=RECORD_CAP * interval_seconds(interval))


def finest_interval_covering(
    start: datetime, now: datetime | None = None
) -> str | None:
    """The finest-grained interval whose retention window still reaches `start`.

    >>> finest_interval_covering(datetime(2025, 10, 2, tzinfo=timezone.utc),
    ...                          datetime(2026, 8, 19, tzinfo=timezone.utc))
    '2h'
    """
    ordered = sorted(INTERVAL_SECONDS, key=lambda k: INTERVAL_SECONDS[k])
    for iv in ordered:
        if reach(iv, now) <= start:
            return iv
    return None


def log_returns(rows: list[dict[str, object]], step: int) -> list[dict[str, object]]:
    """Attach log_return = ln(c_t / c_{t-1}); None where the step is broken."""
    out = []
    prev: tuple[datetime, float] | None = None
    for r in rows:
        new = dict(r)
        t = r["time"]
        c = float(r["c"])  # type: ignore[arg-type]
        if prev is not None and (t - prev[0]).total_seconds() == step:  # type: ignore[operator]
            new["log_return"] = math.log(c / prev[1])
        else:
            new["log_return"] = None
        out.append(new)
        prev = (t, c)  # type: ignore[assignment]
    return out


def missing_steps(rows: list[dict[str, object]], step: int) -> list[datetime]:
    """Timestamps absent between the first and last observation."""
    if len(rows) < 2:
        return []
    times = sorted(r["time"] for r in rows)  # type: ignore[misc]
    present = set(times)
    gaps, cur = [], times[0]
    while cur <= times[-1]:
        if cur not in present:
            gaps.append(cur)
        cur += timedelta(seconds=step)
    return gaps


def aggregate(
    rows: list[dict[str, object]], src_step: int, dst_step: int
) -> list[dict[str, object]]:
    """Aggregate OHLCV bars up to a coarser grid aligned on the UTC epoch.

    A destination bar is emitted only when every constituent source bar is
    present, so a partial bar can never masquerade as a complete one.
    """
    if dst_step % src_step:
        raise PriceRecoveryError(
            f"{src_step}s does not divide {dst_step}s; cannot aggregate."
        )
    per = dst_step // src_step
    buckets: dict[int, list[dict[str, object]]] = {}
    for r in rows:
        key = int(r["time"].timestamp()) // dst_step * dst_step  # type: ignore[union-attr]
        buckets.setdefault(key, []).append(r)
    out = []
    for key in sorted(buckets):
        grp = sorted(buckets[key], key=lambda r: r["time"])  # type: ignore[arg-type,return-value]
        if len(grp) != per:
            continue
        out.append(
            {
                "time": from_ms(key * 1000),
                "o": float(grp[0]["o"]),  # type: ignore[arg-type]
                "h": max(float(g["h"]) for g in grp),  # type: ignore[arg-type]
                "l": min(float(g["l"]) for g in grp),  # type: ignore[arg-type]
                "c": float(grp[-1]["c"]),  # type: ignore[arg-type]
                "v": sum(float(g["v"]) for g in grp),  # type: ignore[arg-type]
            }
        )
    return out


# --------------------------------------------------------------------------
# Source A: official candleSnapshot (free, verified, primary)
# --------------------------------------------------------------------------


def _post(payload: dict, max_retries: int = 5, timeout: int = 45) -> object:
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            REST_URL, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise PriceRecoveryError(
        f"candleSnapshot request failed after {max_retries} attempts: {last}\n"
        f"The endpoint is {REST_URL} and needs no authentication."
    )


def fetch_candles(
    coin: str, interval: str, start: datetime, end: datetime, verbose: bool = True
) -> list[dict[str, object]]:
    """Paginate candleSnapshot over [start, end). Free; no credentials."""
    step = interval_seconds(interval)
    got: dict[int, dict] = {}
    cursor, end_ms, page = to_ms(start), to_ms(end), 0
    while cursor < end_ms:
        page += 1
        chunk = _post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                },
            }
        )
        if not isinstance(chunk, list) or not chunk:
            break
        for k in chunk:
            got[int(k["t"])] = k
        nxt = int(chunk[-1]["t"]) + step * 1000
        if nxt <= cursor:
            break
        cursor = nxt
        if verbose:
            print(
                "    page {:>3}  +{:<5} total {:>6}  through {:%Y-%m-%d %H:%M}".format(
                    page, len(chunk), len(got), from_ms(int(chunk[-1]["t"]))
                )
            )
        time.sleep(0.25)
    rows = []
    for t in sorted(got):
        k = got[t]
        rows.append(
            {
                "time": from_ms(t),
                "o": float(k["o"]),
                "h": float(k["h"]),
                "l": float(k["l"]),
                "c": float(k["c"]),
                "v": float(k["v"]),
            }
        )
    return [r for r in rows if start <= r["time"] < end]  # type: ignore[operator]


def probe_retention(coin: str = "BTC") -> list[tuple[str, int, datetime | None]]:
    """For each interval, how many records and how far back the endpoint goes."""
    out = []
    lo = to_ms(datetime(2020, 1, 1, tzinfo=timezone.utc))
    hi = to_ms(datetime.now(timezone.utc) + timedelta(days=1))
    for iv in sorted(INTERVAL_SECONDS, key=lambda k: INTERVAL_SECONDS[k]):
        r = _post(
            {
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": iv, "startTime": lo, "endTime": hi},
            }
        )
        if isinstance(r, list) and r:
            out.append((iv, len(r), from_ms(int(r[0]["t"]))))
        else:
            out.append((iv, 0, None))
        time.sleep(0.25)
    return out


# --------------------------------------------------------------------------
# Source B: Tardis.dev free first-of-month tick archive (true hourly, 3 days)
# --------------------------------------------------------------------------


def tardis_sample_days(start: datetime, end: datetime) -> list[datetime]:
    """First-of-month days in [start, end] -- the days Tardis serves free."""
    out, cur = [], datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur <= end:
        if cur >= start:
            out.append(cur)
        cur = datetime(
            cur.year + (cur.month == 12), (cur.month % 12) + 1, 1, tzinfo=timezone.utc
        )
    return out


def fetch_tardis_day(
    coin: str, day: datetime, dtype: str = "trades", cache: bool = True
) -> bytes:
    """Download one free Tardis sample day. No API key; first of month only."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{dtype}_{coin}_{day:%Y%m%d}.csv.gz"
    if cache and dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    url = TARDIS_URL.format(
        dtype=dtype, y=day.year, m=day.month, d=day.day, coin=coin
    )
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as exc:
        raise PriceRecoveryError(
            f"Tardis returned HTTP {exc.code} for {url}.\n"
            "The free tier serves only the FIRST DAY OF EACH MONTH. For other "
            "days you need a paid subscription (minimum order $300)."
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise PriceRecoveryError(f"Tardis download failed for {url}: {exc}") from exc
    if cache:
        dest.write_bytes(blob)
    return blob


def hourly_from_trades(blob: bytes, coin: str = "BTC") -> list[dict[str, object]]:
    """Build exact hourly OHLCV from a Tardis tick-trade CSV."""
    bars: dict[int, list[float]] = {}
    with gzip.open(io.BytesIO(blob), "rt") as f:
        for r in csv.DictReader(f):
            if r.get("symbol", "").upper() != coin.upper():
                continue
            key = int(r["timestamp"]) // 1_000_000 // 3600 * 3600
            p, a = float(r["price"]), float(r["amount"])
            b = bars.get(key)
            if b is None:
                bars[key] = [p, p, p, p, a]
            else:
                b[1] = max(b[1], p)
                b[2] = min(b[2], p)
                b[3] = p
                b[4] += a
    return [
        {
            "time": from_ms(k * 1000),
            "o": v[0], "h": v[1], "l": v[2], "c": v[3], "v": v[4],
        }
        for k, v in sorted(bars.items())
    ]


# --------------------------------------------------------------------------
# Source C: Binance perp -- a labelled PROXY, a different instrument
# --------------------------------------------------------------------------


def fetch_binance(
    symbol: str, start: datetime, end: datetime, interval: str = "1h"
) -> list[dict[str, object]]:
    """Binance USD-M perp klines. Free, unlimited history. NOT Hyperliquid."""
    step = interval_seconds(interval)
    got: dict[int, list] = {}
    cursor, end_ms = to_ms(start), to_ms(end)
    while cursor < end_ms:
        url = (f"{BINANCE_URL}?symbol={symbol}&interval={interval}"
               f"&startTime={cursor}&endTime={end_ms}&limit=1500")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=45) as resp:
                    chunk = json.loads(resp.read())
                break
            except (urllib.error.URLError, OSError, ValueError) as exc:
                if attempt == 4:
                    raise PriceRecoveryError("Binance request failed: " + url) from exc
                time.sleep(2 ** attempt)
        if not chunk:
            break
        for k in chunk:
            got[int(k[0])] = k
        nxt = int(chunk[-1][0]) + step * 1000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.25)
    rows = [
        {
            "time": from_ms(t),
            "o": float(got[t][1]), "h": float(got[t][2]),
            "l": float(got[t][3]), "c": float(got[t][4]), "v": float(got[t][5]),
        }
        for t in sorted(got)
    ]
    return [r for r in rows if start <= r["time"] < end]  # type: ignore[operator]


def reconstruct_hourly(
    coarse: list[dict[str, object]], shape: list[dict[str, object]], step: int
) -> list[dict[str, object]]:
    """ESTIMATED hourly closes: pin to Hyperliquid bars, borrow intra-bar shape.

    Within one coarse Hyperliquid bar the total log-return ln(c/o) is EXACT.
    Only its split across the constituent hours is estimated, by taking the
    proxy's hourly shape and spreading the (small) discrepancy evenly.

    Backtested on the 1,753 overlapping 2h bars: mean error -0.06 bps,
    MAE 1.01 bps, p99 4.14 bps, max 8.19 bps; correlation with the true hourly
    series 0.99964, volatility ratio 0.9979.

    THIS IS NOT DATA. Rows carry estimated=True. Never present it as observed.
    """
    per = step // 3600
    if per < 2:
        raise PriceRecoveryError("Reconstruction needs a coarse bar of 2h or more.")
    by_hour = {r["time"]: r for r in shape}
    out: list[dict[str, object]] = []
    for bar in coarse:
        t0 = bar["time"]
        hours = [t0 + timedelta(hours=i) for i in range(per)]  # type: ignore[operator]
        legs = [by_hour.get(h) for h in hours]
        if any(x is None for x in legs):
            continue
        r_proxy = [
            math.log(float(x["c"]) / float(x["o"])) for x in legs  # type: ignore[index]
        ]
        total = math.log(float(bar["c"]) / float(bar["o"]))
        adj = (total - sum(r_proxy)) / per
        px = float(bar["o"])
        for h, rp in zip(hours, r_proxy):
            px = px * math.exp(rp + adj)
            out.append({"time": h, "o": None, "h": None, "l": None,
                        "c": px, "v": None, "estimated": True})
        out[-1]["c"] = float(bar["c"])  # kill accumulated float drift
    return out


# --------------------------------------------------------------------------
# Panel I/O and the acceptance test
# --------------------------------------------------------------------------


def load_panel(path: pathlib.Path = PANEL_PATH) -> list[dict[str, object]]:
    if not path.exists():
        raise PriceRecoveryError(f"Panel not found at {path}.")
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "time": datetime.fromisoformat(
                        r["time"].strip().replace("Z", "+00:00")
                    ),
                    "o": float(r["o"]), "h": float(r["h"]), "l": float(r["l"]),
                    "c": float(r["c"]), "v": float(r["v"]),
                }
            )
    rows.sort(key=lambda r: r["time"])  # type: ignore[arg-type,return-value]
    return rows


def compare_bars(
    fetched: list[dict[str, object]], reference: list[dict[str, object]]
) -> dict[str, object]:
    """Field-by-field comparison of two OHLCV series on their shared stamps."""
    ref = {r["time"]: r for r in reference}
    fields = ("o", "h", "l", "c", "v")
    worst = {f: 0.0 for f in fields}
    exact = {f: 0 for f in fields}
    n = 0
    worst_at: dict[str, datetime | None] = {f: None for f in fields}
    for row in fetched:
        r = ref.get(row["time"])
        if r is None:
            continue
        n += 1
        for f in fields:
            d = abs(float(row[f]) - float(r[f]))  # type: ignore[arg-type]
            if d == 0.0:
                exact[f] += 1
            if d > worst[f]:
                worst[f] = d
                worst_at[f] = row["time"]  # type: ignore[assignment]
    return {"n": n, "max_abs": worst, "exact": exact, "worst_at": worst_at,
            "fetched_rows": len(fetched), "reference_rows": len(reference)}


def write_prices_csv(
    rows: list[dict[str, object]], path: pathlib.Path, estimated: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = PRICE_COLUMNS + (["estimated"] if estimated else [])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            out = []
            for k in cols:
                v = r.get(k)
                if k == "time":
                    out.append(v.isoformat())  # type: ignore[union-attr]
                elif v is None:
                    out.append("")
                elif k == "estimated":
                    out.append("1")
                else:
                    out.append(repr(float(v)) if isinstance(v, float) else v)
            w.writerow(out)


def summarise(rows: list[dict[str, object]], step: int, label: str) -> None:
    print(f"  {label}")
    if not rows:
        print("    (empty)")
        return
    ts = [r["time"] for r in rows]
    rets = [r["log_return"] for r in rows if r.get("log_return") is not None]
    print(f"    rows          : {len(rows)}")
    print(f"    span          : {min(ts):%Y-%m-%d %H:%M} -> {max(ts):%Y-%m-%d %H:%M} UTC")  # type: ignore[arg-type]
    print(f"    missing steps : {len(missing_steps(rows, step))}")
    if rets:
        lo, hi = min(rets), max(rets)  # type: ignore[type-var]
        at_lo = next(r["time"] for r in rows if r.get("log_return") == lo)
        at_hi = next(r["time"] for r in rows if r.get("log_return") == hi)
        mean = sum(rets) / len(rets)  # type: ignore[arg-type]
        sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / (len(rets) - 1))  # type: ignore[operator]
        print(f"    log-ret min   : {lo * 1e4:+.4f} bps at {at_lo:%Y-%m-%d %H:%M}")  # type: ignore[operator,arg-type]
        print(f"    log-ret max   : {hi * 1e4:+.4f} bps at {at_hi:%Y-%m-%d %H:%M}")  # type: ignore[operator,arg-type]
        print(f"    log-ret sd    : {sd * 1e4:.4f} bps")


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def run_dry_run(args, start: datetime, end: datetime) -> int:
    now = datetime.now(timezone.utc)
    print("=" * 74)
    print("DRY RUN -- candleSnapshot retention, coverage and deadlines")
    print("=" * 74)
    print(f"requested : {start:%Y-%m-%d} .. {end:%Y-%m-%d}  coin={args.coin}")
    print(f"now       : {now:%Y-%m-%d %H:%M} UTC")
    print()
    print(f"Retention is a cap on RECORD COUNT (~{RECORD_CAP}/interval), not on age,")
    print(f"so reach = {RECORD_CAP} x interval. Predicted from that model:")
    print()
    print("  {:<5} {:>10}  {:<12} {:<12} {}".format(
        "iv", "reach(d)", "earliest", "covers?", "start drops out"))
    print("  " + "-" * 62)
    best = None
    for iv in sorted(INTERVAL_SECONDS, key=lambda k: INTERVAL_SECONDS[k]):
        e = reach(iv, now)
        ok = e <= start
        if ok and best is None:
            best = iv
        print("  {:<5} {:>10.0f}  {:<12} {:<12} {}".format(
            iv,
            RECORD_CAP * interval_seconds(iv) / 86400.0,
            e.strftime("%Y-%m-%d"),
            "YES" if ok else "no",
            expiry(iv, start).strftime("%Y-%m-%d") if ok else "-",
        ))
    print()
    if best is None:
        print(f"NO interval reaches {start:%Y-%m-%d}. This window is beyond the")
        print("endpoint entirely; only paid tick archives reach further back.")
        return 1
    days_left = (expiry(best, start) - now).days
    print(f"finest interval covering the window : {best}")
    print(f"*** it stops covering {start:%Y-%m-%d} on "
          f"{expiry(best, start):%d %B %Y} -- {days_left} days from now ***")
    if days_left < 180:
        print("    ACT NOW: fetch and COMMIT the CSV before that date.")
    print()
    if args.no_network:
        print("(--no-network: model only, endpoint not contacted)")
        return 0
    print("probing the live endpoint to confirm the model ...")
    print()
    print("  {:<5} {:>8}  {:<20} {}".format("iv", "records", "earliest", "model"))
    print("  " + "-" * 58)
    for iv, n, e in probe_retention(args.coin):
        pred = reach(iv, now)
        mark = "-"
        if e is not None:
            mark = "ok" if abs((e - pred).total_seconds()) < 2 * 86400 else "DIVERGES"
            if n < RECORD_CAP - 50:
                mark = "not capped"
        print("  {:<5} {:>8}  {:<20} {}".format(
            iv, n, e.strftime("%Y-%m-%d %H:%M") if e else "-", mark))
    return 0


def run_validate(args) -> int:
    """Acceptance test: does the source reproduce the archived panel exactly?"""
    panel = load_panel()
    p_start, p_end = panel[0]["time"], panel[-1]["time"]
    step = interval_seconds(args.interval)
    print("=" * 74)
    print("VALIDATE -- acceptance test against data/processed/panel_BTC_subsample.csv")
    print("=" * 74)
    print(f"panel span : {p_start:%Y-%m-%d %H:%M} -> {p_end:%Y-%m-%d %H:%M} UTC "  # type: ignore[arg-type]
          f"({len(panel)} rows, 1h)")
    print(f"source     : {args.source}   interval: {args.interval}")
    print()

    if args.source == "candles":
        print("fetching the same span from candleSnapshot ...")
        fetched = fetch_candles(
            args.coin, args.interval, p_start, p_end + timedelta(seconds=step),  # type: ignore[operator]
            verbose=False)
        reference = aggregate(panel, 3600, step) if step != 3600 else panel
        note = f"panel aggregated 1h -> {args.interval}"
    elif args.source == "tardis-sample":
        days = [d for d in tardis_sample_days(p_start, p_end)]  # type: ignore[arg-type]
        if not days:
            print("FAIL: no first-of-month day inside the panel span.")
            return 1
        print("free Tardis sample days inside the panel: {}".format(
            ", ".join(d.strftime("%Y-%m-%d") for d in days)))
        rows: list[dict[str, object]] = []
        for d in days:
            rows.extend(hourly_from_trades(fetch_tardis_day(args.coin, d), args.coin))
        fetched = aggregate(rows, 3600, step) if step != 3600 else rows
        reference = aggregate(panel, 3600, step) if step != 3600 else panel
        note = f"panel aggregated 1h -> {args.interval}"
    else:
        print(f"FAIL: --validate compares against Hyperliquid. Source {args.source!r} is a")
        print("      different instrument and must never be validated this way.")
        return 1

    res = compare_bars(fetched, reference)
    print()
    print(f"RESULT   ({note})")
    print("  bars compared : {}".format(res["n"]))
    if not res["n"]:
        print("\nFAIL: no overlapping bars. The source is not serving this window.")
        return 1
    # Prices are READ from both sides, so they must agree bit-for-bit.
    # Volume on the reference side is SUMMED during aggregation, so it carries
    # float rounding of order 1e-12 that is an artefact of our own arithmetic,
    # not a disagreement between the sources. Judge it relatively.
    bad = []
    for f in ("o", "h", "l", "c", "v"):
        d = res["max_abs"][f]  # type: ignore[index]
        tol = args.volume_tolerance if f == "v" else args.tolerance
        ok = d <= tol
        if not ok:
            bad.append((f, d, res["worst_at"][f]))  # type: ignore[index]
        print("  max |d {}|     : {:<14.10g} exact: {:>4}/{}  {}".format(
            f, d, res["exact"][f], res["n"], "ok" if ok else "OVER TOLERANCE"))  # type: ignore[index]
    # log_return depends on the CLOSE alone, so call that out separately: a
    # source can be unusable for OHLCV yet exact for the series we model.
    close_exact = res["exact"]["c"] == res["n"]  # type: ignore[index]
    print()
    print("  close-only verdict: {}  ({}/{} exact)".format(
        "EXACT -- log-returns from this source are identical"
        if close_exact else "NOT exact -- log-returns would differ",
        res["exact"]["c"], res["n"]))  # type: ignore[index]
    print()
    if not bad:
        print("PASS: reproduces the archived panel within tolerance")
        print(f"      (prices {args.tolerance:g}, volume {args.volume_tolerance:g}).")
        print("      Same object. The recovered Oct-Nov 2025 bars are usable.")
        return 0
    print(f"FAIL: {len(bad)} field(s) over tolerance:")
    for f, d, at in bad:
        print(f"      {f}: max |d| = {d:.10g} at {at}")
    if close_exact:
        print()
        print("      NOTE: every CLOSE matches exactly. The disagreement is in")
        print("      open/high/low/volume, which a trade-feed capture rebuilds")
        print("      slightly differently from the exchange's candle builder.")
        print("      This source is SAFE for log-returns and UNSAFE for OHLCV.")
    else:
        print("      DO NOT splice this source into the panel until resolved.")
    return 1


def run_fetch(args, start: datetime, end: datetime) -> int:
    step = interval_seconds(args.interval)
    end_x = end + timedelta(days=1)
    print("=" * 74)
    print(f"FETCH  coin={args.coin}  {start:%Y-%m-%d} .. {end:%Y-%m-%d}  "
          f"source={args.source}  interval={args.interval}")
    print("=" * 74)

    estimated = False
    if args.source == "candles":
        if reach(args.interval) > start:
            raise PriceRecoveryError(
                "interval {} only reaches back to {:%Y-%m-%d}, but you asked for "
                "{:%Y-%m-%d}.\nUse --interval {} (the finest that covers it), or "
                "run --dry-run to see the table.".format(
                    args.interval, reach(args.interval), start,
                    finest_interval_covering(start) or "1d"))
        rows = fetch_candles(args.coin, args.interval, start, end_x)
    elif args.source == "tardis-sample":
        days = tardis_sample_days(start, end)
        if not days:
            raise PriceRecoveryError(
                "The free Tardis tier serves only the first day of each month; "
                f"there is none in {start:%Y-%m-%d}..{end:%Y-%m-%d}.")
        print("free sample days: {}".format(
            ", ".join(d.strftime("%Y-%m-%d") for d in days)))
        rows = []
        for d in days:
            rows.extend(hourly_from_trades(fetch_tardis_day(args.coin, d), args.coin))
        if step != 3600:
            rows = aggregate(rows, 3600, step)
    elif args.source == "binance":
        print("NOTE: Binance BTCUSDT perp is a DIFFERENT INSTRUMENT from the")
        print("      Hyperliquid BTC perp. Use only as a labelled proxy.")
        rows = fetch_binance(args.symbol, start, end_x, args.interval)
    elif args.source == "reconstruct":
        print("NOTE: output is ESTIMATED, not observed. Every row is flagged.")
        coarse = fetch_candles(args.coin, args.interval, start, end_x, verbose=False)
        shape = fetch_binance(args.symbol, start, end_x, "1h")
        rows = reconstruct_hourly(coarse, shape, step)
        step, estimated = 3600, True
    else:
        raise PriceRecoveryError(f"Unknown source {args.source!r}.")

    rows = log_returns(rows, step)
    print()
    summarise(rows, step, f"{args.source} series")
    if not rows:
        return 1
    out = pathlib.Path(args.out) if args.out else (
        INTERIM_DIR / f"prices_{args.coin}_{args.interval}"
                      f"_{start:%Y%m%d}_{end:%Y%m%d}_{args.source}.csv")
    write_prices_csv(rows, out, estimated=estimated)
    print()
    print(f"wrote {out}")
    print("columns: {}".format(
        ",".join(PRICE_COLUMNS + (["estimated"] if estimated else []))))
    gaps = missing_steps(rows, step)
    if gaps:
        print()
        print(f"WARNING: {len(gaps)} missing steps, e.g.")
        for g in gaps[:5]:
            print(f"  {g:%Y-%m-%d %H:%M} UTC")
        return 1
    return 0


# --------------------------------------------------------------------------
# Offline self-test
# --------------------------------------------------------------------------


def run_self_test() -> int:
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        verdict = "PASS" if cond else "FAIL"
        note = f"  [{detail}]" if detail else ""
        print(f"  {verdict} {name}{note}")
        if not cond:
            failures.append(name)

    print("=" * 74)
    print("SELF-TEST -- offline; no network, no credentials")
    print("=" * 74)
    NOW = datetime(2026, 8, 19, 7, 0, tzinfo=timezone.utc)

    print("\n[1] retention model (record cap, not age)")
    check("2h reaches 2025-06-28", reach("2h", NOW).date().isoformat() == "2025-06-28",
          reach("2h", NOW).date().isoformat())
    check("4h reaches 2024-05-07/08", reach("4h", NOW).date().isoformat().startswith("2024-05-0"),
          reach("4h", NOW).date().isoformat())
    check("1h does NOT reach 2025-10-02",
          reach("1h", NOW) > datetime(2025, 10, 2, tzinfo=timezone.utc))
    check("finest interval covering 2025-10-02 is 2h",
          finest_interval_covering(datetime(2025, 10, 2, tzinfo=timezone.utc), NOW) == "2h")
    check("2 Oct 2025 expires from 2h on 2026-11-22",
          expiry("2h", datetime(2025, 10, 2, tzinfo=timezone.utc)).date().isoformat()
          == "2026-11-22")
    check("unknown interval rejected",
          _raises(lambda: interval_seconds("7h")))

    print("\n[2] aggregation 1h -> 2h")
    base = [
        {"time": from_ms((1759363200 + 3600 * i) * 1000), "o": 100.0 + i,
         "h": 110.0 + i, "l": 90.0 + i, "c": 105.0 + i, "v": 1.0 + i}
        for i in range(6)
    ]
    agg = aggregate(base, 3600, 7200)
    check("3 bars from 6 hours", len(agg) == 3, str(len(agg)))
    check("open = first open", agg[0]["o"] == 100.0)
    check("close = last close", agg[0]["c"] == 106.0)
    check("high = max high", agg[0]["h"] == 111.0)
    check("low = min low", agg[0]["l"] == 90.0)
    check("volume summed", agg[0]["v"] == 3.0)
    check("epoch-aligned bucket start",
          int(agg[0]["time"].timestamp()) % 7200 == 0)  # type: ignore[union-attr]
    partial = aggregate(base[:5], 3600, 7200)
    check("incomplete bar dropped, not emitted", len(partial) == 2, str(len(partial)))
    check("non-divisible step rejected",
          _raises(lambda: aggregate(base, 3600, 5400)))
    shuffled = aggregate(list(reversed(base)), 3600, 7200)
    check("row order irrelevant", shuffled == agg)

    print("\n[3] log-returns and gap detection")
    lr = log_returns(base, 3600)
    check("first return is None", lr[0]["log_return"] is None)
    check("return value correct",
          abs(lr[1]["log_return"] - math.log(106.0 / 105.0)) < 1e-15)  # type: ignore[arg-type]
    broken = [base[0], base[1], base[4], base[5]]
    lrb = log_returns(broken, 3600)
    check("return suppressed across a gap", lrb[2]["log_return"] is None)
    check("gap detected", len(missing_steps(broken, 3600)) == 2,
          str(len(missing_steps(broken, 3600))))
    check("no false gaps", missing_steps(base, 3600) == [])

    print("\n[4] tick trades -> hourly OHLCV")
    csv_txt = (
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "hyperliquid,BTC,1759276800000000,0,1,buy,114000,0.5\n"
        "hyperliquid,BTC,1759278000000000,0,2,buy,114500,0.25\n"
        "hyperliquid,BTC,1759279500000000,0,3,sell,113800,0.25\n"
        "hyperliquid,ETH,1759276800000000,0,4,buy,4000,10\n"
        "hyperliquid,BTC,1759280400000000,0,5,buy,114200,1.0\n"
    )
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(csv_txt.encode())
    bars = hourly_from_trades(buf.getvalue(), "BTC")
    check("2 hourly bars", len(bars) == 2, str(len(bars)))
    check("other coins excluded", all(b["v"] != 10 for b in bars))
    check("open is first trade", bars[0]["o"] == 114000.0)
    check("close is last trade of the hour", bars[0]["c"] == 113800.0)
    check("high correct", bars[0]["h"] == 114500.0)
    check("low correct", bars[0]["l"] == 113800.0)
    check("volume summed", abs(bars[0]["v"] - 1.0) < 1e-12)  # type: ignore[arg-type]

    print("\n[5] reconstruction is exact at the bar close")
    coarse = [{"time": from_ms(1759363200 * 1000), "o": 100.0, "h": 0.0,
               "l": 0.0, "c": 102.0, "v": 0.0}]
    shape = [
        {"time": from_ms(1759363200 * 1000), "o": 50.0, "h": 0, "l": 0, "c": 50.5, "v": 0},
        {"time": from_ms(1759366800 * 1000), "o": 50.5, "h": 0, "l": 0, "c": 51.0, "v": 0},
    ]
    rec = reconstruct_hourly(coarse, shape, 7200)
    check("one row per hour", len(rec) == 2, str(len(rec)))
    check("terminal close is EXACT", rec[-1]["c"] == 102.0)
    check("all rows flagged estimated", all(r["estimated"] for r in rec))
    check("intermediate close bracketed", 100.0 < rec[0]["c"] < 102.0)  # type: ignore[operator]
    check("missing shape -> no output", reconstruct_hourly(coarse, shape[:1], 7200) == [])

    print("\n[6] Tardis free-sample day enumeration")
    days = tardis_sample_days(parse_day("2025-10-02"), parse_day("2025-12-02"))
    check("Oct 2 .. Dec 2 -> 11-01 and 12-01",
          [d.strftime("%Y-%m-%d") for d in days] == ["2025-11-01", "2025-12-01"],
          ",".join(d.strftime("%Y-%m-%d") for d in days))
    d2 = tardis_sample_days(parse_day("2025-10-01"), parse_day("2025-12-31"))
    check("Oct 1 .. Dec 31 -> 3 days", len(d2) == 3, str(len(d2)))
    check("year rollover handled",
          len(tardis_sample_days(parse_day("2025-12-01"), parse_day("2026-02-01"))) == 3)

    print("\n[7] panel schema round-trip")
    if PANEL_PATH.exists():
        panel = load_panel()
        check("panel loads", len(panel) > 0, f"{len(panel)} rows")
        check("panel is hourly",
              all((panel[i]["time"] - panel[i - 1]["time"]).total_seconds() == 3600  # type: ignore[operator]
                  for i in range(1, min(50, len(panel)))))
        agg2 = aggregate(panel, 3600, 7200)
        check("panel aggregates to 2h", len(agg2) > 0, f"{len(agg2)} bars")
        check("aggregate never invents bars", len(agg2) <= len(panel) // 2 + 1)
    else:
        print("  SKIP panel checks (panel not present)")

    print("\n[8] date parsing")
    check("YYYY-MM-DD", parse_day("2025-10-02").isoformat() == "2025-10-02T00:00:00+00:00")
    check("YYYYMMDD", parse_day("20251002") == parse_day("2025-10-02"))
    check("garbage rejected", _raises(lambda: parse_day("not-a-date")))

    print()
    print("=" * 74)
    if failures:
        print("{} FAILURE(S): {}".format(len(failures), ", ".join(failures)))
        return 1
    print("all offline checks passed")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except PriceRecoveryError:
        return True
    except Exception:
        return False
    return False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m regime_engine.fetch_prices",
        description="Recover Hyperliquid BTC price history for Oct-Dec 2025.",
        epilog=(
            "The Oct-Nov 2025 prices remain available. candleSnapshot caps records,\n"
            "not age, so 2h and coarser still serve that window -- free, and\n"
            "bit-exact against the archived panel. Deadline: 2 Oct 2025 leaves\n"
            "the 2h window on 22 Nov 2026.\n\n"
            "  --dry-run   retention table, coverage and deadlines\n"
            "  --validate  acceptance test against the archived panel\n"
            "  --self-test offline checks, no network\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--start", help="first UTC day, YYYY-MM-DD")
    ap.add_argument("--end", help="last UTC day, YYYY-MM-DD (inclusive)")
    ap.add_argument("--coin", default="BTC")
    ap.add_argument("--symbol", default="BTCUSDT", help="Binance symbol (proxy only)")
    ap.add_argument("--interval", default="2h",
                    help="candle interval (default 2h -- the finest that covers "
                         "Oct 2025 today)")
    ap.add_argument("--source", default="candles",
                    choices=["candles", "tardis-sample", "binance", "reconstruct"],
                    help="candles = official candleSnapshot (free, verified); "
                         "tardis-sample = free first-of-month ticks, true hourly; "
                         "binance = labelled proxy; reconstruct = ESTIMATED hourly")
    ap.add_argument("--dry-run", action="store_true",
                    help="report retention and coverage; downloads nothing")
    ap.add_argument("--validate", action="store_true",
                    help="acceptance test against the archived panel")
    ap.add_argument("--self-test", action="store_true", help="offline checks")
    ap.add_argument("--no-network", action="store_true",
                    help="--dry-run without contacting the endpoint")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="max acceptable |difference| on PRICE fields in "
                         "--validate (default 0: the candle route is bit-exact "
                         "and should stay that way)")
    ap.add_argument("--volume-tolerance", type=float, default=1e-6,
                    help="max acceptable |difference| on volume, which the "
                         "reference side SUMS during aggregation and so carries "
                         "float rounding of order 1e-12 (default 1e-6)")
    ap.add_argument("--out", help="output CSV path")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            return run_self_test()
        if args.validate:
            return run_validate(args)
        if not args.start or not args.end:
            raise PriceRecoveryError(
                "--start and --end are required for --dry-run and for fetching.\n"
                f"The missing segment is:  --start {MISSING_START} --end {MISSING_END}")
        start, end = parse_day(args.start), parse_day(args.end)
        if end < start:
            raise PriceRecoveryError("--end is before --start.")
        if args.dry_run:
            return run_dry_run(args, start, end)
        return run_fetch(args, start, end)
    except PriceRecoveryError as exc:
        sys.stdout.flush()
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
