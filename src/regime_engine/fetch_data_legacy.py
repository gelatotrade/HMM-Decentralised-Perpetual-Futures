"""
Hyperliquid funding rate + price fetcher.

Funding is paid hourly on Hyperliquid (1/8 of the 8h-equivalent rate).
The fundingHistory endpoint returns max ~500 records per call,
so we paginate forward in time.

References
----------
https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals
https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from .panel import build_hourly_panel
from .paths import REPO_ROOT

API = "https://api.hyperliquid.xyz/info"
HEADERS = {"Content-Type": "application/json"}
# Cache into the repository's data/interim, not into the installed package.
DATA_DIR = REPO_ROOT / "data" / "interim"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _post(payload: dict, max_retries: int = 5) -> list | dict:
    for attempt in range(max_retries):
        try:
            r = requests.post(API, json=payload, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[retry {attempt+1}] {e} — sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed after {max_retries} retries")


def fetch_funding(coin: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    """Fetch funding history with forward pagination."""
    if end_ms is None:
        end_ms = int(time.time() * 1000)

    rows: list[dict] = []
    cursor = start_ms
    page = 0
    while cursor < end_ms:
        page += 1
        payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": cursor,
            "endTime": end_ms,
        }
        chunk = _post(payload)
        if not chunk:
            break
        rows.extend(chunk)
        last_t = chunk[-1]["time"]
        if last_t <= cursor:
            break
        cursor = last_t + 1
        if page % 10 == 0:
            at = datetime.fromtimestamp(cursor / 1000, tz=timezone.utc)
            print(f"  funding page {page}: {len(rows)} rows so far, "
                  f"cursor={at:%Y-%m-%d %H:%M}")
        time.sleep(0.15)  # be polite

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def fetch_candles(coin: str, interval: str, start_ms: int,
                  end_ms: int | None = None) -> pd.DataFrame:
    """Fetch OHLCV candles with forward pagination (max 5000 per call)."""
    if end_ms is None:
        end_ms = int(time.time() * 1000)

    rows: list[dict] = []
    cursor = start_ms
    page = 0
    while cursor < end_ms:
        page += 1
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
            },
        }
        chunk = _post(payload)
        if not chunk:
            break
        rows.extend(chunk)
        last_t = chunk[-1]["t"]
        if last_t <= cursor:
            break
        cursor = last_t + 1
        if page % 2 == 0:
            at = datetime.fromtimestamp(last_t / 1000, tz=timezone.utc)
            print(f"  candle page {page}: {len(rows)} rows so far, "
                  f"last_t={at:%Y-%m-%d %H:%M}")
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for col in ("o", "h", "l", "c", "v"):
        df[col] = df[col].astype(float)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df[["time", "o", "h", "l", "c", "v"]]


def get_dataset(coin: str = "BTC", lookback_days: int = 365) -> pd.DataFrame:
    """Build aligned hourly funding + log-return panel, cached on disk."""
    cache = DATA_DIR / f"panel_{coin}_{lookback_days}d.csv"
    if cache.exists():
        print(f"[cache] loading {cache.name}")
        df = pd.read_csv(cache, parse_dates=["time"])
        return df

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_days * 24 * 3600 * 1000

    since = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    print(f"[fetch] funding for {coin} from {since:%Y-%m-%d}")
    funding = fetch_funding(coin, start_ms, end_ms)
    print(f"  → {len(funding)} funding observations")

    print(f"[fetch] 1h candles for {coin}")
    candles = fetch_candles(coin, "1h", start_ms, end_ms)
    print(f"  → {len(candles)} candle observations")

    # Align on the hourly UTC grid (inner join, no forward-fill) and derive
    # log-returns; see regime_engine.panel.build_hourly_panel.
    panel = build_hourly_panel(candles, funding)

    panel.to_csv(cache, index=False)
    print(f"[cache] saved {cache.name}: {len(panel)} aligned hourly observations")
    return panel


if __name__ == "__main__":
    df = get_dataset("BTC", lookback_days=365)
    print(df.head())
    print(df.tail())
    print(df.describe())
