"""Panel loading, assembly and fingerprinting.

The canonical hourly panel has the schema

    time, o, h, l, c, v, fundingRate, premium, log_return, funding_bps

with ``time`` a tz-aware UTC hourly stamp.  ``build_hourly_panel`` is the single
implementation of the alignment rules:

  * candles and funding are joined on the hourly UTC grid with an **inner** join
    -- a funding stamp with no candle (or vice versa) is dropped, never
    forward-filled from a neighbouring hour;
  * ``log_return`` at ``t`` uses only closes at ``t`` and ``t-1`` -- no forward
    information enters;
  * consequently the first row of a freshly assembled panel has no lagged close
    and is dropped.

``tests/test_panel.py`` asserts these invariants against this function and
against the shipped panel.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pandas as pd

from .paths import DEFAULT_PREMIUM_FULL, PAPER_WINDOW

__all__ = [
    "PANEL_COLUMNS",
    "load_paper_window_premium",
    "build_hourly_panel",
    "load_panel",
    "panel_fingerprint",
    "describe_series",
]

PANEL_COLUMNS = (
    "time", "o", "h", "l", "c", "v",
    "fundingRate", "premium", "log_return", "funding_bps",
)


def build_hourly_panel(candles: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Inner-join hourly candles with funding/premium and add derived columns.

    Parameters
    ----------
    candles : DataFrame with columns time, o, h, l, c, v (hourly, UTC).
    funding : DataFrame with columns time, fundingRate, premium.

    Notes
    -----
    No forward-fill anywhere.  The first surviving row is dropped because its
    log-return would require a close from before the sample.

    WARNING -- the inner join truncates the panel to the candle window.
    Hyperliquid's ``candleSnapshot`` endpoint is retention-capped at ~5,000
    candles while ``fundingHistory`` is not, so joining funding onto candles
    discards every funding/premium row older than the candle window.  This is
    why the shipped panel has no Oct-Nov 2025 rows although the premium series
    covers them.  This function therefore reports how many funding rows the
    join drops; if that number is large, the funding series reaches further
    back than the candles and should be preserved separately (see
    ``fetch_official.py``).
    """
    candles = candles.copy()
    funding = funding.copy()
    candles["time"] = pd.to_datetime(candles["time"], utc=True).dt.floor("h")
    funding["time"] = pd.to_datetime(funding["time"], utc=True).dt.floor("h")
    candles = candles.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    funding = funding.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    # r_t = log(c_t / c_{t-1}): strictly backward-looking, and computed on the
    # CANDLE grid before the funding join, so that a dropped funding stamp can
    # never turn the next return into a silent two-hour return.
    candles["log_return"] = np.log(candles["c"] / candles["c"].shift(1))
    spans_gap = candles["time"].diff() != pd.Timedelta("1h")
    candles.loc[spans_gap, "log_return"] = np.nan   # incl. the first row

    panel = candles.merge(
        funding[["time", "fundingRate", "premium"]], on="time", how="inner"
    )
    panel["funding_bps"] = panel["fundingRate"] * 1e4

    n_funding_dropped = len(funding) - len(panel)
    if n_funding_dropped > 0:
        earlier = int((funding["time"] < candles["time"].min()).sum())
        print(f"[panel] inner join dropped {n_funding_dropped} funding rows "
              f"({earlier} of them older than the first candle). Candle retention, "
              f"not funding availability, sets the panel start.")

    # Rows without a well-defined one-hour return are dropped. Nothing is imputed.
    panel = panel.dropna(subset=["log_return"]).reset_index(drop=True)
    return panel


def load_panel(path) -> pd.DataFrame:
    """Load a panel from CSV or parquet and validate its schema."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"panel not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        panel = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        panel = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported panel format: {path.suffix} (use .csv or .parquet)")

    missing = [c for c in ("time", "c", "log_return", "premium", "funding_bps")
               if c not in panel.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    panel["time"] = pd.to_datetime(panel["time"], utc=True)
    panel = panel.sort_values("time").reset_index(drop=True)
    if panel["time"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicated timestamps")
    if panel[["log_return", "premium", "funding_bps", "c"]].isna().any().any():
        raise ValueError(f"{path.name} contains NaNs in modelled columns")
    return panel


def panel_fingerprint(panel: pd.DataFrame, n_chars: int = 16) -> str:
    """Content hash of a panel: timestamps + every numeric column.

    Used as the HMM cache key so that re-running on different data can never
    return a stale fit.
    """
    h = hashlib.sha256()
    cols = [c for c in panel.columns if c != "time"]
    h.update(",".join(map(str, cols)).encode("utf-8"))
    stamps = pd.to_datetime(panel["time"], utc=True).dt.tz_convert(None)
    h.update(np.ascontiguousarray(stamps.values.astype("datetime64[ns]").astype("int64")).tobytes())
    for col in cols:
        values = pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=float)
        h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()[:n_chars]


def describe_series(x, name: str = "") -> dict:
    """mean / std / min / median / max / skew / excess kurtosis of a series."""
    from scipy.stats import kurtosis, skew

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    out = {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)),
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
        "skew": float(skew(a)),
        "excess_kurtosis": float(kurtosis(a, fisher=True)),
    }
    if name:
        out["name"] = name
    return out


def load_paper_window_premium(path=None) -> np.ndarray:
    """The funding premium over the paper's estimation window, in basis points.

    Returns the T = 5,001 hourly premium observations from 2 Oct 2025 00:00 UTC
    to 28 Apr 2026 08:00 UTC, read from the shipped ``premium_BTC_full.csv``.

    ``fundingHistory`` carries no retention limit, so unlike the hourly return
    panel this window is both shipped in full *and* re-fetchable. Everything the
    paper reports on the funding side -- the funding rows of the summary-statistics
    table, the MS-AR parameter table and the funding figure -- is computed from
    exactly this array.
    """
    path = pathlib.Path(path) if path is not None else DEFAULT_PREMIUM_FULL
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is tracked in git; restore it with "
            "`git checkout -- data/interim/premium_BTC_full.csv`."
        )
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    start, end = PAPER_WINDOW
    mask = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[mask, "premium"].to_numpy(dtype=float) * 1e4
