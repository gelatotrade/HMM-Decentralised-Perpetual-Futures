"""The premium gap: Hyperliquid's averaged premium against Binance's and Bybit's.

    python3 -m regime_engine.premium_gap            # artifact, tables, figure from data/premium_gap
    python3 -m regime_engine.premium_gap --fetch    # fetch and aggregate the data first

The three venues set funding by the same rule, F = P + clamp(r - P, +-0.05%), so
each settles exactly at the interest floor r while its averaged premium P lies
in the closed band [-4, +6] bps; at either edge the clamp binds exactly and F
still equals r. The premiums are built differently (reference price, impact
notional, sampling, averaging window; :data:`CONSTRUCTION` lists what the
venues document), and the venues' floor shares differ. This module measures the
difference between the premiums at Hyperliquid's own resolution and links it to
the floor shares.

Construction. Hyperliquid's record stamped t carries its premium averaged over
[t - 1h, t). Each exchange's premium for the same hour is the mean of the
closes of its one-minute premium-index klines that open in [t - 1h, t)
(:mod:`regime_engine.fetch_premium`). The hourly gap is

    D_t = P_HL,t - P_X,t        (bps, 8-hour equivalent).

The exchanges settle on an eight-hour average of their minute premium indices,
rebuilt here from the hourly aggregates: Binance with linearly rising weights
throughout, Bybit with equal weights up to 7 Mar 2024 08:00 UTC and linearly
rising ones from 16:00 that day (:func:`linear_blocks`).

Decomposition. Hyperliquid's hourly floor share and an exchange's settlement
floor share differ in three respects: where the premium sits (location, the
quarter's median gap), over what it is averaged (one hour against eight) and how
widely it is spread (dispersion). :func:`decompose` evaluates the in-band share
at all eight combinations and splits the difference symmetrically (Shapley); the
factors interact strongly, so each factor's range over the six orders is kept.

Phases. A quarter is classed by where the exchange's median hourly premium sat
(:func:`phase_of`): at or below the band's lower edge, above it but not above
zero, or above zero. The split is reported per class, for the below-band
quarters of 2024Q3-2025Q4 and of 2026 apart, and per quarter.

Spans. The comparison runs over the common rule: Hyperliquid's hours stamped
11 Dec 2023 23:00 to 1 Oct 2026 00:00 UTC (the first record that only the
+-0.05% clamp reproduces; the records from 19 Jul 2023 up to it fit +-0.03%) and the
exchanges' settlements stamped 12 Dec 2023 00:00 to 1 Oct 2026 00:00. A record
belongs to the period in which its premium accrued, so 2023 and its fourth
quarter run from 11 Dec 22:00.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import shutil
import sys
from decimal import Decimal

import numpy as np
import pandas as pd

from . import censoring, fetch_premium
from .paths import DEFAULT_OUT, DEFAULT_PREMIUM_FULL, REPO_ROOT

DATA = fetch_premium.OUT_DIR
INTERIM = REPO_ROOT / "data" / "interim"
ARTIFACT = DEFAULT_OUT / "premium_gap.json"
OUTPUT_TABLE = DEFAULT_OUT / "tables" / "table12_premium_gap.tex"
PAPER_TABLE = REPO_ROOT / "paper" / "tables" / "table12_premium_gap.tex"
FIGURE_STEM = "fig11_premium_gap"

ASSETS = fetch_premium.ASSETS
EXCHANGES = ("binance", "bybit")
LABELS = censoring.LABELS
FLOOR_8H = censoring.FLOOR_8H
FLOOR_HOURLY = censoring.FLOOR_HOURLY
BAND = censoring.BAND_BPS
RULE_START = fetch_premium.RULE_START  # first Hyperliquid hour under the +-0.05% clamp
FIRST_SETTLEMENT = "2023-12-12 00:00"
END = fetch_premium.END
WINDOWS = censoring.WINDOWS
CRASH_DAY = "2025-10-10"
# Minute j (1..480) of an eight-hour block weighs j; summed over the minutes of
# hour h (0 = earliest) that is 3600 h + 1830.
LINEAR_WEIGHTS = 3600 * np.arange(8) + 1830
FIRST_LINEAR_SETTLEMENT = {"binance": None, "bybit": "2024-03-07 16:00"}
PREMIUM_BINS = (-np.inf, -6.0, -5.0, -4.5, -4.0, -3.0, -2.0, 0.0, 2.0, 4.0, 6.0, np.inf)
# Where the exchange's median hourly premium sat in a quarter (see phase_of).
PHASES = ("below", "inside_nonpositive", "positive")
# The below-band quarters, early and late (both ends included).
SUBPERIODS = {"below_2024Q3_2025Q4": ("2024Q3", "2025Q4"), "below_2026": ("2026Q1", "2026Q4")}
PARTIAL_QUARTERS = {"2023Q4": "premiums accrued from 2023-12-11 22:00 UTC (482 hours)"}
PARTIAL_MONTHS = {"2023-12": "premiums accrued from 2023-12-11 22:00 UTC (482 hours)"}
# Panels (c), (d) of fig11_premium_gap: a below-band quarter of the paper window whose difference
# location accounts for, and the full quarter in which the exchanges' median
# premium was positive (the other such quarter, 2023Q4, covers 11-31 December).
MECHANISM_QUARTERS = {"location": "2025Q4", "positive": "2024Q1"}
# Moving-block bootstrap of monthly and quarterly medians of the hourly gap. The
# daily mean gap has an AR(1) half-life of about 2.5-2.8 days, so the headline
# intervals use one-week blocks; 24- and 72-hour blocks are kept as sensitivity.
BOOTSTRAP = {"block_hours": 168, "sensitivity_block_hours": [24, 72], "reps": 2000,
             "level": 0.95, "seed": 20261008}
# Hyperliquid's oracle page read "spot prices" in the snapshot of 6 Oct 2025 and
# "spot mid prices" in that of 10 Dec 2025 (Internet Archive); the days compared
# around that interval, see level_shift.
ORACLE_WORDING = {"last_spot_prices": "2025-10-06", "first_spot_mid_prices": "2025-12-10",
                  "days": 28}


# ----------------------------------------------------------------------
#  Loading
# ----------------------------------------------------------------------
def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _index(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True, format="ISO8601"), name="time")


def load_hyperliquid(coin: str, data_dir=DATA) -> pd.DataFrame:
    """Hyperliquid's hourly records: the rate as printed and the premium in bps."""
    df = pd.read_csv(pathlib.Path(data_dir) / f"hyperliquid_{coin}_funding.csv", dtype=str)
    return pd.DataFrame({"rate": df["fundingRate"].str.strip().to_numpy(),
                         "premium": df["premium"].astype(float).to_numpy() * 1e4},
                        index=_index(df["time"]))


def exchange_hourly_frame(df: pd.DataFrame) -> pd.DataFrame:
    """An exchange's hourly aggregates (strings, as written) as numbers, premiums in bps."""
    num = {c: pd.to_numeric(df[c].replace("", np.nan)) * 1e4 for c in ("mean", "lwmean", "close")}
    return pd.DataFrame({"n": pd.to_numeric(df["n"]).astype(int).to_numpy(),
                         **{c: v.to_numpy() for c, v in num.items()}},
                        index=_index(df["time"]))


def load_exchange_hourly(venue: str, symbol: str, data_dir=DATA) -> pd.DataFrame:
    path = pathlib.Path(data_dir) / f"{venue}_{symbol}_premium_hourly.csv"
    return exchange_hourly_frame(pd.read_csv(path, dtype=str, keep_default_na=False))


def settlement_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Settlements with the rate as printed, the floor flag and the premium they identify."""
    rates = df["fundingRate"].astype(str).str.strip()
    return pd.DataFrame({"rate": rates.to_numpy(),
                         "at_floor": [Decimal(r) == FLOOR_8H for r in rates],
                         "premium": censoring.recovered_premium_bps(rates).to_numpy()},
                        index=_index(df["time"]))


def load_settlements(venue: str, symbol: str, data_dir=DATA) -> pd.DataFrame:
    path = pathlib.Path(data_dir) / f"{venue}_{symbol}_funding.csv"
    return settlement_frame(pd.read_csv(path, dtype=str))


def load_prices(symbol: str, data_dir=DATA) -> pd.Series:
    """Hourly closes, stamped when the hour closes (open time + 1 h)."""
    df = pd.read_csv(pathlib.Path(data_dir) / f"binance_{symbol}_price_1h.csv", dtype=str)
    idx = _index(df["open_time"]) + pd.Timedelta(hours=1)
    return pd.Series(df["close"].astype(float).to_numpy(), index=idx, name="close")


def load_usdc_usdt(data_dir=DATA) -> pd.Series:
    """ln of the USDC price in USDT (Binance spot), in bps, stamped when the hour closes."""
    path = pathlib.Path(data_dir) / f"binance_{fetch_premium.STABLECOIN_PAIR}_spot_1h.csv"
    df = pd.read_csv(path, dtype=str)
    idx = _index(df["open_time"]) + pd.Timedelta(hours=1)
    return pd.Series(np.log(df["close"].astype(float).to_numpy()) * 1e4, index=idx)


def in_band(bps) -> pd.Series:
    """Inside the closed band [-4, +6] bps, where the rule pays the floor; NaN stays NaN.

    At either edge the clamp binds exactly, so the rate still equals the floor.
    """
    x = pd.Series(bps, dtype=float)
    lo, hi = BAND
    return ((x >= lo) & (x <= hi)).astype(float).where(x.notna())


# ----------------------------------------------------------------------
#  Periods and statistics
# ----------------------------------------------------------------------
def accrual_period(stamps, freq: str) -> pd.Index:
    """The period ('2024Q1', '2024-01', '2024-01-31', '2024') each record's premium accrued in.

    A record stamped t covers the hour before t, so a stamp at midnight belongs
    to the day, month and quarter that end with it.
    """
    t = (pd.DatetimeIndex(stamps) - pd.Timedelta(hours=1)).tz_convert("UTC").tz_localize(None)
    if freq == "D":
        return pd.Index(t.strftime("%Y-%m-%d"))
    return pd.Index(t.to_period(freq).astype(str))


def gap_stats(x) -> dict:
    """Mean, quartiles and share above zero of a gap series in bps."""
    x = pd.Series(x, dtype=float).dropna()
    if x.empty:
        return {"n": 0}
    q25, med, q75 = np.quantile(x.to_numpy(), [0.25, 0.5, 0.75])
    return {"n": int(len(x)), "mean_bps": float(x.mean()), "median_bps": float(med),
            "q25_bps": float(q25), "q75_bps": float(q75),
            "share_positive": float((x > 0).mean())}


def gap_by(gap: pd.Series, freq: str) -> dict:
    """:func:`gap_stats` for each period of ``freq`` ('Y', 'Q', 'M')."""
    gap = gap.dropna()
    return {k: gap_stats(g) for k, g in gap.groupby(accrual_period(gap.index, freq))}


def daily_means(gap: pd.Series, min_hours: int = 20) -> pd.Series:
    """Mean gap per UTC day, for days with at least ``min_hours`` hours."""
    gap = gap.dropna()
    g = gap.groupby(accrual_period(gap.index, "D"))
    return g.mean()[g.size() >= min_hours]


def _lagged_pairs(x: pd.Series, lag: int = 1):
    """Pairs (x_t, x_{t+lag}); with a date index, only pairs ``lag`` calendar days apart."""
    x = x.dropna()
    if isinstance(x.index, pd.DatetimeIndex) or (len(x) and isinstance(x.index[0], str)):
        days = pd.to_datetime(pd.Index(x.index)).to_numpy()
        ok = (days[lag:] - days[:-lag]) == np.timedelta64(lag, "D")
        return x.to_numpy()[:-lag][ok], x.to_numpy()[lag:][ok]
    return x.to_numpy()[:-lag], x.to_numpy()[lag:]


def autocorrelations(x: pd.Series, lags=(1, 7, 30)) -> dict:
    """Sample autocorrelations (the usual estimator, deviations from the full-sample mean)."""
    x = x.dropna()
    d = x - x.mean()
    den = float((d ** 2).sum())
    out = {}
    for k in lags:
        a, b = _lagged_pairs(d, k)
        out[k] = float((a * b).sum() / den)
    return out


def ar1_half_life(x: pd.Series) -> dict:
    """OLS of x_t on x_{t-1} with an intercept; half-life ln 0.5 / ln phi when 0 < phi < 1."""
    z, y = _lagged_pairs(pd.Series(x), 1)
    beta = np.linalg.lstsq(np.column_stack([np.ones_like(z), z]), y, rcond=None)[0]
    phi = float(beta[1])
    return {"n": int(len(y)), "phi": phi,
            "half_life": float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else None}


def negative_runs(daily: pd.Series) -> dict:
    """The longest run of consecutive calendar days with a negative mean gap."""
    daily = daily.dropna()
    days = pd.to_datetime(pd.Index(daily.index))
    best, best_span, run, start, prev = 0, (None, None), 0, None, None
    for day, value in zip(days, daily.to_numpy()):
        if value < 0 and run and (day - prev) == pd.Timedelta(days=1):
            run += 1
        elif value < 0:
            run, start = 1, day
        else:
            run = 0
        if run > best:
            best, best_span = run, (start, day)
        prev = day
    start_day, end_day = (None if d is None else d.strftime("%Y-%m-%d") for d in best_span)
    return {"days": int(len(daily)), "negative_days": int((daily < 0).sum()),
            "share_negative": float((daily < 0).mean()), "longest_days": int(best),
            "longest_start": start_day, "longest_end": end_day}


def negative_spells(values: pd.Series, days: int = 1, min_length: int = 1) -> list:
    """Spells of at least ``min_length`` consecutive negative values, in time order.

    ``values`` is indexed by dates (a period's first day); consecutive means
    ``days`` apart, so a missing period ends a spell.
    """
    values = values.dropna()
    stamps = pd.to_datetime(pd.Index(values.index))
    step = pd.Timedelta(days=days)
    spells, current = [], None
    for stamp, value in zip(stamps, values.to_numpy()):
        if value < 0 and current is not None and stamp - current[1] == step:
            current[1], current[2] = stamp, current[2] + 1
        elif value < 0:
            current = [stamp, stamp, 1]
            spells.append(current)
        else:
            current = None
    return [{"start": a.strftime("%Y-%m-%d"), "end": b.strftime("%Y-%m-%d"), "length": n}
            for a, b, n in spells if n >= min_length]


def rolling_mean_stats(daily: pd.Series, days: int = 28) -> dict:
    """Means over windows of ``days`` consecutive calendar days: how many are negative.

    Only complete windows count; ``min_bps`` is the lowest mean and
    ``min_end_day`` the last day of its window.
    """
    s = pd.Series(daily.to_numpy(dtype=float), index=pd.to_datetime(pd.Index(daily.index)))
    roll = s.asfreq("D").rolling(days, min_periods=days).mean().dropna()
    return {"days": int(days), "windows": int(len(roll)),
            "negative_windows": int((roll < 0).sum()), "min_bps": float(roll.min()),
            "min_end_day": roll.idxmin().strftime("%Y-%m-%d")}


def within_period(daily: pd.Series, freq: str = "Q") -> pd.Series:
    """Daily values minus the mean of their period (``freq``).

    A level that drifts from one period to the next drops out.
    """
    period = pd.to_datetime(pd.Index(daily.index)).to_period(freq)
    return daily - daily.groupby(period).transform("mean")


def block_bootstrap_median_ci(x, rng, block: int = 24, reps: int = 2000,
                              level: float = 0.95) -> tuple:
    """Percentile interval for the median of an hourly series, by the moving-block bootstrap.

    Each replicate joins randomly placed runs of ``block`` consecutive values
    (cut to the series' length), so dependence within a day is kept.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, size=(reps, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    medians = np.median(x[idx], axis=1)
    lo, hi = np.quantile(medians, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def median_intervals(gap: pd.Series, freq: str, rng, block: int = 24, reps: int = 2000,
                     level: float = 0.95) -> dict:
    """:func:`block_bootstrap_median_ci` of the gap's median in each period of ``freq``."""
    gap = gap.dropna()
    return {k: list(block_bootstrap_median_ci(g.to_numpy(), rng, block, reps, level))
            for k, g in gap.groupby(accrual_period(gap.index, freq))}


def level_shift(daily: pd.Series, last_before: str, first_after: str, days: int = 28) -> dict:
    """The change in the mean daily value across an interval, against every placement of it.

    ``before`` averages the ``days`` days up to and including ``last_before``,
    ``after`` the ``days`` days from ``first_after``; the shift is after minus
    before. The same difference is taken for every placement of the two windows
    and the days between them that fits in the sample (the observed one among
    them); ``share_as_large`` is the share of placements whose shift is at least
    as large in absolute value (two-sided), ``share_same_direction_as_large`` the
    share whose shift is at least as large in the observed direction (a fall at
    least as large for a fall, a rise for a rise).
    """
    s = pd.Series(daily.to_numpy(dtype=float), index=pd.to_datetime(pd.Index(daily.index)))
    s = s.asfreq("D")
    last, first = pd.Timestamp(last_before), pd.Timestamp(first_after)
    between = (first - last).days - 1
    trailing = s.rolling(days, min_periods=days).mean()         # the window ending each day
    shifts = (trailing.shift(-(between + days)) - trailing).dropna()
    observed = float(shifts.loc[last])
    others = shifts.to_numpy()
    q05, q50, q95 = np.quantile(others, [0.05, 0.5, 0.95])
    return {"last_before": last_before, "first_after": first_after, "window_days": int(days),
            "between_days": int(between), "before_bps": float(trailing.loc[last]),
            "after_bps": float(trailing.loc[first + pd.Timedelta(days=days - 1)]),
            "shift_bps": observed, "placements": int(len(others)),
            "share_as_large": float((np.abs(others) >= abs(observed) - 1e-12).mean()),
            "share_same_direction_as_large": float(
                (others <= observed + 1e-12).mean() if observed < 0
                else (others >= observed - 1e-12).mean()),
            "placements_q05_q50_q95_bps": [float(q05), float(q50), float(q95)]}


def binned_medians(y: pd.Series, x: pd.Series, edges) -> list:
    """Median and quartiles of ``y`` in right-closed bins of ``x``."""
    df = pd.DataFrame({"y": pd.Series(y, dtype=float).to_numpy(),
                       "x": pd.Series(x, dtype=float).to_numpy()}).dropna()
    cut = pd.cut(df["x"], list(edges), right=True)
    out = []
    for interval, g in df.groupby(cut, observed=False)["y"]:
        lo, hi = interval.left, interval.right
        row = {"lo": None if np.isinf(lo) else float(lo), "hi": None if np.isinf(hi) else float(hi),
               "n": int(len(g))}
        if len(g):
            q25, med, q75 = np.quantile(g.to_numpy(), [0.25, 0.5, 0.75])
            row.update(median_bps=float(med), q25_bps=float(q25), q75_bps=float(q75),
                       share_positive=float((g > 0).mean()))
        out.append(row)
    return out


def daily_realised_vol(close: pd.Series) -> pd.Series:
    """Daily realised volatility in percent: sqrt of the day's 24 squared hourly log returns."""
    r = np.log(close.astype(float)).diff().dropna()
    g = (r ** 2).groupby(accrual_period(r.index, "D"))
    return 100 * np.sqrt(g.sum()[g.size() == 24])


# ----------------------------------------------------------------------
#  Eight-hour averages and settlements
# ----------------------------------------------------------------------
def _block(series: pd.Series, stamps) -> np.ndarray:
    """The hours stamped S - 7h .. S for each block stamp S, earliest first, as columns."""
    stamps = pd.DatetimeIndex(stamps)
    return np.column_stack([series.reindex(stamps - pd.Timedelta(hours=7 - h)).to_numpy(dtype=float)
                            for h in range(8)])


def linear_blocks(venue: str, stamps) -> np.ndarray:
    """Whether each settlement averaged its premium with linearly rising weights.

    Rebuilding every settlement from the minute klines shows that Binance used the
    linearly weighted eight-hour average throughout, and Bybit the plain average
    up to the settlement of 7 Mar 2024 08:00 UTC and the linearly weighted one
    from 16:00 that day (:data:`FIRST_LINEAR_SETTLEMENT`).
    """
    first = FIRST_LINEAR_SETTLEMENT.get(venue)
    stamps = pd.DatetimeIndex(stamps)
    return np.ones(len(stamps), bool) if first is None else np.asarray(stamps >= _utc(first))


def eight_hour_average(hourly: pd.Series, stamps, linear=True) -> pd.DataFrame:
    """Hourly premiums averaged over the eight-hour block closing at each stamp.

    With linear weights (``linear`` True, or True for that block), hour h of the
    block (0 = earliest) carries the weight its minutes carry in the exchanges'
    average, 3600 h + 1830; otherwise every hour weighs the same. A missing hour
    is left out and the others reweighted; ``hours`` counts the hours present.
    """
    v = _block(hourly, stamps)
    present = ~np.isnan(v)
    lin = np.broadcast_to(np.asarray(linear, bool).reshape(-1, 1), v.shape)
    w = np.where(present, np.where(lin, LINEAR_WEIGHTS, 1.0), 0.0)
    den = w.sum(axis=1)
    num = (w * np.where(present, v, 0.0)).sum(axis=1)
    value = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return pd.DataFrame({"value": value, "hours": present.sum(axis=1)},
                        index=pd.DatetimeIndex(stamps))


def exchange_eight_hour(frame: pd.DataFrame, stamps, linear=True) -> pd.Series:
    """An exchange's eight-hour average at each stamp, rebuilt from its hours.

    The linearly weighted average (minute j of the block weighs j) comes exactly
    from the hours' plain and position-weighted means (see
    :func:`regime_engine.fetch_premium.eight_hour_linear_mean`), the plain one
    from the plain means; ``linear`` chooses per block. A block with an incomplete
    hour is left out.
    """
    mean, lw, n = (_block(frame[c], stamps) for c in ("mean", "lwmean", "n"))
    weighted = (3600 * np.arange(8) * mean + 1830 * lw).sum(axis=1) / (480 * 481 / 2)
    value = np.where(np.asarray(linear, bool), weighted, mean.mean(axis=1))
    complete = (n == 60).all(axis=1)
    return pd.Series(np.where(complete, value, np.nan), index=pd.DatetimeIndex(stamps))


def settlement_fit(settlements: pd.DataFrame, rebuilt: pd.Series) -> dict:
    """How closely rebuilt eight-hour averages reproduce the settlements.

    Errors are taken on settlements off the floor, whose premium the rate
    identifies exactly; ``floor_agreement`` is the share of all settlements whose
    rebuilt average lies in the band exactly when the settlement is at the floor.
    """
    r = rebuilt.reindex(settlements.index)
    off = ~settlements["at_floor"]
    err = (r[off] - settlements.loc[off, "premium"]).dropna()
    both = r.notna()
    agree = (in_band(r[both]).astype(bool).to_numpy() == settlements.loc[both, "at_floor"]
             .to_numpy())
    a = err.abs()
    return {"n_settlements": int(len(settlements)), "n_identified": int(off.sum()),
            "n_compared": int(len(err)), "median_abs_error_bps": float(a.median()),
            "mean_error_bps": float(err.mean()), "p95_abs_error_bps": float(a.quantile(0.95)),
            "max_abs_error_bps": float(a.max()), "share_within_0.1bp": float((a <= 0.1).mean()),
            "n_floor_compared": int(both.sum()), "floor_agreement": float(agree.mean())}


# ----------------------------------------------------------------------
#  Decomposition of the floor-share difference
# ----------------------------------------------------------------------
def phase_of(median_bps: float) -> str:
    """Where a quarter's median exchange premium sat: 'below', 'inside_nonpositive' or 'positive'.

    'below': at or below the band's lower edge (-4 bps), where a higher premium
    lifts hours into the band; 'inside_nonpositive': above that edge and at most
    zero; 'positive': above zero, near or beyond the upper edge (+6 bps), where a
    higher premium lifts hours out of the band.
    """
    if median_bps <= BAND[0]:
        return "below"
    return "inside_nonpositive" if median_bps <= 0 else "positive"


def quarter_medians(exchange: pd.Series) -> dict:
    """The exchange's median hourly premium in each quarter, bps."""
    x = exchange.dropna()
    return {q: float(m) for q, m in x.groupby(accrual_period(x.index, "Q")).median().items()}


def quarter_phases(exchange: pd.Series) -> dict:
    """:func:`phase_of` the exchange's median hourly premium, quarter by quarter."""
    return {q: phase_of(m) for q, m in quarter_medians(exchange).items()}


def phase_groups(phases: dict) -> dict:
    """The quarters of each phase, and the below-band ones of each span in :data:`SUBPERIODS`."""
    out = {ph: [q for q, f in phases.items() if f == ph] for ph in PHASES}
    for name, (lo, hi) in SUBPERIODS.items():
        out[name] = [q for q in out["below"] if lo <= q <= hi]
    return out


FACTORS = ("location", "averaging", "dispersion")


def shapley(corners: dict) -> dict:
    """Split v(0,0,0) - v(1,1,1) over three factors switched from 0 to 1, symmetrically.

    ``corners`` maps (location, averaging, dispersion) in {0, 1}^3 to a share.
    Each factor gets its marginal effect averaged over the six orders in which
    the factors can be switched (the Shapley value); the parts add up to the
    total. ``range`` holds the smallest and largest marginal effect over the orders.
    """
    out = {f: [] for f in FACTORS}
    for order in itertools.permutations(range(3)):
        state = [0, 0, 0]
        for f in order:
            before = corners[tuple(state)]
            state[f] = 1
            out[FACTORS[f]].append(before - corners[tuple(state)])
    return {**{f: float(np.mean(v)) for f, v in out.items()},
            "range": {f: [float(min(v)), float(max(v))] for f, v in out.items()}}


def decompose(hl: pd.Series, exchange: pd.Series, exchange_8h: pd.Series,
              settlements: pd.DataFrame, quarters=None, start=None, linear=True,
              span=None) -> dict:
    """Split Hyperliquid's hourly floor share minus an exchange's settlement floor share.

    Three things differ between the two: where the premium sits (location: the
    quarter's median hourly gap G), over what it is averaged (one hour against
    the exchanges' eight-hour blocks with linearly rising weights) and how widely
    it is spread around its location (dispersion). Each corner of the cube is an
    in-band share:

        a  Hyperliquid hourly              b  Hyperliquid hourly - G
        c  Hyperliquid eight-hour          e  Hyperliquid eight-hour - G
        f  exchange hourly + G             g  exchange hourly
        h  exchange eight-hour + G         d  exchange settlements at the floor

    a - d is split by :func:`shapley`. ``path`` is one of its six orders, location
    then averaging then dispersion (a -> b -> e -> d), whose steps add up to a - d.
    ``contrasts`` are single comparisons that need not add up: shifting
    Hyperliquid's hourly premium by -G (a - b, location at Hyperliquid's spread),
    shifting the exchange's eight-hour average by +G (h - d, location at the
    exchange's spread), averaging Hyperliquid's hours over the exchange's blocks
    (a - c), and what is left once Hyperliquid's premium is shifted and averaged
    (e - d). The hourly shares and the gaps use the hours stamped from ``start``
    on; Hyperliquid's blocks draw on every hour of ``hl``, weighted as the
    exchange weighted each block (``linear``, see :func:`linear_blocks`).
    ``quarters`` (a list) or ``span`` (a name in :data:`SPANS`) restrict the
    shares, not the gaps.
    """
    blocks_from = hl.dropna()
    hl = blocks_from if start is None else blocks_from[blocks_from.index >= _utc(start)]
    x = exchange.reindex(hl.index)
    gap = (hl - x).dropna()
    by_q = gap.groupby(accrual_period(gap.index, "Q")).median()
    q_hours = accrual_period(hl.index, "Q")
    g_hours = by_q.reindex(q_hours).to_numpy()
    stamps = settlements.index
    q_blocks = accrual_period(stamps, "Q")
    g_blocks = by_q.reindex(q_blocks).to_numpy()
    hl8 = eight_hour_average(blocks_from, stamps, linear)["value"].to_numpy()
    x8 = exchange_8h.reindex(stamps).to_numpy()
    keep_h = np.ones(len(hl), bool) if quarters is None else np.asarray(q_hours.isin(quarters))
    keep_b = (np.ones(len(stamps), bool) if quarters is None
              else np.asarray(q_blocks.isin(quarters)))
    if span is not None:
        keep_h &= _span_mask(hl.index, span)
        keep_b &= _span_mask(stamps, span)

    def hourly(v):
        return float(in_band(np.asarray(v)[keep_h]).mean())

    def blocks(v):
        return float(in_band(np.asarray(v)[keep_b]).mean())

    s = {"a_hl_hourly": hourly(hl), "b_hl_hourly_shifted": hourly(hl.to_numpy() - g_hours),
         "c_hl_eight_hour": blocks(hl8), "e_hl_eight_hour_shifted": blocks(hl8 - g_blocks),
         "f_exchange_hourly_shifted": hourly(x.to_numpy() + g_hours),
         "g_exchange_hourly": hourly(x),
         "h_exchange_eight_hour_shifted": blocks(x8 + g_blocks),
         "exchange_eight_hour": blocks(x8),
         "d_exchange_settlements": float(settlements["at_floor"].to_numpy()[keep_b].mean())}
    corners = {(0, 0, 0): s["a_hl_hourly"], (1, 0, 0): s["b_hl_hourly_shifted"],
               (0, 1, 0): s["c_hl_eight_hour"], (1, 1, 0): s["e_hl_eight_hour_shifted"],
               (0, 0, 1): s["f_exchange_hourly_shifted"], (1, 0, 1): s["g_exchange_hourly"],
               (0, 1, 1): s["h_exchange_eight_hour_shifted"],
               (1, 1, 1): s["d_exchange_settlements"]}
    a, b, c, e = (s[k] for k in ("a_hl_hourly", "b_hl_hourly_shifted", "c_hl_eight_hour",
                                 "e_hl_eight_hour_shifted"))
    h, d = s["h_exchange_eight_hour_shifted"], s["d_exchange_settlements"]
    return {"n_hours": int(keep_h.sum()), "n_blocks": int(keep_b.sum()), "shares": s,
            "difference": a - d, "shapley": shapley(corners),
            "path": {"location": a - b, "averaging": b - e, "dispersion": e - d},
            "contrasts": {"shift_hl_hourly": a - b, "shift_exchange_eight_hour": h - d,
                          "average_hl": a - c, "remainder": e - d},
            "gap_by_quarter": {k: float(v) for k, v in by_q.items()}}


# ----------------------------------------------------------------------
#  The artifact
# ----------------------------------------------------------------------
MIN_MINUTES = 60                      # an exchange hour enters the gap only when complete
SPANS = {
    "common": (RULE_START, END),
    "before_paper": (RULE_START, "2025-10-01 23:00"),
    "paper": WINDOWS["paper"],
    "later": ("2026-04-28 09:00", END),
}


def _span_mask(index, name: str) -> np.ndarray:
    lo, hi = (_utc(t) for t in SPANS[name])
    idx = pd.DatetimeIndex(index)
    return np.asarray((idx >= lo) & (idx <= hi))


def _share(x) -> float:
    return float(pd.Series(x, dtype=float).mean())


def _decimal_floor_count(rates, floor: Decimal) -> int:
    return int(sum(Decimal(str(r).strip()) == floor for r in rates))


def _venue_data(coin: str, venue: str, data_dir) -> dict:
    """Hourly premium, close, completeness and settlements of one exchange."""
    symbol = ASSETS[coin]
    frame = load_exchange_hourly(venue, symbol, data_dir)
    settle = load_settlements(venue, symbol, data_dir)
    settle = settle[(settle.index >= _utc(FIRST_SETTLEMENT)) & (settle.index <= _utc(END))]
    return {"frame": frame, "mean": frame["mean"].where(frame["n"] >= MIN_MINUTES),
            "close": frame["close"], "settlements": settle}


def _interim_checks(venues: dict, data_dir, interim_dir) -> dict:
    """The BTC series of the censoring comparison (data/interim) against the files here.

    Hyperliquid's records and the exchanges' settlements must be the same on the
    overlap; the hourly close built from the minutes must equal the close of the
    hourly kline that opens an hour before its stamp, on the paper window.
    """
    interim_dir, data_dir = pathlib.Path(interim_dir), pathlib.Path(data_dir)
    full = pd.read_csv(interim_dir / DEFAULT_PREMIUM_FULL.name, dtype=str)
    full.index = _index(full["time"])
    mine = pd.read_csv(data_dir / "hyperliquid_BTC_funding.csv", dtype=str)
    mine.index = _index(mine["time"])
    shared = full.index.intersection(mine.index)

    def equal(col):
        return int(sum(Decimal(a.strip()) == Decimal(b.strip()) for a, b in
                       zip(full.loc[shared, col], mine.loc[shared, col])))
    out = {"hyperliquid": {"hours": int(len(shared)), "premiums_equal": equal("premium"),
                           "rates_equal": equal("fundingRate")}}
    for venue, v in venues.items():
        f = pd.read_csv(interim_dir / f"{venue}_BTCUSDT_funding.csv", dtype=str)
        f.index = _index(f["time"])
        sh = f.index.intersection(v["settlements"].index)
        eq = sum(Decimal(a.strip()) == Decimal(b) for a, b in
                 zip(f.loc[sh, "fundingRate"], v["settlements"].loc[sh, "rate"]))
        k = pd.read_csv(interim_dir / f"{venue}_BTCUSDT_premium_1h.csv", dtype=str)
        k.index = _index(k["time"]) + pd.Timedelta(hours=1)      # open time -> closing stamp
        k = k[_span_mask(k.index, "paper")]
        mine = pd.read_csv(data_dir / f"{venue}_BTCUSDT_premium_hourly.csv", dtype=str,
                           keep_default_na=False)
        mine.index = _index(mine["time"])
        mine = mine["close"].reindex(k.index).fillna("")
        equal = sum(bool(b) and Decimal(a.strip()) == Decimal(b)
                    for a, b in zip(k["close"], mine))
        kd = (k["close"].astype(float) - pd.to_numeric(mine.replace("", np.nan))).abs() * 1e4
        out[venue] = {"settlements": int(len(sh)), "rates_equal": int(eq),
                      "paper_hours": int(len(k)), "closes_equal": int(equal),
                      "closes_max_abs_diff_bps": float(kd.max())}
    return out


def _window_block(hl: pd.DataFrame, x: pd.Series, settle: pd.DataFrame, name: str) -> dict:
    """In-band and floor shares of both venues and the gap over one span."""
    h = hl[_span_mask(hl.index, name)]
    xs = x.reindex(h.index)
    s = settle[_span_mask(settle.index, name)]
    return {"hours": int(len(h)),
            "hl_floor_share": _decimal_floor_count(h["rate"], FLOOR_HOURLY) / len(h),
            "hl_in_band": _share(in_band(h["premium"])),
            "exchange_hourly_in_band": _share(in_band(xs)),
            "exchange_hours": int(xs.notna().sum()),
            "settlements": int(len(s)), "settlements_at_floor": int(s["at_floor"].sum()),
            "settlement_floor_share": float(s["at_floor"].mean()),
            "gap": gap_stats(h["premium"] - xs)}


def _persistence(gap: pd.Series) -> dict:
    """How the daily mean gap persists, and when it was negative."""
    daily = daily_means(gap)
    week = daily.groupby(pd.to_datetime(pd.Index(daily.index)).to_period("W").astype(str))
    weekly = week.mean()[week.size() >= 5]
    weekly_by_start = pd.Series(weekly.to_numpy(), index=[w[:10] for w in weekly.index])
    return {"daily": {"days": int(len(daily)), "mean_bps": float(daily.mean()),
                      "median_bps": float(daily.median()), "min_bps": float(daily.min()),
                      "min_day": str(daily.idxmin())},
            "acf_daily": {str(k): v for k, v in autocorrelations(daily, (1, 7, 30)).items()},
            "ar1_daily": ar1_half_life(daily),
            "ar1_daily_within_quarter": ar1_half_life(within_period(daily, "Q")),
            "ar1_hourly": ar1_half_life(gap.dropna().reset_index(drop=True)),
            "negative_days": negative_runs(daily),
            "negative_day_spells": negative_spells(daily, 1, min_length=5),
            "weeks": int(len(weekly)), "negative_weeks": int((weekly < 0).sum()),
            "negative_weeks_list": [str(w) for w in weekly[weekly < 0].index],
            "negative_week_spells": negative_spells(weekly_by_start, 7, min_length=2),
            "rolling_28d": rolling_mean_stats(daily, 28)}


def _phase_dependence(gap: pd.Series, x: pd.Series, rv: pd.Series) -> dict:
    lo, hi = BAND
    position = pd.Series(np.select([x < lo, x > hi], ["below", "above"], "inside"),
                         index=x.index).where(x.notna())
    by_position = {}
    for name in ("below", "inside", "above"):
        g = gap[(position == name).to_numpy()]
        by_position[name] = gap_stats(g)
    year = accrual_period(gap.index, "Y")
    by_position_by_year = {
        str(y): {name: gap_stats(gap[np.asarray((year == y) & (position == name).to_numpy())])
                 for name in ("below", "inside", "above")}
        for y in sorted(set(year))}
    daily_gap = daily_means(gap)
    daily_x = daily_means(x)
    both = pd.concat([daily_gap.rename("gap"), rv.rename("rv"), daily_x.rename("x")],
                     axis=1).dropna()
    edges = [-np.inf, *np.quantile(both["rv"], [0.2, 0.4, 0.6, 0.8]), np.inf]
    return {"by_exchange_premium": binned_medians(gap, x.reindex(gap.index), PREMIUM_BINS),
            "by_band_position": by_position,
            "by_band_position_by_year": by_position_by_year,
            "daily_by_realised_vol_quintile": binned_medians(both["gap"], both["rv"], edges),
            "realised_vol_quintile_edges_pct": [float(e) for e in edges[1:-1]],
            "spearman_daily": {"gap_vs_realised_vol": float(both["gap"].corr(both["rv"],
                                                                             "spearman")),
                               "gap_vs_exchange_premium": float(both["gap"].corr(both["x"],
                                                                                 "spearman")),
                               "days": int(len(both))}}


USDC_HAC_LAGS = 20                    # Newey-West lags (days) of the usdc_pricing slopes


def _hac_slope(y: pd.Series, x: pd.Series, lags: int = USDC_HAC_LAGS) -> dict:
    """OLS slope of ``y`` on ``x`` with a Newey-West (HAC) standard error."""
    import statsmodels.api as sm
    fit = sm.OLS(y.to_numpy(float), sm.add_constant(x.to_numpy(float))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    slope, se = float(fit.params[1]), float(fit.bse[1])
    return {"slope": slope, "slope_hac_se": se,
            "slope_ci95": [slope - 1.96 * se, slope + 1.96 * se],
            "intercept_bps": float(fit.params[0])}


def usdc_pricing_check(gap: pd.Series, usdc: pd.Series) -> dict:
    """How the daily mean gap moves with the daily mean USDC/USDT rate: a behavioural check.

    Hyperliquid's perpetuals are USDT-denominated linear contracts margined in
    USDC without conversion (quanto contracts): the oracle, and so the premium,
    are in USDT terms like the exchanges'. By construction the USDC/USDT rate
    does not enter the premium. Traders who nonetheless priced the book in USDC
    would pull the premium down one for one as USDC gains on USDT (slope -1);
    pricing in USDT terms, as the contract specifies, gives a slope of zero.
    ``usdc`` is ln(USDC/USDT) in bps; ``basis`` below refers to it. The slope
    carries a Newey-West standard error (:data:`USDC_HAC_LAGS` days); the
    within-quarter slope uses both daily series net of their quarter's mean, so
    a common drift between quarters drops out.
    """
    d = pd.concat([daily_means(gap).rename("gap"),
                   daily_means(usdc.reindex(gap.index)).rename("basis")], axis=1).dropna()
    full = _hac_slope(d["gap"], d["basis"])
    within = _hac_slope(within_period(d["gap"], "Q"), within_period(d["basis"], "Q"))
    within.pop("intercept_bps")
    r = float(d["gap"].corr(d["basis"]))
    return {"days": int(len(d)), "basis_mean_bps": float(d["basis"].mean()),
            "basis_q05_bps": float(d["basis"].quantile(0.05)),
            "basis_q95_bps": float(d["basis"].quantile(0.95)),
            "pearson_daily": r, "r2_daily": r * r,
            "spearman_daily": float(d["gap"].corr(d["basis"], "spearman")),
            "slope_daily": full["slope"], "intercept_daily_bps": full["intercept_bps"],
            "hac_lags": USDC_HAC_LAGS, "slope_hac_se": full["slope_hac_se"],
            "slope_ci95": full["slope_ci95"], "within_quarter": within}


def _unmatched(hl: pd.Series, settle: pd.DataFrame) -> dict:
    """Quarterly median of Hyperliquid's hourly premium minus the exchange's censored median.

    The exchange median comes from its settlements, the in-band ones known only
    to lie in the band; a quarter whose median falls on them has none.
    """
    out = {}
    q_s = accrual_period(settle.index, "Q")
    for q, h in hl.groupby(accrual_period(hl.index, "Q")):
        med = censoring.censored_quantile(settle["premium"][q_s == q], 0.5)
        out[q] = None if med is None else float(h.median() - med)
    return out


def _robustness(hl: pd.DataFrame, hl_all: pd.Series, v: dict, linear) -> dict:
    """The gap from kline closes, without the crash day, and on eight-hour blocks.

    Hyperliquid's eight-hour blocks draw on every hour of ``hl_all``, so the
    block closing at the first settlement uses the six hours before the rule.
    """
    gap = hl["premium"] - v["mean"].reindex(hl.index)
    close_gap = hl["premium"] - v["close"].reindex(hl.index)
    crash = np.asarray(accrual_period(hl.index, "D") == CRASH_DAY)
    stamps = v["settlements"].index
    hl8 = eight_hour_average(hl_all, stamps, linear)["value"]
    x8 = exchange_eight_hour(v["frame"], stamps, linear)
    gap8 = hl8 - x8
    out = {}
    for name in ("common", "paper"):
        m = _span_mask(hl.index, name)
        m8 = _span_mask(stamps, name)
        out[name] = {"one_minute_mean": gap_stats(gap[m]),
                     "kline_close": gap_stats(close_gap[m]),
                     "excluding_2025_10_10": gap_stats(gap[m & ~crash]),
                     "eight_hour_blocks": gap_stats(gap8[m8])}
    return out


def _median_cis(gap: pd.Series, seed_key) -> dict:
    """Bootstrap intervals of the monthly and quarterly medians, and those that cover zero.

    The headline intervals use ``BOOTSTRAP["block_hours"]``; every block length
    in ``sensitivity_block_hours`` is run from the same seed and kept under
    ``block_sensitivity``, and the periods whose coverage of zero differs from
    the 24-hour blocks are listed. The partial month and quarter (December 2023,
    2023Q4) are listed apart.
    """
    b = BOOTSTRAP

    def run(block: int) -> dict:
        out = {}
        for freq, name in (("M", "monthly"), ("Q", "quarterly")):
            rng = np.random.default_rng([b["seed"], *seed_key, ord(freq)])
            out[name] = median_intervals(gap, freq, rng, block, b["reps"], b["level"])
        for name, key in (("monthly", "months_covering_zero"),
                          ("quarterly", "quarters_covering_zero")):
            out[key] = [k for k, (lo, hi) in out[name].items() if lo <= 0 <= hi]
        out["full_months_covering_zero"] = [m for m in out["months_covering_zero"]
                                            if m not in PARTIAL_MONTHS]
        out["partial_months_covering_zero"] = [m for m in out["months_covering_zero"]
                                               if m in PARTIAL_MONTHS]
        out["full_quarters_covering_zero"] = [q for q in out["quarters_covering_zero"]
                                              if q not in PARTIAL_QUARTERS]
        out["partial_quarters_covering_zero"] = [q for q in out["quarters_covering_zero"]
                                                 if q in PARTIAL_QUARTERS]
        full = [lo for q, (lo, _) in out["quarterly"].items() if q not in PARTIAL_QUARTERS]
        out["lowest_full_quarter_lower_bound"] = min(full) if full else None
        return out

    out = {"block_hours": b["block_hours"], **run(b["block_hours"])}
    out["block_sensitivity"] = {str(h): run(h) for h in b["sensitivity_block_hours"]}
    ref = out["block_sensitivity"].get("24")
    if ref is not None:
        out["coverage_change_vs_24h"] = {
            freq: {"now_covering_zero": sorted(set(out[key]) - set(ref[key])),
                   "no_longer_covering_zero": sorted(set(ref[key]) - set(out[key]))}
            for freq, key in (("monthly", "months_covering_zero"),
                              ("quarterly", "quarters_covering_zero"))}
    return out


def _decomposition(hl_all: pd.Series, x: pd.Series, x8: pd.Series, settle: pd.DataFrame,
                   gap: pd.Series, lin) -> dict:
    """:func:`decompose` over all quarters, each phase group, the two windows and each quarter."""
    phases = quarter_phases(x)
    args = (hl_all, x, x8, settle)
    dec = {"phases": phases, "exchange_median_bps": quarter_medians(x),
           "all": decompose(*args, start=RULE_START, linear=lin)}
    q_gap = accrual_period(gap.index, "Q")
    for name, qs in phase_groups(phases).items():
        if not qs:
            continue
        r = decompose(*args, quarters=qs, start=RULE_START, linear=lin)
        r.pop("gap_by_quarter")
        r["quarters"] = qs
        r["gap"] = gap_stats(gap[np.asarray(q_gap.isin(qs))])
        dec[name] = r
    for w in ("paper", "later"):
        dec[w] = decompose(*args, start=RULE_START, linear=lin, span=w)
        dec[w].pop("gap_by_quarter")
        dec[w]["gap"] = gap_stats(gap[_span_mask(gap.index, w)])
    dec["by_quarter"] = {}
    for q in sorted(phases):
        r = decompose(*args, quarters=[q], start=RULE_START, linear=lin)
        r.pop("gap_by_quarter")
        dec["by_quarter"][q] = r
    if "positive" in dec:
        dec["positive"]["reversed_quarters"] = [
            q for q in dec["positive"]["quarters"] if dec["by_quarter"][q]["difference"] < 0]
    return dec


NOTE = (
    "Premium gap D_t = P_HL,t - P_X,t in bps (8-hour equivalent). P_HL,t: Hyperliquid's "
    "fundingHistory premium of the record stamped t (its average over [t-1h, t)). P_X,t: "
    "mean close of exchange X's one-minute premium-index klines opening in [t-1h, t); an "
    "hour enters only when all 60 minutes are present. Spans under the common rule: hours "
    "stamped 2023-12-11 23:00 to 2026-10-01 00:00 UTC, settlements 2023-12-12 00:00 to "
    "2026-10-01 00:00. A record belongs to the period its premium accrued in (stamp minus "
    "one hour), so 2023 and 2023Q4 start 2023-12-11 22:00. Band [-4, +6] bps with its edges, "
    "where the rule pays the floor; floors compared as decimals. Decomposition: a = "
    "Hyperliquid hourly in band, b = shifted by minus G, c = averaged over the exchanges' "
    "8-hour blocks with the exchange's weights (hour h of the block 3600h + 1830; equal "
    "weights for Bybit blocks before 2024-03-07 16:00 UTC, as Bybit settled), e = c "
    "shifted, d = exchange settlements at the floor; f, g, h = exchange hourly + G, hourly, "
    "8-hour + G (G = the quarter's median hourly gap). a - d is split over location, "
    "averaging and dispersion by the Shapley value over the eight corners; 'range' gives "
    "each factor's smallest and largest effect over the six orders; 'path' is the order "
    "location, averaging, dispersion (a to b to e to d), whose steps add up; 'contrasts' are "
    "single comparisons (a - b, h - d, a - c, e - d) that do not add up. Phase of a quarter, "
    "from the exchange's median hourly premium m: 'below' if m <= -4 bps, "
    "'inside_nonpositive' if -4 < m <= 0, 'positive' if m > 0; the below-band quarters are "
    "also split into 2024Q3-2025Q4 and 2026. median_ci: 95% percentile intervals of the "
    "monthly and quarterly medians of D_t, moving-block bootstrap with 168-hour blocks and "
    "2,000 replicates (block_sensitivity: the same with 24- and 72-hour blocks, from the "
    "same seed; coverage_change_vs_24h: the periods whose coverage of zero differs from the "
    "24-hour blocks; December 2023 and 2023Q4 are partial and are listed apart: "
    "full_months_covering_zero, partial_months_covering_zero and the same for quarters). "
    "checks.usdc_pricing: daily mean gap on the daily mean ln(USDC/USDT) (Binance spot, "
    "bps); Hyperliquid's contracts are USDT-denominated and USDC-margined without "
    "conversion, so a slope of -1 would mean traders price the book in USDC, a slope of 0 "
    "that they price it in USDT as specified; slope_hac_se is a Newey-West standard error "
    "(20 days), within_quarter the slope of both series net of their quarter's mean. "
    "checks.oracle_wording_shift: mean daily gap over the 28 days from 2025-12-10 minus "
    "that over the 28 days to 2025-10-06 (the oracle documentation read 'spot prices' on "
    "the earlier date and 'spot mid prices' on the later), against every placement of the "
    "same windows in the sample; share_as_large counts placements whose change is at "
    "least as large in either direction, share_same_direction_as_large those at least as "
    "large in the observed direction. phase.by_band_position_by_year: the gap by the "
    "exchange's band position within each accrual year. hyperliquid_premium_ar1: OLS "
    "AR(1) with an intercept on Hyperliquid's hourly premium over each span, half-life "
    "ln 0.5 / ln phi in hours.")


def build_payload(data_dir=DATA, interim_dir=INTERIM) -> dict:
    """Every number of the premium-gap analysis, per asset and exchange."""
    p = {"_note": NOTE,
         "band_bps": list(BAND), "floors": {"8h": str(FLOOR_8H), "hourly": str(FLOOR_HOURLY)},
         "spans": {k: list(v) for k, v in SPANS.items()},
         "first_settlement": FIRST_SETTLEMENT, "crash_day": CRASH_DAY,
         "partial_quarters": PARTIAL_QUARTERS, "phases": list(PHASES),
         "subperiods": {k: list(v) for k, v in SUBPERIODS.items()},
         "mechanism_quarters": MECHANISM_QUARTERS, "bootstrap": BOOTSTRAP,
         "oracle_wording": ORACLE_WORDING,
         "data": {}, "windows": {}, "validation": {"settlements": {}}, "gap": {},
         "monthly": {}, "median_ci": {}, "persistence": {}, "phase": {}, "decomposition": {},
         "robustness": {}, "unmatched_reference": {}, "mechanism": {}, "checks": {},
         "hyperliquid_premium_ar1": {}}
    usdc = load_usdc_usdt(data_dir)
    for ci, (coin, symbol) in enumerate(ASSETS.items()):
        hl_all = load_hyperliquid(coin, data_dir)
        hl = hl_all[hl_all.index >= _utc(RULE_START)]
        expected = pd.date_range(RULE_START, END, freq="1h", tz="UTC")
        p["data"][coin] = {"hyperliquid": {
            "first": str(hl_all.index.min()), "last": str(hl_all.index.max()),
            "hours": int(len(hl)),
            "missing_hours": [str(t) for t in expected.difference(hl.index)],
            "floor_hours": _decimal_floor_count(hl["rate"], FLOOR_HOURLY),
            "in_band_hours": int(in_band(hl["premium"]).sum())}}
        p["hyperliquid_premium_ar1"][coin] = {
            w: ar1_half_life(hl["premium"][_span_mask(hl.index, w)].reset_index(drop=True))
            for w in SPANS}
        rv = daily_realised_vol(load_prices(symbol, data_dir))
        venues = {venue: _venue_data(coin, venue, data_dir) for venue in EXCHANGES}
        if coin == "BTC":
            checks = _interim_checks(venues, data_dir, interim_dir)
            p["data"][coin]["hyperliquid"]["interim_overlap"] = checks.pop("hyperliquid")
            p["validation"]["hourly_close_vs_interim"] = {
                k: {"hours": c["paper_hours"], "equal": c["closes_equal"],
                    "max_abs_diff_bps": c["closes_max_abs_diff_bps"]} for k, c in checks.items()}
        for key in ("windows", "gap", "monthly", "median_ci", "persistence", "phase",
                    "decomposition", "robustness", "unmatched_reference", "mechanism", "checks"):
            p[key][coin] = {}
        p["validation"]["settlements"][coin] = {}
        for vi, (venue, v) in enumerate(venues.items()):
            frame, settle = v["frame"], v["settlements"]
            hours = frame[frame.index >= _utc(RULE_START)]
            d = {"hours": int(len(hours)), "complete_hours": int((hours["n"] == 60).sum()),
                 "hours_without_minutes": int((hours["n"] == 0).sum()),
                 "settlements": int(len(settle)),
                 "spacing_hours": censoring.spacing_hours(settle.index.to_series()),
                 "max_abs_rate": str(max((Decimal(r) for r in settle["rate"]), key=abs))}
            if coin == "BTC":
                c = checks[venue]
                d["interim_overlap"] = {"settlements": c["settlements"],
                                        "rates_equal": c["rates_equal"]}
            p["data"][coin][venue] = d
            x = v["mean"].reindex(hl.index)
            gap = hl["premium"] - x
            p["windows"][coin][venue] = {w: _window_block(hl, x, settle, w) for w in SPANS}
            lin = linear_blocks(venue, settle.index)
            x8 = exchange_eight_hour(frame, settle.index, lin)
            fit = settlement_fit(settle, x8)
            fit["weights"] = ("linear throughout" if lin.all() else
                              f"plain before {FIRST_LINEAR_SETTLEMENT[venue]} UTC, linear from it")
            fit["linear_throughout"] = settlement_fit(
                settle, exchange_eight_hour(frame, settle.index, True))
            if not lin.all():
                early = settle[~lin]
                fit["before_switch"] = {
                    "settlements": int(len(early)), "last": str(early.index.max()),
                    "plain": settlement_fit(early, exchange_eight_hour(frame, early.index, False)),
                    "linear": settlement_fit(early, exchange_eight_hour(frame, early.index, True))}
            paper = _span_mask(settle.index, "paper")
            fit["paper"] = settlement_fit(settle[paper], x8[paper])
            p["validation"]["settlements"][coin][venue] = fit
            p["gap"][coin][venue] = {"common": gap_stats(gap), "by_year": gap_by(gap, "Y"),
                                     "by_quarter": gap_by(gap, "Q")}
            p["monthly"][coin][venue] = gap_by(gap, "M")
            p["median_ci"][coin][venue] = _median_cis(gap, (ci, vi))
            p["persistence"][coin][venue] = _persistence(gap)
            p["phase"][coin][venue] = _phase_dependence(gap, x, rv)
            dec = _decomposition(hl_all["premium"], x, x8, settle, gap, lin)
            p["decomposition"][coin][venue] = dec
            p["robustness"][coin][venue] = _robustness(hl, hl_all["premium"], v, lin)
            p["unmatched_reference"][coin][venue] = _unmatched(hl["premium"], settle)
            w = ORACLE_WORDING
            p["checks"][coin][venue] = {
                "usdc_pricing": usdc_pricing_check(gap, usdc),
                "oracle_wording_shift": level_shift(daily_means(gap), w["last_spot_prices"],
                                                    w["first_spot_mid_prices"], w["days"])}
            for name, q in MECHANISM_QUARTERS.items():
                m = np.asarray(accrual_period(hl.index, "Q") == q)
                p["mechanism"][coin].setdefault(name, {"quarter": q,
                                                       "hyperliquid": premium_summary(
                                                           hl["premium"][m])})
                summary = premium_summary(x[m])
                summary["settlement_floor_share"] = (
                    dec["by_quarter"][q]["shares"]["d_exchange_settlements"])
                p["mechanism"][coin][name][venue] = summary
    return p


def premium_summary(bps) -> dict:
    """Median, quartiles and the shares below, inside (edges included) and above the band."""
    x = pd.Series(bps, dtype=float).dropna()
    lo, hi = BAND
    q25, med, q75 = np.quantile(x.to_numpy(), [0.25, 0.5, 0.75])
    return {"n": int(len(x)), "median_bps": float(med), "q25_bps": float(q25),
            "q75_bps": float(q75), "share_below": float((x < lo).mean()),
            "share_in_band": float(((x >= lo) & (x <= hi)).mean()),
            "share_above": float((x > hi).mean())}


# ----------------------------------------------------------------------
#  Tables
# ----------------------------------------------------------------------
YEARS = ("2023", "2024", "2025", "2026")
SPLIT_ASSETS = ("BTC",)                # assets in Panel B of table12_premium_gap.tex
GROUP_ROWS = ("all", "below", "below_2024Q3_2025Q4", "below_2026", "inside_nonpositive",
              "positive", "paper")
GROUP_LABELS = {"all": "all quarters", "below": "$\\tilde P_X\\le-4$",
                "below_2024Q3_2025Q4": "\\quad below band, 2024Q3--2025Q4",
                "below_2026": "\\quad below band, 2026",
                "inside_nonpositive": "$-4<\\tilde P_X\\le0$", "positive": "$\\tilde P_X>0$",
                "paper": "estimation window"}


def _bps(x) -> str:
    return "" if x is None else ("$-$" if round(x, 2) < 0 else "") + f"{abs(x):.2f}"


def _pp(x: float) -> str:
    return ("$-$" if round(100 * x, 1) < 0 else "") + f"{abs(100 * x):.1f}"


def sample_label(sample: str, dec: dict) -> str:
    """The row label of a group in Panel B, with its number of quarters."""
    if sample == "paper":
        return GROUP_LABELS["paper"]
    n = len(dec["phases"]) if sample == "all" else len(dec[sample]["quarters"])
    return f"{GROUP_LABELS[sample]} ({n})"


def write_table(payload: dict) -> str:
    """The LaTeX table the paper inputs, generated from the artifact: two panels.

    Panel A: the median hourly gap over the common span, by calendar year (2023
    is 11-31 December, 2026 January to September), over the paper window, and
    the share of hours with a positive gap. Panel B: Hyperliquid's hourly and the
    exchange's settlement floor shares, their difference and its Shapley split,
    with location's smallest and largest effect over the six orders, over all
    quarters, the phase groups, the below-band quarters of 2024Q3-2025Q4 and of
    2026, and the paper window, for :data:`SPLIT_ASSETS`. ETH's Panel B is
    :func:`write_group_table`; the split quarter by quarter, with the ranges of
    location and dispersion, is :func:`write_quarter_table`.
    """
    head = ["% Generated by regime_engine.premium_gap -- do not edit by hand.",
            "% Hourly premium gap and the split of the floor-share difference; see",
            "% output/premium_gap.json."]
    panels = [_gap_panel(payload), _split_panel(payload)]
    return "\n".join(head) + "\n" + "\n\\medskip\n\n".join(panels)


def _note_row(ncols: int, text: str) -> str:
    return f"\\multicolumn{{{ncols}}}{{l}}{{\\footnotesize {text}}} \\\\"


def _gap_panel(payload: dict) -> str:
    """Panel A: the median hourly gap by year and over the paper window."""
    tex = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llrrrrrrr}",
           "\\toprule",
           "\\multicolumn{9}{l}{\\textit{A. Hourly gap, Hyperliquid minus exchange: median "
           "(bps)}} \\\\",
           "Asset & Exchange & All & " + " & ".join(YEARS) + " & Estim. & Hours $>0$ \\\\",
           "\\midrule"]
    for coin in ASSETS:
        for venue in EXCHANGES:
            g = payload["gap"][coin][venue]
            years = [g["by_year"].get(y, {}).get("median_bps") for y in YEARS]
            paper = payload["windows"][coin][venue]["paper"]["gap"]
            tex.append(f"{coin} & {LABELS[venue]} & {_bps(g['common']['median_bps'])} & "
                       + " & ".join(_bps(y) for y in years)
                       + f" & {_bps(paper['median_bps'])} & "
                       f"{100 * g['common']['share_positive']:.0f}\\% \\\\")
    tex += ["\\bottomrule", "\\end{tabular*}", ""]
    return "\n".join(tex)


def _split_panel(payload: dict, assets=SPLIT_ASSETS, letter: str = "B") -> str:
    """Panel B: floor shares and the split of their difference, by group of quarters."""
    tex = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrrr}",
           "\\toprule",
           f"\\multicolumn{{8}}{{l}}{{\\textit{{{letter}. Share at the floor (\\%) and the "
           "split of the difference (percentage points)}} \\\\",
           "Quarters & HL & Exch. & Diff. & Location & [min, max] & Averaging & Dispersion \\\\",
           "\\midrule"]
    for k, (coin, venue) in enumerate((c, v) for c in assets for v in EXCHANGES):
        dec = payload["decomposition"][coin][venue]
        if k:
            tex.append("\\addlinespace")
        tex.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{coin}, Hyperliquid against "
                   f"{LABELS[venue]}}}}} \\\\")
        for sample in (g for g in GROUP_ROWS if g in dec):
            r = dec[sample]
            s, sh = r["shares"], r["shapley"]
            lo, hi = sh["range"]["location"]
            tex.append(f"{sample_label(sample, dec)} & {100 * s['a_hl_hourly']:.1f} & "
                       f"{100 * s['d_exchange_settlements']:.1f} & {_pp(r['difference'])} & "
                       f"{_pp(sh['location'])} & [{_pp(lo)}, {_pp(hi)}] & "
                       f"{_pp(sh['averaging'])} & {_pp(sh['dispersion'])} \\\\")
    tex += ["\\midrule",
            _note_row(8, "$\\tilde P_X$: the exchange's median hourly premium in a quarter "
                         "(bps)."),
            _note_row(8, "Shapley values; [min, max]: location's effect over the six orders."),
            "\\bottomrule", "\\end{tabular*}", ""]
    return "\n".join(tex)


ETH_GROUPS_STEM = "table12b_premium_gap_groups.tex"
QUARTERS_STEM = "table12c_premium_gap_quarters.tex"
DETAIL_ASSETS = tuple(c for c in ASSETS if c not in SPLIT_ASSETS)


def _range(lo: float, hi: float) -> str:
    return f"[{_pp(lo)}, {_pp(hi)}]"


def _quarter_range_panel(payload: dict) -> str:
    """Every quarter of every pair, with the ranges of location and dispersion."""
    tex = ["\\begin{tabular}{lrrrrrrrrr}",
           "\\toprule",
           "\\multicolumn{10}{l}{\\textit{By quarter: share at the floor (\\%) and the split "
           "of the difference (percentage points)}} \\\\",
           "Quarter & $\\tilde P_X$ & HL & Exch. & Diff. & Location & [min, max] & Averaging & "
           "Dispersion & [min, max] \\\\",
           "\\midrule"]
    for k, (coin, venue) in enumerate((c, v) for c in ASSETS for v in EXCHANGES):
        dec = payload["decomposition"][coin][venue]
        if k:
            tex.append("\\addlinespace")
        tex.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{coin}, Hyperliquid against "
                   f"{LABELS[venue]}}}}} \\\\")
        for q, r in dec["by_quarter"].items():
            sh = r["shapley"]
            tex.append(" & ".join([
                f"{q}{'$^{*}$' if q in PARTIAL_QUARTERS else ''}",
                _bps(dec["exchange_median_bps"][q]),
                f"{100 * r['shares']['a_hl_hourly']:.1f}",
                f"{100 * r['shares']['d_exchange_settlements']:.1f}",
                _pp(r["difference"]), _pp(sh["location"]), _range(*sh["range"]["location"]),
                _pp(sh["averaging"]), _pp(sh["dispersion"]),
                _range(*sh["range"]["dispersion"])]) + " \\\\")
    tex += ["\\midrule",
            _note_row(10, "$^{*}$ From 11 Dec 2023. $\\tilde P_X$: the exchange's median hourly "
                          "premium in the quarter (bps)."),
            _note_row(10, "Shapley values; [min, max]: the effect of the step to its left over "
                          "the six orders of the three steps."),
            "\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(tex)


def write_group_table(payload: dict) -> str:
    """Panel B of table12_premium_gap.tex for the assets it leaves out (DETAIL_ASSETS)."""
    head = ["% Generated by regime_engine.premium_gap -- do not edit by hand.",
            "% Panel B of table12_premium_gap.tex for "
            + ", ".join(DETAIL_ASSETS) + "; see output/premium_gap.json."]
    return "\n".join(head) + "\n" + _split_panel(payload, DETAIL_ASSETS, "B")


def write_quarter_table(payload: dict) -> str:
    """Every quarter of every asset and exchange, with the Shapley split and ranges.

    Each row gives the smallest and largest effect of location and of
    dispersion over the six orders of the three steps, so that a step called
    the largest can be checked against every order.
    """
    head = ["% Generated by regime_engine.premium_gap -- do not edit by hand.",
            "% The split of table12_premium_gap.tex by quarter for both assets and exchanges;",
            "% see output/premium_gap.json."]
    return "\n".join(head) + "\n" + _quarter_range_panel(payload)


# What the venues document about the premium that enters the common rule, as
# retrieved on 8 Oct 2026; rows and notes of table13_construction.tex. The facts of note a come
# from Hyperliquid's records in data/longhistory (longhistory.detect_switch),
# those of note c from the settlement fits of this module, not from documents.
CONSTRUCTION_STEM = "table13_construction.tex"
CONSTRUCTION = (
    ("Premium",
     "$[\\max(b-O,0)-{}$\\allowbreak$\\max(O-a,0)]/O$; $b$, $a$: average fill prices "
     "for the impact notional on the bid and ask; $O$: oracle",
     "same form against the price index",
     "same form against the index price"),
    ("Sampling", "every 5 s", "every 5 s", "every minute"),
    ("Averaging",
     "over the hour (equal weights implied)",
     "8 h, weights $1,\\dots,n$ rising to settlement, $n=5{,}760$; equal weights when "
     "settling hourly",
     "8 h, weights $1,\\dots,480$ rising to settlement; equal weights until 7 Mar 2024"
     "\\textsuperscript{c}"),
    ("Reference price",
     "oracle: stake-weighted median of the validators' weighted medians of spot mid prices "
     "on Binance, OKX, Bybit, Kraken, KuCoin, Gate, MEXC (weights 3, 2, 2, 1, 1, 1, 1), "
     "every 3 s\\textsuperscript{b}",
     "price index: weighted average of spot prices; BTCUSDT\\textsuperscript{d}: Binance "
     "0.435, OKX 0.130, Coinbase 0.130, Bitget, Bybit, KuCoin, MEXC 0.065 each, Gate 0.043",
     "index price: weighted average of the six largest spot pairs by 24-h volume; "
     "BTCUSDT\\textsuperscript{d}: Binance 0.339--0.348, Bybit 0.220--0.234, Gate "
     "0.215--0.219, OKX 0.138--0.147, KuCoin 0.070--0.071, Coinbase 0.001"),
    ("Impact notional", "20,000 USDC (BTC, ETH)",
     "200 USDT / initial margin rate at maximum leverage",
     "set per contract in USDT (value not in the pages used)"),
    ("Interest $r$", "0.01\\% per 8 h (0.00125\\% per hour)", "0.01\\% per 8 h",
     "0.01\\% per 8 h"),
    ("Clamp $c$", "$\\pm0.05\\%$ from the record of 11 Dec 2023 23:00 UTC\\textsuperscript{a}",
     "$\\pm0.05\\%$", "$\\pm0.05\\%$"),
    ("Cap", "4\\% per hour",
     "$\\pm0.75\\times$ maintenance margin rate; API: $\\pm0.300\\%$ (BTC, ETH)",
     "$\\pm0.75\\times(\\mathrm{IMR}-\\mathrm{MMR})$, at most $\\pm\\mathrm{MMR}$; "
     "API: $\\pm0.333\\%$ (BTC, ETH)"),
    ("Settlement", "every hour, $F/8$",
     "every 8 h (00, 08, 16 UTC); hourly while the rate is capped",
     "every 8 h; hourly while the rate is at its limit"),
    ("Margin, quote",
     "USDC margin; contract and oracle in USDT, whose profit and loss is paid in USDC "
     "without conversion (quanto)",
     "USDT margin and settlement; contract in USDT",
     "USDT margin and settlement; contract in USDT"),
    ("Changes since 2023",
     "8-hourly until 8 Jun 2023; premium only, 16 Jun--15 Jul 2023; "
     "$\\pm0.03\\%$ until 11 Dec 2023\\textsuperscript{a}; oracle wording, late "
     "2025\\textsuperscript{b}",
     "none seen in the settlements from Dec 2023 on",
     "from equal to rising weights, 7 Mar 2024\\textsuperscript{c}"),
)
CONSTRUCTION_NOTES = (
    "\\textsuperscript{a} Inferred from Hyperliquid's published hourly rates; no announcement "
    "found. Settlement was every 8 h until 8 June 2023 and hourly since. From 16 June 21:00 to "
    "15 July 02:00 UTC 2023 every rate equals the premium alone ($F=\\bar P$, paid as "
    "$\\bar P/8$ each hour: no interest, no clamp), and on 16 July a few records fit a "
    "$\\pm0.02\\%$ clamp instead; otherwise the "
    "rates fit $\\pm0.03\\%$ (band $-2$ to $+4$ bps) up to the record stamped 11 December "
    "2023 22:00 UTC and $\\pm0.05\\%$ from the one stamped 23:00.",
    "\\textsuperscript{b} The oracle page read ``spot prices'' in an Internet Archive "
    "snapshot of 6 October 2025 and ``spot mid prices'' from one of 10 December 2025 on; "
    "whether the method changed is not documented.",
    "\\textsuperscript{c} Found in the data: Bybit's settlements are reproduced by equal "
    "weights through the one of 7 March 2024 08:00 UTC and by rising weights from 16:00; not "
    "in the documentation retrieved.",
    "\\textsuperscript{d} Weights from the public APIs on 8 October 2026: Binance's at 16:29 "
    "and 16:52 UTC (identical), Bybit's from three snapshots between 16:29 and 18:29 UTC, "
    "printed as the range over the three. Bybit's weights vary over time: they follow "
    "24-hour volumes (``real-time weight'', updated hourly according to its help page) and "
    "moved between the snapshots; no snapshot need describe the sample period.",
    "Sources, retrieved 8 October 2026: Hyperliquid documentation (Funding, Oracle, Contract "
    "specifications); Binance FAQ ``Introduction to Binance Futures Funding Rates'' (updated "
    "6 March 2026); Bybit Help Center ``Introduction to Funding Rate'' (22 May 2026) and "
    "``Index Price Calculation'' (19 June 2026); Binance and Bybit public APIs.",
)
CONSTRUCTION_WIDTHS = (0.16, 0.28, 0.28, 0.28)       # shares of the line width


def write_construction_table() -> str:
    """table13_construction.tex: how each venue builds the premium of the common rule."""
    col = "{{>{{\\raggedright\\arraybackslash}}p{{\\dimexpr {w}\\linewidth-2\\tabcolsep\\relax}}}}"
    spec = "".join(col.format(w=w)[1:-1] for w in CONSTRUCTION_WIDTHS)
    full = "{>{\\raggedright\\arraybackslash}p{\\dimexpr\\linewidth-2\\tabcolsep\\relax}}"
    tex = ["% Generated by regime_engine.premium_gap -- do not edit by hand.",
           "% How the venues build the premium of the common rule, as documented on",
           "% 8 October 2026; see the notes for what the data add.",
           f"\\begin{{tabular}}{{{spec}}}",
           "\\toprule",
           " & Hyperliquid & Binance (USDT-M) & Bybit (linear) \\\\",
           "\\midrule"]
    for i, row in enumerate(CONSTRUCTION):
        if i:
            tex.append("\\addlinespace[2pt]")
        tex.append(" & ".join(row) + " \\\\")
    tex.append("\\midrule")
    for note in CONSTRUCTION_NOTES:
        tex.append(f"\\multicolumn{{4}}{full}{{\\footnotesize {note}}} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(tex)


# ----------------------------------------------------------------------
#  Figure
# ----------------------------------------------------------------------
PALETTE = {"hyperliquid": "#1B7F79", "binance": "#C98A0B", "bybit": "#5A4FCF"}  # as in fig4
# identity not by colour alone, and the same styles as fig10_long_history
LINESTYLE = {"hyperliquid": "-", "binance": "--", "bybit": ":"}
BAND_COLOR, INK, MUTED = "#E6E6E6", "#333333", "#555555"
FIG_SIZE_IN = (6.3, 5.1)
PRINT_WIDTH_CM = 15.8                 # the paper prints figures at \textwidth
MECHANISM_XLIM = {"location": (-10.0, 8.0), "positive": (-8.0, 22.0)}
MECHANISM_TITLE = {"location": "location", "positive": "positive premium"}


def monthly_rows(payload: dict, coin: str) -> dict:
    """Per exchange: month midpoints and the month's median and quartiles of the gap."""
    out = {}
    for venue in EXCHANGES:
        months = payload["monthly"][coin][venue]
        mid = [pd.Period(m, "M").start_time.tz_localize("UTC") + pd.Timedelta(days=14)
               for m in months]
        out[venue] = {"months": list(months), "mid": mid,
                      "median": [months[m]["median_bps"] for m in months],
                      "q25": [months[m]["q25_bps"] for m in months],
                      "q75": [months[m]["q75_bps"] for m in months]}
    return out


def mechanism_series(coin: str, quarter: str, data_dir=DATA) -> dict:
    """Each venue's hourly premium (bps) over one quarter, at Hyperliquid's resolution."""
    hl = load_hyperliquid(coin, data_dir)["premium"]
    hl = hl[(hl.index >= _utc(RULE_START)) & np.asarray(accrual_period(hl.index, "Q") == quarter)]
    out = {"hyperliquid": hl}
    for venue in EXCHANGES:
        frame = load_exchange_hourly(venue, ASSETS[coin], data_dir)
        out[venue] = frame["mean"].where(frame["n"] >= MIN_MINUTES).reindex(hl.index).dropna()
    return out


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4, "figure.dpi": 130, "savefig.dpi": 200,
        "savefig.bbox": "tight", "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def _summary_block(ax, rows, header=("median", "in band"), right=0.99, top=0.97,
                   step=0.1, gap=0.035) -> None:
    """A right-aligned table of (label, median, share) rows in the top corner of ``ax``.

    Column positions come from the rendered widths of the texts, so the columns
    never run into each other whatever the font.
    """
    renderer = ax.figure.canvas.get_renderer()
    axes_width = ax.get_window_extent(renderer).width

    def width(text):
        t = ax.text(0, 0, text, transform=ax.transAxes)
        w = t.get_window_extent(renderer).width / axes_width
        t.remove()
        return w

    columns = list(zip(*rows))
    w_med = max(width(t) for t in (header[0], *columns[1]))
    w_share = max(width(t) for t in (header[1], *columns[2]))
    w_label = max(width(t) for t in columns[0])
    x_share = right
    x_med = x_share - w_share - gap
    x_label = x_med - w_med - gap - w_label
    for x, text in ((x_med, header[0]), (x_share, header[1])):
        ax.text(x, top, text, transform=ax.transAxes, ha="right", va="top", color=MUTED)
    for i, row in enumerate(rows, start=1):
        for x, text, ha in zip((x_label, x_med, x_share), row, ("left", "right", "right")):
            ax.text(x, top - step * i, text, transform=ax.transAxes, ha=ha, va="top", color=INK)


def draw_figure(payload: dict, data_dir=DATA):
    """fig11_premium_gap: the gap month by month, and the three premiums in two quarters."""
    plt = _style()
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=FIG_SIZE_IN)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.55, wspace=0.22)
    paper = [_utc(t) for t in WINDOWS["paper"]]
    top = None
    for col, coin in enumerate(ASSETS):
        ax = fig.add_subplot(gs[0, col], sharey=top)
        top = top or ax
        ax.axvspan(*paper, color="#2E86AB", alpha=0.08, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        for venue, r in monthly_rows(payload, coin).items():
            ax.fill_between(r["mid"], r["q25"], r["q75"], color=PALETTE[venue], alpha=0.12,
                            lw=0)
            ax.plot(r["mid"], r["median"], color=PALETTE[venue], lw=1.3, ls=LINESTYLE[venue],
                    label=f"minus {LABELS[venue]}")
            # full months filled; the first month, from 11 December 2023, open
            ax.plot(r["mid"][1:], r["median"][1:], ls="none", marker="o", ms=2.4,
                    color=PALETTE[venue])
            ax.plot(r["mid"][:1], r["median"][:1], ls="none", marker="o", ms=3.0,
                    markerfacecolor="white", markeredgewidth=0.9, color=PALETTE[venue])
        ax.set_ylim(-3, 12)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.text(paper[0] + (paper[1] - paper[0]) / 2, 11.7, "estimation\nwindow", fontsize=7.5,
                color=MUTED, ha="center", va="top", linespacing=1.0)
        if col == 0:
            ax.set_ylabel("Hyperliquid minus exchange (bps)")
            ax.legend(frameon=False, loc="upper left", handlelength=2.2)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"({'ab'[col]}) {coin}: monthly median and IQR", loc="left",
                     fontweight="bold")
    for col, (name, quarter) in enumerate(MECHANISM_QUARTERS.items()):
        ax = fig.add_subplot(gs[1, col])
        lo, hi = MECHANISM_XLIM[name]
        bins = np.arange(lo, hi + 1e-9, 0.5)
        ax.axvspan(*BAND, color=BAND_COLOR, zorder=0)
        mech = payload["mechanism"]["BTC"][name]
        rows = []
        for venue, x in mechanism_series("BTC", quarter, data_dir).items():
            # density over all hours, so a curve's area is its share inside the axis range
            w = np.full(len(x), 1.0 / (len(x) * 0.5))
            if venue == "hyperliquid":
                ax.hist(x, bins=bins, weights=w, histtype="stepfilled", lw=0,
                        color=PALETTE[venue], alpha=0.18)
            ax.hist(x, bins=bins, weights=w, histtype="step", lw=1.3, color=PALETTE[venue],
                    ls=LINESTYLE[venue])
            m = mech[venue]
            rows.append((LABELS[venue], f"{m['median_bps']:+.1f}".replace("-", "\u2212"),
                         f"{100 * m['share_in_band']:.0f}%"))
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.75)
        _summary_block(ax, rows)
        ax.set_xlabel("Hourly premium (bps, 8-hour equivalent)")
        if col == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"({'cd'[col]}) BTC, {quarter} "
                     f"({MECHANISM_TITLE[name]})", loc="left", fontweight="bold")
    handles = [Line2D([], [], color=PALETTE[v], ls=LINESTYLE[v], lw=1.3, label=LABELS[v])
               for v in ("hyperliquid", *EXCHANGES)]
    handles.append(Patch(facecolor=BAND_COLOR, edgecolor=MUTED, lw=0.4,
                         label="band: funding at the floor"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.045), handlelength=2.2, columnspacing=1.6)
    return fig


def figure(payload: dict, data_dir=DATA, out_dir=DEFAULT_OUT / "figures") -> pathlib.Path:
    """Draw fig11_premium_gap and save it as PDF and PNG; return the PDF's path."""
    import matplotlib.pyplot as plt
    fig = draw_figure(payload, data_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{FIGURE_STEM}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{FIGURE_STEM}.png")
    plt.close(fig)
    return pdf


def smallest_font_pt(fig) -> float:
    """The smallest font size (pt) of any visible, non-empty text in ``fig``."""
    from matplotlib.text import Text
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    return float(min(sizes))


def print_scale(pdf) -> float:
    """How much the paper scales a saved PDF figure to print it at :data:`PRINT_WIDTH_CM`."""
    import re
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]",
                    pathlib.Path(pdf).read_bytes())
    return PRINT_WIDTH_CM / 2.54 * 72 / float(box.group(1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m regime_engine.premium_gap",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DATA), help="directory of the premium-gap files")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch and aggregate the data into --data first (about an hour)")
    ap.add_argument("--raw-dir", help="with --fetch: keep (and reuse) the one-minute klines here")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch_premium.fetch_all(out_dir=args.data, raw_dir=args.raw_dir)
    payload = build_payload(args.data)
    ARTIFACT.write_text(json.dumps(payload, indent=1) + "\n")
    tables = {OUTPUT_TABLE.name: write_table(payload),
              ETH_GROUPS_STEM: write_group_table(payload),
              QUARTERS_STEM: write_quarter_table(payload),
              CONSTRUCTION_STEM: write_construction_table()}
    for name, text in tables.items():
        for folder in (OUTPUT_TABLE.parent, PAPER_TABLE.parent):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(text)
    pdf = figure(payload, args.data)
    shutil.copyfile(pdf, REPO_ROOT / "paper" / "figures" / pdf.name)
    for coin in ASSETS:
        for venue in EXCHANGES:
            g = payload["gap"][coin][venue]["common"]
            d = payload["decomposition"][coin][venue]["all"]
            print(f"{coin} {LABELS[venue]:<8} gap median {g['median_bps']:+.2f} bps "
                  f"({100 * g['share_positive']:.0f}% of hours > 0); floor "
                  f"{100 * d['shares']['a_hl_hourly']:.1f}% vs "
                  f"{100 * d['shares']['d_exchange_settlements']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
