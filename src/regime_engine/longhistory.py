"""Long history of funding-rate censoring on Hyperliquid, Binance and Bybit, BTC and ETH.

The paper compares the three venues over October 2025 - September 2026. This
module extends the comparison back to the first Hyperliquid funding record and
to a second asset, ETH. It has three parts.

Data. ``fetch_all`` pages through Hyperliquid's ``fundingHistory`` and the
Binance and Bybit funding endpoints and writes one CSV per venue and asset to
``data/longhistory/``, every rate kept as the decimal string the venue prints.
Hyperliquid's records are stamped at settlement and floored to the hour (the
record stamped t carries the premium averaged over [t - 1h, t)); the exchanges'
stamps are rounded to the minute, as in ``fetch_cex``. Hyperliquid's pages are
paced to its per-IP weight budget.

Rule. Hyperliquid did not always apply the exchanges' rule. Each of its
records is classified by the distance between the eight-hour-equivalent rate
and the premium: |8F - P| = 0.05% is the exchanges' clamp, 0.03% the narrower
one Hyperliquid used before, and 0 an unclamped premium. ``detect_switch``
finds the first record of the common rule from the data, and ``rule_history``
lists every era before it.

Comparison. From that record onward the floor shares, the shares below and
above the band, and the median premium are computed for the whole period,
each calendar year, half-year, quarter and month, and the paper's windows. A
settlement belongs to the period in which its averaging interval ends, so a
stamp at 00:00 on 1 January belongs to December. Hyperliquid's mean premium
over each 8-hour block is ordered against the exchange's, with bounds that
need no assumption about the blocks the band leaves undecided, and the two
premium distributions are compared quantile by quantile.

    python3 -m regime_engine.longhistory            # offline, from the CSVs
    python3 -m regime_engine.longhistory --fetch    # refetch, then rebuild
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd

from . import censoring as cz
from .fetch_cex import BINANCE, BYBIT, _dedup, _iso, _ms, _span, get_json
from .paths import DEFAULT_OUT, REPO_ROOT

HL_INFO = "https://api.hyperliquid.xyz/info"
HL_PAGE = 500
ASSETS = ("BTC", "ETH")
VENUES = ("hyperliquid", "binance", "bybit")
EXCHANGES = cz.EXCHANGES
END = "2026-10-01 00:00"
# Earliest stamps requested; each endpoint returns its history from its first record.
FIRST = {"hyperliquid": "2023-05-01 00:00", "binance": "2019-01-01 00:00",
         "bybit": "2019-01-01 00:00"}
PAUSE = 0.3                 # between Binance and Bybit pages
# Hyperliquid weighs fundingHistory at 20 plus 1 per 20 records returned, about 45 per
# 500-record page, against 1,200 per minute and IP: at most about 26 pages a minute.
HL_PAUSE = 2.5
RATE_LIMIT_WAIT = 30.0      # after HTTP 429 without a Retry-After header
MAX_LATENESS = pd.Timedelta(minutes=30)

DATA = REPO_ROOT / "data" / "longhistory"
ARTIFACT = DEFAULT_OUT / "censoring_long_history.json"
OUTPUT_TABLE = DEFAULT_OUT / "tables" / "table11_long_history.tex"
PAPER_TABLE = REPO_ROOT / "paper" / "tables" / "table11_long_history.tex"
FIGURE_STEM = "fig10_long_history"

FLOOR_8H, FLOOR_HOURLY, CLAMP = cz.FLOOR_8H, cz.FLOOR_HOURLY, cz.CLAMP
CLAMP_OLD = Decimal("0.0003")          # Hyperliquid's clamp before the common rule
# Rates carry ten decimals, so 8F - P misses the clamp by at most a few 1e-10.
TOLERANCE = Decimal("1e-8")
SETTLEMENT_SPACING_8H = pd.Timedelta(hours=4)   # a gap this long marks 8-hourly settlement
PAPER_START = pd.Timestamp(cz.WINDOWS["paper"][0], tz="UTC")


def symbol(venue: str, asset: str) -> str:
    return asset if venue == "hyperliquid" else f"{asset}USDT"


def file_name(venue: str, asset: str) -> str:
    return f"{venue}_{symbol(venue, asset)}_funding.csv"


# ----------------------------------------------------------------------
#  Fetching
# ----------------------------------------------------------------------
def _retry_after(exc) -> float:
    try:
        return float(exc.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return RATE_LIMIT_WAIT


def post_json(payload: dict, retries: int = 6):
    """POST ``payload`` to Hyperliquid's info endpoint, retrying with back-off.

    HTTP 429 (the per-IP weight budget is spent) waits for the Retry-After
    header, or ``RATE_LIMIT_WAIT`` seconds; other errors back off exponentially.
    """
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(HL_INFO, data=body, headers={
                "Content-Type": "application/json", "User-Agent": "regime-engine/1.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            if attempt == retries - 1:
                raise
            limited = isinstance(exc, urllib.error.HTTPError) and exc.code == 429
            time.sleep(_retry_after(exc) if limited else 2 ** attempt)


def _hour(ms) -> str:
    """The settlement hour of a record: its stamp floored to the hour.

    Records arrive a few milliseconds, at times minutes, after their hour. A
    stamp half an hour or more past the hour could belong to the next one (or be
    an hour late) and is refused rather than assigned silently.
    """
    t = pd.Timestamp(int(ms), unit="ms", tz="UTC")
    hour = t.floor("h")
    if t - hour >= MAX_LATENESS:
        raise ValueError(f"record stamped {t} lies {t - hour} past the hour")
    return str(hour)


def _first_per_stamp(rows: list, start, end) -> list:
    """The first record per stamp inside [start, end], in time order."""
    lo, hi = _iso(_ms(start)), _iso(_ms(end))
    by_time = {}
    for r in rows:
        if lo <= r["time"] <= hi:
            by_time.setdefault(r["time"], r)
    return [by_time[t] for t in sorted(by_time)]


def hyperliquid_funding(coin: str, start, end, post=post_json,
                        pause: float = HL_PAUSE) -> list:
    """Hyperliquid's funding records for ``coin``: stamp, rate and premium as strings.

    Where two records fall in the same hour the first is kept and a warning
    names the count.
    """
    rows, (cursor, stop) = [], _span(start, end)
    while cursor <= stop:
        page = post({"type": "fundingHistory", "coin": coin, "startTime": cursor,
                     "endTime": stop})
        if not page:
            break
        page = sorted(page, key=lambda x: int(x["time"]))
        rows += [{"time": _hour(x["time"]), "fundingRate": x["fundingRate"],
                  "premium": x["premium"]} for x in page]
        if len(page) < HL_PAGE:
            break
        cursor = int(page[-1]["time"]) + 1
        time.sleep(pause)
    lo, hi = _iso(_ms(start)), _iso(_ms(end))
    out = _first_per_stamp(rows, start, end)
    dup = sum(lo <= r["time"] <= hi for r in rows) - len(out)
    if dup:
        warnings.warn(f"{coin}: {dup} record{'s' if dup > 1 else ''} share"
                      f"{'' if dup > 1 else 's'} an hour with an earlier one; "
                      "the first is kept", stacklevel=2)
    return out


def binance_funding(sym: str, start, end, get=get_json, pause: float = PAUSE) -> list:
    rows, (cursor, stop) = [], _span(start, end)
    while cursor <= stop:
        page = get(f"{BINANCE}/fundingRate?symbol={sym}"
                   f"&startTime={cursor}&endTime={stop}&limit=1000")
        if not page:
            break
        rows += [{"time": _iso(x["fundingTime"]), "fundingRate": x["fundingRate"],
                  "markPrice": x.get("markPrice", "")} for x in page]
        if len(page) < 1000:
            break
        cursor = int(page[-1]["fundingTime"]) + 1
        time.sleep(pause)
    return _dedup(rows, start, end)


def bybit_funding(sym: str, start, end, get=get_json, pause: float = PAUSE) -> list:
    rows, (first, cursor) = [], _span(start, end)
    while cursor >= first:
        page = get(f"{BYBIT}/funding/history?category=linear&symbol={sym}"
                   f"&startTime={first}&endTime={cursor}&limit=200")["result"]["list"]
        if not page:
            break
        rows += [{"time": _iso(x["fundingRateTimestamp"]), "fundingRate": x["fundingRate"]}
                 for x in page]
        if len(page) < 200:
            break
        cursor = min(int(x["fundingRateTimestamp"]) for x in page) - 1
        time.sleep(pause)
    return _dedup(rows, start, end)


def write_csv(rows: list, path) -> None:
    with pathlib.Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fetch_all(out_dir=DATA, end=END, get=get_json, post=post_json,
              pause: float = PAUSE, hl_pause: float = HL_PAUSE) -> dict:
    """Fetch the six series and write them to ``out_dir``; return the row counts."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for asset in ASSETS:
        for venue in VENUES:
            sym, start = symbol(venue, asset), FIRST[venue]
            if venue == "hyperliquid":
                rows = hyperliquid_funding(sym, start, end, post=post, pause=hl_pause)
            elif venue == "binance":
                rows = binance_funding(sym, start, end, get=get, pause=pause)
            else:
                rows = bybit_funding(sym, start, end, get=get, pause=pause)
            if not rows:
                raise RuntimeError(f"{venue} {sym}: the endpoint returned no records")
            write_csv(rows, out / file_name(venue, asset))
            counts[file_name(venue, asset)] = len(rows)
    return counts


# ----------------------------------------------------------------------
#  Loading and the clamp rule
# ----------------------------------------------------------------------
def load(venue: str, asset: str, data_dir=DATA) -> pd.DataFrame:
    """One shipped series: UTC stamps, rates (and Hyperliquid's premium) as strings."""
    cols = {"fundingRate": str, "premium": str, "markPrice": str}
    df = pd.read_csv(pathlib.Path(data_dir) / file_name(venue, asset), dtype=cols)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def clamp_regimes(hl: pd.DataFrame) -> pd.DataFrame:
    """Classify each Hyperliquid record by the rule its rate reveals.

    A record that follows the previous one by four hours or more settled
    eight-hourly (rate per 8 h, floor 0.01%); otherwise hourly (rate per hour,
    floor 0.00125%). With s the number of hours a rate covers (8 / 1) the
    eight-hour-equivalent rate is (8/s)·F, and the label is
      ``floor``      F equals the floor of its schedule;
      ``clamp_5bp``  |(8/s)·F - P| = 0.05%, the exchanges' clamp;
      ``clamp_3bp``  |(8/s)·F - P| = 0.03%;
      ``unclamped``  (8/s)·F = P;
      ``other``      anything else.
    """
    t = hl["time"]
    gap_before = t.diff()
    gap = gap_before.fillna(t.diff(-1).abs())
    eight_hourly = (gap >= SETTLEMENT_SPACING_8H).to_numpy()
    labels, devs = [], []
    for rate, prem, eh in zip(hl["fundingRate"], hl["premium"], eight_hourly):
        f, p = Decimal(str(rate).strip()), Decimal(str(prem).strip())
        floor, scale = (FLOOR_8H, 1) if eh else (FLOOR_HOURLY, 8)
        dev = scale * f - p
        devs.append(float(dev))
        if f == floor:
            labels.append("floor")
        elif abs(abs(dev) - CLAMP) <= TOLERANCE:
            labels.append("clamp_5bp")
        elif abs(abs(dev) - CLAMP_OLD) <= TOLERANCE:
            labels.append("clamp_3bp")
        elif abs(dev) <= TOLERANCE:
            labels.append("unclamped")
        else:
            labels.append("other")
    return pd.DataFrame({"time": t, "schedule": np.where(eight_hourly, "8h", "1h"),
                         "regime": labels, "deviation": devs})


def _stamp(ts) -> str:
    return None if pd.isna(ts) else f"{ts:%Y-%m-%d %H:%M}"


RULES = ("clamp_3bp", "clamp_5bp", "unclamped")
RULE_CLAMP = {"clamp_3bp": CLAMP_OLD, "clamp_5bp": CLAMP, "unclamped": None}


def _decimal(series: pd.Series) -> pd.Series:
    return series.map(lambda s: Decimal(str(s).strip()))


def _whole_bps_clamp(dev: float):
    """The clamp in whole bps that |8F - P| equals exactly, or None."""
    c = round(abs(dev) * 1e4)
    return int(c) if c >= 1 and abs(abs(dev) - c * 1e-4) <= float(TOLERANCE) else None


def rule_history(hl: pd.DataFrame) -> list:
    """Hyperliquid's funding rule, era by era, as its records reveal it.

    An era is a run of records whose schedule (8h or 1h) and revealing label
    (``clamp_3bp``, ``clamp_5bp`` or ``unclamped``) stay the same. Records that
    reveal no rule (``floor``, ``other``) between two eras go to the one that
    can produce them: the new schedule where the schedule changes, the clamped
    era where the other prints no floor, and otherwise the earlier era, since
    an era starts at its first revealing record (as the switch does); their
    count is then ``undetermined_after``. Each era gives its span, counts, the
    records that fit no rule (8F - P in bps) and, for a clamp c, the records
    at the floor with the premium outside [floor - c, floor + c] and those
    inside it but off the floor.
    """
    reg = clamp_regimes(hl).reset_index(drop=True)
    prem = _decimal(hl["premium"]).reset_index(drop=True)
    reveal = reg.index[reg["regime"].isin(RULES)]
    key = (reg["schedule"] + " " + reg["regime"])[reveal]
    run = (key != key.shift()).cumsum()
    runs = [(k.iloc[0], k.index[0], k.index[-1]) for _, k in key.groupby(run)]

    cuts, undetermined = [0], []
    for (k0, _, end0), (k1, start1, _) in zip(runs, runs[1:]):
        (s0, r0), (s1, r1) = k0.split(), k1.split()
        gap = reg.loc[end0 + 1:start1 - 1]
        if s0 != s1:
            new = gap.index[gap["schedule"] == s1]
            cut, n_open = (new[0] if len(new) else start1), 0
        elif r0 == "unclamped":
            cut, n_open = end0 + 1, 0
        else:
            cut, n_open = start1, (0 if r1 == "unclamped" else len(gap))
        cuts.append(cut)
        undetermined.append(n_open)
    cuts.append(len(reg))
    undetermined.append(0)

    out = []
    for (k, _, _), lo, hi, n_open in zip(runs, cuts[:-1], cuts[1:], undetermined):
        schedule, rule = k.split()
        seg, p = reg.iloc[lo:hi], prem.iloc[lo:hi]
        at_floor = seg["regime"] == "floor"
        other = seg["regime"] == "other"
        c = RULE_CLAMP[rule]
        if c is None:
            band_counts = (int(at_floor.sum()), None)
        else:
            in_band = (p >= FLOOR_8H - c) & (p <= FLOOR_8H + c)
            band_counts = (int((at_floor & ~in_band).sum()), int((~at_floor & in_band).sum()))
        out.append({
            "schedule": schedule, "rule": rule,
            "from": _stamp(seg["time"].iloc[0]), "to": _stamp(seg["time"].iloc[-1]),
            "n_records": int(len(seg)), "n_floor": int(at_floor.sum()),
            "n_other": int(other.sum()),
            "floor_not_in_band": band_counts[0], "in_band_not_floor": band_counts[1],
            "undetermined_after": n_open,
            "fits_no_rule": [{"time": _stamp(t), "premium_bps": round(float(q) * 1e4, 4),
                              "deviation_bps": round(d * 1e4, 4),
                              "fits_clamp_bps": _whole_bps_clamp(d)}
                             for t, q, d in zip(seg.loc[other, "time"], p[other],
                                                seg.loc[other, "deviation"])],
        })
    return out


def detect_switch(hl: pd.DataFrame) -> dict:
    """The record from which Hyperliquid applies the 0.05% clamp, and the eras before it.

    The switch is the first record classified ``clamp_5bp``; it is accepted only
    if no ``clamp_3bp`` or ``unclamped`` record follows it. From the switch on,
    a record must sit at the floor exactly when its premium lies inside the
    closed band [-4, +6] bps (the clamp saturates exactly at its edges); the
    count of records that break this is returned. Before the switch the same
    check runs against [-2, +4] bps, the band of the 0.03% clamp, outside the
    unclamped span. ``history`` lists every era (``rule_history``).
    """
    reg = clamp_regimes(hl)
    first5 = reg.loc[reg["regime"] == "clamp_5bp", "time"].min()
    after = reg["time"] >= first5
    late_old = int(reg.loc[after, "regime"].isin(["clamp_3bp", "unclamped"]).sum())
    if late_old:
        raise ValueError(f"{late_old} records after {first5} follow an older rule")
    hourly_start = reg.loc[(reg["schedule"] == "1h") & (reg["time"] > reg["time"].iloc[0]),
                           "time"].min()
    era = {}
    for name in ("clamp_3bp", "unclamped", "clamp_5bp", "other"):
        x = reg.loc[reg["regime"] == name, "time"]
        era[name] = {"n": int(len(x)), "first": _stamp(x.min()), "last": _stamp(x.max())}

    prem = _decimal(hl["premium"])
    lo, hi = FLOOR_8H - CLAMP, FLOOR_8H + CLAMP
    in_band = (prem >= lo) & (prem <= hi)
    at_floor = reg["regime"] == "floor"
    common = after & (reg["schedule"] == "1h")
    lo3, hi3 = FLOOR_8H - CLAMP_OLD, FLOOR_8H + CLAMP_OLD
    in_narrow = (prem >= lo3) & (prem <= hi3)
    unclamped = reg.loc[reg["regime"] == "unclamped", "time"]
    narrow = (reg["schedule"] == "1h") & (reg["time"] < first5)
    if len(unclamped):
        narrow &= ~reg["time"].between(unclamped.min(), unclamped.max())
    return {
        "switch": _stamp(first5),
        "last_before_switch": _stamp(reg.loc[reg["time"] < first5, "time"].max()),
        "hourly_from": _stamp(hourly_start),
        "eight_hourly_records": int((reg["schedule"] == "8h").sum()),
        "regimes": era,
        "common_rule_hours": int(common.sum()),
        "common_rule_floor_hours": int((common & at_floor).sum()),
        "common_rule_floor_not_in_band": int((common & at_floor & ~in_band).sum()),
        "common_rule_in_band_not_floor": int((common & ~at_floor & in_band).sum()),
        "common_rule_non_floor_not_5bp": int((common & ~at_floor &
                                              (reg["regime"] != "clamp_5bp")).sum()),
        "clamp_3bp_era_hours": int(narrow.sum()),
        "clamp_3bp_era_floor_not_in_narrow_band": int((narrow & at_floor & ~in_narrow).sum()),
        "clamp_3bp_era_in_narrow_band_not_floor": int((narrow & ~at_floor & in_narrow).sum()),
        "history": rule_history(hl),
    }


# ----------------------------------------------------------------------
#  Periods and summaries
# ----------------------------------------------------------------------
def period_of(times: pd.Series, freq: str) -> pd.Series:
    """Calendar period of each settlement: 'Y', 'H' (half-year), 'Q' or 'M'.

    A settlement stamped t averages the interval ending at t, so it belongs to
    the period containing t - 1 s: 1 Jan 00:00 belongs to December.
    """
    t = (times - pd.Timedelta(seconds=1)).dt.tz_localize(None)
    if freq == "H":
        return t.dt.year.astype(str) + "H" + np.where(t.dt.month <= 6, "1", "2")
    return t.dt.to_period(freq).astype(str)


def premium_quartiles(bps) -> dict:
    """Quartiles of a premium series in bps; NaN values are censored inside the band."""
    return {f"{k}_bps": cz.censored_quantile(bps, q)
            for k, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75))}


def summarise(rates, floor: Decimal, premium_bps) -> dict:
    """Floor, below- and above-band counts and shares, and premium quartiles."""
    out = cz.classify(list(rates), floor)
    return {**out, **premium_quartiles(premium_bps)}


def hl_block_means(hl: pd.DataFrame) -> pd.Series:
    """Mean of Hyperliquid's hourly premium over each eight-hour settlement block, in bps.

    The block that an exchange settles at t averages [t - 8h, t), which
    Hyperliquid's records stamped t - 7h, ..., t cover. Only full blocks of
    eight records are kept; the result is indexed by the settlement stamp.
    """
    s = hl.set_index("time")["premium"].astype(float) * 1e4
    grouped = s.resample("8h", origin="epoch", label="right", closed="right")
    means, counts = grouped.mean(), grouped.count()
    return means[counts == 8]


def block_comparison(hl_means: pd.Series, exchange: pd.DataFrame) -> pd.DataFrame:
    """Order Hyperliquid's 8-hour mean premium against an exchange's, block by block.

    An exchange settlement off the floor identifies its premium; one at the
    floor says only that the premium lay in [-4, +6] bps. Hyperliquid's mean is
    above (below) the exchange's when that is decided by the exact values or
    by the band, and undecided otherwise.
    """
    ex = exchange.set_index("time")
    rec = cz.recovered_premium_bps(ex["fundingRate"]).to_numpy()
    df = pd.DataFrame({"exchange_bps": rec}, index=ex.index)
    df["hl_bps"] = hl_means.reindex(df.index)
    df = df.dropna(subset=["hl_bps"])
    lo, hi = cz.BAND_BPS
    known = df["exchange_bps"].notna()
    diff = df["hl_bps"] - df["exchange_bps"]
    df["diff_bps"] = diff
    df["order"] = np.select(
        [known & (diff > 0), known & (diff < 0), ~known & (df["hl_bps"] > hi),
         ~known & (df["hl_bps"] < lo)],
        ["above", "below", "above", "below"], default="undecided")
    return df.rename_axis("time").reset_index()


def comparison_summary(cmp: pd.DataFrame) -> dict:
    """Counts of the block ordering and the share of blocks with Hyperliquid higher.

    The share over decided blocks assumes the undecided ones split alike, which
    fails when the two premia differ in spread. The bounds need no assumption:
    the lower one counts every undecided block against Hyperliquid being
    higher, the upper one for it. The median difference uses only blocks whose
    exchange premium is identified, mostly below the band, and is conditional
    on that.
    """
    n = len(cmp)
    above = int((cmp["order"] == "above").sum())
    below = int((cmp["order"] == "below").sum())
    decided = above + below
    diff = cmp["diff_bps"].dropna()
    return {"n_blocks": n, "hl_above": above, "hl_below": below,
            "undecided": n - decided,
            "share_hl_above_lower": above / n if n else None,
            "share_hl_above_upper": (n - below) / n if n else None,
            "share_hl_above_of_decided": above / decided if decided else None,
            "identified": int(len(diff)),
            "median_diff_given_exchange_identified_bps":
                float(diff.median()) if len(diff) else None}


QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)


def _qkey(p: float) -> str:
    return f"q{round(100 * p):02d}"


def quantile_gaps(cmp: pd.DataFrame, probs=QUANTILES) -> dict:
    """Quantiles of Hyperliquid's 8-hour block means and of the exchange's premium over
    the same blocks, and their difference (Hyperliquid minus exchange), in bps.

    The exchange's premium is censored inside the band, so its quantile and the
    difference are None where the quantile falls on a censored value. A constant
    offset between the venues would make the difference the same at every
    quantile.
    """
    hl = cmp["hl_bps"].to_numpy(dtype=float)
    out = {"n_blocks": int(len(cmp))}
    for p in probs:
        h = float(np.quantile(hl, p)) if len(hl) else None
        e = cz.censored_quantile(cmp["exchange_bps"], p) if len(hl) else None
        out[_qkey(p)] = {"hyperliquid_bps": h, "exchange_bps": e,
                         "gap_bps": None if h is None or e is None else h - e}
    return out


def shift_test(cmp: pd.DataFrame) -> dict:
    """Would a constant offset explain the floor shares? Move Hyperliquid's block means
    down by the median gap and count those in the closed band, against the share of
    blocks at which the exchange printed the floor.

    The median gap is the difference of the two medians (Hyperliquid's block
    means minus the exchange's censored median over the same blocks). The
    ``_by_block_diff`` variant shifts by the median of the per-block
    differences instead, over the blocks whose exchange premium is identified."""
    gap = quantile_gaps(cmp, (0.5,))["q50"]["gap_bps"]
    hl = cmp["hl_bps"].to_numpy(dtype=float)
    lo, hi = cz.BAND_BPS

    def in_band(x):
        return float(((x >= lo) & (x <= hi)).mean())

    diffs = (cmp["hl_bps"] - cmp["exchange_bps"]).dropna()
    block = float(diffs.median()) if len(diffs) else None
    return {"median_gap_bps": gap,
            "exchange_at_floor": float(cmp["exchange_bps"].isna().mean()),
            "hyperliquid_in_band": in_band(hl),
            "hyperliquid_shifted_in_band": None if gap is None else in_band(hl - gap),
            "median_block_diff_bps": block,
            "hyperliquid_shifted_by_block_diff_in_band":
                None if block is None else in_band(hl - block)}


class History:
    """The six series of one data directory, restricted to the analysis span."""

    def __init__(self, data_dir=DATA, end=END):
        self.end = pd.Timestamp(end, tz="UTC")
        self.raw = {(v, a): load(v, a, data_dir) for a in ASSETS for v in VENUES}
        self.rule = {a: detect_switch(self.raw[("hyperliquid", a)]) for a in ASSETS}
        switches = {self.rule[a]["switch"] for a in ASSETS}
        if len(switches) != 1:
            raise ValueError(f"the assets switch at different stamps: {switches}")
        self.switch = pd.Timestamp(switches.pop(), tz="UTC")
        self.hourly_from = pd.Timestamp(self.rule[ASSETS[0]]["hourly_from"], tz="UTC")

    def frames(self, asset: str, start, end, left_closed: bool = True) -> dict:
        """The four rows of the comparison for stamps in [start, end] (or (start, end])."""
        def cut(df):
            lo = df["time"] >= start if left_closed else df["time"] > start
            return df[lo & (df["time"] <= end)]
        hl = cut(self.raw[("hyperliquid", asset)])
        out = {"hyperliquid": hl,
               "hyperliquid_at_8h_stamps": hl[cz.at_settlement_stamps(hl["time"])]}
        for venue in EXCHANGES:
            out[venue] = cut(self.raw[(venue, asset)])
        return out

    @staticmethod
    def summary(frames: dict) -> dict:
        out = {}
        for name, df in frames.items():
            if name.startswith("hyperliquid"):
                prem = df["premium"].astype(float) * 1e4
                out[name] = summarise(df["fundingRate"], FLOOR_HOURLY, prem.to_numpy())
            else:
                prem = cz.recovered_premium_bps(df["fundingRate"])
                out[name] = summarise(df["fundingRate"], FLOOR_8H, prem)
        return out

    def common(self, asset: str) -> dict:
        return self.frames(asset, self.switch, self.end)

    def by_period(self, asset: str, freq: str) -> dict:
        frames = self.common(asset)
        keys = {name: period_of(df["time"], freq) for name, df in frames.items()}
        periods = sorted(set(keys["hyperliquid"]))
        out = {}
        for p in periods:
            out[p] = self.summary({n: df[keys[n] == p] for n, df in frames.items()})
            out[p]["hl_vs_exchanges"] = rank_against_exchanges(out[p])
        return out

    def windows(self, asset: str) -> dict:
        paper_lo, paper_hi = (pd.Timestamp(t, tz="UTC") for t in cz.WINDOWS["paper"])
        spans = {
            "common_rule": (self.switch, self.end, True),
            "before_paper": (self.switch, PAPER_START - pd.Timedelta(minutes=1), True),
            "paper": (paper_lo, paper_hi, True),
            "later": (paper_hi, self.end, False),
            "pre_switch_hourly": (self.hourly_from, self.switch - pd.Timedelta(minutes=1),
                                  True),
        }
        return {name: self.summary(self.frames(asset, lo, hi, closed))
                for name, (lo, hi, closed) in spans.items()}

    def blocks(self, asset: str) -> dict:
        """Block-by-block comparison against each exchange, common-rule period."""
        frames = self.common(asset)
        means = hl_block_means(frames["hyperliquid"])
        return {venue: block_comparison(means, frames[venue]) for venue in EXCHANGES}


def rank_against_exchanges(row: dict, key: str = "hyperliquid") -> str:
    """'above_both', 'below_both' or 'between': Hyperliquid's floor share against the exchanges'."""
    hl = row[key]["share_at_floor"]
    ex = [row[v]["share_at_floor"] for v in EXCHANGES]
    if hl > max(ex):
        return "above_both"
    if hl < min(ex):
        return "below_both"
    return "between"


def block_median_gaps(blocks: dict, freq: str = "Q") -> dict:
    """Per period and exchange: the median of Hyperliquid's 8-hour block means minus the
    median of the exchange's premium over the same blocks, in bps; None where the
    exchange's median is censored inside the band."""
    out = {}
    for venue, cmp in blocks.items():
        per = period_of(cmp["time"], freq)
        for k in sorted(set(per)):
            out.setdefault(k, {})[venue] = quantile_gaps(cmp[per == k], (0.5,))["q50"]["gap_bps"]
    return out


def _period_bounds(ts, freq: str) -> tuple:
    """(label, start, end) of the period of ``freq`` containing ``ts`` (UTC).

    'H' is the half-year, labelled as :func:`period_of` labels it ('2024H1');
    the other frequencies are pandas periods.
    """
    if freq == "H":
        half = 1 if ts.month <= 6 else 2
        lo = pd.Timestamp(year=ts.year, month=6 * half - 5, day=1, tz="UTC")
        hi = (pd.Timestamp(year=ts.year + 1, month=1, day=1, tz="UTC") if half == 2
              else pd.Timestamp(year=ts.year, month=7, day=1, tz="UTC"))
        return f"{ts.year}H{half}", lo, hi
    p = pd.Period(ts.tz_localize(None), freq)
    return str(p), p.start_time.tz_localize("UTC"), (p + 1).start_time.tz_localize("UTC")


def partial_periods(start, end, freq: str) -> list:
    """The periods at either end of [start, end] that the span covers only in part.

    Hourly settlements are stamped at the end of their hour, so a period is
    covered in full from the stamp one hour after its start to its end."""
    first, first_lo, _ = _period_bounds(start - pd.Timedelta(seconds=1), freq)
    last, _, last_hi = _period_bounds(end - pd.Timedelta(seconds=1), freq)
    out = []
    if start > first_lo + pd.Timedelta(hours=1):
        out.append(first)
    if end < last_hi:
        out.append(last)
    return out


def _gap_range(gaps: dict, skip=()) -> dict:
    rows = [row for k, row in gaps.items() if k not in skip]
    vals = [g for row in rows for g in row.values() if g is not None]
    return {"min_bps": min(vals), "max_bps": max(vals), "n_identified": len(vals),
            "n_cells": sum(len(row) for row in rows), "all_positive": all(g > 0 for g in vals),
            "excludes": list(skip)}


# ----------------------------------------------------------------------
#  The artifact
# ----------------------------------------------------------------------
def _block_windows(cmp: pd.DataFrame) -> dict:
    return {"common_rule": cmp, "paper": cmp[cz.in_window(cmp["time"], "paper")],
            "later": cmp[cz.in_window(cmp["time"], "later")]}


def _lowest(shares: dict) -> dict | None:
    if not shares:
        return None
    m = min(shares, key=lambda k: shares[k])
    return {"month": m, "share": shares[m]}


def month_majority(months: dict) -> dict:
    """In how many months Hyperliquid is higher in most blocks: of the decided ones, and
    under the lower bound (in more than half of all blocks)."""
    dec = {m: r["share_hl_above_of_decided"] for m, r in months.items()}
    low = {m: r["share_hl_above_lower"] for m, r in months.items()}
    return {
        "n_months": len(months),
        "hl_above_in_most_decided_blocks": sum(x is not None and x > 0.5 for x in dec.values()),
        "months_not": sorted(m for m, x in dec.items() if x is None or x <= 0.5),
        "hl_above_in_most_blocks_lower_bound": sum(x > 0.5 for x in low.values()),
        "months_not_lower_bound": sorted(m for m, x in low.items() if x <= 0.5),
        "lowest": _lowest({m: x for m, x in dec.items() if x is not None}),
        "lowest_lower_bound": {"month": min(low, key=lambda m: low[m]),
                               "share": min(low.values())},
    }


def build_payload(data_dir=DATA) -> dict:
    h = History(data_dir)
    partial = {f: partial_periods(h.switch, h.end, f) for f in ("Y", "H", "Q", "M")}
    assets = {}
    for asset in ASSETS:
        months = h.by_period(asset, "M")
        full = {m: r for m, r in months.items() if m not in partial["M"]}
        ranks = {m: r["hl_vs_exchanges"] for m, r in full.items()}
        ranks8 = {m: rank_against_exchanges(r, "hyperliquid_at_8h_stamps")
                  for m, r in full.items()}
        not_above = [m for m, r in ranks.items() if r != "above_both"]
        below = [{"month": m,
                  "hyperliquid": r["hyperliquid"]["share_at_floor"],
                  "hyperliquid_at_8h_stamps": r["hyperliquid_at_8h_stamps"]["share_at_floor"],
                  "binance": r["binance"]["share_at_floor"],
                  "bybit": r["bybit"]["share_at_floor"],
                  "hyperliquid_share_above_band": r["hyperliquid"]["share_above"],
                  "hyperliquid_median_bps": r["hyperliquid"]["median_bps"]}
                 for m, r in full.items() if ranks[m] == "below_both"]
        quarters = h.by_period(asset, "Q")
        blocks = h.blocks(asset)
        q_gaps = block_median_gaps(blocks, "Q")
        block_summary = {}
        for venue, cmp in blocks.items():
            wins = _block_windows(cmp)
            block_summary[venue] = {w: comparison_summary(c) for w, c in wins.items()}
            for key, freq in (("years", "Y"), ("half_years", "H"), ("quarters", "Q"),
                              ("months", "M")):
                per = period_of(cmp["time"], freq)
                block_summary[venue][key] = {k: comparison_summary(cmp[per == k])
                                             for k in sorted(set(per))}
            block_summary[venue]["month_majority"] = month_majority(
                {m: r for m, r in block_summary[venue]["months"].items()
                 if m not in partial["M"]})
            block_summary[venue]["quantile_gaps"] = {w: quantile_gaps(c)
                                                     for w, c in wins.items()}
            block_summary[venue]["shift_test"] = {w: shift_test(c) for w, c in wins.items()}
        shares = [r["hyperliquid"]["share_at_floor"] for r in full.values()]
        assets[asset] = {
            "rule": h.rule[asset],
            "windows": h.windows(asset),
            "years": h.by_period(asset, "Y"),
            "half_years": h.by_period(asset, "H"),
            "quarters": quarters,
            "months": months,
            "month_ranking": {
                "first_full_month": min(full), "last_month": max(full),
                "n_full_months": len(full),
                "hl_above_both": sum(r == "above_both" for r in ranks.values()),
                "hl_between": sum(r == "between" for r in ranks.values()),
                "hl_below_both": sum(r == "below_both" for r in ranks.values()),
                "hl_at_8h_stamps_above_both": sum(r == "above_both" for r in ranks8.values()),
                "months_hl_not_above_both": not_above,
                "above_both_every_month_from": _run_start(ranks),
                "hl_share_min": {"month": min(full, key=lambda m:
                                              full[m]["hyperliquid"]["share_at_floor"]),
                                 "share": min(shares)},
                "hl_share_max": {"month": max(full, key=lambda m:
                                              full[m]["hyperliquid"]["share_at_floor"]),
                                 "share": max(shares)},
            },
            "months_hl_below_both": below,
            "quarter_median_gaps_bps": q_gaps,
            "quarter_median_gap_range": _gap_range(q_gaps, skip=partial["Q"]),
            "block_comparison": block_summary,
            "data": {v: {"file": f"data/longhistory/{file_name(v, asset)}",
                         "rows": int(len(h.raw[(v, asset)])),
                         "first": _stamp(h.raw[(v, asset)]["time"].min()),
                         "last": _stamp(h.raw[(v, asset)]["time"].max()),
                         "spacing_hours_common_rule": cz.spacing_hours(
                             h.common(asset)[v]["time"])}
                     for v in VENUES},
        }
    return {
        "_note": ("Funding settlements exactly at the interest floor (0.01% per 8h; "
                  "0.00125% per hour on Hyperliquid), compared as decimals, for BTC and ETH "
                  "on Hyperliquid, Binance and Bybit. Periods and windows run over the common "
                  "rule, from Hyperliquid's first record with the 0.05% clamp to the stamp "
                  "end; a settlement belongs to the calendar period in which its averaging "
                  "interval ends, and partial_periods lists the periods the span covers only "
                  "in part (left out of month counts and gap ranges). Premium: Hyperliquid's "
                  "published hourly premium; for the exchanges the premium each settlement "
                  "identifies outside the band, with floor settlements ranked inside it "
                  "(quantiles that fall on them are null). Band: the closed interval "
                  "[-4, +6] bps, on whose edges the clamp saturates. rule.history lists "
                  "Hyperliquid's eras: 8-hourly settlement to 8 Jun 2023 00:00, then hourly; "
                  "the 0.03% clamp, except for an unclamped span (rate = premium / 8) from "
                  "16 Jun 2023 21:00 to 15 Jul 2023 02:00; records that fit no rule (their "
                  "fits_clamp_bps gives the whole-bps clamp they fit exactly, if any: the "
                  "16 Jul 2023 records fit 0.02%, which looks like a clamp applied for about "
                  "14 hours, an inference from the data); the 0.05% clamp from the switch. "
                  "block_comparison orders Hyperliquid's mean "
                  "premium over each 8-hour settlement block against the exchange's premium "
                  "for that block, decided exactly or by the band; share_hl_above_lower and "
                  "_upper count the undecided blocks against and for Hyperliquid being "
                  "higher, and need no assumption, unlike the share of decided blocks; "
                  "median_diff_given_exchange_identified_bps uses only blocks whose exchange "
                  "premium lies outside the band. quantile_gaps: quantiles of Hyperliquid's "
                  "block means and of the exchange's premium over the same blocks, and their "
                  "difference; quarter_median_gaps_bps is their q50 difference by quarter. "
                  "shift_test moves Hyperliquid's block means down by that median difference "
                  "(the difference of the two medians) and counts those in the band; its "
                  "_by_block_diff variant shifts by the median of the identified per-block "
                  "differences instead. partial_periods H: the half-years. "
                  "pre_switch_hourly is under Hyperliquid's "
                  "earlier rule and is not a like-for-like comparison."),
        "end": _stamp(h.end),
        "common_rule": {"from": _stamp(h.switch), "to": _stamp(h.end)},
        "floors": {"8h": str(FLOOR_8H), "hourly": str(FLOOR_HOURLY)},
        "band_bps": list(cz.BAND_BPS),
        "partial_periods": partial,
        "windows": {"paper": cz.WINDOWS["paper"], "later": cz.WINDOWS["later"],
                    "before_paper": [_stamp(h.switch), "before " + cz.WINDOWS["paper"][0]],
                    "pre_switch_hourly": [_stamp(h.hourly_from),
                                          "before " + _stamp(h.switch)]},
        "assets": assets,
    }


def _run_start(ranks: dict):
    """The first month of the final run of 'above_both' months, or None."""
    start = None
    for m in sorted(ranks):
        if ranks[m] == "above_both":
            start = start or m
        else:
            start = None
    return start


# ----------------------------------------------------------------------
#  Table
# ----------------------------------------------------------------------
COLUMNS = ("hyperliquid", "hyperliquid_at_8h_stamps", "binance", "bybit")


def table_rows(payload: dict, asset: str) -> list:
    """(label, row) pairs for one asset: the common-rule period, each calendar year with
    its half-years beneath it, then the paper's two windows. A year the period covers
    in one half only has no half-year rows."""
    a = payload["assets"][asset]
    start = pd.Timestamp(payload["common_rule"]["from"])
    end = pd.Timestamp(payload["common_rule"]["to"]) - pd.Timedelta(minutes=1)
    rows = [(f"{start.day} {start:%b %Y} -- {end.day} {end:%b %Y}",
             a["windows"]["common_rule"])]
    for year, row in a["years"].items():
        halves = [k for k in a["half_years"] if k.startswith(year + "H")]
        if year == str(start.year):
            label = f"{year}, from {start.day} {start:%b}"
        elif year == str(end.year) and end.month != 12:
            label = f"{year}, Jan--{end:%b}"
        else:
            label = year
        rows.append((label, row))
        if len(halves) < 2:
            continue
        for key in halves:
            last = end if key == max(a["half_years"]) and end.month != 12 else None
            span = ("Jan--Jun" if key.endswith("1") else
                    f"Jul--{last:%b}" if last is not None else "Jul--Dec")
            rows.append((f"\\quad {span}", a["half_years"][key]))
    for w in ("paper", "later"):
        rows.append((cz.window_label(w), a["windows"][w]))
    return rows


SHARE_COLUMNS = COLUMNS
MEDIAN_COLUMNS = ("hyperliquid", "binance", "bybit")


def _share_cell(d: dict) -> str:
    return f"{100 * d['share_at_floor']:.1f}"


def _median_cell(d: dict) -> str:
    m = d["median_bps"]
    if m is None:
        return "in band"
    return ("$-$" if m < 0 else "") + f"{abs(m):.2f}"


def write_table(payload: dict) -> str:
    """The LaTeX table: floor shares and median premia by year and half-year, one panel
    per asset. Sized for the text width when set in \\footnotesize."""
    ncol = 1 + len(SHARE_COLUMNS) + len(MEDIAN_COLUMNS)
    tex = ["% Generated by regime_engine.longhistory -- do not edit by hand.",
           "% Settlements exactly at the interest floor, compared as decimals, and median",
           "% premia (bps, 8h-equivalent; Hyperliquid hourly); see",
           "% output/censoring_long_history.json. Set in \\footnotesize.",
           f"\\begin{{tabular}}{{@{{}}l{'r' * (ncol - 1)}@{{}}}}",
           "\\toprule",
           " & \\multicolumn{4}{c}{Settlements at the floor (\\%)}"
           " & \\multicolumn{3}{c}{Median premium (bps)} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(l){6-8}",
           " & HL & HL & & & HL & & \\\\",
           "Period & hourly & 00/08/16 & Binance & Bybit & hourly & Binance & Bybit \\\\",
           "\\midrule"]
    for i, asset in enumerate(ASSETS):
        if i:
            tex.append("\\addlinespace")
        tex.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{'AB'[i]}. {asset}}}}} \\\\")
        rows = table_rows(payload, asset)
        for j, (label, row) in enumerate(rows):
            if j == len(rows) - 2:
                tex.append("\\addlinespace")
            cells = ([_share_cell(row[c]) for c in SHARE_COLUMNS]
                     + [_median_cell(row[c]) for c in MEDIAN_COLUMNS])
            tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(tex)


# ----------------------------------------------------------------------
#  Figure
# ----------------------------------------------------------------------
# Venue colours as in the censoring figure (fig4); line style and marker repeat the
# identity for print in grey.
PALETTE = {"hyperliquid": "#1B7F79", "binance": "#C98A0B", "bybit": "#5A4FCF"}
STYLE = {"hyperliquid": "-", "binance": "--", "bybit": ":"}
MARKER = {"hyperliquid": "o", "binance": "s", "bybit": "^"}
LABELS = {"hyperliquid": "Hyperliquid (hourly)", "binance": "Binance", "bybit": "Bybit"}
FIGURE_WIDTH_IN = 15.8 / 2.54          # the text width the figure is printed at
MIN_FONT_PT = 7.0
#: Paper-window shading, as in the premium-gap figure (premium_gap.draw_figure).
WINDOW_SHADE = {"color": "#2E86AB", "alpha": 0.08}
FIGURE_DIR = DEFAULT_OUT / "figures"
PAPER_FIGURE_DIR = REPO_ROOT / "paper" / "figures"


def figure_data(payload: dict) -> dict:
    """Per asset and full month: floor shares by venue, and the bounds on the share of
    8-hour blocks in which Hyperliquid's mean premium lies above each exchange's."""
    partial = set(payload["partial_periods"]["M"])
    out = {}
    for asset in ASSETS:
        a = payload["assets"][asset]
        months = [m for m in a["months"] if m not in partial]
        blocks = a["block_comparison"]
        out[asset] = {
            "months": months,
            "x": [pd.Timestamp(m) + pd.Timedelta(days=14) for m in months],
            "share": {v: [a["months"][m][v]["share_at_floor"] for m in months]
                      for v in VENUES},
            "lower": {v: [blocks[v]["months"][m]["share_hl_above_lower"] for m in months]
                      for v in EXCHANGES},
            "upper": {v: [blocks[v]["months"][m]["share_hl_above_upper"] for m in months]
                      for v in EXCHANGES},
        }
    return out


def draw_figure(data: dict):
    """Floor shares by month (top) and the bounds on the premium ordering (bottom),
    BTC left and ETH right, sized for the text width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4, "figure.dpi": 130, "savefig.dpi": 220,
        "savefig.bbox": "tight", "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN + 0.3, 4.7), sharex=True,
                             sharey="row", gridspec_kw={"height_ratios": [1.15, 1],
                                                        "hspace": 0.3, "wspace": 0.1})
    paper = [pd.Timestamp(t) for t in cz.WINDOWS["paper"]]
    muted = "#555555"
    offset = {"binance": pd.Timedelta(days=-5), "bybit": pd.Timedelta(days=5)}
    for col, asset in enumerate(ASSETS):
        d = data[asset]
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            ax.axvspan(*paper, zorder=0, lw=0, **WINDOW_SHADE)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        for v in VENUES:
            top.plot(d["x"], d["share"][v], STYLE[v], color=PALETTE[v], lw=1.3,
                     marker=MARKER[v], ms=2.8, label=LABELS[v])
        for v in EXCHANGES:
            x = [t + offset[v] for t in d["x"]]
            bottom.vlines(x, d["lower"][v], d["upper"][v], color=PALETTE[v], lw=2.0,
                          label=f"than {LABELS[v]}")
            # end marks, so that a month whose bounds coincide still shows
            for key in ("lower", "upper"):
                bottom.plot(x, d[key][v], ls="none", marker="_", ms=3.5, mew=1.2,
                            color=PALETTE[v])
        bottom.axhline(0.5, color=muted, lw=0.8, zorder=1)
        top.set_title(f"({'ab'[col]}) {asset}: settlements at the floor", loc="left",
                      fontweight="bold")
        # read with the legend: "Hyperliquid premium higher than Binance"
        bottom.set_title(f"({'cd'[col]}) {asset}: Hyperliquid premium higher",
                         loc="left", fontweight="bold")
        bottom.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        # left-aligned in the shading, clear of the September 2025 peak just before it
        top.text(paper[0] + pd.Timedelta(days=4), 0.98, "estimation\nwindow", fontsize=7.5,
                 color=muted, ha="left", va="top")
    axes[0, 0].set_ylabel("Share of settlements")
    axes[1, 0].set_ylabel("Share of 8-hour blocks,\nlower to upper bound")
    handles = [Line2D([], [], ls=STYLE[v], color=PALETTE[v], marker=MARKER[v], ms=3,
                      lw=1.3, label=LABELS[v]) for v in VENUES]
    fig.subplots_adjust(top=0.88)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, borderaxespad=0, handlelength=2.6)
    axes[1, 0].legend(loc="lower left", frameon=False, handlelength=1.2, ncol=2)
    return fig


def printed_font_sizes(fig) -> tuple:
    """(smallest font in pt as printed at the text width, saved width in inches).

    The saved width is the tight bounding box plus savefig's padding on both
    sides, which is what the paper scales to the text width."""
    import matplotlib
    fig.canvas.draw()
    pad = float(matplotlib.rcParams["savefig.pad_inches"])
    width = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
    scale = FIGURE_WIDTH_IN / width
    sizes = [t.get_fontsize() for t in fig.findobj(
        lambda a: hasattr(a, "get_fontsize") and hasattr(a, "get_text")) if t.get_text()]
    return min(sizes) * scale, width


def figure(payload: dict, out_dir=None) -> pathlib.Path:
    """Draw the figure from the artifact and save it as PDF and PNG."""
    import matplotlib.pyplot as plt

    fig = draw_figure(figure_data(payload))
    out_dir = pathlib.Path(out_dir or FIGURE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{FIGURE_STEM}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{FIGURE_STEM}.png")
    plt.close(fig)
    return pdf


# ----------------------------------------------------------------------
#  Command line
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m regime_engine.longhistory",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true",
                    help="refetch the six series from the venues before rebuilding")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args(argv)
    if args.fetch:
        for name, n in fetch_all(args.data).items():
            print(f"{name}: {n} rows")
    payload = build_payload(args.data)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=1) + "\n")
    table = write_table(payload)
    for path in (OUTPUT_TABLE, PAPER_TABLE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table)
    pdf = figure(payload)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, PAPER_FIGURE_DIR / pdf.name)
    print(f"common rule from {payload['common_rule']['from']} to {payload['end']}")
    for asset, a in payload["assets"].items():
        w = a["windows"]["common_rule"]
        print(asset + "  " + "   ".join(
            f"{name}: {w[name]['at_floor']}/{w[name]['n']} ({100 * w[name]['share_at_floor']:.1f}%)"
            for name in COLUMNS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
