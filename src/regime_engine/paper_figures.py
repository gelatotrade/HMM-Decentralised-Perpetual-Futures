"""The explanatory figures of the paper, drawn from the shipped data and artifacts.

    python3 -m regime_engine.paper_figures                 # every figure
    python3 -m regime_engine.paper_figures --only windows  # one of them

Each figure has a pure function that assembles what it shows (tested in
``tests/test_paper_figures.py``) and a function that only draws. Drawings go to
``output/figures/<stem>.{pdf,png}``; the PDF of every figure the paper prints is
copied to ``paper/figures/``. fig7_october_hour and fig8_rolling, which show the
linear delta intervals of the earlier version, stay in output/ only: the paper
prints fig7b and fig8b. PDFs carry no creation date, so unchanged inputs give
byte-identical files.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import sys

import pandas as pd

from . import censoring
from .paths import DEFAULT_OUT, DEFAULT_PANEL, PAPER_WINDOW, REPO_ROOT

DATA = REPO_ROOT / "data" / "interim"
FIG_OUT = DEFAULT_OUT / "figures"
FIG_PAPER = REPO_ROOT / "paper" / "figures"
FULL_SUMMARY = DEFAULT_OUT / "summary_FULLSAMPLE_2025-10-02_to_2026-04-28.json"
BOOTSTRAP = DEFAULT_OUT / "bootstrap_hmm.json"
OOS = DEFAULT_OUT / "oos_robustness.json"
SUMMARY = DEFAULT_OUT / "summary.json"
SEED_VARIATION = DEFAULT_OUT / "msar_seed_variation.json"
OPTIMA = DEFAULT_OUT / "msar_optima.json"
DIAGNOSTICS = DEFAULT_OUT / "diagnostics.json"
A31_EVIDENCE = DEFAULT_OUT / "a31_evidence.json"

# matplotlib's cmsy10.ttf carries pre-1970 header dates; fontTools reports them on
# every subset, which says nothing about the figures.
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

STYLE = {
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "figure.dpi": 130, "savefig.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 1.1,
    # Math (R, the p headers) in the serif face of the text around it.
    "mathtext.fontset": "dejavuserif",
    # Read only by the "custom" fontset, which Figure 7's legend uses: there
    # \mathcal{L} is the Computer Modern glyph of the tables and extracts as L.
    "mathtext.rm": "DejaVu Serif", "mathtext.it": "DejaVu Serif:italic",
    "mathtext.bf": "DejaVu Serif:bold", "mathtext.cal": "cmsy10",
    # Type 42 rather than Type 3 fonts, as in analysis.py.
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
WINDOW_COLORS = {"estimation window": "#2E86AB", "later window": "#C98A0B"}
REGIME_COLORS = ["#2E86AB", "#A8A8A8", "#C73E1D"]   # low / moderate / high volatility
INK, MUTED = "#333333", "#777777"


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _times(path, col: str = "time") -> pd.Series:
    return pd.to_datetime(pd.read_csv(path, usecols=[col])[col], utc=True, format="ISO8601")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt


def save(fig, stem: str, paper: bool = True) -> pathlib.Path:
    """Write ``<stem>.pdf`` and ``<stem>.png`` to output/; copy the PDF to paper/ if printed."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    pdf = FIG_OUT / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(FIG_OUT / f"{stem}.png")
    if paper:
        FIG_PAPER.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, FIG_PAPER / pdf.name)
    return pdf


# ----------------------------------------------------------------------
#  fig5_windows: windows and data
# ----------------------------------------------------------------------
LONG = REPO_ROOT / "data" / "longhistory"
GAP = REPO_ROOT / "data" / "premium_gap"
#: Hyperliquid's first record with the exchanges' +-0.05% clamp (censoring_long_history.json).
COMMON_RULE_FROM = "2023-12-11 23:00"


def window_spans() -> list:
    """The windows the paper measures on, with their length in Hyperliquid hours."""
    prem = _times(DATA / "premium_BTC_full.csv")
    longh = _times(LONG / "hyperliquid_BTC_funding.csv")
    rule = (_utc(COMMON_RULE_FROM), _utc(censoring.WINDOWS["later"][1]))
    est = (_utc(PAPER_WINDOW[0]), _utc(PAPER_WINDOW[1]))
    later = tuple(_utc(t) for t in censoring.WINDOWS["later"])
    panel = _times(DEFAULT_PANEL)
    return [
        {"label": "common rule", "start": rule[0], "end": rule[1],
         "hours": int(((longh >= rule[0]) & (longh <= rule[1])).sum())},
        {"label": "estimation window", "start": est[0], "end": est[1],
         "hours": int(((prem >= est[0]) & (prem <= est[1])).sum())},
        {"label": "hourly sub-period", "start": panel.min(), "end": panel.max(),
         "hours": int(len(panel))},
        {"label": "later window", "start": later[0], "end": later[1],
         "hours": int(((prem > later[0]) & (prem <= later[1])).sum())},
    ]


def availability_rows() -> list:
    """Which series the package ships, from first to last stamp."""
    hl = _times(LONG / "hyperliquid_BTC_funding.csv")
    settlements = pd.concat([_times(LONG / f"{v}_{c}USDT_funding.csv")
                             for v in ("binance", "bybit") for c in ("BTC", "ETH")])
    index = pd.concat([_times(GAP / f"{v}_{c}USDT_premium_hourly.csv")
                       for v in ("binance", "bybit") for c in ("BTC", "ETH")])
    prices_2h = _times(DATA / "prices_BTC_2h_full.csv")
    panel = _times(DEFAULT_PANEL)
    extended = _times(REPO_ROOT / "data" / "processed" / "panel_BTC_extended.csv")
    return [
        {"label": "hyperliquid funding and premium", "start": hl.min(), "end": hl.max(),
         "common_rule_from": _utc(COMMON_RULE_FROM),
         "note": "hourly from June 2023, BTC and ETH"},
        {"label": "exchange settlements", "start": settlements.min(), "end": settlements.max(),
         "note": "Binance and Bybit, every 8 h, BTC and ETH"},
        {"label": "exchange premium index", "start": index.min(), "end": index.max(),
         "note": "one-minute klines aggregated to hours"},
        {"label": "two-hourly prices", "start": prices_2h.min(), "end": prices_2h.max(),
         "note": "Hyperliquid candles, re-fetchable"},
        {"label": "hourly prices", "start": panel.min(), "end": extended.max(),
         "note": "archived panel continued; the earlier hours are no longer served"},
    ]


def price_history() -> pd.DataFrame:
    """Binance's BTCUSDT perpetual over the long history, for context only.

    Hourly klines are stamped when they open; each close is placed at the end of
    its hour, the convention of Hyperliquid's records.
    """
    px = pd.read_csv(GAP / "binance_BTCUSDT_price_1h.csv")
    t = pd.to_datetime(px["open_time"], utc=True, format="ISO8601") + pd.Timedelta(hours=1)
    return pd.DataFrame({"time": t, "close": px["close"].astype(float)})


def price_extremes() -> tuple:
    """The estimation window's closing-price high and low, located in the 2h candles.

    The values are the hourly closes of the full-window extract, as archived in
    the full-sample summary; the two-hourly candles only place them in time.
    """
    lo_usd, hi_usd = json.loads(FULL_SUMMARY.read_text())["price_range_usd"]
    px = pd.read_csv(DATA / "prices_BTC_2h_full.csv")
    px["time"] = pd.to_datetime(px["time"], utc=True, format="ISO8601")
    est = (px["time"] >= _utc(PAPER_WINDOW[0])) & (px["time"] <= _utc(PAPER_WINDOW[1]))
    w = px.loc[est]
    return ({"price": float(hi_usd), "time": w.loc[w["c"].idxmax(), "time"]},
            {"price": float(lo_usd), "time": w.loc[w["c"].idxmin(), "time"]})


def october_peak() -> dict:
    """The premium's largest hour over the whole series."""
    prem = pd.read_csv(DATA / "premium_BTC_full.csv")
    prem["time"] = pd.to_datetime(prem["time"], utc=True, format="ISO8601")
    i = prem["premium"].idxmax()
    return {"time": prem.loc[i, "time"], "bps": float(prem.loc[i, "premium"] * 1e4)}


def figure_windows() -> pathlib.Path:
    plt = _plt()
    import matplotlib.dates as mdates

    px = price_history()
    spans = {s["label"]: s for s in window_spans()}
    rows = {r["label"]: r for r in availability_rows()}
    peak = october_peak()
    x0, x1 = _utc("2023-04-15"), _utc("2026-10-20")
    teal, grey, light = "#1B7F79", "#BBBBBB", "#DDDDDD"

    fig, (ax, gantt) = plt.subplots(2, 1, figsize=(6.35, 3.55), sharex=True,
                                    gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.1})
    for label in ("estimation window", "later window"):
        s = spans[label]
        ax.axvspan(s["start"], s["end"], color=WINDOW_COLORS[label], alpha=0.12, lw=0)
        ax.text(s["start"] + (s["end"] - s["start"]) / 2, 131, label.replace(" ", "\n"),
                fontsize=7.5, color=WINDOW_COLORS[label], ha="center", va="top")
    ax.plot(px["time"], px["close"] / 1e3, color=INK, lw=0.6)
    ax.axvline(peak["time"], color="#C73E1D", lw=0.8, ls="--")
    ax.text(peak["time"] - pd.Timedelta(days=12), 22, "10 Oct 2025", fontsize=7.5,
            color="#C73E1D", ha="right", va="bottom")
    rule = spans["common rule"]["start"]
    ax.axvline(rule, color=teal, lw=0.8, ls=":")
    ax.text(rule + pd.Timedelta(days=10), 118, "common rule\nfrom 11 Dec 2023",
            fontsize=7.5, color=teal, ha="left", va="top")
    ax.set_ylabel("BTC (USD 000)", fontsize=8)
    ax.set_ylim(15, 140)
    ax.tick_params(labelsize=7.5)

    def bar(y, start, end, color, text="", inside=True, text_color=INK, height=0.62):
        start, end = max(start, x0), min(end, x1)
        gantt.barh(y, end - start, left=start, height=height, color=color, lw=0)
        if text and inside:
            gantt.text(start + pd.Timedelta(days=8), y, text, fontsize=7.5, va="center",
                       color=text_color)
        elif text:
            gantt.text(start - pd.Timedelta(days=8), y, text, fontsize=7.5, va="center",
                       ha="right", color=INK)

    hl = rows["hyperliquid funding and premium"]
    bar(4, hl["start"], hl["common_rule_from"], light, "earlier rules", text_color=MUTED)
    bar(4, hl["common_rule_from"], hl["end"], teal,
        "Hyperliquid, hourly, BTC and ETH: funding and premium", text_color="white")
    ex = rows["exchange settlements"]
    bar(3, ex["start"], ex["end"], grey, "Binance and Bybit: funding, every 8 h, BTC and ETH")
    ix = rows["exchange premium index"]
    bar(2, ix["start"], ix["end"], grey, "Binance and Bybit: premium index, one-minute")
    two, one = rows["two-hourly prices"], rows["hourly prices"]
    bar(1, two["start"], two["end"], grey, "Hyperliquid prices: two-hourly; hourly",
        inside=False)
    bar(1, one["start"], one["end"], "#7FA7BE", height=0.34)
    est, later = spans["estimation window"], spans["later window"]
    bar(0, est["start"], est["end"], WINDOW_COLORS["estimation window"],
        f"{est['hours']:,} h", text_color="white")
    bar(0, later["start"], later["end"], WINDOW_COLORS["later window"],
        f"{later['hours']:,} h", text_color="white")
    gantt.text(est["start"] - pd.Timedelta(days=8), 0, "estimation and later windows",
               fontsize=7.5, va="center", ha="right", color=INK)
    gantt.set_yticks([])
    gantt.set_ylim(-0.6, 4.6)
    gantt.grid(axis="y", visible=False)
    gantt.spines["left"].set_visible(False)
    gantt.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    gantt.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 10)))
    gantt.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    gantt.tick_params(labelsize=7.5)
    gantt.set_xlim(x0, x1)
    out = save(fig, "fig5_windows")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
#  fig6_transitions: transition geometry of the return HMM
# ----------------------------------------------------------------------
def transition_geometry(path=BOOTSTRAP) -> dict:
    """Hourly transition probabilities of the K = 3 return model, sigma ascending.

    Point estimates and bootstrap intervals come from the full estimation window.
    The archive of that fit kept only diag(A) and the stationary distribution, so
    the off-diagonal cells are reconstructed with the high-to-low cell A31 set to
    zero, as the original fit printed it (``A_provenance`` in bootstrap_hmm.json):
    the zero is an input here, not a result. The figure quotes the
    profile-likelihood bound on it instead (``high_low_label``); ``bound_high_low``
    keeps the rule-of-three count over the state-3 hours the stationary mass
    implies.
    """
    fs = json.loads(pathlib.Path(path).read_text())["full_sample"]
    s = fs["sigma_ascending"]
    pi = [c["point"] for c in s["stationary"]]
    d = [c["point"] for c in s["expected_duration_hours"]]
    hours_high = fs["T"] * pi[2]
    return {
        "A": [[c["point"] for c in row] for row in s["A"]],
        "A_ci": [[c["ci95"] for c in row] for row in s["A"]],
        "pi": pi, "duration_h": d,
        "vol_pct": [c["point"] for c in s["annualised_sigma_pct"]],
        "T": int(fs["T"]),
        "entries_high": hours_high / d[2],
        "hours_high": hours_high,
        "bound_high_low": 3.0 / hours_high,
        "provenance": fs.get("A_provenance", ""),
    }


def high_low_label(path=A31_EVIDENCE) -> str:
    """The annotation of the high-to-low arrow in the transitions figure.

    The zero on that arrow is imposed by the reconstruction, not counted, so the
    label says so, and the bound it quotes is the one-sided 95% profile-likelihood
    bound of the hourly sub-period (``profile_likelihood.upper_bound_one_sided_95``
    in a31_evidence.json), not a rule-of-three count.
    """
    b = json.loads(pathlib.Path(path).read_text())["profile_likelihood"][
        "upper_bound_one_sided_95"]
    return (f"high \u2192 low: zero imposed (original fit)\n"
            f"bound {100 * b:.2f}%/h (profile likelihood, sub-period)")


def draw_transitions():
    """The transitions figure, drawn but not saved."""
    plt = _plt()
    import math

    from matplotlib.patches import Circle, FancyArrowPatch

    g = transition_geometry()
    A, ci = g["A"], g["A_ci"]
    x = [0.0, 2.8, 5.6]
    r = 0.72
    names = ["Low", "Moderate", "High"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.set_xlim(-1.0, 6.6)
    ax.set_ylim(-2.45, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in range(3):
        ax.add_patch(Circle((x[k], 0), r, facecolor=REGIME_COLORS[k], alpha=0.18,
                            edgecolor=REGIME_COLORS[k], lw=1.2))
        ax.text(x[k], 0.25, names[k], ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x[k], -0.05, f"{g['vol_pct'][k]:.1f}% vol", ha="center", va="center",
                fontsize=7.5)
        ax.text(x[k], -0.32, f"stays {g['duration_h'][k]:.1f} h", ha="center", va="center",
                fontsize=7, color=MUTED)

    def edge(k, deg, side):
        """A point on node k's rim, ``deg`` above (+) or below (-) the axis, on ``side``."""
        a = math.radians(deg)
        return (x[k] + side * r * math.cos(a), r * math.sin(a))

    def label(i, j):
        lo, hi = ci[i][j]
        return f"{100 * A[i][j]:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"

    def arc(i, j, deg, rad, text, y_text, color=INK, ls="-", lw=None, dx=0.0):
        side = 1 if j > i else -1
        ax.add_patch(FancyArrowPatch(edge(i, deg, side), edge(j, deg, -side),
                                     connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=9, lw=lw or 0.6 + 10 * A[i][j],
                                     color=color, ls=ls, shrinkA=0, shrinkB=0))
        ax.text((x[i] + x[j]) / 2 + dx, y_text, text, ha="center", va="center",
                fontsize=7, color=color, linespacing=1.15)

    # The labels of the short upper arcs sit inward, clear of the long arc above them,
    # and the long arc's label sits under its crest.
    arc(0, 1, 40, -0.35, label(0, 1), 0.93, dx=0.25)
    arc(1, 2, 40, -0.35, label(1, 2), 0.93, dx=-0.3)
    arc(1, 0, -40, -0.35, label(1, 0), -1.02)
    arc(2, 1, -40, -0.35, label(2, 1), -1.02, dx=-0.3)
    arc(0, 2, 75, -0.40, label(0, 2), 1.5)
    arc(2, 0, -75, -0.50, high_low_label(), -2.25, color="#C73E1D", ls=(0, (3, 2)),
        lw=0.8)
    return fig


def figure_transitions() -> pathlib.Path:
    plt = _plt()
    fig = draw_transitions()
    out = save(fig, "fig6_transitions")
    plt.close(fig)
    return out

# ----------------------------------------------------------------------
#  fig7_october_hour: the 10 October hour and what it carries
# ----------------------------------------------------------------------
def _premium() -> pd.DataFrame:
    prem = pd.read_csv(DATA / "premium_BTC_full.csv")
    prem["time"] = pd.to_datetime(prem["time"], utc=True, format="ISO8601")
    prem["bps"] = prem["premium"] * 1e4
    return prem


def _day(t) -> str:
    return f"{t.day} {t:%b %Y}"


def october_slice(start: str = "2025-10-08", end: str = "2025-10-14") -> pd.DataFrame:
    """Hourly premium in bps over the days around the crash."""
    prem = _premium()
    m = (prem["time"] >= _utc(start)) & (prem["time"] < _utc(end))
    return prem.loc[m, ["time", "bps"]].reset_index(drop=True)


def injection_times() -> list:
    """When the overwritten hour falls: the positions oos.py used, as timestamps."""
    oos = json.loads(OOS.read_text())
    positions = [int(r["variant"].split("@")[1]) for r in oos["injection"][1:]]
    prem = _premium()
    after = prem.loc[prem["time"] > _utc(PAPER_WINDOW[1]), "time"].reset_index(drop=True)
    return [after[p] for p in positions]


def forest_rows() -> list:
    """Every estimate of R the robustness section reports, in display order.

    A delta-method lower end below zero is not a half-life ratio; like the rolling table
    the figure leaves it open (``open_lo``) instead of drawing a negative bound.
    """
    oos = json.loads(OOS.read_text())
    sub = json.loads(SUMMARY.read_text())["funding_MSAR"]["half_life_ratio"]
    spans = {s["label"]: s for s in window_spans()}

    def row(group, label, ratio, ci):
        lo, hi = ci
        open_lo = lo is None or lo < 0
        return {"group": group, "label": label, "ratio": float(ratio),
                "lo": None if open_lo else float(lo), "hi": float(hi), "open_lo": open_lo}

    sens = {r["variant"]: r for r in oos["sensitivity"]}
    names = {"baseline": "as estimated", "replace_max": "maximum \u2192 second largest",
             "drop_max": "maximum deleted", "winsorise_top1": "upper 1% winsorised"}
    rows = [row("paper", names[k], sens[k]["ratio"], sens[k]["ci95"]) for k in names]
    hourly = spans["hourly sub-period"]
    ext, post = oos["extended_window"], oos["injection"][0]
    rows += [
        row("windows", f"hourly sub-period, {_day(hourly['start'])} \u2013 "
                       f"{_day(hourly['end'])}", sub["value"], sub["ci95_delta"]),
        row("windows", "whole series, " + ext["label"].replace(" - ", " \u2013 "),
            ext["ratio"], ext["ci95"]),
        row("windows", "after the window, "
            + censoring.window_label("later").replace(" -- ", " \u2013 "),
            post["ratio"], post["ci95"]),
    ]
    for t, rec in zip(injection_times(), oos["injection"][1:]):
        rows.append(row("overwritten", f"at {_day(t)}", rec["ratio"], rec["ci95"]))
    return rows


def figure_october() -> pathlib.Path:
    plt = _plt()
    import matplotlib.dates as mdates

    d = october_slice()
    rows = forest_rows()
    peak = d.loc[d["bps"].idxmax()]
    fig, (ax, fx) = plt.subplots(2, 1, figsize=(6.6, 5.1),
                                 gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.38})

    # (a) the hour
    ax.axhspan(*censoring.BAND_BPS, color="#E6E6E6", lw=0, zorder=0)
    ax.plot(d["time"], d["bps"], color="#1B7F79", lw=0.9)
    ax.annotate(f"{peak['bps']:+.2f} bps\n{peak['time']:%d %b, %H:%M} UTC",
                xy=(peak["time"], peak["bps"]),
                xytext=(peak["time"] + pd.Timedelta(hours=14), peak["bps"] - 4),
                fontsize=7.5, color="#C73E1D", va="center",
                arrowprops={"arrowstyle": "-", "color": "#C73E1D", "lw": 0.6})
    ax.text(d["time"].iloc[-1], 1.0, "band", fontsize=7, color=MUTED, ha="right", va="center")
    ax.set_ylabel("Premium (bps)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("(a) The premium around 10 Oct 2025", loc="left", fontweight="bold")

    # (b) R under every test, log axis
    colors = {"paper": "#2E86AB", "windows": "#555555", "overwritten": "#C73E1D"}
    headers = {"paper": "paper window, "
                        + censoring.window_label("paper").replace(" -- ", " \u2013 "),
               "windows": "other windows",
               "overwritten": f"after the window, one hour set to {october_peak()['bps']:+.2f} bps"}
    left, right = 0.02, 150
    y, ticks, labels, seen = 0.0, [], [], set()
    for r in rows:
        if r["group"] not in seen:
            seen.add(r["group"])
            y -= 0.35 if seen != {r["group"]} else 0.0
            fx.text(left * 1.15, y, headers[r["group"]], fontsize=7.5, fontweight="bold",
                    color=colors[r["group"]], va="center", zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
            y -= 0.75
        c = colors[r["group"]]
        lo = r["lo"] if r["lo"] is not None else left
        fx.plot([lo, r["hi"]], [y, y], color=c, lw=1.2)
        if r["open_lo"]:
            fx.annotate("", xy=(left, y), xytext=(left * 2.2, y),
                        arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.2,
                                    "mutation_scale": 7})
        fx.plot(r["ratio"], y, "o", ms=4, color=c)
        ci = ("[\u00b7, " if r["open_lo"] else f"[{r['lo']:.2f}, ") + f"{r['hi']:.2f}]"
        fx.text(right * 1.25, y, f"{r['ratio']:.2f} {ci}", fontsize=7.5, va="center",
                color=c, clip_on=False)
        ticks.append(y)
        labels.append(r["label"])
        y -= 0.75
    fx.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    fx.set_xscale("log")
    fx.set_xlim(left, right)
    fx.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    fx.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30", "100"])
    fx.minorticks_off()
    fx.set_ylim(y + 0.2, 0.6)
    fx.set_yticks(ticks, labels, fontsize=7.8)
    fx.tick_params(axis="y", length=0)
    fx.grid(axis="y", visible=False)
    fx.set_xlabel(r"$R = \hat\tau_{\mathrm{calm}} / \hat\tau_{\mathrm{stressed}}$ (log scale)")
    fx.set_title("(b) The ratio under every test", loc="left", fontweight="bold")
    out = save(fig, "fig7_october_hour", paper=False)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
#  fig8_rolling: rolling windows
# ----------------------------------------------------------------------
def rolling_rows() -> dict:
    """The rolling estimates of table8_rolling, as columns; open lower ends flagged."""
    recs = json.loads(OOS.read_text())["rolling"]
    open_lo = [r["ci95"][0] is None or r["ci95"][0] < 0 for r in recs]
    return {
        "mid": [_utc(r["mid"]) for r in recs],
        "start": [_utc(r["start"]) for r in recs],
        "end": [_utc(r["end"]) for r in recs],
        "ratio": [float(r["ratio"]) for r in recs],
        "lo": [None if o else float(r["ci95"][0]) for r, o in zip(recs, open_lo)],
        "hi": [float(r["ci95"][1]) for r in recs],
        "open_lo": open_lo,
        "significance": [r["significance"] for r in recs],
        "drawdown": [float(r["drawdown_mean"]) for r in recs],
    }


def figure_rolling() -> pathlib.Path:
    plt = _plt()
    import matplotlib.dates as mdates

    r = rolling_rows()
    colors = {"above": "#2E86AB", "below": "#C98A0B", "ns": "#888888"}
    floor = 0.006
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    dd = ax.twinx()
    dd.plot(r["mid"], [100 * d for d in r["drawdown"]], color="#AAAAAA", lw=1.0, ls=":",
            marker="s", ms=2.5)
    dd.set_ylim(-100, 0)
    dd.set_ylabel("Mean drawdown (%)", color=MUTED)
    dd.tick_params(axis="y", colors=MUTED)
    dd.grid(False)
    dd.spines["right"].set_visible(True)
    ax.set_zorder(dd.get_zorder() + 1)
    ax.patch.set_visible(False)
    for i, t in enumerate(r["mid"]):
        c = colors[r["significance"][i]]
        lo = r["lo"][i] if r["lo"][i] is not None else floor
        ax.plot([t, t], [lo, r["hi"][i]], color=c, lw=1.3)
        if r["open_lo"][i]:
            ax.annotate("", xy=(t, floor), xytext=(t, floor * 2.2),
                        arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.2,
                                    "mutation_scale": 7})
        ax.plot(t, r["ratio"][i], "o", ms=4.5, color=c, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(floor, 150)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_ylabel(r"$R = \hat\tau_{\mathrm{calm}} / \hat\tau_{\mathrm{stressed}}$")
    ax.annotate("the window that\ncontains 10 Oct 2025",
                xy=(r["mid"][0], r["ratio"][0]),
                xytext=(r["mid"][0] + pd.Timedelta(days=22), 60),
                fontsize=7.5, color=colors["above"], va="center",
                arrowprops={"arrowstyle": "-", "color": colors["above"], "lw": 0.6})
    for key, text in (("above", "interval above 1"), ("ns", "covers 1"), ("below", "below 1")):
        ax.plot([], [], "o", color=colors[key], label=text)
    ax.plot([], [], color="#AAAAAA", ls=":", marker="s", ms=2.5, label="drawdown (right)")
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Window mid (each window 2,500 hours)")
    out = save(fig, "fig8_rolling", paper=False)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
#  fig9_optima: two optima
# ----------------------------------------------------------------------
def optima_record() -> dict:
    """What each optimum of the paper-window likelihood implies (scripts/msar_optima.py)."""
    return json.loads(OPTIMA.read_text())


def optima_by_block() -> list:
    """How each block of 40 seeds splits between the maximum and the inferior optimum."""
    rec = optima_record()
    best, inferior = (o["llf"] for o in rec["optima"])
    out = []
    for run in json.loads(SEED_VARIATION.read_text())["runs"]:
        counts = {o["llf"]: o["count"] for o in run["optima_found"]}
        out.append({"seeds": run["seeds"], "best": counts.get(best, 0),
                    "inferior": counts.get(inferior, 0), "failures": run["failures"]})
    return out


def figure_optima() -> pathlib.Path:
    plt = _plt()

    rec = optima_record()
    blocks = optima_by_block()
    best, inferior = rec["optima"]
    c_inf, c_best = "#C98A0B", "#2E86AB"
    fig, ax = plt.subplots(figsize=(6.2, 1.95))
    for i, b in enumerate(blocks[::-1]):
        ax.barh(i, b["inferior"], color=c_inf, height=0.66)
        ax.barh(i, b["best"], left=b["inferior"], color=c_best, height=0.66)
        ax.text(b["inferior"] / 2, i, str(b["inferior"]), color="white", fontsize=7.5,
                ha="center", va="center", fontweight="bold")
        ax.text(b["inferior"] + b["best"] / 2, i, str(b["best"]), color="white", fontsize=7.5,
                ha="center", va="center", fontweight="bold")
    ax.set_yticks(range(len(blocks)), [f"block {len(blocks) - i}" for i in range(len(blocks))],
                  fontsize=7.5)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Estimations out of 40")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch

    def text(o, what):
        llf = f"{o['llf']:,.2f}".replace("-", "\u2212")
        return (f"{what}: $\\log\\mathcal{{L}}$ {llf}; half-lives {o['tau_calm']:.1f} h and "
                f"{o['tau_stressed']:.1f} h, $R$ = {o['ratio']:.1f}")
    ax.legend(handles=[Patch(color=c_inf, label=text(inferior, "inferior optimum")),
                       Patch(color=c_best, label=text(best, "maximum"))],
              prop=FontProperties(size=7, math_fontfamily="custom"), frameon=False,
              loc="upper center", bbox_to_anchor=(0.45, -0.31), ncol=1)
    out = save(fig, "fig9_optima")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
#  fig7b_october_hour_robust and fig8b_rolling_robust: robust intervals
# ----------------------------------------------------------------------
#: Colour of an estimate by the Wald test of phi_calm = phi_stressed at 5%:
#: the two poles of a diverging pair for R above and below unity, neutral gray
#: when the test does not reject. The marker repeats the verdict in its shape,
#: so it never rests on colour alone.
VERDICT_COLORS = {"above": "#2a78d6", "below": "#e34948", "ns": "#898781",
                  "undetermined": "#898781"}
VERDICT_MARKERS = {"horizontal": {"above": ">", "below": "<", "ns": "o", "undetermined": "o"},
                   "vertical": {"above": "^", "below": "v", "ns": "o", "undetermined": "o"}}
VERDICT_TEXT = {"above": r"$R > 1$", "below": r"$R < 1$", "ns": "not significant"}
COVS_ROBUST = ("hessian", "robust")
COV_NAMES = {"hessian": "Hessian", "robust": "sandwich"}
#: The smallest font any of the two figures uses, in points at the size the
#: paper prints them (text width, 15.8 cm).
MIN_FONT_PT = 7.0
TEXT_WIDTH_IN = 15.8 / 2.54


def positive_pieces(form: dict, lo_limit: float, hi_limit: float) -> list:
    """The part of a confidence set for a positive ratio that a log axis can draw.

    ``form`` is an interval form of ``msar.ratio_intervals`` (``shape`` and
    ``bounds``). Returns a list of pieces ``{"lo", "hi", "open_lo",
    "open_hi"}`` clipped to [``lo_limit``, ``hi_limit``]; an end is open when
    the set runs past the axis there (to zero or below, or to infinity). A
    bounded set below zero, or an undefined one, has nothing to draw.
    """
    shape = form.get("shape")
    lo, hi = (form.get("bounds") or [None, None])[:2]

    def piece(a, b):
        a_open = a is None or a < lo_limit
        b_open = b is None or b > hi_limit
        a = lo_limit if a_open else float(a)
        b = hi_limit if b_open else float(b)
        return [] if a > b else [{"lo": a, "hi": b, "open_lo": a_open, "open_hi": b_open}]

    if shape == "whole_line":
        return piece(None, None)
    if shape == "bounded" and lo is not None and hi is not None:
        return [] if hi <= 0 else piece(lo, hi)
    if shape == "two_rays" and lo is not None and hi is not None:
        out = piece(None, lo) if lo > 0 else []        # (0, lo] when lo is positive
        return out + piece(max(hi, 0.0) or None, None)  # [hi, inf)
    return []


def _wald(rec: dict, cov: str) -> dict:
    w = rec["inference"][cov]["ratio_wald"]
    return {"z": w["z"], "p": w["p"], "side": w["significance"]}


def _robust_entry(rec: dict) -> dict:
    """What both figures draw for one fit: R, R^sw and, per covariance, the
    Fieller set and the Wald test."""
    return {
        "ratio": float(rec.get("ratio", rec.get("half_life_ratio"))),
        "ratio_switching": rec["ratio_switching_interp"],
        "ratio_switching_hours": rec["ratio_switching"],
        "set": {cov: rec["inference"][cov]["ratio_ci95"]["fieller"] for cov in COVS_ROBUST},
        "wald": {cov: _wald(rec, cov) for cov in COVS_ROBUST},
        "side": {cov: rec["inference"][cov]["ratio_wald"]["significance"]
                 for cov in COVS_ROBUST},
    }


#: Row labels of fig7b; short enough for the figure to print at text width.
ROBUST_LABELS = {"baseline": "as estimated", "replace_max": "maximum → 2nd largest",
                 "drop_max": "maximum deleted", "winsorise_top1": "upper 1% winsorised"}


def forest_rows_robust() -> list:
    """fig7's estimates of R, with the Wald test and Fieller set under both covariances.

    Same groups, order and estimates as :func:`forest_rows`, with shorter
    labels. Each row carries R, the switching-adjusted ratio (interpolated
    hours; ``ratio_switching_hours`` is the whole-hour ratio), the Fieller
    set under the Hessian and the sandwich covariance, and the Wald test of
    phi_calm = phi_stressed whose verdict (``side``) colours the row. The
    hourly sub-period is the sub-sample fit recorded in diagnostics.json, the
    same fit summary.json reports.
    """
    from .msar_tables import msar_block

    oos = json.loads(OOS.read_text())
    diag = json.loads(DIAGNOSTICS.read_text())
    sens = {r["variant"]: r for r in oos["sensitivity"]}
    rows = [dict(_robust_entry(sens[k]), group="paper", label=ROBUST_LABELS[k])
            for k in ("baseline", "replace_max", "drop_max", "winsorise_top1")]
    rows += [dict(_robust_entry(rec), group="windows", label=label) for label, rec in (
        ("hourly sub-period", msar_block(diag, "sub-sample")),
        ("whole series", oos["extended_window"]),
        ("later window", oos["injection"][0]))]
    rows += [dict(_robust_entry(rec), group="overwritten", label=f"at {_day(t)}")
             for t, rec in zip(injection_times(), oos["injection"][1:])]
    return rows


def rolling_rows_robust() -> dict:
    """The rolling estimates of fig8 with the Wald test and Fieller set per covariance."""
    recs = json.loads(OOS.read_text())["rolling"]
    entries = [_robust_entry(r) for r in recs]
    out = {"mid": [_utc(r["mid"]) for r in recs]}
    for key in ("ratio", "ratio_switching", "ratio_switching_hours"):
        out[key] = [e[key] for e in entries]
    for key in ("set", "wald", "side"):
        out[key] = {cov: [e[key][cov] for e in entries] for cov in COVS_ROBUST}
    return out


LOG_LADDER = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000)


def log_axis_limits(lows, highs, margin: float = 1.1) -> tuple:
    """Log-axis limits on the 1-3-10 ladder that clear every interval by ``margin``.

    Undefined interval ends (``None``) are ignored.
    """
    lo = max(t for t in LOG_LADDER if t <= min(v for v in lows if v is not None) / margin)
    hi = min(t for t in LOG_LADDER if t >= max(v for v in highs if v is not None) * margin)
    return lo, hi


def set_axis_limits(entries, floor: float = 0.01) -> tuple:
    """Log-axis limits for the Fieller sets and estimates of ``entries``.

    The axis clears every finite positive end and every estimate, but stops at
    ``floor``: a set reaching further down is drawn open there.
    """
    lows, highs = [], []
    for e in entries:
        lows += [e["ratio"]] + ([e["ratio_switching"]] if e["ratio_switching"] else [])
        highs += [e["ratio"]] + ([e["ratio_switching"]] if e["ratio_switching"] else [])
        for form in e["set"].values():
            lo, hi = (form.get("bounds") or [None, None])[:2]
            if form.get("shape") == "bounded" and lo is not None and lo > 0:
                lows.append(lo)
            if form.get("shape") == "bounded" and hi is not None and hi > 0:
                highs.append(hi)
    lo, hi = log_axis_limits(lows, highs)
    return max(lo, floor), hi


def _log_ticks(axis, lo: float, hi: float, decades: bool = False) -> None:
    ladder = (0.01, 0.1, 1, 10, 100) if decades else (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,
                                                       100, 300)
    ticks = [t for t in ladder if lo <= t <= hi]
    labels = [f"{t:g}" for t in ticks]
    if axis.axis_name == "x":
        axis.axes.set_xticks(ticks)
        axis.axes.set_xticklabels(labels)
    else:
        axis.axes.set_yticks(ticks)
        axis.axes.set_yticklabels(labels)
    axis.axes.minorticks_off()


def _october_panel(ax) -> None:
    """Panel (a) of fig7_october_hour: the premium around 10 Oct 2025."""
    import matplotlib.dates as mdates

    d = october_slice()
    peak = d.loc[d["bps"].idxmax()]
    ax.axhspan(*censoring.BAND_BPS, color="#E6E6E6", lw=0, zorder=0)
    ax.plot(d["time"], d["bps"], color="#1B7F79", lw=0.9)
    # a funding record is stamped at settlement, after the hour it averages
    ax.annotate(f"{peak['bps']:+.2f} bps, record\nstamped {peak['time']:%d %b %H:%M} UTC",
                xy=(peak["time"], peak["bps"]),
                xytext=(peak["time"] + pd.Timedelta(hours=14), peak["bps"] - 4),
                fontsize=7.5, color=INK, va="center",
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6})
    ax.text(d["time"].iloc[-1], 1.0, "band", fontsize=7.5, color=MUTED, ha="right",
            va="center")
    ax.set_ylabel("Premium (bps)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("(a) The premium around 10 Oct 2025", loc="left", fontweight="bold")


def _draw_set(ax, pieces, at, color, vertical: bool) -> None:
    """Draw a set's pieces at one position, with an arrow head at every open end."""
    for p in pieces:
        span = [p["lo"], p["hi"]]
        if vertical:
            ax.plot([at, at], span, color=color, lw=1.4, solid_capstyle="butt")
        else:
            ax.plot(span, [at, at], color=color, lw=1.4, solid_capstyle="butt")
        for end, other, is_open in ((p["lo"], p["hi"], p["open_lo"]),
                                    (p["hi"], p["lo"], p["open_hi"])):
            if not is_open:
                continue
            # a short head pointing outward, at the axis edge
            tail = end * (1.9 if end < other else 1 / 1.9)
            xy, xytext = ((at, end), (at, tail)) if vertical else ((end, at), (tail, at))
            ax.annotate("", xy=xy, xytext=xytext,
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2,
                                    "mutation_scale": 7, "shrinkA": 0, "shrinkB": 0})


def _verdict_legend(fig, orientation: str, y: float) -> None:
    from matplotlib.lines import Line2D

    marks = VERDICT_MARKERS[orientation]
    handles = [Line2D([], [], color=VERDICT_COLORS[k], lw=1.4, marker=marks[k], ms=5,
                      label=VERDICT_TEXT[k]) for k in ("above", "below", "ns")]
    handles.append(Line2D([], [], ls="none", marker="D", ms=4.2, mfc="white", mec=INK,
                          mew=1.0, label="switching-adjusted ratio"))
    # No sub- or superscripts anywhere: mathtext sets them at 70% of the font
    # size, below 7 pt in print. The title takes two lines; the second is set
    # below ``y`` so that the legend does not grow into the axis labels.
    y -= 1.25 * 7.5 / 72 / fig.get_figheight()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=4,
               fontsize=7.5, frameon=False, handlelength=1.8, columnspacing=1.4,
               title="Wald test of equal calm and stressed coefficients at 5%;\n"
                     "bars are 95% Fieller sets, arrows mark sets that run past the axis",
               title_fontsize=7.5)


def _fmt_p(p) -> str:
    if p is None:
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.0995 else f"{p:.2f}"


def build_october_robust():
    """fig7b: the premium around 10 Oct 2025, and R under every test with its
    Fieller set coloured by the Wald test, under the Hessian and the sandwich."""
    plt = _plt()

    rows = forest_rows_robust()
    lo_lim, hi_lim = set_axis_limits(rows)
    headers = {"paper": "estimation window",
               "windows": "other windows",
               "overwritten": f"hour set to {october_peak()['bps']:+.2f} bps"}
    # rows top to bottom, a header line before each group
    ys, labels, bold, entries = [], [], [], []
    y, seen = 0.0, None
    for r in rows:
        if r["group"] != seen:
            seen = r["group"]
            if ys:
                y -= 0.35
            ys.append(y)
            labels.append(headers[seen])
            bold.append(True)
            entries.append(None)
            y -= 1.0
        ys.append(y)
        labels.append(r["label"])
        bold.append(False)
        entries.append(r)
        y -= 1.0

    fig = plt.figure(figsize=(6.2, 5.35))
    left, width, gap = 0.27, 0.285, 0.1
    ax = fig.add_axes([left, 0.78, 2 * width + gap, 0.1885])
    _october_panel(ax)
    panels = []
    for i, cov in enumerate(COVS_ROBUST):
        fx = fig.add_axes([left + i * (width + gap), 0.157, width, 0.5185])
        panels.append(fx)
        for yy, e in zip(ys, entries):
            if e is None:
                continue
            side = e["side"][cov]
            color = VERDICT_COLORS[side]
            _draw_set(fx, positive_pieces(e["set"][cov], lo_lim, hi_lim), yy, color, False)
            fx.plot(e["ratio"], yy, VERDICT_MARKERS["horizontal"][side], ms=5, color=color,
                    zorder=3)
            if e["ratio_switching"] is not None:
                fx.plot(e["ratio_switching"], yy, "D", ms=4.2, mfc="white", mec=INK, mew=1.0,
                        zorder=3)
            fx.text(1.04, yy, _fmt_p(e["wald"][cov]["p"]), transform=fx.get_yaxis_transform(),
                    fontsize=7.5, va="center", ha="left", color=INK)
        fx.text(1.04, 1.0, "$p$", transform=fx.transAxes, fontsize=7.5, va="bottom", ha="left",
                color=INK)
        fx.axvline(1.0, color=MUTED, lw=0.8, ls="--")
        fx.set_xscale("log")
        fx.set_xlim(lo_lim, hi_lim)
        _log_ticks(fx.xaxis, lo_lim, hi_lim, decades=True)
        fx.tick_params(axis="x", labelsize=7.5)
        fx.set_ylim(ys[-1] - 0.7, ys[0] + 0.7)
        fx.grid(axis="y", visible=False)
        fx.tick_params(axis="y", length=0)
        fx.set_xlabel(r"$R$, calm over stressed half-life", fontsize=8)
        fx.set_title(f"({'bc'[i]}) {COV_NAMES[cov].capitalize()} covariance", loc="left",
                     fontweight="bold", fontsize=8.5)
        if i == 0:
            fx.set_yticks(ys, labels, fontsize=7.5)
            for tick, b in zip(fx.get_yticklabels(), bold):
                if b:
                    tick.set_fontweight("bold")
        else:
            fx.set_yticks(ys, [""] * len(ys))
    _verdict_legend(fig, "horizontal", 0.0)
    return fig


def figure_october_robust() -> pathlib.Path:
    fig = build_october_robust()
    out = save(fig, "fig7b_october_hour_robust")
    _plt().close(fig)
    return out


def build_rolling_robust():
    """fig8b: R in each rolling window with its Fieller set coloured by the Wald
    test, one panel per covariance."""
    plt = _plt()
    import matplotlib.dates as mdates

    r = rolling_rows_robust()
    entries = [{"ratio": r["ratio"][i], "ratio_switching": r["ratio_switching"][i],
                "set": {c: r["set"][c][i] for c in COVS_ROBUST}} for i in range(len(r["mid"]))]
    lo_lim, hi_lim = set_axis_limits(entries)
    # The two covariances side by side, on one log scale, so the figure takes a
    # third of a page.
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True,
                             gridspec_kw={"wspace": 0.06, "left": 0.1, "right": 0.98,
                                          "top": 0.91, "bottom": 0.3})
    for i, (ax, cov) in enumerate(zip(axes, COVS_ROBUST)):
        for k, t in enumerate(r["mid"]):
            side = r["side"][cov][k]
            color = VERDICT_COLORS[side]
            _draw_set(ax, positive_pieces(r["set"][cov][k], lo_lim, hi_lim), t, color, True)
            ax.plot(t, r["ratio"][k], VERDICT_MARKERS["vertical"][side], ms=5, color=color,
                    zorder=3)
            sw = r["ratio_switching"][k]
            if sw is not None:
                ax.plot(t, sw, "D", ms=4.2, mfc="white", mec=INK, mew=1.0, zorder=3)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(lo_lim, hi_lim)
        ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(f"({'ab'[i]}) {COV_NAMES[cov].capitalize()} covariance", loc="left",
                     fontweight="bold", fontsize=8.5)
    _log_ticks(axes[0].yaxis, lo_lim, hi_lim)
    axes[0].set_ylabel(r"$R$, calm over stressed", fontsize=8)
    axes[0].annotate("the window that\ncontains 10 Oct 2025",
                     xy=(r["mid"][0], r["ratio"][0]),
                     xytext=(r["mid"][0] + pd.Timedelta(days=24), r["ratio"][0] * 1.4),
                     fontsize=7.5, color=INK, va="center",
                     arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6})
    fig.supxlabel("Window mid (each window 2,500 hours)", fontsize=8, y=0.155)
    _verdict_legend(fig, "vertical", 0.0)
    return fig


def figure_rolling_robust() -> pathlib.Path:
    fig = build_rolling_robust()
    out = save(fig, "fig8b_rolling_robust")
    _plt().close(fig)
    return out


FIGURES = {"windows": figure_windows, "transitions": figure_transitions,
           "october": figure_october, "rolling": figure_rolling, "optima": figure_optima,
           "october_robust": figure_october_robust, "rolling_robust": figure_rolling_robust}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m regime_engine.paper_figures",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(FIGURES), action="append")
    args = ap.parse_args(argv)
    for name in args.only or sorted(FIGURES):
        print(f"[paper_figures] {name}: {FIGURES[name]()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
