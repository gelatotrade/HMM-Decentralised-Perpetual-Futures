"""The explanatory figures of the paper, drawn from the shipped data and artifacts.

    python3 -m regime_engine.paper_figures                 # every figure
    python3 -m regime_engine.paper_figures --only windows  # one of them

Each figure has a pure function that assembles what it shows (tested in
``tests/test_paper_figures.py``) and a function that only draws. Drawings go to
``output/figures/<stem>.{pdf,png}`` and the PDF is copied to ``paper/figures/``.
PDFs carry no creation date, so unchanged inputs give byte-identical files.
"""
from __future__ import annotations

import argparse
import json
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

STYLE = {
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "figure.dpi": 130, "savefig.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 1.1,
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


def save(fig, stem: str) -> pathlib.Path:
    """Write ``<stem>.pdf`` and ``<stem>.png`` to output/ and copy the PDF to paper/."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    FIG_PAPER.mkdir(parents=True, exist_ok=True)
    pdf = FIG_OUT / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(FIG_OUT / f"{stem}.png")
    shutil.copyfile(pdf, FIG_PAPER / pdf.name)
    return pdf


# ----------------------------------------------------------------------
#  fig5_windows: windows and data
# ----------------------------------------------------------------------
def window_spans() -> list:
    """The three windows the paper estimates on, with their length in hours."""
    prem = _times(DATA / "premium_BTC_full.csv")
    est = (_utc(PAPER_WINDOW[0]), _utc(PAPER_WINDOW[1]))
    later = tuple(_utc(t) for t in censoring.WINDOWS["later"])
    panel = _times(DEFAULT_PANEL)
    return [
        {"label": "estimation window", "start": est[0], "end": est[1],
         "hours": int(((prem >= est[0]) & (prem <= est[1])).sum())},
        {"label": "hourly sub-period", "start": panel.min(), "end": panel.max(),
         "hours": int(len(panel))},
        {"label": "later window", "start": later[0], "end": later[1],
         "hours": int(((prem > later[0]) & (prem <= later[1])).sum())},
    ]


def availability_rows() -> list:
    """Which series the package ships, from first to last stamp."""
    funding = pd.concat([_times(DATA / "premium_BTC_full.csv"),
                         _times(DATA / "binance_BTCUSDT_funding.csv"),
                         _times(DATA / "bybit_BTCUSDT_funding.csv")])
    prices_2h = _times(DATA / "prices_BTC_2h_full.csv")
    panel = _times(DEFAULT_PANEL)
    extended = _times(REPO_ROOT / "data" / "processed" / "panel_BTC_extended.csv")
    return [
        {"label": "hourly prices", "start": panel.min(), "end": extended.max(),
         "note": "archived panel continued; the earlier hours are no longer served"},
        {"label": "two-hourly prices", "start": prices_2h.min(), "end": prices_2h.max(),
         "note": "candles, re-fetchable"},
        {"label": "funding and premium", "start": funding.min(), "end": funding.max(),
         "note": "Hyperliquid hourly; Binance, Bybit eight-hourly"},
    ]


def price_extremes() -> tuple:
    """The estimation window's closing-price high and low, located in the 2h candles.

    The values are the hourly closes of the full-window extract, as in the
    summary-statistics table; the two-hourly candles only place them in time.
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

    px = pd.read_csv(DATA / "prices_BTC_2h_full.csv")
    px["time"] = pd.to_datetime(px["time"], utc=True, format="ISO8601")
    spans = {s["label"]: s for s in window_spans()}
    rows = {r["label"]: r for r in availability_rows()}
    hi, lo = price_extremes()
    peak = october_peak()

    fig, (ax, gantt) = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=True,
                                    gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.08})
    for label in ("estimation window", "later window"):
        s = spans[label]
        ax.axvspan(s["start"], s["end"], color=WINDOW_COLORS[label], alpha=0.10, lw=0)
    ax.plot(px["time"], px["c"] / 1e3, color=INK, lw=0.8)
    ax.axvline(peak["time"], color="#C73E1D", lw=0.8, ls="--")
    ax.annotate(f"10 Oct 2025: premium {peak['bps']:+.2f} bps\nin one hour of the crash",
                xy=(peak["time"], 117), xytext=(_utc("2025-11-04"), 121),
                fontsize=7.5, color="#C73E1D", va="center",
                arrowprops={"arrowstyle": "-", "color": "#C73E1D", "lw": 0.6})
    for ext, dy, va in ((hi, 2.5, "bottom"), (lo, -3, "top")):
        y = float(px.loc[px["time"] == ext["time"], "c"].iloc[0]) / 1e3
        ax.plot(ext["time"], y, "o", ms=3, color=INK)
        ax.text(ext["time"] + pd.Timedelta(days=4), y + dy, f"${ext['price']:,.0f}",
                fontsize=7.5, ha="left", va=va, color=INK)
    ax.set_ylabel("BTC close (USD 000)")
    ax.set_ylim(55, 135)
    ax.text(_utc("2025-12-12"), 66, "estimation window", fontsize=7.5,
            color=WINDOW_COLORS["estimation window"], va="bottom", ha="center")
    ax.text(_utc("2026-08-01"), 131, "later window", fontsize=7.5,
            color=WINDOW_COLORS["later window"], va="top", ha="center")

    bars = [
        ("Estimation window", spans["estimation window"], WINDOW_COLORS["estimation window"],
         f"{spans['estimation window']['hours']:,} h"),
        ("Hourly prices", rows["hourly prices"], "#9DB9C9",
         f"{spans['hourly sub-period']['hours']:,} h to {_day(spans['hourly sub-period']['end'])}"
         f", continued to {_day(rows['hourly prices']['end'])}"),
        ("Two-hourly prices", rows["two-hourly prices"], "#BBBBBB", "candles, re-fetchable"),
        ("Funding and premium", rows["funding and premium"], "#BBBBBB",
         "Hyperliquid hourly; Binance, Bybit every 8 h"),
        ("Later window", spans["later window"], WINDOW_COLORS["later window"],
         f"{spans['later window']['hours']:,} h"),
    ]
    for i, (_label, r, color, note) in enumerate(bars):
        y = len(bars) - 1 - i
        gantt.barh(y, r["end"] - r["start"], left=r["start"], height=0.62, color=color,
                   alpha=0.85, lw=0)
        gantt.text(r["start"] + pd.Timedelta(days=3), y, note, fontsize=6.8, va="center",
                   color="white" if color in WINDOW_COLORS.values() else INK)
    gantt.set_yticks(range(len(bars)), [b[0] for b in bars][::-1], fontsize=7.5)
    gantt.grid(axis="y", visible=False)
    gantt.spines["left"].set_visible(False)
    gantt.tick_params(axis="y", length=0)
    gantt.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    gantt.xaxis.set_minor_locator(mdates.MonthLocator())
    gantt.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    gantt.set_xlim(_utc("2025-09-25"), _utc("2026-10-08"))
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
    the off-diagonal cells are reconstructed with the original fit's boundary
    estimate A31 = 0 imposed (``A_provenance`` in bootstrap_hmm.json): the zero is
    an input here, not a result. The paper bounds it with the rule of three over
    the state-3 hours the stationary mass implies.
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


def figure_transitions() -> pathlib.Path:
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
    ax.set_ylim(-2.45, 2.05)
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
        ax.text((x[i] + x[j]) / 2 + dx, y_text, text, ha="center", va="center", fontsize=7,
                color=color)

    arc(0, 1, 40, -0.35, label(0, 1), 1.02)
    arc(1, 2, 40, -0.35, label(1, 2), 1.02)
    arc(1, 0, -40, -0.35, label(1, 0), -1.02)
    arc(2, 1, -40, -0.35, label(2, 1), -1.02, dx=-0.3)
    arc(0, 2, 75, -0.40, label(0, 2), 1.82)
    arc(2, 0, -75, -0.50,
        f"high \u2192 low: 0 of some {g['entries_high']:.0f} entries; "
        f"bound {100 * g['bound_high_low']:.1f}%/h (rule of three)",
        -2.28, color="#C73E1D", ls=(0, (3, 2)), lw=0.8)
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
    out = save(fig, "fig7_october_hour")
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
    out = save(fig, "fig8_rolling")
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
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    for i, b in enumerate(blocks[::-1]):
        ax.barh(i, b["inferior"], color=c_inf, height=0.6)
        ax.barh(i, b["best"], left=b["inferior"], color=c_best, height=0.6)
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

    from matplotlib.patches import Patch

    def text(o, what):
        llf = f"{o['llf']:,.2f}".replace("-", "\u2212")
        return (f"{what}: logL {llf}; half-lives {o['tau_calm']:.1f} h and "
                f"{o['tau_stressed']:.1f} h, R = {o['ratio']:.1f}")
    ax.legend(handles=[Patch(color=c_inf, label=text(inferior, "inferior optimum")),
                       Patch(color=c_best, label=text(best, "maximum"))],
              fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.28),
              ncol=1)
    out = save(fig, "fig9_optima")
    plt.close(fig)
    return out


FIGURES = {"windows": figure_windows, "transitions": figure_transitions,
           "october": figure_october, "rolling": figure_rolling, "optima": figure_optima}


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
