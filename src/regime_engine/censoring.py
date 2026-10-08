"""Funding-rate censoring across perpetual venues.

Hyperliquid, Binance and Bybit compute funding with the same rule,

    F = P + clamp(r - P, -0.05%, +0.05%),   r = 0.01% per 8h,

so each pays exactly the interest floor, 0.01% per 8h, whenever its premium P
lies inside (-4, +6) bps (8h-equivalent). Hyperliquid settles every hour, at
F / 8, so its floor is 0.125 bps/h; Binance and Bybit settle every eight hours.
A settlement at the floor says only that the premium sat inside the band; one
above or below it identifies the premium exactly.

This module counts, per venue and window, how many settlements sit exactly at
the floor, below it and above it, and describes where each venue's premium lies
relative to the band. Rates are compared as decimals, never with a float
tolerance: the venues print the floor exactly, and so must the count.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from decimal import Decimal

import numpy as np
import pandas as pd

from .paths import DEFAULT_OUT, REPO_ROOT

FLOOR_8H = Decimal("0.0001")          # 0.01% per 8h
FLOOR_HOURLY = Decimal("0.0000125")   # Hyperliquid: 0.01% / 8, paid every hour
BAND_BPS = (-4.0, 6.0)                # premium band inside which F equals r
CLAMP = Decimal("0.0005")             # the clamp saturates at +-0.05%
SETTLEMENT_HOURS = (0, 8, 16)         # Binance and Bybit settlement stamps, UTC

# The paper window is closed at both ends; the later window starts after it.
WINDOWS = {
    "paper": ("2025-10-02 00:00", "2026-04-28 08:00"),
    "later": ("2026-04-28 08:00", "2026-10-01 00:00"),
}


def classify(rates, floor: Decimal = FLOOR_8H) -> dict:
    """Count settlements exactly at, below and above ``floor``."""
    at = below = above = 0
    for rate in rates:
        d = Decimal(str(rate).strip())
        if d == floor:
            at += 1
        elif d < floor:
            below += 1
        else:
            above += 1
    n = at + below + above
    if n == 0:
        raise ValueError("no settlements to classify")
    return {"n": n, "at_floor": at, "below": below, "above": above,
            "share_at_floor": at / n, "share_below": below / n, "share_above": above / n}


def in_window(times: pd.Series, name: str) -> pd.Series:
    """Membership of each UTC timestamp in the named window."""
    start, end = (pd.Timestamp(t, tz="UTC") for t in WINDOWS[name])
    if name == "paper":
        return (times >= start) & (times <= end)
    return (times > start) & (times <= end)


def window_label(name: str) -> str:
    """'28 Apr -- 30 Sep 2026'. An end at midnight closes the day before."""
    start, end = (pd.Timestamp(t) for t in WINDOWS[name])
    last = end - pd.Timedelta(minutes=1)
    year = "" if start.year == last.year else f" {start.year}"
    return f"{start.day} {start:%b}{year} -- {last.day} {last:%b %Y}"


def at_settlement_stamps(times: pd.Series) -> pd.Series:
    """True at 00:00, 08:00 and 16:00 UTC, where the eight-hourly venues settle."""
    return times.dt.hour.isin(SETTLEMENT_HOURS) & (times.dt.minute == 0)


def spacing_hours(times: pd.Series) -> list:
    """The distinct gaps between consecutive settlements, in whole hours."""
    gaps = times.sort_values().diff().dropna().dt.total_seconds() / 3600
    return sorted({int(round(g)) for g in gaps})


def premium_location(bps) -> dict:
    """Mean, standard deviation and share inside the band of a premium series in bps."""
    x = pd.Series(bps, dtype=float).dropna()
    lo, hi = BAND_BPS
    return {"n": int(len(x)), "mean_bps": float(x.mean()), "sd_bps": float(x.std(ddof=1)),
            "share_in_band": float(((x > lo) & (x < hi)).mean())}


def funding_rule(premium_bps, floor_bps: float = float(FLOOR_8H) * 1e4,
                 clamp_bps: float = float(CLAMP) * 1e4) -> np.ndarray:
    """Funding per eight hours against the averaged premium, in bps.

    F = P + clamp(r - P, -c, +c): exactly the floor r while the premium lies in
    the band (r - c, r + c) of BAND_BPS, and the premium shifted by c outside it.
    """
    p = np.asarray(premium_bps, dtype=float)
    return p + np.clip(floor_bps - p, -clamp_bps, clamp_bps)


def recovered_premium_bps(rates, floor: Decimal = FLOOR_8H) -> pd.Series:
    """The averaged premium each settlement implies, in bps; NaN where it printed the floor.

    Outside the band the clamp saturates, so a settlement F below the floor equals
    P + 0.05% and one above it equals P - 0.05%: the premium is recovered exactly.
    A settlement at the floor says only that the premium lay inside the band.
    """
    out = []
    for rate in rates:
        d = Decimal(str(rate).strip())
        if d < floor:
            out.append(float((d - CLAMP) * 10_000))
        elif d > floor:
            out.append(float((d + CLAMP) * 10_000))
        else:
            out.append(float("nan"))
    return pd.Series(out, dtype=float)


def censored_quantile(bps, p: float):
    """Quantile of a premium series whose NaNs are values known only to lie in the band.

    The censored values sort between those below and above the band, so a quantile
    whose order statistics avoid them is exact; one that falls on them is unknown
    and returned as None. Linear interpolation, as numpy's default.
    """
    x = pd.Series(bps, dtype=float).reset_index(drop=True)
    inside = (BAND_BPS[0] + BAND_BPS[1]) / 2
    values = x.fillna(inside).to_numpy()
    order = np.argsort(values, kind="stable")
    v, censored = values[order], x.isna().to_numpy()[order]
    h = (len(v) - 1) * p
    i, j = int(np.floor(h)), int(np.ceil(h))
    if censored[i] or censored[j]:
        return None
    return float(v[i] + (h - i) * (v[j] - v[i]))


def _quartiles(bps) -> dict:
    return {f"{k}_bps": censored_quantile(bps, q)
            for k, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75))}


def eight_hour_means(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Mean of hourly premium-index readings over each 00/08/16 UTC block, in bps.

    A reading is stamped when its hour opens, a settlement when its block closes,
    so each block is labelled by the settlement stamp that closes it.
    """
    s = df.set_index("time")[col].astype(float) * 1e4
    return s.resample("8h", origin="epoch", label="right", closed="left").mean().dropna()


# ----------------------------------------------------------------------
#  The comparison: artifact, table and figure
# ----------------------------------------------------------------------
DATA = REPO_ROOT / "data" / "interim"
ARTIFACT = DEFAULT_OUT / "censoring_comparison.json"
OUTPUT_TABLE = DEFAULT_OUT / "tables" / "table10_censoring_venues.tex"
PAPER_TABLE = REPO_ROOT / "paper" / "tables" / "table10_censoring_venues.tex"
FIGURE_STEM = "fig4_censoring_venues"
EXCHANGES = ("binance", "bybit")
LABELS = {"hyperliquid": "Hyperliquid", "binance": "Binance", "bybit": "Bybit"}


def _load(path, rate_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={rate_col: str})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def build_payload(data_dir=DATA) -> dict:
    """Censoring counts and premium location per venue and window."""
    data_dir = str(data_dir)
    hl = _load(f"{data_dir}/premium_BTC_full.csv", "fundingRate")
    venues = {"hyperliquid": {"settles": "hourly"},
              "hyperliquid_at_8h_stamps": {"settles": "hourly, read at 00/08/16 UTC"}}
    for w in WINDOWS:
        m = in_window(hl["time"], w)
        hourly = hl.loc[m, "premium"] * 1e4
        venues["hyperliquid"][w] = {**classify(hl.loc[m, "fundingRate"], FLOOR_HOURLY),
                                    "premium": {"source": "hourly average premium",
                                                **premium_location(hourly), **_quartiles(hourly)}}
        stamped = m & at_settlement_stamps(hl["time"])
        venues["hyperliquid_at_8h_stamps"][w] = classify(hl.loc[stamped, "fundingRate"],
                                                         FLOOR_HOURLY)
    for venue in EXCHANGES:
        funding = _load(f"{data_dir}/{venue}_BTCUSDT_funding.csv", "fundingRate")
        means = eight_hour_means(_load(f"{data_dir}/{venue}_BTCUSDT_premium_1h.csv", "close"))
        blocks = means.index.to_series()
        venues[venue] = {"settles": "8-hourly", "spacing_hours": spacing_hours(funding["time"])}
        for w in WINDOWS:
            m = in_window(funding["time"], w)
            recovered = recovered_premium_bps(funding.loc[m, "fundingRate"])
            venues[venue][w] = {
                **classify(funding.loc[m, "fundingRate"], FLOOR_8H),
                "premium": {"source": "settlements", "n": int(len(recovered)),
                            "identified": int(recovered.notna().sum()), **_quartiles(recovered)},
                "premium_index_8h_means": premium_location(means[in_window(blocks, w).values]),
            }
    return {
        "_note": ("Funding settlements exactly at the interest floor (0.01% per 8h; "
                  "0.125 bps/h on Hyperliquid), compared as decimals, per venue and window. "
                  "Premium: Hyperliquid's hourly average premium; for Binance and Bybit, the "
                  "premium each settlement identifies outside the band (F - 0.05% below it), "
                  "with in-band settlements treated as censored; premium_index_8h_means is an "
                  "approximation from the hourly premium index. Band: (-4, +6) bps."),
        "floors": {"8h": str(FLOOR_8H), "hourly": str(FLOOR_HOURLY)},
        "band_bps": list(BAND_BPS),
        "windows": WINDOWS,
        "venues": venues,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def _signed(x: float) -> str:
    return ("$-$" if x < 0 else "") + f"{abs(x):.2f}"


def write_table(payload: dict) -> str:
    """The LaTeX table the paper inputs, generated from the artifact."""
    v = payload["venues"]
    rows = [("Hyperliquid", "hourly", v["hyperliquid"]),
            ("Hyperliquid", "at 00/08/16 UTC", v["hyperliquid_at_8h_stamps"]),
            ("Binance", "every 8 h", v["binance"]),
            ("Bybit", "every 8 h", v["bybit"])]
    tex = ["% Generated by regime_engine.censoring -- do not edit by hand.",
           "% Settlements exactly at the interest floor, compared as decimals; see",
           "% output/censoring_comparison.json.",
           "\\begin{tabular}{llrrrrrrr}",
           "\\toprule",
           " & & \\multicolumn{5}{c}{" + window_label("paper") + "} & "
           "\\multicolumn{2}{c}{" + window_label("later") + "} \\\\",
           "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}",
           "Venue & Settles & $n$ & At floor & Below & Median & IQR & $n$ & At floor \\\\",
           "\\midrule"]
    for name, settles, d in rows:
        p, later = d["paper"], d["later"]
        prem = p.get("premium")
        median = _signed(prem["median_bps"]) if prem else ""
        iqr = f"[{_signed(prem['q25_bps'])}, {_signed(prem['q75_bps'])}]" if prem else ""
        tex.append(f"{name} & {settles} & {p['n']:,} & {_pct(p['share_at_floor'])} & "
                   f"{_pct(p['share_below'])} & {median} & {iqr} & {later['n']:,} & "
                   f"{_pct(later['share_at_floor'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(tex)


def figure_layout():
    """The rule (a) above the premiums (b) on one premium axis; the shares (c) beside."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4, "figure.dpi": 130, "savefig.dpi": 200,
        "savefig.bbox": "tight", "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(8.0, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1.5],
                          hspace=0.16, wspace=0.42)
    rule = fig.add_subplot(gs[0, 0])
    density = fig.add_subplot(gs[1, 0], sharex=rule)
    bars = fig.add_subplot(gs[:, 1])
    return fig, {"rule": rule, "density": density, "bars": bars}


def figure(payload: dict, data_dir=DATA, out_dir=DEFAULT_OUT / "figures") -> pathlib.Path:
    """The rule, where each venue's premium sits, and the censoring that follows."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    v = payload["venues"]
    fig, axes = figure_layout()
    ax_rule, ax_b, ax_a = axes["rule"], axes["density"], axes["bars"]
    band_color, ink, muted = "#E6E6E6", "#333333", "#555555"
    xlim = (-14, 10)

    # (a) the rule
    grid = np.linspace(*xlim, 481)
    floor = float(FLOOR_8H) * 1e4
    clamp = float(CLAMP) * 1e4
    ax_rule.axvspan(*BAND_BPS, color=band_color, zorder=0)
    ax_rule.plot(grid, funding_rule(grid), color=ink, lw=1.5)
    ax_rule.text(-13.4, 1.0, f"below the band:\npremium + {clamp:.0f} bps",
                 fontsize=7, color=muted, ha="left", va="bottom")
    ax_rule.text(9.8, -2.0, f"above:\npremium \u2212 {clamp:.0f} bps",
                 fontsize=7, color=muted, ha="right", va="top")
    ax_rule.text(1.0, floor + 1.4, f"in the band:\nfunding = floor, {floor:.0f} bp per 8 h",
                 fontsize=7, color=muted, ha="center", va="bottom")
    ax_rule.set_ylim(-10, 7)
    ax_rule.set_ylabel("Funding (bps per 8 h)")
    ax_rule.set_title("(a) The rule: funding against premium", loc="left", fontsize=9,
                      fontweight="bold")
    ax_rule.tick_params(labelbottom=False)

    # (b) premium distributions against the band
    hl = _load(f"{data_dir}/premium_BTC_full.csv", "fundingRate")
    hourly = hl.loc[in_window(hl["time"], "paper"), "premium"] * 1e4
    series = {"hyperliquid": (hourly, len(hourly))}
    for venue in EXCHANGES:
        funding = _load(f"{data_dir}/{venue}_BTCUSDT_funding.csv", "fundingRate")
        recovered = recovered_premium_bps(funding.loc[in_window(funding["time"], "paper"),
                                                      "fundingRate"])
        series[venue] = (recovered.dropna(), len(recovered))
    ax_b.axvspan(*BAND_BPS, color=band_color, zorder=0)
    palette = {"hyperliquid": "#1B7F79", "binance": "#C98A0B", "bybit": "#5A4FCF"}
    bins = [x / 2 for x in range(-28, 21)]                      # -14 ... +10 bps, 0.5 bp bins
    for venue, (x, n) in series.items():
        prem = v[venue]["paper"]["premium"]
        kind = "hourly" if venue == "hyperliquid" else "from settlements"
        # density over all settlements, so a venue's curve integrates to its identified share
        ax_b.hist(x.clip(*xlim), bins=bins, weights=np.full(len(x), 1.0 / (n * 0.5)),
                  histtype="step", linewidth=1.4, color=palette[venue],
                  label=f"{LABELS[venue]}, {kind}\n(median {prem['median_bps']:.2f} bps)"
                        .replace("-", "\u2212"))
    shares = "\n".join(f"{LABELS[e]} {100 * v[e]['paper']['share_at_floor']:.1f}%"
                        for e in EXCHANGES)
    ax_b.text(1.3, 0.32, f"in band, value\nnot identified:\n{shares}",
              transform=ax_b.get_xaxis_transform(), ha="center", va="center", fontsize=7,
              color=muted)
    ax_b.set_xlim(*xlim)
    ax_b.set_xlabel("Premium (bps, 8h-equivalent)")
    ax_b.set_ylabel("Density")
    ax_b.set_ylim(0, ax_b.get_ylim()[1] * 1.1)
    ax_b.legend(fontsize=7, frameon=False, loc="upper right")      # the right half is empty
    ax_b.set_title("(b) Where each venue's premium sits", loc="left", fontsize=9,
                   fontweight="bold")

    # (c) composition of settlements, paper window
    rows = [("Hyperliquid\nhourly", v["hyperliquid"]["paper"]),
            ("Hyperliquid\nat 00/08/16 UTC", v["hyperliquid_at_8h_stamps"]["paper"]),
            ("Binance\nevery 8 h", v["binance"]["paper"]),
            ("Bybit\nevery 8 h", v["bybit"]["paper"])]
    colors = {"below": "#2E86AB", "at": "#3D3D3D", "above": "#C73E1D"}
    for i, (_label, d) in enumerate(rows[::-1]):
        below, at, above = d["share_below"], d["share_at_floor"], d["share_above"]
        ax_a.barh(i, below, color=colors["below"], height=0.5)
        ax_a.barh(i, at, left=below, color=colors["at"], height=0.5)
        ax_a.barh(i, above, left=below + at, color=colors["above"], height=0.5)
        ax_a.text(below + at / 2 if at > 0.08 else below + at + 0.01, i,
                  f"{100 * at:.1f}%", va="center",
                  ha="center" if at > 0.08 else "left",
                  color="white" if at > 0.08 else "#3D3D3D", fontsize=8, fontweight="bold")
    ax_a.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax_a.set_xlim(0, 1)
    ax_a.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_a.grid(axis="y", visible=False)
    ax_a.legend(handles=[Patch(color=colors["below"], label="below the band"),
                         Patch(color=colors["at"], label="at the floor (in band)"),
                         Patch(color=colors["above"], label="above the band")],
                loc="upper center", bbox_to_anchor=(0.42, -0.07), ncol=2, fontsize=7,
                frameon=False)
    ax_a.set_title("(c) Settlements, " + window_label("paper").replace(" -- ", " \u2013 "),
                   loc="left", fontsize=9, fontweight="bold")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{FIGURE_STEM}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{FIGURE_STEM}.png")
    plt.close(fig)
    return pdf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m regime_engine.censoring",
                                 description="Funding-rate censoring across venues.")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args(argv)
    payload = build_payload(args.data)
    ARTIFACT.write_text(json.dumps(payload, indent=1) + "\n")
    table = write_table(payload)
    for path in (OUTPUT_TABLE, PAPER_TABLE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table)
    pdf = figure(payload, args.data)
    shutil.copyfile(pdf, REPO_ROOT / "paper" / "figures" / pdf.name)
    for name, d in payload["venues"].items():
        p, later = d["paper"], d["later"]
        print(f"{name:<26} paper: {p['at_floor']:>5}/{p['n']:<5} at floor "
              f"({100 * p['share_at_floor']:.1f}%)   later: {100 * later['share_at_floor']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
