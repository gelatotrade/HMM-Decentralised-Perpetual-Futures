"""Every number the paper's abstract and Sections 2 to 8 print, against its artifact.

Each claim names a part of ``paper/paper.tex``, a fragment of its text with
``{}`` where numbers stand, and the artifact values that fill them at printed
precision. The test formats the fragment from the artifacts and looks for it in
the plain text of that part (math dollars, thousands braces and ties removed),
so an artifact that moves a printed number fails here until the text moves
with it. ``CLAIMS`` doubles as the paper's numbers ledger: every value names its
file, its key and how it is rounded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache

import pytest

from regime_engine.paths import DEFAULT_OUT, repo_root

TEX = repo_root() / "paper" / "paper.tex"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 14: "fourteen"}

CC = "censoring_comparison.json"
LH = "censoring_long_history.json"
PG = "premium_gap.json"
DG = "diagnostics.json"
OOS = "oos_robustness.json"
OPT = "msar_optima.json"
SEED = "msar_seed_variation.json"
RATE = "msar_on_rate.json"
A31 = "a31_evidence.json"
MS = "model_selection_extended.json"
BOOT = "bootstrap_hmm.json"
SUM = "summary.json"
ROB = "robustness.json"
TWO = "two_hourly_refit.json"
BEN = "benchmarks.json"
PAPER_MSAR = "tests/4/msar"          # diagnostics.json: the paper-window MS-AR block
SUB_MSAR = "tests/3/msar"            # diagnostics.json: the hourly sub-period MS-AR block


# ---------------------------------------------------------------------------
#  the manuscript as plain text, by part
# ---------------------------------------------------------------------------
def plain(tex: str) -> str:
    """The text a reader sees, near enough: no comments, math or citation keys."""
    s = re.sub(r"(?<!\\)%.*", "", tex)
    s = s.replace(r"\$", "\x00").replace("{,}", ",").replace("{=}", "=").replace(r"\%", "%")
    s = s.replace("~", " ").replace(r"\ ", " ").replace(r"\pm", "±").replace(r"\,", " ")
    s = s.replace("$", "").replace("\x00", "$")
    s = re.sub(r"\\cite[a-z]*\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:eq)?ref\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\(?:emph|textbf|text|mathrm)\{([^{}]*)\}", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


@cache
def parts() -> dict:
    tex = TEX.read_text()
    out = {"abstract": plain(tex.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0])}
    body = tex.split(r"\end{abstract}")[1]
    marks = [(m.start(), m.group(1)) for m in
             re.finditer(r"\\section\{[^}]*\}\\label\{sec:([a-z]+)\}", body)]
    ends = [m.start() for m in re.finditer(r"\\section\*|\\begin\{thebibliography\}|"
                                           r"\\appendix|\\section\{", body)]
    for start, name in marks:
        end = min([e for e in ends if e > start] + [len(body)])
        out[name] = plain(body[start:end])
    out["availability"] = plain(body.split(r"\section*{Data and Code Availability}")[1]
                                .split(r"\section*")[0])
    out["appendix"] = plain(body.split(r"\appendix")[1])
    return out


# ---------------------------------------------------------------------------
#  artifact values
# ---------------------------------------------------------------------------
@cache
def load(name: str):
    path = DEFAULT_OUT / name
    if not path.exists():                       # pragma: no cover
        pytest.skip(f"{name} has not been generated")
    return json.loads(path.read_text())


def get(obj, path: str):
    for part in path.split("/"):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


@dataclass(frozen=True)
class V:
    """One printed number: where it comes from and how it is printed."""

    file: str
    keys: tuple
    fmt: str = "{}"
    scale: float = 1.0
    op: object = None
    how: str = ""

    def raw(self):
        if self.file.endswith(".json"):
            vals = [get(load(self.file), k) for k in self.keys]
        else:
            vals = []
        return self.op(*vals) if self.op else vals[0]

    def printed(self) -> str:
        x = self.raw()
        if isinstance(x, (int, float)) and not isinstance(x, bool) and self.scale != 1:
            x = x * self.scale
        return self.fmt.format(x)


def v(file, *keys, fmt="{}", scale=1.0, op=None, how=""):
    return V(file, tuple(keys), fmt, scale, op, how)


def pct(file, *keys, dec=1, op=None, how=""):
    """A share printed as a percentage."""
    return V(file, tuple(keys), f"{{:.{dec}f}}%", 100.0, op, how or f"x100, {dec} dp")


def same(fmt):
    """Several values that the text prints once: they must agree at that precision."""
    def op(*xs):
        printed = {fmt.format(x) for x in xs}
        assert len(printed) == 1, printed
        return xs[0]
    return op


def word(n) -> str:
    return WORDS[n]


def day(stamp: str, year: bool = True) -> str:
    t = datetime.strptime(stamp[:16], "%Y-%m-%d %H:%M" if len(stamp) > 10 else "%Y-%m-%d")
    return f"{t.day} {t:%B}" + (f" {t.year}" if year else "")


def month(stamp: str) -> str:
    t = datetime.strptime(stamp[:7], "%Y-%m")
    return f"{t:%B %Y}"


def month_list(months) -> str:
    """'January, February and November 2024 and July 2025' from YYYY-MM stamps."""
    by_year = {}
    for m in months:
        t = datetime.strptime(m[:7], "%Y-%m")
        by_year.setdefault(t.year, []).append(f"{t:%B}")
    groups = []
    for y, names in by_year.items():
        names_text = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        groups.append(f"{names_text} {y}")
    return " and ".join(groups)


def _fn(desc, func, fmt="{}"):
    """A value computed from shipped data or from several artifacts."""
    return V(desc, (), fmt, 1.0, func, desc)


def _next_largest_premium():
    from regime_engine.panel import load_paper_window_premium
    y = sorted(load_paper_window_premium(), reverse=True)
    return y[1]


#: The window's price extremes are Hyperliquid's archived hourly closes, dated by
#: Hyperliquid's two-hourly candles.
PRICE_EXTREMES_SOURCE = (
    "summary_FULLSAMPLE price_range_usd (archived Hyperliquid hourly closes), dated by "
    "Hyperliquid 2h candles in data/interim/prices_BTC_2h_full.csv via "
    "paper_figures.price_extremes()")


def _price_extremes():
    from regime_engine import paper_figures as pf
    hi, lo = pf.price_extremes()
    return hi, lo


def _stamp(s: str) -> str:
    """'15 August 2024 13:00' from '2024-08-15 13:00[...]'."""
    return day(s[:16]) + " " + s[11:16]


def _hour(s: str) -> datetime:
    return datetime.strptime(s[:16], "%Y-%m-%d %H:%M")


def _missing_in_span():
    """Hourly stamps of the common-rule span without a Hyperliquid record, both assets."""
    lo, hi = (_hour(t) for t in load(PG)["spans"]["common"])
    stamps = int((hi - lo) / timedelta(hours=1)) + 1
    return same("{}")(*[stamps - get(load(LH), f"assets/{c}/windows/common_rule/hyperliquid/n")
                        for c in ("BTC", "ETH")])


def _hours_before_switch():
    """Hours of the first exchange settlement's eight-hour block stamped before the switch."""
    first = _hour(load(PG)["first_settlement"])
    switch = _hour(same("{}")(*[get(load(LH), f"assets/{c}/rule/switch") for c in ("BTC", "ETH")]))
    return sum(first - timedelta(hours=k) < switch for k in range(8))


def _block_end(stamp: str) -> str:
    """The exchange settlement (00, 08 or 16 UTC) whose block (s - 8h, s] holds the hour."""
    t = _hour(stamp)
    end = t.replace(hour=0, minute=0) + timedelta(hours=8 * -(-t.hour // 8))
    return f"{end:%Y-%m-%d %H:%M}"


def _below_both_months():
    """(month record, (asset, month)) for each asset-month with Hyperliquid below both exchanges."""
    lh = load(LH)
    return [(lh["assets"][c]["months"][rec["month"]], (c, rec["month"]))
            for c in ("BTC", "ETH") for rec in lh["assets"][c]["months_hl_below_both"]]


def _exchange_medians_above_band():
    """((asset, exchange), (month, median)) for the exchange medians of those months that lie
    above the band; every other exchange median lies in it, and none above Hyperliquid's."""
    hi = load(LH)["band_bps"][1]
    out = []
    for m, (c, mon) in _below_both_months():
        for x in ("binance", "bybit"):
            med = m[x]["median_bps"]
            if med is None:                     # the median falls on censored settlements
                continue
            assert hi < med < m["hyperliquid"]["median_bps"], (c, mon, x, med)
            out.append(((c, x), (mon, med)))
    return out


def _quarter_lowest():
    """(lower bound, asset, exchange, quarter) of the lowest quarterly lower bound."""
    lh = load(LH)
    return min((r["share_hl_above_lower"], c, x, q) for c in ("BTC", "ETH")
               for x in ("binance", "bybit")
               for q, r in lh["assets"][c]["block_comparison"][x]["quarters"].items())


def _months_reversed():
    """Months in which even the upper bound lies below one half against both exchanges."""
    lh = load(LH)
    hits = [(c, m) for c in ("BTC", "ETH")
            for m in lh["assets"][c]["block_comparison"]["binance"]["months"]
            if all(lh["assets"][c]["block_comparison"][x]["months"][m]["share_hl_above_upper"]
                   < 0.5 for x in ("binance", "bybit"))]
    assert {c for c, _ in hits} == {"ETH"}, hits
    return [m for _, m in hits]


def _a32():
    from regime_engine import paper_figures as pf
    return pf.transition_geometry()["A"][2][1]


def _msar_fits():
    """The 24 MS-AR fits of the paper, as test_msar_robust_artifacts counts them."""
    diag, oos = load(DG), load(OOS)
    fits = [get(diag, PAPER_MSAR), get(diag, SUB_MSAR), oos["extended_window"],
            oos["injection"][0]]
    fits += list(oos["rolling"]) + list(oos["sensitivity"][1:]) + list(oos["injection"][1:])
    return fits


def _interval_errors():
    """Fit-covariance cases in which each interval form contradicts the Wald test."""
    wrong = {"linear": 0, "log": 0, "fieller": 0}
    fits = _msar_fits()
    for rec in fits:
        for cov in ("hessian", "robust"):
            verdict = rec["inference"][cov]["ratio_wald"]["significance"]
            for form in wrong:
                if rec["inference"][cov]["ratio_ci95"][form]["significance"] != verdict:
                    wrong[form] += 1
    return len(fits), 2 * len(fits), wrong


def _rolling_sides():
    roll = load(OOS)["rolling"]

    def where(cov, side):
        return {r["index"] for r in roll
                if r["inference"][cov]["ratio_wald"]["significance"] == side}
    return where("hessian", "above"), where("robust", "above"), \
        where("hessian", "below") & where("robust", "below")


def _reversal_count():
    """Pair-quarters of the positive-premium phase in which Hyperliquid's share is lower."""
    dec = load(PG)["decomposition"]
    hits, total, exceptions = 0, 0, []
    for coin in ("BTC", "ETH"):
        for venue in ("binance", "bybit"):
            d = dec[coin][venue]
            for q in [q for q, ph in d["phases"].items() if ph == "positive"]:
                s = d["by_quarter"][q]["shares"]
                total += 1
                if s["a_hl_hourly"] < s["d_exchange_settlements"]:
                    hits += 1
                else:
                    exceptions.append(f"{coin}/{venue}/{q}")
    return hits, total, exceptions


def _optimum_count(rec) -> int:
    """How many estimations reach the reported optimum (optima are stored at 2 dp)."""
    return same("{}")(*[o["count"] for o in rec["optima_found"]
                       if o["llf"] is not None and abs(o["llf"] - rec["llf"]) < 0.005])


def _admissible(rec) -> int:
    n = load(OOS)["protocol"]["n_starts"]
    return n - rec["failures"] - sum(rec["excluded"].values())


def _reached():
    """(estimations at the reported optimum, completed estimations, window mid, gap to the most
    frequent optimum) for each rolling window."""
    out = []
    for r in load(OOS)["rolling"]:
        done = sum(o["count"] for o in r["optima_found"])
        assert done == load(OOS)["protocol"]["n_starts"] - r["failures"], r["mid"]
        modal = max((o for o in r["optima_found"] if o["llf"] is not None),
                    key=lambda o: o["count"])
        out.append((_optimum_count(r), done, r["mid"], r["llf"] - modal["llf"]))
    return out


def _worst_injection():
    """(injected hour, record) of the injection fit with the most numerical failures."""
    from regime_engine.msar_tables import injection_dates
    inj = load(OOS)["injection"][1:]
    pairs = list(zip(injection_dates(load(OOS)), inj))
    return max(pairs, key=lambda p: p[1]["failures"])


def _most_common_year(months) -> str:
    """The year that holds more than half of the YYYY-MM stamps."""
    years = [m[:4] for m in months]
    top = max(set(years), key=years.count)
    assert 2 * years.count(top) > len(years), years
    return top


def _a31_two_hours():
    """The high-to-low cell of the reconstructed hourly kernel over two hours, in %."""
    import numpy as np

    from regime_engine import paper_figures as pf
    A = np.asarray(pf.transition_geometry()["A"])
    assert A[2][0] == 0.0
    return 100 * (A @ A)[2][0]


def _positive_quarters():
    return {q for c in ("BTC", "ETH") for x in ("binance", "bybit")
            for q, f in load(PG)["decomposition"][c][x]["phases"].items() if f == "positive"}


def _isolated_months():
    """Months with Hyperliquid below both exchanges outside the positive-premium quarters."""
    def quarter(m):
        return f"{m[:4]}Q{(int(m[5:7]) - 1) // 3 + 1}"
    lh = load(LH)
    return {c: [r["month"] for r in lh["assets"][c]["months_hl_below_both"]
                if quarter(r["month"]) not in _positive_quarters()] for c in ("BTC", "ETH")}


def _quarters_below_binance():
    """Quarters outside the positive phase in which Hyperliquid's share lies below Binance's,
    for both assets."""
    out = []
    for c in ("BTC", "ETH"):
        bq = load(PG)["decomposition"][c]["binance"]["by_quarter"]
        below = [q for q, r in bq.items() if q not in _positive_quarters()
                 and r["shares"]["a_hl_hourly"] < r["shares"]["d_exchange_settlements"]]
        assert len(below) == 1, (c, below)
        out += below
    return out


def _crash_record() -> str:
    """The stamp of the record that carries most of the sandwich variance of R."""
    top = get(load(DG), PAPER_MSAR + "/ratio_sandwich_top_hours/0")
    start = datetime.strptime(load(CC)["windows"]["paper"][0], "%Y-%m-%d %H:%M")
    return f"{start + timedelta(hours=top['index']):%Y-%m-%d %H:%M}"


def _sandwich_hour():
    t = _hour(_crash_record())
    return f"{t.day} {t:%B %Y %H:%M}"


def _garch_bound():
    caveat = load(BEN)["rows"][0]["caveat"]
    return re.search(r"at the (0\.\d+) bound", caveat).group(1)


def _shapley_share_below_half(*quarters):
    """Location's Shapley value below half the difference, against both exchanges."""
    def op(dec):
        for venue in ("binance", "bybit"):
            for q in quarters:
                rec = dec[venue]["by_quarter"][q]
                assert rec["shapley"]["location"] < 0.5 * rec["difference"], (venue, q)
        return "less than half"
    return op


def _eth_location_shares():
    """ETH: location's Shapley value as a percentage of the difference, 2026Q1 and 2026Q2."""
    dec = load(PG)["decomposition"]["ETH"]
    return [100 * dec[x]["by_quarter"][q]["shapley"]["location"]
            / dec[x]["by_quarter"][q]["difference"]
            for x in ("binance", "bybit") for q in ("2026Q1", "2026Q2")]


#: The span 2024Q3-2025Q4 with 2024Q4, which the below-band group leaves out against Binance.
SIX = ("2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4")
CORNERS = {(0, 0, 0): "a_hl_hourly", (1, 0, 0): "b_hl_hourly_shifted",
           (0, 1, 0): "c_hl_eight_hour", (1, 1, 0): "e_hl_eight_hour_shifted",
           (0, 0, 1): "f_exchange_hourly_shifted", (1, 0, 1): "g_exchange_hourly",
           (0, 1, 1): "h_exchange_eight_hour_shifted", (1, 1, 1): "d_exchange_settlements"}
HOURLY_CORNERS = {"a_hl_hourly", "b_hl_hourly_shifted", "f_exchange_hourly_shifted",
                  "g_exchange_hourly"}


def _pool(coin, venue, quarters):
    """(difference, location) of quarters pooled from the per-quarter corner shares, hourly
    corners weighted by hours and block corners by blocks."""
    from regime_engine.premium_gap import shapley
    bq = load(PG)["decomposition"][coin][venue]["by_quarter"]
    corners = {}
    for corner, key in CORNERS.items():
        w = "n_hours" if key in HOURLY_CORNERS else "n_blocks"
        corners[corner] = (sum(bq[q]["shares"][key] * bq[q][w] for q in quarters)
                           / sum(bq[q][w] for q in quarters))
    return corners[(0, 0, 0)] - corners[(1, 1, 1)], shapley(corners)["location"]


def _pooled_split(coin, venue, quarters):
    """:func:`_pool`, once it reproduces the artifact's own below-band group of the span (an
    exchange hour without minutes leaves the pooled exchange corners off by about 1e-6)."""
    group = load(PG)["decomposition"][coin][venue]["below_2024Q3_2025Q4"]
    diff, loc = _pool(coin, venue, group["quarters"])
    assert abs(diff - group["difference"]) < 1e-5, diff
    assert abs(loc - group["shapley"]["location"]) < 1e-5, loc
    return _pool(coin, venue, quarters)


SHIFT = "checks/{}/{}/oracle_wording_shift"
OTHER_PAIRS = (("BTC", "bybit"), ("ETH", "binance"), ("ETH", "bybit"))


def _gap_years_and_quarters():
    """BTC's yearly and 2026 quarterly gap medians, and ETH's in 2026Q3 against its 2025 ones."""
    def gap(coin, period, *venues):
        return [GAP.format(coin, x) + f"/{period}/median_bps" for x in venues]
    both = ("binance", "bybit")
    return (
        v(PG, *gap("BTC", "by_year/2024", *both), fmt="{:.1f}", op=same("{:.1f}"),
          how="1 dp, both exchanges"),
        v(PG, *gap("BTC", "by_year/2025", *both), fmt="{:.1f}", op=min, how="min, 1 dp"),
        v(PG, *gap("BTC", "by_year/2025", *both), fmt="{:.1f}", op=max, how="max, 1 dp"),
        v(PG, *gap("BTC", "by_year/2026", *both), fmt="{:.1f}", op=same("{:.1f}"),
          how="1 dp, both exchanges"),
        v(PG, *gap("BTC", "by_quarter/2026Q1", *both), *gap("BTC", "by_quarter/2026Q2", *both),
          fmt="{:.1f}", op=min, how="min over 2026Q1-Q2 and both exchanges, 1 dp"),
        v(PG, *gap("BTC", "by_quarter/2026Q1", *both), *gap("BTC", "by_quarter/2026Q2", *both),
          fmt="{:.1f}", op=max, how="max over 2026Q1-Q2 and both exchanges, 1 dp"),
        v(PG, *gap("BTC", "by_quarter/2026Q3", *both), fmt="{:.1f}", op=min, how="min, 1 dp"),
        v(PG, *gap("BTC", "by_quarter/2026Q3", *both), fmt="{:.1f}", op=max, how="max, 1 dp"),
        v(PG, *gap("ETH", "by_quarter/2026Q3", *both), *gap("ETH", "by_year/2025", *both),
          fmt="{:.1f}", op=lambda q1, q2, y1, y2: same("{:.1f}")(q1, q2)
          if min(q1, q2) > max(y1, y2) else float("nan"),
          how="1 dp, both exchanges; above both 2025 yearly medians"),
    )


def _months_covering_zero_everywhere():
    """The months whose interval covers zero against both exchanges at every block length; the
    other months that cover zero differ between block lengths."""
    common = None
    for x in ("binance", "bybit"):
        m = load(PG)["median_ci"]["BTC"][x]
        sets = [set(m["full_months_covering_zero"])] + [
            set(b["full_months_covering_zero"]) for b in m["block_sensitivity"].values()]
        everywhere = set.intersection(*sets)
        assert len({frozenset(s - everywhere) for s in sets}) > 1, x
        common = everywhere if common is None else common & everywhere
    assert common == {"2026-04", "2026-05"}, common
    return "April and May 2026"


# ---------------------------------------------------------------------------
#  the claims
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Claim:
    where: str
    text: str
    values: tuple

    @property
    def id(self) -> str:
        return f"{self.where}: {self.text[:48]}"

    def expected(self) -> str:
        return self.text.format(*(x.printed() for x in self.values))


def C(where, text, *values):
    return Claim(where, text, tuple(values))


HL_P, BN_P, BY_P = ("venues/hyperliquid/paper", "venues/binance/paper", "venues/bybit/paper")
HL8 = "venues/hyperliquid_at_8h_stamps/paper"
LHW = "assets/{}/windows/common_rule/{}/share_at_floor"
GAP = "gap/{}/{}"
DEC = "decomposition/BTC/{}/{}"
BC = "assets/{}/block_comparison/{}/common_rule/share_hl_above_{}"
FULL_SIG = "full_sample/sigma_ascending"


def _lhw(coin, venue, dec=1):
    return pct(LH, LHW.format(coin, venue), dec=dec)


def _lhw_0(coin, venue):
    return _lhw(coin, venue, dec=0)


def _two_dec(file, key):
    return v(file, key, fmt="{:.2f}", how="2 dp")


CLAIMS = [
    # ---- abstract ---------------------------------------------------------
    C("abstract", "while the averaged premium lies in a {} bp band",
      v(CC, "band_bps", op=lambda b: round(b[1] - b[0]), how="band width")),
    C("abstract", "{} of Hyperliquid's hourly BTC settlements sit at that rate, against {} on "
      "Binance and {} on Bybit",
      pct(CC, HL_P + "/share_at_floor"), pct(CC, BN_P + "/share_at_floor"),
      pct(CC, BY_P + "/share_at_floor")),
    C("abstract", "From 11 December 2023 to 30 September 2026 the shares are {}, {} and {} for "
      "BTC and {}, {} and {} for ETH",
      _lhw("BTC", "hyperliquid"), _lhw("BTC", "binance"), _lhw("BTC", "bybit"),
      _lhw("ETH", "hyperliquid"), _lhw("ETH", "binance"), _lhw("ETH", "bybit")),
    C("abstract", "Hyperliquid's BTC premium lies a median {} bps above both (ETH {}), in {} "
      "hours out of five and in {} quarter",
      v(PG, GAP.format("BTC", "binance") + "/common/median_bps",
        GAP.format("BTC", "bybit") + "/common/median_bps", fmt="{:.1f}", op=same("{:.1f}"),
        how="1 dp, both exchanges"),
      v(PG, GAP.format("ETH", "binance") + "/common/median_bps",
        GAP.format("ETH", "bybit") + "/common/median_bps", fmt="{:.1f}", op=same("{:.1f}"),
        how="1 dp, both exchanges"),
      v(PG, GAP.format("BTC", "binance") + "/common/share_positive",
        GAP.format("BTC", "bybit") + "/common/share_positive",
        op=lambda a, b: word(same("{:.0f}")(round(5 * a), round(5 * b))),
        how="share of hours with a positive gap, in fifths"),
      v(PG, GAP.format("BTC", "binance") + "/by_quarter",
        GAP.format("BTC", "bybit") + "/by_quarter",
        op=lambda a, b: "every" if all(q["median_bps"] > 0 for q in [*a.values(), *b.values()])
        else "not every", how="quarterly medians all positive")),
    C("abstract", "For BTC the gap persists but varies: {} bps in 2024, {} to {} in 2025, {} to {} "
      "in 2026Q1 and 2026Q2 and {} to {} in 2026Q3",
      *(x for k, x in enumerate(_gap_years_and_quarters()) if k not in (3, 8))),
    C("abstract", "By a Shapley split it accounts for {} of the difference in shares from 2024Q3 "
      "to 2025Q4, but for {} in 2026Q1 and 2026Q2 (ETH: {} to {}), when all three median "
      "premiums lay {} the band",
      v(PG, "decomposition/BTC", op=lambda d: "about all" if all(
          0.9 <= d[x]["below_2024Q3_2025Q4"]["shapley"]["location"]
          / d[x]["below_2024Q3_2025Q4"]["difference"] <= 1.1 for x in ("binance", "bybit"))
        and all(0.9 <= _pooled_split("BTC", x, SIX)[1] / _pooled_split("BTC", x, SIX)[0] <= 1.1
                for x in ("binance", "bybit")) else "?",
        how="location within 10% of the difference, below-band quarters and all six"),
      v(PG, "decomposition/BTC", op=_shapley_share_below_half("2026Q1", "2026Q2"),
        how="Shapley location < half the difference, 2026Q1 and 2026Q2, both exchanges"),
      _fn("premium_gap.json decomposition/ETH/*/by_quarter/2026Q1-Q2 location / difference",
          lambda: f"{min(_eth_location_shares()):.0f}%"),
      _fn("premium_gap.json decomposition/ETH/*/by_quarter/2026Q1-Q2 location / difference",
          lambda: f"{max(_eth_location_shares()):.0f}%"),
      v(LH, "assets/BTC/quarters", op=lambda qs: "below" if all(
          qs[q][venue]["median_bps"] < load(LH)["band_bps"][0] for q in ("2026Q1", "2026Q2")
          for venue in ("hyperliquid", "binance", "bybit")) else "?",
        how="Hyperliquid's and the exchanges' medians below -4 bps")),
    C("abstract", "reverts faster under stress only around {}: allowing for switching, the "
      "half-life ratio is {}, {} one under a misspecification-robust covariance that one hour of "
      "the {} crash {}, and later data {} it",
      _fn("diagnostics.json ratio_sandwich_top_hours/0/index + paper-window start",
          lambda: f"{_hour(_crash_record()):%B %Y}"),
      v(DG, PAPER_MSAR + "/ratio_switching_interp", fmt="{:.1f}", how="1 dp"),
      v(DG, PAPER_MSAR + "/inference/robust/ratio_wald/significance",
        op=lambda s: {"ns": "not significantly different from"}.get(s, s),
        how="Wald verdict, sandwich"),
      _fn("diagnostics.json ratio_sandwich_top_hours/0/index + paper-window start",
          lambda: f"{_hour(_crash_record()).day} {_hour(_crash_record()):%B}"),
      v(DG, PAPER_MSAR + "/ratio_sandwich_top_hours/0/share",
        op=lambda s: "dominates" if s > 0.5 else "?", how="share of the sandwich variance > 1/2"),
      v(OOS, "injection/0/inference", op=lambda i: "reverse" if all(
          i[c]["ratio_wald"]["significance"] == "below" for c in ("hessian", "robust")) else "?",
        how="later window below unity under both covariances")),
    C("abstract", "separates hourly volatility into {}, {} and {} annualised states",
      *(v(BOOT, f"{FULL_SIG}/annualised_sigma_pct/{k}/point", fmt="{:.1f}%", how="1 dp")
        for k in range(3))),
    C("abstract", "estimates the direct move from the high to the low state at zero to three "
      "decimals, as {} its fits to simulated single-regime Student-t GARCH paths",
      v(A31, "garch_generic/garch_t_fitted/share_mle_zero/share",
        op=lambda s: "do" if s == 1.0 else "?", how="share of zero estimates = 1")),

    # ---- Section 1 --------------------------------------------------------
    C("intro", "its share fell below both exchanges' in {} months for BTC and {} for ETH, mostly "
      "in {}, in each of which {} to {} of its hours lay above the band",
      v(LH, "assets/BTC/month_ranking/hl_below_both", op=word, how="count in words"),
      v(LH, "assets/ETH/month_ranking/hl_below_both", op=word, how="count in words"),
      v(LH, "assets/BTC/months_hl_below_both", "assets/ETH/months_hl_below_both",
        op=lambda a, b: _most_common_year([m["month"] for m in a + b]),
        how="the year of most of the months, more than half"),
      pct(LH, "assets/BTC/months_hl_below_both", "assets/ETH/months_hl_below_both", dec=0,
          op=lambda a, b: min(m["hyperliquid_share_above_band"] for m in a + b),
          how="min, x100, 0 dp"),
      pct(LH, "assets/BTC/months_hl_below_both", "assets/ETH/months_hl_below_both", dec=0,
          op=lambda a, b: max(m["hyperliquid_share_above_band"] for m in a + b),
          how="max, x100, 0 dp")),

    # ---- Section 2 --------------------------------------------------------
    C("rule", "that is, when \\bar P_t lies in the band [r-c, r+c] = [{}, {}] bps, centred on {} "
      "bp",
      v(CC, "band_bps/0", fmt="{:+.0f}", how="band edge"),
      v(CC, "band_bps/1", fmt="{:+.0f}", how="band edge"),
      v(CC, "band_bps", fmt="{:+.0f}", op=lambda b: (b[0] + b[1]) / 2, how="band centre")),
    C("rule", "classified record by record, settle every eight hours until {} and hourly since",
      v(LH, "assets/BTC/rule/history/0/to", "assets/ETH/rule/history/0/to",
        op=lambda a, b: day(same("{}")(a, b)), how="end of the 8-hourly era, both assets")),
    C("rule", "from {} to {} 2023 they equal the premium alone, and a few records on {} fit a ± {} "
      "clamp",
      v(LH, "assets/BTC/rule/regimes/unclamped/first", op=lambda s: day(s, year=False),
        how="first unclamped record"),
      v(LH, "assets/BTC/rule/regimes/unclamped/last", op=lambda s: day(s, year=False),
        how="last unclamped record"),
      v(LH, "assets/BTC/rule/regimes/other/first", "assets/ETH/rule/regimes/other/last",
        op=lambda a, b: day(same("{:.10}")(a, b)), how="day of the records that fit no rule"),
      v(LH, "assets/BTC/rule/history", "assets/ETH/rule/history",
        op=lambda a, b: "{:.2f}%".format(same("{}")(*{r["fits_clamp_bps"] / 100 for era in a + b
                                                         for r in era["fits_no_rule"]})),
        how="whole-bps clamp the records fit, in %")),
    C("rule", "From the record stamped {} UTC they fit the exchanges' ± 0.05%",
      v(LH, "assets/BTC/rule/switch", "assets/ETH/rule/switch",
        op=lambda a, b: day(same("{}")(a, b)) + " " + a[11:16], how="first 0.05% record")),
    C("rule", "Every long-run comparison below starts at the record of {} UTC",
      v(LH, "assets/BTC/rule/switch", "assets/ETH/rule/switch",
        op=lambda a, b: day(same("{}")(a, b)) + " " + a[11:16], how="first 0.05% record")),
    C("rule", "with no exception in {} hours for BTC or ETH",
      v(LH, "assets/BTC/rule", "assets/ETH/rule", fmt="{:,}",
        op=lambda a, b: same("{}")(*[r["common_rule_hours"] for r in (a, b)
                                     if r["common_rule_floor_not_in_band"] == 0
                                     and r["common_rule_in_band_not_floor"] == 0]),
        how="common-rule hours with floor exactly in band")),

    # ---- Section 3 --------------------------------------------------------
    C("data", "for BTC and ETH from {} to the settlement stamped {} UTC",
      v(LH, "assets/BTC/data/hyperliquid/first", op=day, how="first record"),
      v(LH, "assets/BTC/data/hyperliquid/last", op=lambda s: day(s) + " " + s[11:16],
        how="last record")),
    C("data", "{} hourly Hyperliquid settlements and {} settlements per exchange",
      v(CC, HL_P + "/n", fmt="{:,}"),
      v(CC, BN_P + "/n", BY_P + "/n", fmt="{:,}", op=same("{}"), how="both exchanges")),
    C("data", "The later window runs to 30 September 2026 ({} hours, {} settlements)",
      v(CC, "venues/hyperliquid/later/n", fmt="{:,}"),
      v(CC, "venues/binance/later/n", "venues/bybit/later/n", fmt="{:,}", op=same("{}"),
        how="both exchanges")),
    C("data", "the long history from 11 December 2023 23:00 ({} hours, {} settlements per "
      "exchange)",
      v(LH, "assets/BTC/windows/common_rule/hyperliquid/n", fmt="{:,}"),
      v(LH, "assets/BTC/windows/common_rule/binance/n", "assets/BTC/windows/common_rule/bybit/n",
        fmt="{:,}", op=same("{}"), how="both exchanges")),
    C("data", "Hyperliquid publishes no record stamped {} UTC for either asset, so the long "
      "history holds {} hour fewer than it spans, and the block comparison of Section "
      "[ssec:history] uses the {} "
      "settlement blocks with all eight hourly records under the common rule: the first "
      "settlement, {}, averages {} hours from before the switch, and the block ending {} lacks "
      "that hour (the Shapley split averages Hyperliquid's hours in all {} blocks)",
      v(PG, "data/BTC/hyperliquid/missing_hours", "data/ETH/hyperliquid/missing_hours",
        op=lambda a, b: _stamp(same("{}")(a, b)[0]) if len(a) == 1 else "?",
        how="the one missing record, both assets"),
      _fn("premium_gap.json spans/common (inclusive hourly stamps) minus "
          "censoring_long_history.json common-rule records", lambda: word(_missing_in_span())),
      v(LH, *(f"assets/{c}/block_comparison/{x}/common_rule/n_blocks" for c in ("BTC", "ETH")
              for x in ("binance", "bybit")), fmt="{:,}", op=same("{}"),
        how="complete blocks, every pair"),
      v(PG, "first_settlement", op=_stamp, how="first exchange settlement under the common rule"),
      _fn("premium_gap.json first_settlement and censoring_long_history.json rule/switch",
          lambda: word(_hours_before_switch())),
      v(PG, "data/BTC/hyperliquid/missing_hours", op=lambda m: _stamp(_block_end(m[0])),
        how="the settlement whose block holds the missing hour"),
      v(PG, *(f"decomposition/{c}/{x}/all/n_blocks" for c in ("BTC", "ETH")
              for x in ("binance", "bybit")), fmt="{:,}", op=same("{}"),
        how="blocks over which the split averages Hyperliquid's hours, every pair")),
    C("data", "the return-side ones on the hourly sub-period, {} to 28 April 2026 ({} hours)",
      v(SUM, "start", op=day, how="first hour of the sub-period"),
      v(SUM, "n_obs", fmt="{:,}")),
    C("data", "BTC perpetual traded between hourly closes of ${} ({}) and ${} ({})",
      _fn(PRICE_EXTREMES_SOURCE,
          lambda: f"{round(_price_extremes()[0]['price']):,}"),
      _fn(PRICE_EXTREMES_SOURCE,
          lambda: day(str(_price_extremes()[0]["time"])[:16])),
      _fn(PRICE_EXTREMES_SOURCE,
          lambda: f"{round(_price_extremes()[1]['price']):,}"),
      _fn(PRICE_EXTREMES_SOURCE,
          lambda: day(str(_price_extremes()[1]["time"])[:16]))),
    C("data", "The premium averaged {} bps",
      _two_dec(CC, HL_P + "/descriptives/premium_bps/mean")),
    C("data", "funding averaged {} bps per hour, about {} a year",
      v(CC, HL_P + "/descriptives/funding_bps_per_hour/mean", fmt="{:+.3f}", how="3 dp"),
      v(CC, HL_P + "/descriptives/funding_bps_per_hour/mean", fmt="{:.1f}%",
        op=lambda m: m * 8760 / 100, how="x 8,760 h / 100, 1 dp")),
    C("data", "Premium (bps) & estimation & {} & {} & {} & {} & {} & {}",
      *(_two_dec(CC, HL_P + f"/descriptives/premium_bps/{k}")
        for k in ("mean", "std", "min", "median", "max", "skew"))),
    C("data", "Funding rate (bps/h) & estimation & {} & {} & {} & {} & {} & {}",
      *(v(CC, HL_P + f"/descriptives/funding_bps_per_hour/{k}", fmt="{:.3f}", how="3 dp")
        for k in ("mean", "std", "min", "median", "max")),
      _two_dec(CC, HL_P + "/descriptives/funding_bps_per_hour/skew")),
    C("data", "Log-return (bps/h) & sub-period & {} & {} & {} & {} & {} & {}",
      _two_dec(SUM, "descriptives/log_return_bps_per_hour/mean"),
      _two_dec(SUM, "descriptives/log_return_bps_per_hour/std"),
      v(SUM, "descriptives/log_return_bps_per_hour/min", fmt="{:.1f}", how="1 dp"),
      _two_dec(SUM, "descriptives/log_return_bps_per_hour/median"),
      v(SUM, "descriptives/log_return_bps_per_hour/max", fmt="{:.1f}", how="1 dp"),
      _two_dec(SUM, "descriptives/log_return_bps_per_hour/skew")),

    # ---- Section 4 --------------------------------------------------------
    C("censoring", "Binance prints the floor in {} of {} settlements ({}) and Bybit in {} ({}), "
      "against {} of {} hours ({}) on Hyperliquid",
      v(CC, BN_P + "/at_floor"), v(CC, BN_P + "/n"), pct(CC, BN_P + "/share_at_floor"),
      v(CC, BY_P + "/at_floor"), pct(CC, BY_P + "/share_at_floor"),
      v(CC, HL_P + "/at_floor", fmt="{:,}"), v(CC, HL_P + "/n", fmt="{:,}"),
      pct(CC, HL_P + "/share_at_floor")),
    C("censoring", "{} exchange settlement lies above the floor, and {} of Binance's and {} of "
      "Bybit's lie below it",
      v(CC, BN_P + "/above", BY_P + "/above", op=lambda a, b: "No" if a == b == 0 else "Some",
        how="settlements above the floor = 0"),
      pct(CC, BN_P + "/share_below"), pct(CC, BY_P + "/share_below")),
    C("censoring", "The five months after the window repeat the pattern: {} on Hyperliquid "
      "against {} and {}",
      pct(CC, "venues/hyperliquid/later/share_at_floor"),
      pct(CC, "venues/binance/later/share_at_floor"),
      pct(CC, "venues/bybit/later/share_at_floor")),
    C("censoring", "with medians of {} and {} bps and interquartile ranges of {} and {} bps",
      _two_dec(CC, BN_P + "/premium/median_bps"), _two_dec(CC, BY_P + "/premium/median_bps"),
      v(CC, BN_P + "/premium/q25_bps", BN_P + "/premium/q75_bps", fmt="{:.2f}",
        op=lambda a, b: b - a, how="q75 - q25, 2 dp"),
      v(CC, BY_P + "/premium/q25_bps", BY_P + "/premium/q75_bps", fmt="{:.2f}",
        op=lambda a, b: b - a, how="q75 - q25, 2 dp")),
    C("censoring", "its median is {} bps, {} of its hours lie below the band, and a long tail "
      "into the band lifts its mean to {}",
      _two_dec(CC, HL_P + "/premium/median_bps"), pct(CC, HL_P + "/share_below"),
      _two_dec(CC, HL_P + "/premium/mean_bps")),
    C("censoring", "at the floor in {} to {} of hours depending on the phase, and in {} at the "
      "exchanges' 00, 08 and 16 UTC stamps",
      pct(CC, HL_P + "/phase_shares/min"), pct(CC, HL_P + "/phase_shares/max"),
      pct(CC, HL8 + "/share_at_floor")),
    C("censoring", "its premium lies in the band {} of the time ({} with equal weights), "
      "against {} of hours",
      pct(PG, DEC.format("binance", "paper/shares/c_hl_eight_hour"),
          DEC.format("bybit", "paper/shares/c_hl_eight_hour"), op=same("{}"),
          how="8-h block means with the exchanges' rising weights in the closed band, x100, 1 dp"),
      pct(LH, "assets/BTC/block_comparison/binance/shift_test/paper/hyperliquid_in_band",
          "assets/BTC/block_comparison/bybit/shift_test/paper/hyperliquid_in_band",
          op=same("{}"), how="equal-weight 8-h block means in the closed band, x100, 1 dp"),
      pct(CC, HL_P + "/share_at_floor")),
    C("censoring", "Hyperliquid's BTC funding is at the floor in {} of hours, against {} of "
      "Binance's and {} of Bybit's settlements; ETH gives {}, {} and {}",
      _lhw("BTC", "hyperliquid"), _lhw("BTC", "binance"), _lhw("BTC", "bybit"),
      _lhw("ETH", "hyperliquid"), _lhw("ETH", "binance"), _lhw("ETH", "bybit")),
    C("censoring", "Hyperliquid's share exceeds both exchanges' in {} of the {} full months for "
      "either asset, including every month since {}, but lies below both in {} months for BTC "
      "({}) and {} for ETH",
      v(LH, "assets/BTC/month_ranking/hl_above_both", "assets/ETH/month_ranking/hl_above_both",
        op=same("{}"), how="months, both assets"),
      v(LH, "assets/BTC/month_ranking/n_full_months", "assets/ETH/month_ranking/n_full_months",
        op=same("{}"), how="full months, both assets"),
      v(LH, "assets/BTC/month_ranking/above_both_every_month_from",
        "assets/ETH/month_ranking/above_both_every_month_from",
        op=lambda a, b: month(same("{}")(a, b)), how="first month of the run"),
      v(LH, "assets/BTC/month_ranking/hl_below_both", op=word, how="count in words"),
      v(LH, "assets/BTC/months_hl_below_both", op=lambda ms: month_list(m["month"] for m in ms),
        how="the months, named"),
      v(LH, "assets/ETH/month_ranking/hl_below_both", op=word, how="count in words")),
    C("censoring", "In those months Hyperliquid's median premium was positive, {} to {} bps, with "
      "{} to {} of its hours above the band",
      v(LH, "assets/BTC/months_hl_below_both", "assets/ETH/months_hl_below_both", fmt="{:+.1f}",
        op=lambda a, b: min(m["hyperliquid_median_bps"] for m in a + b), how="min, 1 dp"),
      v(LH, "assets/BTC/months_hl_below_both", "assets/ETH/months_hl_below_both", fmt="{:+.1f}",
        op=lambda a, b: max(m["hyperliquid_median_bps"] for m in a + b), how="max, 1 dp"),
      pct(LH, "assets/BTC/months_hl_below_both", "assets/ETH/months_hl_below_both", dec=0,
          op=lambda a, b: min(m["hyperliquid_share_above_band"] for m in a + b),
          how="min, x100, 0 dp"),
      pct(LH, "assets/BTC/months_hl_below_both", "assets/ETH/months_hl_below_both", dec=0,
          op=lambda a, b: max(m["hyperliquid_share_above_band"] for m in a + b),
          how="max, x100, 0 dp")),
    C("censoring", "of its hours above the band, {} a larger share than of either exchange's "
      "settlements. The exchanges' median premiums lay in the band, except in {}, when {} of the "
      "four lay above it as well, though below Hyperliquid's",
      _fn("censoring_long_history.json months/<below-both month>/<venue>/share_above",
          lambda: "in each month" if all(
              m[v]["share_above"] < m["hyperliquid"]["share_above"]
              for m, _ in _below_both_months() for v in ("binance", "bybit")) else "?"),
      _fn("censoring_long_history.json months/<below-both month>/<exchange>/median_bps",
          lambda: month(same("{}")(*{m for _, (m, _) in _exchange_medians_above_band()}))),
      _fn("censoring_long_history.json months/<below-both month>/<exchange>/median_bps",
          lambda: word(len(_exchange_medians_above_band())))),
    C("censoring", "By quarter the lower bound {} (lowest {}, {} in {}); by month the order "
      "reverses for ETH in {}, when Hyperliquid's premium was higher in only {} to {} of the "
      "blocks against Binance and {} to {} against Bybit",
      _fn("censoring_long_history.json block_comparison/*/*/quarters/*/share_hl_above_lower",
          lambda: "never falls below one half" if _quarter_lowest()[0] > 0.5 else "?"),
      _fn("censoring_long_history.json block_comparison/*/*/quarters/*/share_hl_above_lower",
          lambda: f"{100 * _quarter_lowest()[0]:.1f}%"),
      _fn("censoring_long_history.json block_comparison/*/*/quarters, the lowest lower bound",
          lambda: f"{_quarter_lowest()[1]} against {_quarter_lowest()[2].title()}"),
      _fn("censoring_long_history.json block_comparison/*/*/quarters, the lowest lower bound",
          lambda: _quarter_lowest()[3]),
      _fn("censoring_long_history.json block_comparison/*/*/months: both bounds below one half "
          "against both exchanges", lambda: month(same("{}")(*_months_reversed()))),
      *(pct(LH, f"assets/ETH/block_comparison/{x}/months/2025-04/share_hl_above_{b}", dec=0)
        for x in ("binance", "bybit") for b in ("lower", "upper"))),
    C("censoring", "Hyperliquid's premium is higher in at least {} and at most {} of the {} blocks "
      "against Binance, and in {} to {} against Bybit (ETH: {} to {} and {} to {})",
      pct(LH, BC.format("BTC", "binance", "lower")), pct(LH, BC.format("BTC", "binance", "upper")),
      v(LH, "assets/BTC/block_comparison/binance/common_rule/n_blocks", fmt="{:,}"),
      pct(LH, BC.format("BTC", "bybit", "lower")), pct(LH, BC.format("BTC", "bybit", "upper")),
      pct(LH, BC.format("ETH", "binance", "lower")), pct(LH, BC.format("ETH", "binance", "upper")),
      pct(LH, BC.format("ETH", "bybit", "lower")), pct(LH, BC.format("ETH", "bybit", "upper"))),

    # ---- Section 5 --------------------------------------------------------
    C("gap", "in all {} hours of the estimation window; and from {} to {}, eight-hour averages "
      "with each exchange's weights reproduce the settlements that identify the premium",
      v(PG, "validation/hourly_close_vs_interim/binance",
        "validation/hourly_close_vs_interim/bybit",
        fmt="{:,}", op=lambda a, b: same("{}")(*[x["equal"] for x in (a, b)
                                                 if x["equal"] == x["hours"]]),
        how="hours equal, both exchanges"),
      v(PG, "first_settlement", op=lambda s: day(f"{_hour(s) - timedelta(hours=1):%Y-%m-%d}"),
        how="the day the first settlement pays for"),
      v(PG, "spans/common/1", op=lambda s: day(f"{_hour(s) - timedelta(hours=1):%Y-%m-%d}"),
        how="the day the last settlement pays for")),
    C("gap", "to a median absolute error of {} bps on Binance ({} within 0.1 bps; ETH {} bps and "
      "{}) and {} bps on Bybit ({} within 0.1 bps for both assets), once Bybit's switch from "
      "equal to rising weights on {} is allowed for",
      v(PG, "validation/settlements/BTC/binance/median_abs_error_bps", fmt="{:.2f}", how="2 dp"),
      pct(PG, "validation/settlements/BTC/binance/share_within_0.1bp", dec=0),
      v(PG, "validation/settlements/ETH/binance/median_abs_error_bps", fmt="{:.2f}", how="2 dp"),
      pct(PG, "validation/settlements/ETH/binance/share_within_0.1bp", dec=0),
      v(PG, "validation/settlements/BTC/bybit/median_abs_error_bps",
        "validation/settlements/ETH/bybit/median_abs_error_bps", fmt="{:.3f}", op=same("{:.3f}"),
        how="3 dp, both assets"),
      v(PG, "validation/settlements/BTC/bybit/share_within_0.1bp",
        "validation/settlements/ETH/bybit/share_within_0.1bp",
        op=lambda a, b: "all" if a == b == 1.0 else "?", how="share = 1, both assets"),
      v(PG, "validation/settlements/BTC/bybit/weights",
        op=lambda s: day(re.search(r"\d{4}-\d\d-\d\d", s).group(0)),
        how="date of the switch")),
    C("gap", "Hyperliquid's BTC premium exceeded Binance's in {} of hours and Bybit's in {}, by "
      "medians of {} and {} bps; for ETH the medians are {} and {} bps, in {} of hours",
      pct(PG, GAP.format("BTC", "binance") + "/common/share_positive", dec=0),
      pct(PG, GAP.format("BTC", "bybit") + "/common/share_positive", dec=0),
      _two_dec(PG, GAP.format("BTC", "binance") + "/common/median_bps"),
      _two_dec(PG, GAP.format("BTC", "bybit") + "/common/median_bps"),
      _two_dec(PG, GAP.format("ETH", "binance") + "/common/median_bps"),
      _two_dec(PG, GAP.format("ETH", "bybit") + "/common/median_bps"),
      pct(PG, GAP.format("ETH", "binance") + "/common/share_positive",
          GAP.format("ETH", "bybit") + "/common/share_positive", dec=0, op=same("{:.2f}"),
          how="x100, 0 dp, both exchanges")),
    C("gap", "The quarterly median is positive in all {} quarters for every pair, and its 95% "
      "interval, from a moving-block bootstrap with one-week blocks, excludes zero in {} full "
      "quarter",
      v(PG, "gap", op=lambda g: word(same("{}")(*[
          sum(q["median_bps"] > 0 for q in g[c][x]["by_quarter"].values())
          for c in ("BTC", "ETH") for x in ("binance", "bybit")])),
        how="quarters with a positive median, the same for every pair"),
      v(PG, "median_ci", op=lambda m: "every" if all(
          m[c][x]["full_quarters_covering_zero"] == [] and m[c][x]["block_hours"] == 168
          for c in ("BTC", "ETH") for x in ("binance", "bybit")) else "not every",
        how="168-h blocks; no full quarter covers zero")),
    C("gap", "for BTC its yearly median was {} bps in 2024, {} to {} bps in 2025 and {} bps in "
      "2026 (to September); by quarter it fell to {} to {} bps in 2026Q1 and 2026Q2 and widened "
      "again to {} to {} bps in 2026Q3 (ETH {}, above its 2025 median)",
      *_gap_years_and_quarters()),
    C("gap", "For BTC, the interval of the monthly median covers zero in {} against both exchanges "
      "under "
      "every block length, and in a few other months that differ with the block length (with "
      "one-week blocks, {} full months in all against Binance and {} against Bybit)",
      _fn("premium_gap.json median_ci/BTC/*/full_months_covering_zero and block_sensitivity",
          _months_covering_zero_everywhere),
      v(PG, "median_ci/BTC/binance", op=lambda m: word(len(m["full_months_covering_zero"]))
        if m["block_hours"] == 168 else "?", how="full months, one-week blocks, in words"),
      v(PG, "median_ci/BTC/bybit", op=lambda m: word(len(m["full_months_covering_zero"]))
        if m["block_hours"] == 168 else "?", how="full months, one-week blocks, in words")),
    C("gap", "Deviations of the daily mean gap from its average halve in {} three days ({} for "
      "BTC against Binance; about {} days around each quarter's mean), though the autocorrelation "
      "at 30 days is still {}: the level itself moves slowly",
      v(PG, "persistence", op=lambda p: "under" if all(
          p[c][x]["ar1_daily"]["half_life"] < 3 for c in ("BTC", "ETH")
          for x in ("binance", "bybit")) else "over", how="AR(1) half-life < 3 days, every pair"),
      v(PG, "persistence/BTC/binance/ar1_daily/half_life", fmt="{:.1f}", how="1 dp"),
      v(PG, "persistence", op=lambda p: word(round(same("{:.0f}")(*[
          p[c][x]["ar1_daily_within_quarter"]["half_life"] for c in ("BTC", "ETH")
          for x in ("binance", "bybit")]))),
        how="within-quarter AR(1) half-life, every pair, 0 dp"),
      _two_dec(PG, "persistence/BTC/binance/acf_daily/30")),
    C("gap", "For ETH the weekly mean gap was negative for {} weeks from {} against both "
      "exchanges",
      v(PG, "persistence/ETH/binance/negative_week_spells",
        "persistence/ETH/bybit/negative_week_spells",
        op=lambda a, b: word(same("{}")(*[s["length"] for s in a + b
                                          if s["start"] == "2025-03-31"])),
        how="length of the spell from 31 March 2025, both exchanges"),
      v(PG, "persistence/ETH/binance/negative_week_spells", op=lambda s: day(s[-1]["start"]),
        how="start of the spell")),
    C("gap", "lie {} bps below Binance's at the 5th percentile, {} bps above at the median and {} "
      "bps above at the 90th",
      v(LH, "assets/BTC/block_comparison/binance/quantile_gaps/paper/q05/gap_bps", fmt="{:.2f}",
        op=lambda g: -g if g < 0 else float("nan"), how="minus the q05 gap, 2 dp"),
      _two_dec(LH, "assets/BTC/block_comparison/binance/quantile_gaps/paper/q50/gap_bps"),
      _two_dec(LH, "assets/BTC/block_comparison/binance/quantile_gaps/paper/q90/gap_bps")),
    C("gap", "still leaves {} of them in the band, against Binance's {} at the floor",
      pct(LH, "assets/BTC/block_comparison/binance/shift_test/paper/hyperliquid_shifted_in_band"),
      pct(LH, "assets/BTC/block_comparison/binance/shift_test/paper/exchange_at_floor")),
    C("gap", "(BTC medians of {} and {} bps, against {} and {} below it), but {} in five "
      "below-band hours fall in 2025 and 2026, when the gap was smaller, and within 2024 the "
      "ordering reverses ({} and {} bps below the band against {} and {} inside it)",
      _two_dec(PG, "phase/BTC/binance/by_band_position/inside/median_bps"),
      _two_dec(PG, "phase/BTC/bybit/by_band_position/inside/median_bps"),
      _two_dec(PG, "phase/BTC/binance/by_band_position/below/median_bps"),
      _two_dec(PG, "phase/BTC/bybit/by_band_position/below/median_bps"),
      v(PG, "phase/BTC/binance/by_band_position_by_year",
        "phase/BTC/bybit/by_band_position_by_year",
        op=lambda a, b: word(same("{}")(*[
            round(5 * sum(y[k]["below"]["n"] for k in ("2025", "2026"))
                  / sum(r["below"]["n"] for r in y.values())) for y in (a, b)])),
        how="share of below-band hours in 2025-26, in fifths, both exchanges"),
      *(_two_dec(PG, f"phase/BTC/{x}/by_band_position_by_year/2024/{p}/median_bps")
        for p in ("below", "inside") for x in ("binance", "bybit"))),
    C("gap", "(daily rank correlations of {} to {} for BTC)",
      _two_dec(PG, "phase/BTC/binance/spearman_daily/gap_vs_realised_vol"),
      _two_dec(PG, "phase/BTC/bybit/spearman_daily/gap_vs_realised_vol")),
    C("gap", "In the {} quarters in which Binance's median premium lay at or below the band's "
      "lower edge, Hyperliquid's BTC funding was at the floor in {} of hours against {} of "
      "Binance's settlements, and location accounts for {} of the {} points ({} to {} over the "
      "orders), averaging for {} and dispersion for {}",
      v(PG, DEC.format("binance", "below/quarters"), op=lambda q: word(len(q)),
        how="count in words"),
      pct(PG, DEC.format("binance", "below/shares/a_hl_hourly")),
      pct(PG, DEC.format("binance", "below/shares/d_exchange_settlements")),
      *(v(PG, DEC.format("binance", f"below/{k}"), fmt="{:.1f}", scale=100,
          how="percentage points, 1 dp")
        for k in ("shapley/location", "difference", "shapley/range/location/0",
                  "shapley/range/location/1", "shapley/averaging", "shapley/dispersion"))),
    C("gap", "Of these, in the {} from 2024Q3 to 2025Q4 (all but {}, when Binance's median premium "
      "lay inside the band) location accounts for {} of the difference, {} of {} points ({} of {} "
      "with 2024Q4), but in the {} of 2026 for {} of {}, and for {} in 2026Q1 and 2026Q2 (Table "
      "[tab:gap-quarters]), against {} to {} for ETH",
      v(PG, DEC.format("binance", "below_2024Q3_2025Q4/quarters"), op=lambda q: word(len(q)),
        how="count in words"),
      v(PG, DEC.format("binance", "phases"), op=lambda ph: same("{}")(*[
          q for q, f in ph.items() if "2024Q3" <= q <= "2025Q4" and f != "below"]),
        how="the quarter of the span outside the below-band group"),
      v(PG, DEC.format("binance", "below_2024Q3_2025Q4"),
        op=lambda g: "about all" if 0.9 <= g["shapley"]["location"] / g["difference"] <= 1.1
        else "?", how="location within 10% of the difference"),
      *(v(PG, DEC.format("binance", f"below_2024Q3_2025Q4/{k}"), fmt="{:.1f}", scale=100,
          how="percentage points, 1 dp") for k in ("shapley/location", "difference")),
      _fn("premium_gap.json decomposition/BTC/binance/by_quarter 2024Q3-2025Q4, corners pooled "
          "by hours and blocks", lambda: f"{100 * _pooled_split('BTC', 'binance', SIX)[1]:.1f}"),
      _fn("premium_gap.json decomposition/BTC/binance/by_quarter 2024Q3-2025Q4, corners pooled "
          "by hours and blocks", lambda: f"{100 * _pooled_split('BTC', 'binance', SIX)[0]:.1f}"),
      v(PG, DEC.format("binance", "below_2026/quarters"), op=lambda q: word(len(q)),
        how="count in words"),
      *(v(PG, DEC.format("binance", f"below_2026/{k}"), fmt="{:.1f}", scale=100,
          how="percentage points, 1 dp") for k in ("shapley/location", "difference")),
      v(PG, "decomposition/BTC", op=_shapley_share_below_half("2026Q1", "2026Q2"),
        how="Shapley location < half the difference, both exchanges"),
      _fn("premium_gap.json decomposition/ETH/*/by_quarter/2026Q1-Q2 location / difference",
          lambda: f"{min(_eth_location_shares()):.0f}%"),
      _fn("premium_gap.json decomposition/ETH/*/by_quarter/2026Q1-Q2 location / difference",
          lambda: f"{max(_eth_location_shares()):.0f}%")),
    C("gap", "In those two quarters the gap was {} to {} bps and the median premiums of all three "
      "venues lay {} the band",
      v(PG, *(DEC.format(x, f"all/gap_by_quarter/{q}") for x in ("binance", "bybit")
              for q in ("2026Q1", "2026Q2")), fmt="{:.1f}", op=min, how="min, 1 dp"),
      v(PG, *(DEC.format(x, f"all/gap_by_quarter/{q}") for x in ("binance", "bybit")
              for q in ("2026Q1", "2026Q2")), fmt="{:.1f}", op=max, how="max, 1 dp"),
      v(LH, "assets/BTC/quarters", op=lambda qs: "below" if all(
          qs[q][venue]["median_bps"] < load(LH)["band_bps"][0] for q in ("2026Q1", "2026Q2")
          for venue in ("hyperliquid", "binance", "bybit")) else "?",
        how="Hyperliquid's and the exchanges' medians below -4 bps")),
    C("gap", "in 2026Q3 location again accounts for {} of the difference ({} of {} points)",
      v(PG, DEC.format("binance", "by_quarter/2026Q3"),
        op=lambda r: "about all" if 0.9 <= r["shapley"]["location"] / r["difference"] <= 1.1
        else "?", how="location within 10% of the difference"),
      *(v(PG, DEC.format("binance", f"by_quarter/2026Q3/{k}"), fmt="{:.1f}", scale=100,
          how="percentage points, 1 dp") for k in ("shapley/location", "difference"))),
    C("gap", "Hyperliquid's BTC share is {}, against {} on Binance and {} on Bybit, with location "
      "{} in every order",
      pct(PG, DEC.format("binance", "positive/shares/a_hl_hourly"),
          DEC.format("bybit", "positive/shares/a_hl_hourly"), op=same("{}"),
          how="x100, 1 dp, same hours"),
      pct(PG, DEC.format("binance", "positive/shares/d_exchange_settlements")),
      pct(PG, DEC.format("bybit", "positive/shares/d_exchange_settlements")),
      v(PG, DEC.format("binance", "positive/shapley/range/location/1"),
        DEC.format("bybit", "positive/shapley/range/location/1"),
        op=lambda a, b: "negative" if max(a, b) < 0 else "?", how="largest effect < 0")),
    C("gap", "Quarter by quarter the reversal holds in {} of the {} pair-quarters, BTC against "
      "Bybit in 2023Q4 being the exception",
      _fn("premium_gap.json decomposition/*/*/by_quarter, positive-phase quarters",
          lambda: word(_reversal_count()[0])),
      _fn("premium_gap.json decomposition/*/*/by_quarter, positive-phase quarters",
          lambda: word(_reversal_count()[1]) if _reversal_count()[2] == ["BTC/bybit/2023Q4"]
          else "?")),
    C("gap", "the shares are closer, {} against {} on Binance and {} against {} on Bybit, but "
      "location's part differs: {} of {} points against Binance ({} to {} over the orders), {} of "
      "{} against Bybit, whose group adds {}, when its median lay {} bps inside the band",
      *(pct(PG, DEC.format(x, f"inside_nonpositive/shares/{s}")) for x in ("binance", "bybit")
        for s in ("a_hl_hourly", "d_exchange_settlements")),
      *(v(PG, DEC.format("binance", f"inside_nonpositive/{k}"), fmt="{:.1f}", scale=100,
          how="percentage points, 1 dp")
        for k in ("shapley/location", "difference", "shapley/range/location/0",
                  "shapley/range/location/1")),
      *(v(PG, DEC.format("bybit", f"inside_nonpositive/{k}"), fmt="{:.1f}", scale=100,
          how="percentage points, 1 dp") for k in ("shapley/location", "difference")),
      v(PG, DEC.format("bybit", "inside_nonpositive/quarters"),
        DEC.format("binance", "inside_nonpositive/quarters"),
        op=lambda a, b: same("{}")(*sorted(set(a) - set(b))), how="the quarter Bybit adds"),
      v(PG, DEC.format("bybit", "exchange_median_bps/2025Q3"), fmt="{:.2f}",
        op=lambda m: m - load(PG)["band_bps"][0], how="median minus the lower edge, 2 dp")),
    C("gap", "ETH shows the same reversal and closer shares, but there location exceeds the small "
      "difference, {} of {} and {} of {} points",
      *(v(PG, f"decomposition/ETH/{x}/inside_nonpositive/{k}", fmt="{:.1f}", scale=100,
          how="percentage points, 1 dp")
        for x in ("binance", "bybit") for k in ("shapley/location", "difference"))),
    C("gap", "the daily gap's slope on the daily mean of \\ln(USDC/USDT) (Binance spot, in bps) is "
      "{} for BTC against Binance (HAC s.e. {}) and {} to {} for the other pairs, far from -1, "
      "and the BTC--Binance slope vanishes within quarters ({}, s.e. {})",
      v(PG, "checks/BTC/binance/usdc_pricing/slope_daily", fmt="{:.3f}", how="3 dp"),
      v(PG, "checks/BTC/binance/usdc_pricing/slope_hac_se", fmt="{:.3f}", how="3 dp"),
      v(PG, "checks/BTC/bybit/usdc_pricing/slope_daily",
        "checks/ETH/binance/usdc_pricing/slope_daily",
        "checks/ETH/bybit/usdc_pricing/slope_daily", fmt="{:.2f}", op=min, how="min, 2 dp"),
      v(PG, "checks/BTC/bybit/usdc_pricing/slope_daily",
        "checks/ETH/binance/usdc_pricing/slope_daily",
        "checks/ETH/bybit/usdc_pricing/slope_daily", fmt="{:.2f}", op=max, how="max, 2 dp"),
      v(PG, "checks/BTC/binance/usdc_pricing/within_quarter/slope", fmt="{:.3f}", how="3 dp"),
      v(PG, "checks/BTC/binance/usdc_pricing/within_quarter/slope_hac_se", fmt="{:.3f}",
        how="3 dp")),
    C("gap", "between {} and {} (Table [tab:construction], note b); the BTC gap against Binance "
      "averaged {} bps less in the {} days from {} than in the {} days to {} ({} to {} bps less "
      "for the other pairs), but among all {} placements of two {}-day windows with {} days "
      "between them, the mean moves at least as much, up or down, in {} (a fall at least as "
      "large in {}; {} to {} for the other pairs)",
      v(PG, "oracle_wording/last_spot_prices", op=lambda s: day(s, year=False),
        how="last snapshot with the old wording"),
      v(PG, "oracle_wording/first_spot_mid_prices", op=day,
        how="first snapshot with the new wording"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/shift_bps", fmt="{:.1f}",
        op=lambda s: -s if s < 0 else float("nan"), how="minus the (negative) shift, 1 dp"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/window_days"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/first_after", op=lambda s: day(s, year=False),
        how="first day of the later window"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/window_days"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/last_before", op=lambda s: day(s, year=False),
        how="last day of the earlier window"),
      v(PG, *(f"{SHIFT.format(c, x)}/shift_bps" for c, x in OTHER_PAIRS), fmt="{:.1f}",
        op=lambda *s: -max(s) if max(s) < 0 else float("nan"), how="smallest fall, 1 dp"),
      v(PG, *(f"{SHIFT.format(c, x)}/shift_bps" for c, x in OTHER_PAIRS), fmt="{:.1f}",
        op=lambda *s: -min(s) if max(s) < 0 else float("nan"), how="largest fall, 1 dp"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/placements", fmt="{:,}"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/window_days"),
      v(PG, f"{SHIFT.format('BTC', 'binance')}/between_days"),
      pct(PG, f"{SHIFT.format('BTC', 'binance')}/share_as_large", dec=0,
          how="two-sided: |shift| at least as large, x100, 0 dp"),
      pct(PG, f"{SHIFT.format('BTC', 'binance')}/share_same_direction_as_large", dec=0,
          how="one-sided: a fall at least as large, x100, 0 dp"),
      pct(PG, *(f"{SHIFT.format(c, x)}/share_as_large" for c, x in OTHER_PAIRS), dec=0, op=min,
          how="two-sided, min over the other pairs, x100, 0 dp"),
      pct(PG, *(f"{SHIFT.format(c, x)}/share_as_large" for c, x in OTHER_PAIRS), dec=0, op=max,
          how="two-sided, max over the other pairs, x100, 0 dp")),

    # ---- Section 6.1 ------------------------------------------------------
    C("regimes", "with a variance below 10^{{-31}} in each of {} estimations",
      v(RATE, "fits", op=lambda f: word(len(f)) if max(x["sigma2"][0] for x in f) < 1e-31
        else "?", how="fits whose collapsed variance is below 1e-31")),
    C("regimes", "from {} estimations of {} random initialisations each",
      v(DG, PAPER_MSAR + "/multistart/n_starts"), v(DG, PAPER_MSAR + "/multistart/search_reps")),
    C("regimes", "the premium halves in {} hours when calm and {} hours under stress",
      v(DG, PAPER_MSAR + "/half_life_hours/0", fmt="{:.1f}", how="1 dp"),
      v(DG, PAPER_MSAR + "/half_life_hours/1", fmt="{:.1f}", how="1 dp")),
    C("regimes", "a ratio R = \\taucalm/\\taustr of {}. But the stressed regime lasts {} hours on "
      "average",
      v(DG, PAPER_MSAR + "/half_life_ratio", fmt="{:.1f}", how="1 dp"),
      v(DG, PAPER_MSAR + "/expected_duration_hours/1", fmt="{:.1f}", how="1 dp")),
    C("regimes", "a dislocation halves after {} hours when it arises under stress and {} hours "
      "when it arises in the calm regime, a switching-adjusted ratio \\Rsw of {}",
      v(DG, PAPER_MSAR + "/half_life_switching_interp_hours/1", fmt="{:.1f}", how="1 dp"),
      v(DG, PAPER_MSAR + "/half_life_switching_interp_hours/0", fmt="{:.1f}", how="1 dp"),
      _two_dec(DG, PAPER_MSAR + "/ratio_switching_interp")),
    C("regimes", "A single-regime AR(1) on the same premium gives a half-life of {} hours",
      v(PG, "hyperliquid_premium_ar1/BTC/paper", fmt="{:.1f}",
        op=lambda r: r["half_life"] if r["n"] == load(CC)["venues"]["hyperliquid"]["paper"]["n"] - 1
        else float("nan"), how="AR(1) on the estimation-window premium, 1 dp")),
    C("regimes", "Whole series: the estimation and later windows together, 2 October 2025 to 30 "
      "September 2026 (T = {})",
      v(OOS, "extended_window/T", "paper_window/T", "injection/0/T", fmt="{:,}",
        op=lambda t, a, b: t if t == a + b else float("nan"),
        how="T of the whole series = estimation + later window")),
    C("regimes", "with an excess kurtosis of {}",
      v(DG, "tests/4/jarque_bera/excess_kurtosis", fmt="{:.1f}", how="1 dp")),
    C("regimes", "significant with the Hessian (z = {}, p < 0.001) but not with the sandwich "
      "(z = {}, p = {})",
      v(DG, PAPER_MSAR + "/inference/hessian/ratio_wald", fmt="{:.2f}",
        op=lambda w: w["z"] if w["p"] < 0.001 else float("nan"), how="z, 2 dp; p < 0.001"),
      _two_dec(DG, PAPER_MSAR + "/inference/robust/ratio_wald/z"),
      _two_dec(DG, PAPER_MSAR + "/inference/robust/ratio_wald/p")),
    C("regimes", "and {} of the sandwich variance of R comes from one hour: the record stamped {} "
      "UTC, whose premium fell back to {} bps from the series' maximum of {} an hour earlier",
      pct(DG, PAPER_MSAR + "/ratio_sandwich_top_hours/0/share"),
      _fn("diagnostics.json ratio_sandwich_top_hours/0/index + paper-window start",
          _sandwich_hour),
      _two_dec(DG, PAPER_MSAR + "/ratio_sandwich_top_hours/0/y"),
      v(DG, PAPER_MSAR + "/ratio_sandwich_top_hours/0/y_prev", fmt="{:+.2f}", how="2 dp")),
    C("regimes", "Replacing the maximum by the next largest value, {} bps, lowers R to {} "
      "(Hessian p = {}, sandwich p = {}); deleting it leaves {}, significant under {}; "
      "winsorising the upper 1%, which caps the maximum and {} further hours, {} of them from "
      "October 2025, at {} bps, leaves {}, significant under {}",
      _fn("data/interim/premium_BTC_full.csv via panel.load_paper_window_premium(), 2nd largest",
          _next_largest_premium, fmt="{:.2f}"),
      _two_dec(OOS, "sensitivity/1/ratio"),
      v(OOS, "sensitivity/1/inference/hessian/ratio_wald/p", fmt="{:.3f}", how="3 dp"),
      _two_dec(OOS, "sensitivity/1/inference/robust/ratio_wald/p"),
      _two_dec(OOS, "sensitivity/2/ratio"),
      v(OOS, "sensitivity/2/inference", op=lambda i: "neither" if all(
          i[c]["ratio_wald"]["significance"] == "ns" for c in ("hessian", "robust")) else "?",
        how="Wald ns under Hessian and sandwich"),
      v(OOS, "upper_tail", op=lambda u: u["n_capped"] - 1 if u["month"] == "2025-10"
        and u["maximum_in_month"] else "?", how="capped hours besides the maximum"),
      v(OOS, "upper_tail", op=lambda u: u["n_capped_in_month"] - 1 if u["maximum_in_month"]
        else "?", how="capped October 2025 hours besides the maximum"),
      _two_dec(OOS, "upper_tail/cap_bps"),
      _two_dec(OOS, "sensitivity/3/ratio"),
      v(OOS, "sensitivity/3/inference", op=lambda i: "both" if all(
          i[c]["ratio_wald"]["significance"] == "above" for c in ("hessian", "robust")) else "?",
        how="Wald above under Hessian and sandwich")),
    C("regimes", "capping only October's {} upper-tail hours leaves {} (sandwich p = {}), while "
      "from {} on the winsorised series shows no asymmetry (R = {}; p = {} and {})",
      v(OOS, "upper_tail/n_capped_in_month"),
      _two_dec(OOS, "upper_tail/october_tail_capped/ratio"),
      v(OOS, "upper_tail/october_tail_capped/inference/robust/ratio_wald", fmt="{:.3f}",
        op=lambda w: w["p"] if w["significance"] == "above" else float("nan"),
        how="3 dp; above unity"),
      v(OOS, "upper_tail/from_stamp", op=lambda s: day(s, year=False),
        how="first stamp of the series"),
      _two_dec(OOS, "upper_tail/winsorised_from_11_oct/ratio"),
      *(v(OOS, f"upper_tail/winsorised_from_11_oct/inference/{c}/ratio_wald", fmt="{:.2f}",
          op=lambda w: w["p"] if w["significance"] == "ns" else float("nan"),
          how="2 dp; not significant") for c in ("hessian", "robust"))),
    C("regimes", "On the hourly sub-period, which lacks October, R is {}, below unity with the "
      "Hessian (p {}) but not with the sandwich (p = {}). On the five months after the window it "
      "is {}, below unity under {} covariances (\\Rsw = {}); one hour of that later window set "
      "to {} bps raises it to {}, {} or {}",
      _two_dec(DG, SUB_MSAR + "/half_life_ratio"),
      v(DG, SUB_MSAR + "/inference/hessian/ratio_wald",
        op=lambda w: "< 0.001" if w["significance"] == "below" and w["p"] < 0.001 else "?",
        how="Wald below unity, p < 0.001"),
      v(DG, SUB_MSAR + "/inference/robust/ratio_wald", fmt="{:.3f}",
        op=lambda w: w["p"] if w["significance"] == "ns" else float("nan"),
        how="3 dp; not significant"),
      _two_dec(OOS, "injection/0/ratio"),
      v(OOS, "injection/0/inference", op=lambda i: "both" if all(
          i[c]["ratio_wald"]["significance"] == "below" for c in ("hessian", "robust")) else "?",
        how="Wald below under Hessian and sandwich"),
      _two_dec(OOS, "injection/0/ratio_switching_interp"),
      v(DG, PAPER_MSAR + "/ratio_sandwich_top_hours/0/y_prev", fmt="{:+.2f}",
        how="the October maximum, 2 dp"),
      *(v(OOS, "injection", fmt="{:.1f}", op=lambda inj, k=k: sorted(
          r["ratio"] for r in inj[1:])[k], how=f"injected fits, {k + 1}. smallest, 1 dp")
        for k in range(3))),
    C("regimes", "The second starts on {}, after the hour, and still lies above unity with the "
      "Hessian (R = {}, p = {}), though not with the sandwich (p = {})",
      v(OOS, "rolling/1/start", op=lambda s: day(s, year=False)
        if _hour(s) > _hour(_crash_record()) else "?", how="start of the second window"),
      _two_dec(OOS, "rolling/1/ratio"),
      v(OOS, "rolling/1/inference/hessian/ratio_wald", fmt="{:.3f}",
        op=lambda w: w["p"] if w["significance"] == "above" else float("nan"),
        how="3 dp; above unity"),
      v(OOS, "rolling/1/inference/robust/ratio_wald", fmt="{:.3f}",
        op=lambda w: w["p"] if w["significance"] == "ns" else float("nan"),
        how="3 dp; not significant")),
    C("regimes", "Of {} rolling windows of {} hours",
      v(OOS, "rolling", op=len), v(OOS, "protocol/window", fmt="{:,}")),
    C("regimes", "the {} earliest lie above unity with the Hessian and {} with the sandwich, while "
      "{} lie below it under both",
      _fn("oos_robustness.json rolling/*/inference/*/ratio_wald",
          lambda: word(len(_rolling_sides()[0])) if _rolling_sides()[0] == {0, 1} else "?"),
      _fn("oos_robustness.json rolling/*/inference/*/ratio_wald",
          lambda: "none" if not _rolling_sides()[1] else "?"),
      _fn("oos_robustness.json rolling/*/inference/*/ratio_wald",
          lambda: word(len(_rolling_sides()[2])))),
    C("regimes", "Each of our {} estimations on the estimation window reaches one of {} maxima; "
      "the inferior one, {} log-likelihood points lower, attracts {} to {} of every {} "
      "estimations and implies R = {}",
      v(SEED, "runs", op=lambda r: sum(sum(o["count"] for o in b["optima_found"]) for b in r),
        how="estimations over the six seed blocks"),
      v(SEED, "runs", op=lambda r: word(same("{}")(*[len(b["optima_found"]) for b in r])),
        how="optima per block"),
      _two_dec(OPT, "logL_gap"),
      v(SEED, "starts_at_inferior_optimum/min"), v(SEED, "starts_at_inferior_optimum/max"),
      v(SEED, "protocol/n_starts"),
      v(OPT, "optima", fmt="{:.1f}", op=lambda o: min(o, key=lambda x: x["llf"])["ratio"],
        how="ratio at the lower optimum, 1 dp")),
    C("regimes", "over the {} fits of Tables [tab:rolling] and [tab:sensitivity] and both "
      "covariances, the log-scale delta interval for R "
      "contradicts the Wald test in {} of {} cases, the linear one in {} and Fieller's set in {}",
      _fn("diagnostics.json + oos_robustness.json, the 24 fits", lambda: _interval_errors()[0]),
      _fn("diagnostics.json + oos_robustness.json, ratio_ci95/log vs ratio_wald",
          lambda: _interval_errors()[2]["log"]),
      _fn("diagnostics.json + oos_robustness.json, fit-covariance cases",
          lambda: _interval_errors()[1]),
      _fn("diagnostics.json + oos_robustness.json, ratio_ci95/linear vs ratio_wald",
          lambda: _interval_errors()[2]["linear"]),
      _fn("diagnostics.json + oos_robustness.json, ratio_ci95/fieller vs ratio_wald",
          lambda: _interval_errors()[2]["fieller"])),

    # ---- Section 6.2 ------------------------------------------------------
    C("regimes", "separates the estimation window into {}, {} and {} annualised volatility, with "
      "stationary masses of {}, {} and {} and expected durations of {}, {} and {} hours",
      *(v(BOOT, f"{FULL_SIG}/annualised_sigma_pct/{k}/point", fmt="{:.1f}%", how="1 dp")
        for k in range(3)),
      *(pct(BOOT, f"{FULL_SIG}/stationary/{k}/point", dec=0) for k in range(3)),
      *(v(BOOT, f"{FULL_SIG}/expected_duration_hours/{k}/point", fmt="{:.1f}", how="1 dp")
        for k in range(3))),
    C("regimes", "stopping at 150 iterations would change no annualised volatility by more than "
      "{} percentage points. The converged fit gives {}, {} and {}, and clipping the extreme {} "
      "of returns, {} hours, lowers the high state to {}",
      _two_dec(A31, "convergence/sub_period_cap_effect/max_abs_change_annualised_sigma_pct"),
      *(v(SUM, f"return_K3/annualised_vol_pct/{k}", fmt="{:.1f}%", how="1 dp")
        for k in range(3)),
      v(ROB, "winsor_pct_clipped", fmt="{:.1f}%", how="1 dp"),
      v(ROB, "winsor_n_clipped", op=word, how="count in words"),
      v(ROB, "winsorised/ann_vol_pct_unrounded/2", fmt="{:.1f}%", how="1 dp")),
    C("regimes", "Two-hourly candles over the whole estimation window give {}, {} and {}, with a "
      "high-to-low cell of {} per two-hour bar, as expected when a bar can contain a passage "
      "through the moderate state (the reconstructed hourly chain of Figure [fig:transitions] "
      "gives {} over two hours)",
      *(v(TWO, f"annualised_sigma_pct/{k}", fmt="{:.1f}%", how="1 dp") for k in range(3)),
      pct(TWO, "A/2/0", how="A[3,1] of the two-hourly kernel, x100, 1 dp"),
      _fn("bootstrap_hmm.json full_sample via paper_figures.transition_geometry(), (A^2)[3,1]",
          _a31_two_hours, fmt="{:.1f}%")),
    C("regimes", "(\\hat A_{{32}} = {}; Figure",
      _fn("bootstrap_hmm.json full_sample via paper_figures.transition_geometry(), A[3,2]",
          _a32, fmt="{:.2f}")),
    C("regimes", "the profile likelihood bounds A_{{31}} below {} per hour (one-sided 95%; {} "
      "two-sided), a bound that is conservative within the model ({} rejections at a nominal 5%)",
      v(A31, "profile_likelihood/upper_bound_one_sided_95_pct_per_hour", fmt="{:.2f}%",
        how="2 dp"),
      v(A31, "profile_likelihood/upper_bound_two_sided_95_pct_per_hour", fmt="{:.2f}%",
        how="2 dp"),
      pct(A31, "profile_likelihood/calibration_at_bound/rejection_rate/share")),
    C("regimes", "with A_{{31}} = {} per hour in the estimation-window chain, {} of {} simulated "
      "samples still return exactly zero, and {} at {}",
      pct(A31, "power/full_sample/0.003/alternative_A31"),
      pct(A31, "power/full_sample/0.003/share_mle_zero/share", dec=0),
      v(A31, "power/full_sample/0.003/n_reps", fmt="{:,}"),
      pct(A31, "power/full_sample/0.01/share_mle_zero/share", dec=0),
      pct(A31, "power/full_sample/0.01/alternative_A31", dec=0)),
    C("regimes", "fitted to {} paths of the Student-t GARCH(1,1) of Table [tab:benchmarks], a "
      "single-regime model, the three-state model returns \\hat A_{{31}} = 0 on {}, and on {} of "
      "Gaussian-GARCH paths at the best of {} starts per path ({} at the eight starts)",
      v(A31, "garch_generic/garch_t_fitted/n_paths"),
      v(A31, "garch_generic/garch_t_fitted/share_mle_zero/share",
        op=lambda s: "every one" if s == 1.0 else "?", how="share = 1"),
      pct(A31, "garch_generic/garch_normal_fitted/wide_multistart/share_mle_zero_best_optimum/"
               "share"),
      v(A31, "garch_generic/garch_normal_fitted/wide_multistart/n_starts_per_path"),
      pct(A31, "garch_generic/garch_normal_fitted/wide_multistart/share_mle_zero_8_starts/share",
          "garch_generic/garch_normal_fitted/share_mle_zero/share", op=same("{:.6f}"),
          how="the eight-start share, x100, 1 dp")),
    C("regimes", "with every fit converged from about {} starts per K, BIC prefers K=4 to K=3 by "
      "{} points, while K=3 gains {} on K=2",
      v(MS, "records", op=lambda r: 200 if all(195 <= x["n_starts"] <= 215 and x["converged"]
                                               for x in r) else "?",
        how="starts per K, rounded"),
      v(MS, "records", fmt="{:.1f}", op=lambda r: r[1]["BIC"] - r[2]["BIC"],
        how="BIC(K=3) - BIC(K=4), 1 dp"),
      v(MS, "records", fmt="{:.1f}", op=lambda r: r[0]["BIC"] - r[1]["BIC"],
        how="BIC(K=2) - BIC(K=3), 1 dp")),
    C("regimes", "A Student-t GARCH(1,1) beats the three-state model on BIC with {} parameters "
      "against {}",
      v(BEN, "rows/0", op=lambda r: word(r["k"]) if r["model"] == "GARCH(1,1)-Student-t"
        and load(BEN)["best_by_BIC"] == r["model"] else "?", how="parameters, BIC winner"),
      v(BEN, "rows/1", op=lambda r: word(r["k"]) if r["model"] == "Gaussian HMM, K=3" else "?",
        how="parameters, K = 3 HMM")),

    # ---- Sections 7 and 8 -------------------------------------------------
    C("implications", "the stressed premium regime lasts {} hours on average, and allowing for "
      "switching a dislocation halves in {} to {} hours whichever regime it starts in",
      v(DG, PAPER_MSAR + "/expected_duration_hours/1", fmt="{:.1f}", how="1 dp"),
      v(DG, PAPER_MSAR + "/half_life_switching_interp_hours", fmt="{:.0f}", op=min,
        how="0 dp"),
      v(DG, PAPER_MSAR + "/half_life_switching_interp_hours", fmt="{:.0f}", op=max,
        how="0 dp")),
    C("implications", "sets a series censored about half the time against series whose censored "
      "share depends on the phase: {} to {} for BTC over the estimation window, {} since December "
      "2023, and in the first half of 2024 {} Hyperliquid's or higher",
      pct(CC, BN_P + "/share_at_floor", dec=0), pct(CC, BY_P + "/share_at_floor", dec=0),
      v(LH, "assets", op=lambda a: "about a quarter" if all(
          0.2 < a[c]["windows"]["common_rule"][x]["share_at_floor"] < 0.3 for c in ("BTC", "ETH")
          for x in ("binance", "bybit")) else "?", how="exchange floor shares 20-30%, both assets"),
      v(LH, "assets", op=lambda a: "about as high as" if all(
          a[c]["half_years"]["2024H1"][x]["share_at_floor"]
          > a[c]["half_years"]["2024H1"]["hyperliquid"]["share_at_floor"] - 0.02
          for c in ("BTC", "ETH") for x in ("binance", "bybit")) and all(
          max(a[c]["half_years"]["2024H1"][x]["share_at_floor"] for x in ("binance", "bybit"))
          > a[c]["half_years"]["2024H1"]["hyperliquid"]["share_at_floor"] for c in ("BTC", "ETH"))
        else "?", how="2024H1: each exchange within 2 points of Hyperliquid, one above it")),
    C("implications", "otherwise only in isolated months (Hyperliquid below both exchanges in {} "
      "and, for BTC, {}) and against Binance over {}",
      _fn("censoring_long_history.json months_hl_below_both outside the positive quarters",
          lambda: month(same("{}")(*_isolated_months()["ETH"])) if _isolated_months()["BTC"]
          [:1] == _isolated_months()["ETH"] else "?"),
      _fn("censoring_long_history.json months_hl_below_both outside the positive quarters",
          lambda: month(same("{}")(*_isolated_months()["BTC"][1:]))),
      _fn("premium_gap.json decomposition/*/binance/by_quarter outside the positive quarters",
          lambda: same("{}")(*_quarters_below_binance()))),
    C("conclusion", "Hyperliquid's BTC funding sat there in {} of hours over the estimation "
      "window and {} since December 2023, against {} to {} and {} to {} of the exchanges' "
      "settlements",
      pct(CC, HL_P + "/share_at_floor", dec=0), _lhw_0("BTC", "hyperliquid"),
      pct(CC, BN_P + "/share_at_floor", dec=0), pct(CC, BY_P + "/share_at_floor", dec=0),
      _lhw_0("BTC", "binance"), _lhw_0("BTC", "bybit")),
    C("conclusion", "Hyperliquid's sat a median {} bps above the exchanges' for BTC, persistent in "
      "sign, {} of 2026 and {} in its third quarter",
      v(PG, GAP.format("BTC", "binance") + "/common/median_bps",
        GAP.format("BTC", "bybit") + "/common/median_bps", fmt="{:.1f}", op=same("{:.1f}"),
        how="1 dp, both exchanges"),
      v(PG, "gap/BTC", op=lambda g: "smallest in the first half" if all(
          min(g[x]["by_quarter"], key=lambda q: g[x]["by_quarter"][q]["median_bps"])
          in ("2026Q1", "2026Q2") for x in ("binance", "bybit")) else "?",
        how="the smallest quarterly median falls in 2026Q1 or 2026Q2, both exchanges"),
      v(PG, "gap/BTC", op=lambda g: "wider again" if all(
          g[x]["by_quarter"]["2026Q3"]["median_bps"] > max(g[x]["by_quarter"][q]["median_bps"]
                                                           for q in ("2026Q1", "2026Q2"))
          for x in ("binance", "bybit")) else "?", how="2026Q3 above 2026Q1-Q2, both exchanges")),
    C("conclusion", "on the estimation window it is significant with the {} but not with the {} "
      "covariance, and one hour of the 10 October crash sets most of its size, the ratio falling "
      "from {} to {} without that hour and to {} with the hour set to the next largest value. "
      "With the premium's upper 1% winsorised it is significant under {} covariances (sandwich "
      "p = {}), and the first rolling window to start after the crash, on {}, still finds it "
      "with the Hessian (p = {}); later data {} it",
      v(DG, PAPER_MSAR + "/inference/hessian/ratio_wald/significance",
        op=lambda s: "Hessian" if s == "above" else "?", how="Wald above unity, Hessian"),
      v(DG, PAPER_MSAR + "/inference/robust/ratio_wald/significance",
        op=lambda s: "sandwich" if s == "ns" else "?", how="Wald not significant, sandwich"),
      v(DG, PAPER_MSAR + "/half_life_ratio", fmt="{:.1f}", how="1 dp"),
      v(OOS, "sensitivity/2/ratio", fmt="{:.1f}", how="maximum deleted, 1 dp"),
      v(OOS, "sensitivity/1/ratio", fmt="{:.1f}", how="maximum replaced, 1 dp"),
      v(OOS, "sensitivity/3/inference", op=lambda i: "both" if all(
          i[c]["ratio_wald"]["significance"] == "above" for c in ("hessian", "robust")) else "?",
        how="Wald above under Hessian and sandwich"),
      v(OOS, "sensitivity/3/inference/robust/ratio_wald/p", fmt="{:.3f}", how="3 dp"),
      v(OOS, "rolling/1/start", op=lambda s: day(s, year=False),
        how="start of the second rolling window, the first after the crash"),
      v(OOS, "rolling/1", fmt="{:.3f}", op=lambda r: r["inference"]["hessian"]["ratio_wald"]["p"]
        if r["inference"]["hessian"]["ratio_wald"]["significance"] == "above"
        and _hour(r["start"]) > _hour(_crash_record()) else float("nan"),
        how="second rolling window, Hessian Wald p, 3 dp"),
      v(OOS, "injection/0/inference", op=lambda i: "reverse" if all(
          i[c]["ratio_wald"]["significance"] == "below" for c in ("hessian", "robust")) else "?",
        how="later window below unity under both covariances")),

    # ---- captions and appendices -----------------------------------------
    C("regimes", "Parenthesised figures are parametric-bootstrap standard errors from {} "
      "replications", v(BOOT, "full_sample/n_reps")),
    C("regimes", "Each bar is a block of {} consecutive seeds, the first starting at seed {}",
      v(SEED, "protocol/n_starts"), v(SEED, "runs/0/seed_base")),
    C("appendix", "the labels would switch in {} of bootstrap replications",
      pct(BOOT, "full_sample/label_switching/disagreement_rate")),
    C("appendix", "would be [{}, {}] instead of [{}, {}]",
      *(v(BOOT, f"full_sample/mu_ascending/sigma_bps_per_hour/1/ci95/{i}", fmt="{:.1f}",
          how="1 dp") for i in range(2)),
      *(v(BOOT, f"{FULL_SIG}/sigma_bps_per_hour/0/ci95/{i}", fmt="{:.1f}", how="1 dp")
        for i in range(2))),
    C("appendix", "from {} estimations (seeds {} to {}) of {} random initialisations each",
      v(DG, PAPER_MSAR + "/multistart/n_starts"), v(DG, PAPER_MSAR + "/multistart/seed_base"),
      v(DG, PAPER_MSAR + "/multistart", op=lambda m: m["seed_base"] + m["n_starts"] - 1,
        how="last seed"),
      v(DG, PAPER_MSAR + "/multistart/search_reps")),
    C("appendix", "Its simulated interval draws {} parameter vectors",
      v(DG, PAPER_MSAR + "/inference/robust/ratio_switching_sim/draws", fmt="{:,}")),
    C("appendix", "An hour enters D_t only when both venues have it: {} Binance hour lacks "
      "minutes, which leaves {} hours against Binance and {} against Bybit",
      v(PG, "data/BTC/binance", "data/ETH/binance",
        op=lambda a, b: word(same("{}")(*[r["hours"] - r["complete_hours"] for r in (a, b)])),
        how="exchange hours without all 60 minutes, both assets"),
      v(PG, "windows/BTC/binance/common/exchange_hours",
        "windows/ETH/binance/common/exchange_hours", fmt="{:,}", op=same("{}"),
        how="both assets"),
      v(PG, "windows/BTC/bybit/common/exchange_hours", "windows/ETH/bybit/common/exchange_hours",
        fmt="{:,}", op=same("{}"), how="both assets")),
    C("appendix", "moving-block bootstrap with {}-hour blocks and {} replicates; {}- and {}-hour "
      "blocks",
      v(PG, "bootstrap/block_hours"), v(PG, "bootstrap/reps", fmt="{:,}"),
      v(PG, "bootstrap/sensitivity_block_hours/0"), v(PG, "bootstrap/sensitivity_block_hours/1")),
    C("appendix", "windows of {} hours stepped by {}, {} estimations each, at the window "
      "mid-points; estimations that fail numerically ({} to {} of the {} per window) are dropped, "
      "and the reported optimum is reached by {} to {} of the rest ({} of {} in the window "
      "centred on {}, whose most frequent optimum lies {} log-likelihood points lower)",
      v(OOS, "protocol/window", fmt="{:,}"), v(OOS, "protocol/step"),
      v(OOS, "protocol/n_starts"),
      v(OOS, "rolling", op=lambda r: min(x["failures"] for x in r), how="fewest failures"),
      v(OOS, "rolling", op=lambda r: max(x["failures"] for x in r), how="most failures"),
      v(OOS, "protocol/n_starts"),
      _fn("oos_robustness.json rolling/*/optima_found", lambda: min(_reached())[0]),
      _fn("oos_robustness.json rolling/*/optima_found", lambda: max(_reached())[0]),
      _fn("oos_robustness.json rolling/*/optima_found", lambda: min(_reached())[0]),
      _fn("oos_robustness.json rolling/*/optima_found", lambda: min(_reached())[1]),
      _fn("oos_robustness.json rolling/*/mid", lambda: day(min(_reached())[2])),
      _fn("oos_robustness.json rolling/*/optima_found", lambda: f"{min(_reached())[3]:.2f}")),
    C("appendix", "in the fit of Table [tab:sensitivity] with one hour set to the maximum at {}, "
      "{} of the {} fail numerically, {} more end at the variance floor or with a singular "
      "Hessian, and the reported optimum is reached by {} of the {} admissible ones",
      _fn("oos_robustness.json injection/* via msar_tables.injection_dates()",
          lambda: day(f"{_worst_injection()[0]:%Y-%m-%d}")),
      _fn("oos_robustness.json injection/*/failures", lambda: _worst_injection()[1]["failures"]),
      v(OOS, "protocol/n_starts"),
      _fn("oos_robustness.json injection/*/excluded",
          lambda: _worst_injection()[1]["excluded"]["variance_floor"]
          + _worst_injection()[1]["excluded"]["singular_hessian"]),
      _fn("oos_robustness.json injection/*/optima_found",
          lambda: _optimum_count(_worst_injection()[1])),
      _fn("oos_robustness.json injection/*: estimations neither failed nor excluded",
          lambda: _admissible(_worst_injection()[1]))),
    C("appendix", "The Student-t GARCH, best on BIC, is near-integrated (\\hat\\alpha + "
      "\\hat\\beta = {} against an imposed bound of {}) with \\hat\\nu = {}",
      v(BEN, "rows/0/persistence", fmt="{:.5f}", how="5 dp"),
      _fn("benchmarks.json rows/0/caveat: the bound named there", _garch_bound),
      _two_dec(BEN, "rows/0/params/nu")),
]



@pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
def test_the_text_prints_what_the_artifacts_hold(claim):
    expected = claim.expected()
    assert expected in parts()[claim.where], expected


def test_every_part_a_claim_names_exists():
    assert {c.where for c in CLAIMS} <= set(parts())


def test_the_msar_blocks_are_the_windows_the_claims_name():
    tests = load(DG)["tests"]
    assert "paper window" in get(tests, "4/msar/window")
    assert "sub-sample" in get(tests, "3/msar/window")
