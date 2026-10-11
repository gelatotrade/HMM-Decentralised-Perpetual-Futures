"""Out-of-sample robustness for the funding half-life asymmetry.

Section 6.1 of the paper reports that the calm-to-stress half-life ratio is not
a parameter of the premium series but a property of one episode. Two exercises
support that, and both live here.

The side of unity is decided by the Wald test of phi_calm = phi_stressed
(``msar.ratio_wald_test``): R = 1 exactly when the two coefficients are equal,
so the test needs no delta step. The delta and Fieller intervals in each record
are descriptive.

**Rolling windows.** Re-estimating the MS-AR on overlapping windows shows where
the asymmetry lives. Under the Hessian covariance only the first two windows,
the one containing the 10 October 2025 premium maximum and the one after it,
put the ratio above unity, and most later windows put it below. Under the
sandwich covariance no window puts it above unity.

**Single-observation sensitivity.** Replacing that one hour with the second
largest value of the series moves the full-window estimate from 11.2 to 3.0.
The magnitude of the headline number is carried by a single observation out of
5,001; under the sandwich covariance neither estimate is significantly above
unity.

Labelling
---------
Regime 0 is always the *calm* regime and regime 1 the *stressed* one, ordered by
``sigma2`` ascending -- the convention of :mod:`regime_engine.msar`. Ordering by
half-life instead would assume the very ranking these functions are meant to
test, and would make every window agree with the paper by construction.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

__all__ = [
    "injection_variants",
    "json_safe_optima",
    "VARIANT_SHORT",
    "write_rolling_table",
    "write_sensitivity_table",
    "label_regimes",
    "rolling_windows",
    "sensitivity_variants",
    "span_label",
    "upper_tail_variants",
    "VARIANT_LABELS",
]

#: Human-readable descriptions, used for the artifact and the paper's table.
VARIANT_LABELS = {
    "baseline": "unchanged",
    "replace_max": "premium maximum replaced by the second largest value",
    "drop_max": "premium maximum deleted",
    "winsorise_top1": "upper 1 per cent winsorised",
}

#: Table cells. The full sentence above stays in the artifact and the caption;
#: a column wide enough for it would push the interval off the page.
VARIANT_SHORT = {
    "baseline": "Unchanged",
    "replace_max": "Maximum $\\to$ 2nd largest",
    "drop_max": "Maximum deleted",
    "winsorise_top1": "Winsorised, upper 1\\%",
}


def rolling_windows(n: int, window: int, step: int) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index pairs for overlapping windows.

    A window is emitted only if it fits entirely inside ``n``; the tail that
    cannot fill one is dropped rather than silently shortened, so every
    estimate in the series is computed on the same number of observations.
    """
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    if n < window:
        return []
    return [(s, s + window) for s in range(0, n - window + 1, step)]


def span_label(times) -> str:
    """'2 Oct 2025 - 30 Sep 2026'. A last stamp at midnight closes the day before."""
    import pandas as pd

    first, last = times[0], times[-1] - pd.Timedelta(minutes=1)
    return f"{first.day} {first:%b %Y} - {last.day} {last:%b %Y}"


def sensitivity_variants(y: np.ndarray) -> dict[str, np.ndarray]:
    """The baseline series and three perturbations of its upper tail.

    Every value returned is a fresh array. The baseline is refitted alongside
    the variants, so a variant editing ``y`` in place would quietly turn the
    comparison into a comparison of a series with itself.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        raise ValueError("need at least three observations")

    i = int(np.argmax(y))
    second = float(np.sort(y)[-2])

    replace_max = y.copy()
    replace_max[i] = second

    cap = float(np.quantile(y, 0.99))
    winsorised = np.minimum(y.copy(), cap)

    return {
        "baseline": y.copy(),
        "replace_max": replace_max,
        "drop_max": np.delete(y.copy(), i),
        "winsorise_top1": winsorised,
    }


#: The two upper-tail variants that separate the October 2025 hour from the rest
#: of the upper 1%: the month whose upper-tail hours alone are capped, and the
#: first stamp of the winsorised series' restriction to after the crash.
UPPER_TAIL_MONTH = "2025-10"
UPPER_TAIL_FROM = "2025-10-11 00:00"

UPPER_TAIL_LABELS = {
    "october_tail_capped": "upper 1 per cent winsorised in October 2025 only",
    "winsorised_from_11_oct": "upper 1 per cent winsorised, from 11 October 2025",
}


def upper_tail_variants(y, times) -> tuple[dict[str, np.ndarray], dict]:
    """Where the winsorised series' asymmetry comes from, and what was capped.

    The cap is the series' 99th percentile, as in :func:`sensitivity_variants`.
    ``october_tail_capped`` caps only the hours above it whose premium accrued in
    :data:`UPPER_TAIL_MONTH` (stamp minus one hour), the maximum among them;
    ``winsorised_from_11_oct`` is the winsorised series from the stamp
    :data:`UPPER_TAIL_FROM` on. The facts record the cap and how many hours it
    sets, in that month and outside it.
    """
    import pandas as pd

    y = np.asarray(y, dtype=float)
    times = pd.DatetimeIndex(times)
    if len(times) != y.size:
        raise ValueError("y and times differ in length")
    cap = float(np.quantile(y, 0.99))
    over = y > cap
    month = np.asarray((times - pd.Timedelta(hours=1)).strftime("%Y-%m") == UPPER_TAIL_MONTH)
    after = np.asarray(times >= pd.Timestamp(UPPER_TAIL_FROM, tz="UTC"))
    winsorised = np.minimum(y, cap)
    series = {"october_tail_capped": np.where(over & month, cap, y),
              "winsorised_from_11_oct": winsorised[after]}
    facts = {"cap_bps": cap, "n_capped": int(over.sum()),
             "n_capped_in_month": int((over & month).sum()),
             "n_capped_outside_month": int((over & ~month).sum()),
             "maximum_in_month": bool(month[int(np.argmax(y))]),
             "month": UPPER_TAIL_MONTH, "from_stamp": UPPER_TAIL_FROM,
             "T_from": int(after.sum())}
    return series, facts


def injection_variants(y, value: float, positions) -> dict[str, np.ndarray]:
    """Copies of ``y`` with the observation at each position overwritten by ``value``.

    The constructive counterpart to :func:`sensitivity_variants`: if one hour can
    destroy the asymmetry, one hour should also be able to create it. Placing the
    October-sized excursion at several points in a window that shows the reversed
    ordering tests exactly that, and the position is varied so the result cannot
    be an artifact of where in the sample the spike lands.
    """
    y = np.asarray(y, dtype=float)
    out = {}
    for p in positions:
        p = int(p)
        if not 0 <= p < y.size:
            raise ValueError(f"position {p} is outside a series of length {y.size}")
        z = y.copy()
        z[p] = float(value)
        out[f"inject@{p}"] = z
    return out


def label_regimes(sigma2, half_lives) -> dict:
    """Assign the calm/stressed labels by ``sigma2`` ascending.

    ``sigma2`` and ``half_lives`` are the two regimes' conditional variances and
    implied half-lives, in the estimator's own regime order. The calm regime is
    the one with the smaller conditional variance, whatever its persistence --
    that independence is the whole point, because the ratio of the two
    half-lives is the quantity under test.

    Returns ``ratio = None`` when the stressed half-life is zero or non-finite,
    rather than propagating an infinity into the artifact.
    """
    sigma2 = np.asarray(sigma2, dtype=float)
    half_lives = np.asarray(half_lives, dtype=float)
    if sigma2.shape != (2,) or half_lives.shape != (2,):
        raise ValueError("label_regimes expects exactly two regimes")

    calm, stressed = int(np.argmin(sigma2)), int(np.argmax(sigma2))
    tau_c, tau_s = float(half_lives[calm]), float(half_lives[stressed])
    ratio = tau_c / tau_s if np.isfinite(tau_s) and tau_s > 0 else None
    return {
        "calm_index": calm,
        "tau_calm": tau_c,
        "tau_stressed": tau_s,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Estimation layer
# ---------------------------------------------------------------------------

#: Rolling-window geometry. 2,500 hours is long enough for the MS-AR to
#: identify two regimes and short enough that the October 2025 excursion leaves
#: the window part-way through the sample; 450 hours advances by ~19 days.
ROLL_WINDOW = 2500
ROLL_STEP = 450

#: The paper's own multi-start protocol. The MS-AR likelihood is bimodal and its
#: dominant attractor is not its maximum, so a reduced start count here would
#: report a different mode from the paper's MS-AR table
#: (paper/tables/table3b_funding_robust.tex) and the comparison would be
#: meaningless.
N_STARTS = 40
SEARCH_REPS = 40


def _significance(ci) -> str:
    """Which side of unity the interval falls on, or ``ns`` if it covers it."""
    lo, hi = (None, None) if ci is None else (ci[0], ci[1])
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "undetermined"
    if lo > 1.0:
        return "above"
    if hi < 1.0:
        return "below"
    return "ns"


def json_safe_optima(optima):
    """The multi-start optima with a non-finite log-likelihood written as ``None``.

    A start can end at a non-finite likelihood without raising; it is kept in
    the record, but as ``null`` rather than as a NaN token that strict JSON
    parsers reject.
    """
    if optima is None:
        return None
    return [dict(o, llf=float(o["llf"]) if np.isfinite(o["llf"]) else None) for o in optima]


def fit_window(y, *, n_starts: int = N_STARTS, search_reps: int = SEARCH_REPS,
               label: str = "") -> dict:
    """Fit the MS-AR on one premium slice and return the labelled ratio record.

    Cross-checks the estimator's own ratio block against :func:`label_regimes`.
    They are computed independently -- the block by the delta method inside
    :mod:`regime_engine.msar`, the labels here from ``sigma2`` -- so a mismatch
    means the regime ordering has drifted between the two and the record would
    otherwise be silently mislabelled.

    The record also carries the multi-start protocol behind the estimate
    (``optima_found``, ``failures``, ``best_seed``, ``seed_base`` and the
    ``excluded`` starts of the guarded scan, ``msar.fit_msar_multistart``) and
    the keys of :func:`regime_engine.msar.msar_robust_inference`: staying
    probabilities, expected durations, switching-adjusted half-lives and their
    ratio, and under the Hessian, OPG and sandwich covariances the ratio's
    standard error, its linear, log-scale and Fieller intervals, and the Wald
    test of phi_calm = phi_stressed that decides its side of unity.

    ``ci95`` and ``significance`` are the Hessian linear delta interval and
    its side, the columns of table8_rolling and table9_sensitivity; the
    verdict is ``inference[cov]["ratio_wald"]``.
    """
    from .msar import fit_msar_premium, msar_robust_inference

    y = np.asarray(y, dtype=float)
    res = fit_msar_premium(y, n_pad=len(y), n_starts=n_starts,
                           search_reps=search_reps)
    block = res["half_life_ratio"]
    check = label_regimes(res["sigma2"], res["half_life_hours"])

    if check["ratio"] is not None and block.get("value") is not None:
        if not np.isclose(check["ratio"], block["value"], rtol=1e-6):
            raise AssertionError(
                f"regime ordering disagrees for {label!r}: label_regimes gives "
                f"{check['ratio']!r}, the estimator's ratio block {block['value']!r}"
            )

    ci = block.get("ci95_delta")
    ms = res.get("multistart") or {}
    rec = {
        "label": label,
        "T": int(len(y)),
        "tau_calm": check["tau_calm"],
        "tau_stressed": check["tau_stressed"],
        "ratio": block.get("value"),
        "ratio_se": block.get("se_delta"),
        "ci95": ci,
        "significance": _significance(ci),
        "sigma2": [float(v) for v in np.asarray(res["sigma2"], dtype=float)],
        "phi": [float(v) for v in np.asarray(res["phi"], dtype=float)],
        "llf": res.get("llf"),
        "optima_found": json_safe_optima(ms.get("optima_found")),
        "failures": ms.get("failures"),
        "best_seed": ms.get("best_seed"),
        "seed_base": ms.get("seed_base"),
        "excluded": ms.get("excluded"),
    }
    rec.update(msar_robust_inference(res))
    return rec


def fit_rolling(y, times, *, window: int = ROLL_WINDOW, step: int = ROLL_STEP,
                n_starts: int = N_STARTS, search_reps: int = SEARCH_REPS,
                progress=None) -> list[dict]:
    """Fit one MS-AR per rolling window over the premium series."""
    y = np.asarray(y, dtype=float)
    out = []
    for i, (a, b) in enumerate(rolling_windows(len(y), window, step)):
        rec = fit_window(y[a:b], n_starts=n_starts, search_reps=search_reps,
                         label=f"window {i}")
        rec.update(index=i, start=str(times[a]), end=str(times[b - 1]),
                   mid=str(times[a + (b - a) // 2]))
        out.append(rec)
        if progress is not None:
            progress(rec)
    return out


def fit_sensitivity(y, *, n_starts: int = N_STARTS,
                    search_reps: int = SEARCH_REPS, progress=None) -> list[dict]:
    """Fit the baseline and each upper-tail perturbation of the same window."""
    out = []
    for key, series in sensitivity_variants(y).items():
        rec = fit_window(series, n_starts=n_starts, search_reps=search_reps,
                         label=key)
        rec["variant"] = key
        rec["description"] = VARIANT_LABELS[key]
        out.append(rec)
        if progress is not None:
            progress(rec)
    return out


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

#: The premium maximum of 10 October 2025, in basis points: the single
#: observation the paper-window asymmetry rests on.
OCTOBER_MAXIMUM_BPS = 23.6132


# ---------------------------------------------------------------------------
# Manuscript tables
# ---------------------------------------------------------------------------

_BANNER = "% Generated by regime_engine.oos -- do not edit by hand.\n"


def _ucfirst(s: str) -> str:
    """Capitalise the first letter only; ``str.capitalize`` lowercases the rest."""
    return s[:1].upper() + s[1:]


def _fmt_ci(ci) -> str:
    """Render a delta-method interval, leaving an unusable endpoint open.

    The interval for a ratio of half-lives can come back with a missing or
    negative lower endpoint when one autoregressive coefficient sits close to
    unity; printing ``None`` or a negative half-life would be worse than saying
    the bound is not determined.
    """
    def endpoint(v):
        if v is None or not np.isfinite(v) or v < 0:
            return "$\\cdot$"
        return f"{v:.2f}"

    lo, hi = (None, None) if not ci else (ci[0], ci[1])
    return f"[{endpoint(lo)}, {endpoint(hi)}]"


def _side(significance: str) -> str:
    return {"above": "$>1$", "below": "$<1$"}.get(significance, "n.s.")


def write_rolling_table(payload: dict, out_dir) -> pathlib.Path:
    """Write the rolling-window table of the earlier version (output/ only).

    Its interval and side columns are each record's ``ci95`` and
    ``significance``, the Hessian linear delta interval; the Wald test that
    decides the side of unity is in ``table8b_rolling_robust``
    (:mod:`regime_engine.msar_tables`).
    """
    tab = pathlib.Path(out_dir) / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    p = payload.get("protocol", {})
    rows = []
    for rec in payload["rolling"]:
        mid = str(rec.get("mid", ""))[:10]
        dd = rec.get("drawdown_mean")
        px = rec.get("price_mean")
        rows.append(
            f"{mid} & {rec['tau_calm']:.2f} & {rec['tau_stressed']:.2f} & "
            f"{rec['ratio']:.2f} & {_fmt_ci(rec.get('ci95'))} & {_side(rec['significance'])} & "
            + ("--" if px is None else f"{px:,.0f}".replace(",", "{,}")) + " & "
            + ("--" if dd is None else f"${100 * dd:.0f}\\%$") + " \\\\\n"
        )
    tex = (
        _BANNER
        + f"% Rolling MS-AR on the premium series: window {p.get('window')} h, "
          f"step {p.get('step')} h, {p.get('n_starts')} starts per fit.\n"
        "% Regimes are labelled by sigma2 ascending, so R = tau_calm / tau_stressed.\n"
        "\\begin{tabular}{lrrrlcrr}\n\\toprule\n"
        "Window mid & $\\hat\\tau_{\\text{calm}}$ & $\\hat\\tau_{\\text{str}}$ & $\\hat R$ & "
        "95\\% CI & side & price & DD \\\\\n\\midrule\n"
        + "".join(rows)
        + "\\bottomrule\n\\end{tabular}\n"
    )
    path = tab / "table8_rolling.tex"
    path.write_text(tex)
    return path


def legacy_table_dirs(out_dir) -> tuple:
    """Where table8_rolling and table9_sensitivity go: the output directory only.

    Both tables side the ratio by the Hessian linear delta interval of the earlier
    version. The paper prints table8b_rolling_robust and table9b_sensitivity_robust
    (regime_engine.msar_tables), whose verdict is the Wald test, so these two stay
    as artifacts and are not copied to paper/tables/.
    """
    return (pathlib.Path(out_dir),)


def write_sensitivity_table(payload: dict, out_dir) -> pathlib.Path:
    """Write the single-observation sensitivity table of the earlier version (output/ only).

    As :func:`write_rolling_table`, from the Hessian linear delta interval;
    ``table9b_sensitivity_robust`` carries the Wald test.
    """
    tab = pathlib.Path(out_dir) / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in payload["sensitivity"]:
        rows.append(
            VARIANT_SHORT.get(rec["variant"], _ucfirst(rec["description"]))
            + f" & {rec['T']:,} & ".replace(",", "{,}")
            + f"{rec['tau_calm']:.2f} & {rec['tau_stressed']:.2f} & "
            f"{rec['ratio']:.2f} & {_fmt_ci(rec.get('ci95'))} & {_side(rec['significance'])} \\\\\n"
        )
    tex = (
        _BANNER
        + "% Paper window (T = 5,001). Each row refits the MS-AR after one change\n"
          "% to the upper tail of the premium; the baseline is refitted alongside.\n"
        "\\begin{tabular}{lrrrrlc}\n\\toprule\n"
        "Series & $T$ & $\\hat\\tau_{\\text{calm}}$ & $\\hat\\tau_{\\text{str}}$ & $\\hat R$ & "
        "95\\% CI & side \\\\\n\\midrule\n"
        + "".join(rows)
        + "\\bottomrule\n\\end{tabular}\n"
    )
    path = tab / "table9_sensitivity.tex"
    path.write_text(tex)
    return path


def market_context(prices, start, end) -> dict:
    """Mean price and mean drawdown-from-running-peak over one window.

    The drawdown is measured against the running maximum of the *whole* price
    history, not of the window, so windows remain comparable to one another.
    """
    import pandas as pd

    peak = prices["c"].cummax()
    dd = prices["c"] / peak - 1.0
    m = (prices["time"] >= pd.Timestamp(start)) & (prices["time"] <= pd.Timestamp(end))
    if not m.any():
        return {"price_mean": None, "drawdown_mean": None}
    return {
        "price_mean": float(prices.loc[m, "c"].mean()),
        "drawdown_mean": float(dd[m].mean()),
    }


def build_payload(premium_path=None, prices_path=None, *, window: int = ROLL_WINDOW,
                  step: int = ROLL_STEP, n_starts: int = N_STARTS,
                  search_reps: int = SEARCH_REPS, verbose: bool = True) -> dict:
    """Compute every out-of-sample robustness result Section 6.1 reports."""
    import pandas as pd

    from .hmm import MSAR_SEED
    from .panel import load_paper_window_premium
    from .paths import DEFAULT_PREMIUM_FULL, PAPER_WINDOW, REPO_ROOT

    premium_path = pathlib.Path(premium_path or DEFAULT_PREMIUM_FULL)
    prices_path = pathlib.Path(
        prices_path or REPO_ROOT / "data" / "interim" / "prices_BTC_2h_full.csv")

    prem = pd.read_csv(premium_path)
    prem["time"] = pd.to_datetime(prem["time"], utc=True, format="ISO8601")
    prem = prem.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    y_all = prem["premium"].to_numpy(dtype=float) * 1e4
    times = list(prem["time"])

    prices = pd.read_csv(prices_path)
    prices["time"] = pd.to_datetime(prices["time"], utc=True, format="ISO8601")

    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)

    def fmt(v):
        return "None" if v is None else f"{v:.3f}"

    def show(rec):
        ci = rec.get("ci95") or [None, None]
        say(f"  {rec.get('label', ''):26s} T={rec['T']:5d}  "
            f"tau_c={rec['tau_calm']:7.2f}  tau_s={rec['tau_stressed']:7.2f}  "
            f"R={fmt(rec['ratio']):>9s}  CI=[{fmt(ci[0])}, {fmt(ci[1])}]  "
            f"{rec['significance']}")

    say(f"[oos] premium series: {len(y_all)} hours, "
        f"{times[0]:%Y-%m-%d} to {times[-1]:%Y-%m-%d}")

    say(f"[oos] rolling windows (window={window}, step={step}, starts={n_starts})")
    rolling = fit_rolling(y_all, times, window=window, step=step,
                          n_starts=n_starts, search_reps=search_reps, progress=show)
    for rec in rolling:
        rec.update(market_context(prices, rec["start"], rec["end"]))

    say("[oos] single-observation sensitivity on the paper window")
    y_paper = load_paper_window_premium(premium_path)
    sensitivity = fit_sensitivity(y_paper, n_starts=n_starts,
                                  search_reps=search_reps, progress=show)

    say("[oos] the upper tail of the paper window: October 2025 against the rest")
    paper_mask = (prem["time"] >= pd.Timestamp(PAPER_WINDOW[0], tz="UTC")) & (
        prem["time"] <= pd.Timestamp(PAPER_WINDOW[1]))
    paper_times = list(prem.loc[paper_mask, "time"])
    tail_series, tail_facts = upper_tail_variants(y_paper, paper_times)
    upper_tail = {"_note": (
        "The cap of the 'winsorise_top1' variant (the paper window's 99th percentile) "
        "and the hours it sets, by the month their premium accrued in; "
        "october_tail_capped caps only the October 2025 ones, "
        "winsorised_from_11_oct is the winsorised series from the stamp "
        "2025-10-11 00:00 UTC on."), **tail_facts}
    for key, series in tail_series.items():
        rec = fit_window(series, n_starts=n_starts, search_reps=search_reps, label=key)
        rec["variant"] = key
        rec["description"] = UPPER_TAIL_LABELS[key]
        upper_tail[key] = rec
        show(rec)

    say("[oos] full extended window")
    extended = fit_window(y_all, n_starts=n_starts, search_reps=search_reps,
                          label=span_label(times))
    show(extended)

    say("[oos] placebo injection into the post-April window")
    start_oos = pd.Timestamp(PAPER_WINDOW[1])
    mask = prem["time"] > start_oos
    y_oos = prem.loc[mask, "premium"].to_numpy(dtype=float) * 1e4
    injection = [fit_window(y_oos, n_starts=n_starts, search_reps=search_reps,
                            label="post-April, unchanged")]
    show(injection[0])
    for key, series in injection_variants(
            y_oos, OCTOBER_MAXIMUM_BPS,
            (len(y_oos) // 4, len(y_oos) // 2, 3 * len(y_oos) // 4)).items():
        rec = fit_window(series, n_starts=n_starts, search_reps=search_reps,
                         label=f"post-April + {key}")
        rec["variant"] = key
        injection.append(rec)
        show(rec)

    return {
        "_note": (
            "Out-of-sample robustness for Section 6.1. Every fit uses the paper's "
            "multi-start protocol; regimes are labelled by sigma2 ascending, so "
            "ratio = tau_calm / tau_stressed and a value below unity means the "
            "stressed regime is the more persistent one."
        ),
        "_verdicts": (
            "The side of unity of the ratio is the Wald test of phi_calm = "
            "phi_stressed under each covariance, inference.<cov>.ratio_wald: the "
            "ratio is one exactly when the two coefficients are equal, so the test "
            "needs no delta step. Where p_cc + p_ss - 1 is significantly positive "
            "the switching-adjusted ratio lies on the same side of unity, "
            "inference.<cov>.ratio_switching_side. The linear, log-scale and "
            "Fieller sets in inference.<cov>.ratio_ci95, the delta intervals of "
            "the switching-adjusted ratio and its simulated interval "
            "(ratio_switching_sim) are descriptive. Each record's top-level ci95 "
            "and significance are the Hessian linear delta interval and its side, "
            "the columns of table8_rolling and table9_sensitivity."
        ),
        "premium_series": {
            "path": str(premium_path.relative_to(REPO_ROOT)),
            "T": int(len(y_all)),
            "start": str(times[0]),
            "end": str(times[-1]),
        },
        "paper_window": {"start": PAPER_WINDOW[0], "end": PAPER_WINDOW[1],
                         "T": int(len(y_paper))},
        "protocol": {"n_starts": n_starts, "search_reps": search_reps,
                     "window": window, "step": step, "seed_base": MSAR_SEED},
        "rolling": rolling,
        "extended_window": extended,
        "sensitivity": sensitivity,
        "upper_tail": upper_tail,
        "injection": injection,
    }


def write_artifact(payload: dict, out_dir=None) -> pathlib.Path:
    """Write ``oos_robustness.json`` and return its path."""
    from .paths import DEFAULT_OUT

    out_dir = pathlib.Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "oos_robustness.json"
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--premium", metavar="PATH", default=None)
    ap.add_argument("--prices", metavar="PATH", default=None)
    ap.add_argument("--out", metavar="DIR", default=None)
    ap.add_argument("--window", type=int, default=ROLL_WINDOW)
    ap.add_argument("--step", type=int, default=ROLL_STEP)
    ap.add_argument("--n-starts", type=int, default=N_STARTS)
    args = ap.parse_args(argv)

    payload = build_payload(args.premium, args.prices, window=args.window,
                            step=args.step, n_starts=args.n_starts)
    path = write_artifact(payload, args.out)
    print(f"[oos] wrote {path}")

    from .paths import DEFAULT_OUT

    for dest in legacy_table_dirs(pathlib.Path(args.out or DEFAULT_OUT)):
        for writer in (write_rolling_table, write_sensitivity_table):
            print(f"[oos] wrote {writer(payload, dest)}")
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
