"""
Main analysis pipeline.

1) Fit a 2- and 3-state Gaussian HMM to BTC perp hourly log-returns.
2) Fit a 2-state Markov-Switching AR(1) to the BTC funding premium.
3) Generate the regime-volatility visualisation, regime-conditional
   distribution panels, transition heatmaps, and price/funding overlays.
4) Persist tables (.tex) and parameters (.json) for the paper.

Usage
-----
    python -m regime_engine.analysis --panel data/processed/panel_BTC_subsample.csv \
                                     --out output/

``--panel`` takes a CSV or parquet panel from disk and is the intended entry
point.  With no ``--panel`` the pipeline falls back to fetching from the live
Hyperliquid API (``--fetch``), which cannot reproduce the paper's window: the
``candleSnapshot`` endpoint retains only ~5,000 recent candles.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pathlib
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")           # headless: never require a display

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from .funding import FLOOR_BPS_PER_HOUR, clamp_share  # noqa: E402
from .hmm import MSAR_SEED, GaussianHMM  # noqa: E402
from .msar import fit_msar_premium  # noqa: E402
from .panel import describe_series, load_panel, panel_fingerprint  # noqa: E402
from .paths import DEFAULT_OUT  # noqa: E402

HMM_SEED_BASE = 42          # per-K seed is HMM_SEED_BASE + K
HMM_N_STARTS = 8
HMM_N_ITER = 150

# Output locations. Default to the *repository's* output/ directory, not a
# directory inside the installed package; --out overrides at runtime.
OUT = DEFAULT_OUT
FIG = OUT / "figures"
TAB = OUT / "tables"
CACHE_DIR = OUT / "cache"


def set_output_dir(path) -> pathlib.Path:
    """Point every writer at ``path``; create the directory tree."""
    global OUT, FIG, TAB, CACHE_DIR
    OUT = pathlib.Path(path).expanduser().resolve()
    FIG = OUT / "figures"
    TAB = OUT / "tables"
    CACHE_DIR = OUT / "cache"
    for d in (OUT, FIG, TAB, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return OUT


# -- aesthetics ---------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.1,
    # Embed TrueType (Type 42) rather than Type 3 fonts: Type 3 is accepted by
    # SSRN but rejected by IEEE/ACM submission systems and by PDF/A validators.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REGIME_COLORS_3 = ["#2E86AB", "#A8A8A8", "#C73E1D"]   # low / moderate / high volatility
REGIME_COLORS_2 = ["#2E86AB", "#C73E1D"]
PROB_CMAP = "Blues"      # sequential: probabilities live on [0, 1], not around 0


def _regime_labels(hmm: GaussianHMM) -> list:
    """Legend labels derived from the *fitted* parameters.

    States arrive sorted by sigma ascending (see ``hmm.fit``), so index k is the
    k-th smallest conditional volatility and the descriptor follows the index
    directly.

    The labels name volatility and nothing else. The paper is explicit that the
    fit identifies volatility states and not drift states -- none of the three
    state-conditional means is distinguishable from zero -- so an economic name
    like "crisis" or "trend" would assert exactly what the estimates do not
    support, and the sojourn arithmetic (a mean of roughly seven hours) rules
    out the vocabulary of market phases as well.
    """
    K = hmm.K
    if K == 3:
        vol_words = ["low", "moderate", "high"]
    elif K == 2:
        vol_words = ["low", "high"]
    else:                                   # pragma: no cover - K in (2, 3)
        vol_words = [f"level {k + 1}" for k in range(K)]
    return [
        f"Regime {k + 1} — {vol_words[k]} volatility "
        f"($\\hat\\sigma$ = {hmm.sigma[k] * 1e4:.1f} bps/h)"
        for k in range(K)
    ]


def _annotation_color(cmap_name: str, value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Pick black/white text so it stays legible on any colormap."""
    rgba = matplotlib.colormaps[cmap_name]((value - vmin) / (vmax - vmin))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.55 else "black"


# ----------------------------------------------------------------------
def estimator_fingerprint(n_chars: int = 12) -> str:
    """SHA-256 of the code and hyperparameters that determine a return fit.

    A cache keyed on the panel alone is only safe if the estimator never
    changes. A change to the estimator -- to its label-switching rule, say --
    would leave every earlier entry valid under that key, and regenerating the
    artifacts would reload a fit the current code no longer produces instead of
    refitting. Hashing the estimator's source alongside its hyperparameters
    makes any edit to it a cache miss.
    """
    material = "\n".join([
        inspect.getsource(GaussianHMM),
        repr((HMM_SEED_BASE, HMM_N_STARTS, HMM_N_ITER)),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:n_chars]


def fit_cache_key(panel: pd.DataFrame) -> str:
    """Cache key: what the fit was computed *from* and what computed it."""
    return f"{panel_fingerprint(panel)}_{estimator_fingerprint()}"


def fit_return_hmms(panel: pd.DataFrame, use_cache: bool = True) -> dict:
    """Fit K=2 and K=3 return HMMs. Cached on disk, keyed by panel AND estimator.

    A key on the filename alone would return a stale fit whenever the file's
    contents changed; a key on the panel alone would return a stale fit
    whenever the *code* changed. The key covers both.
    """
    fingerprint = panel_fingerprint(panel)
    key = fit_cache_key(panel)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"return_hmms_{key}.pkl"
    if use_cache and cache.exists():
        print(f"[cache] loading {cache.name} (panel {fingerprint}, "
              f"estimator {estimator_fingerprint()})")
        with open(cache, "rb") as f:
            return pickle.load(f)

    r = panel["log_return"].values

    results = {"panel_fingerprint": fingerprint}
    for K in (2, 3):
        hmm = GaussianHMM(K=K, n_starts=HMM_N_STARTS, n_iter=HMM_N_ITER,
                          random_state=HMM_SEED_BASE + K).fit(r)
        post_filt = hmm.filtered_posterior(r)
        post_smooth = hmm.smoothed_posterior(r)
        results[K] = {
            "model": hmm,
            "filtered": post_filt,
            "smoothed": post_smooth,
            "viterbi": hmm.viterbi(r),
            "ll": hmm.log_likelihood_,
            "aic": hmm.aic(),
            "bic": hmm.bic(len(r)),
            "seed": HMM_SEED_BASE + K,
        }
        print(f"K={K}: LL={hmm.log_likelihood_:.4f}  AIC={hmm.aic():.2f}  "
              f"BIC={hmm.bic(len(r)):.2f}  (seed {HMM_SEED_BASE + K})")
        print(f"      mu (bps/h)    = {hmm.mu * 1e4}")
        print(f"      sigma (bps/h) = {hmm.sigma * 1e4}")
        print(f"      diag(A)       = {np.diag(hmm.A)}")
        print(f"      E[duration]   = {hmm.expected_durations()} hours")
        print(f"      stationary    = {hmm.stationary_distribution()}")

    with open(cache, "wb") as f:
        pickle.dump(results, f)
    print(f"[cache] saved {cache.name}")
    return results


def fit_funding_ms_ar(panel: pd.DataFrame, seed: int = MSAR_SEED) -> dict:
    """Markov-Switching AR(1) on the premium component (in bps).

    The headline funding rate is heavily clamped to the constant interest-rate
    floor (0.125 bps/h) -- 46% of hours on the full sample, 35% on the shipped
    sub-sample -- which causes a pathological zero-variance regime in MS-AR.
    The exchange-published premium index captures the genuine market-positioning
    signal and is well-behaved.  See ``regime_engine.funding``.
    """
    y = panel["premium"].values * 1e4  # bps

    print("\n[Funding] Fitting MS-AR(1) on premium (bps), K=2 ...")
    res = fit_msar_premium(y, seed=seed, n_pad=len(panel))
    print(res["result"].summary())
    print(f"\n[MS-AR] start-search seed = {res['seed']}  "
          f"llf={res['llf']:.6f}  AIC={res['aic']:.2f}  BIC={res['bic']:.2f}  "
          f"nobs={res['nobs']}")
    for block in res["half_lives"]:
        lo, hi = block["half_life_ci95_delta"]
        hi_txt = "inf" if block["phi_ci95_contains_unit_root"] else f"{hi:.2f}"
        print(f"      {block['regime']:>8}: phi={block['phi']:.4f} "
              f"(se {block['phi_se']:.4f})  tau={block['half_life_hours']:.2f} h  "
              f"delta-95% [{lo:.2f}, {hi:.2f}]  (upper from phi: {hi_txt})")
    ratio = res["half_life_ratio"]
    if ratio["se_delta"] is not None:
        print(f"      ratio tau_calm/tau_stressed = {ratio['value']:.3f} "
              f"(se {ratio['se_delta']:.3f}, 95% "
              f"[{ratio['ci95_delta'][0]:.3f}, {ratio['ci95_delta'][1]:.3f}])")
    return res


def _save_figure(fig, stem: str) -> None:
    """Write <stem>.pdf and <stem>.png.

    ``CreationDate: None`` strips the timestamp matplotlib would otherwise embed
    in the PDF, so re-running the pipeline on unchanged data produces
    byte-identical figures. That is what makes "the outputs in this repo are the
    outputs of this code" a checkable claim rather than an assertion.
    """
    fig.savefig(FIG / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / f"{stem}.png")


# ----------------------------------------------------------------------
#             fig1_regime_dashboard — regime dashboard
# ----------------------------------------------------------------------
def figure_regime_dashboard(panel: pd.DataFrame, ret_res: dict, fund_res: dict, K: int = 3):
    """Five-panel dashboard: price, return-regime stripe, funding,
    funding-regime stripe, realised vs regime-implied volatility."""
    times = pd.to_datetime(panel["time"]).values
    price = panel["c"].values
    funding = panel["funding_bps"].values

    post = ret_res[K]["filtered"]
    fund_post = fund_res["filtered"]

    fig = plt.figure(figsize=(11, 9.4))
    gs = fig.add_gridspec(
        5, 1, height_ratios=[2.5, 0.35, 2.0, 0.35, 1.4],
        hspace=0.55,
    )
    fig.suptitle(
        "BTC perpetual on Hyperliquid — return regimes "
        f"(K={K}) and funding regimes",
        x=0.125, ha="left", fontweight="bold", fontsize=11.5, y=0.955,
    )

    # --- Panel A: price + dominant-regime background --------------------
    axA = fig.add_subplot(gs[0])
    axA.plot(times, price, color="#222", lw=0.9)
    # Median-filtered argmax of the filtered posterior (a 7-hour window), not
    # raw Viterbi, to reduce visual noise from hour-by-hour regime flips.
    dominant = np.argmax(post, axis=1)
    from scipy.ndimage import median_filter
    dominant_smooth = median_filter(dominant, size=7, mode="nearest")
    cmap = REGIME_COLORS_3 if K == 3 else REGIME_COLORS_2
    for k in range(K):
        mask = dominant_smooth == k
        axA.fill_between(times, 0, price.max() * 1.02,
                         where=mask, color=cmap[k], alpha=0.13, lw=0)
    axA.set_ylabel("BTC close (USD)")
    axA.set_ylim(price.min() * 0.98, price.max() * 1.02)
    axA.set_xlim(times[0], times[-1])
    axA.set_title("(a) BTC price with median-filtered dominant return regime",
                  loc="left", fontsize=9, fontweight="bold", pad=4)

    from matplotlib.patches import Patch
    labels = _regime_labels(ret_res[K]["model"])
    handles = [Patch(facecolor=cmap[k], alpha=0.45, label=labels[k]) for k in range(K)]
    axA.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

    # --- Panel B: return regime posterior stripe -------------------------
    axB = fig.add_subplot(gs[1], sharex=axA)
    rgb = np.zeros((len(times), 3))
    for k in range(K):
        c = np.array(to_rgba(cmap[k])[:3])
        rgb += post[:, k:k + 1] * c[None, :]
    rgb = np.clip(rgb, 0, 1)
    axB.imshow(rgb[None, :, :], aspect="auto",
               extent=[mdates.date2num(times[0]), mdates.date2num(times[-1]), 0, 1])
    axB.set_yticks([])
    axB.set_ylabel("Return\nregime", rotation=0, ha="right", va="center", fontsize=9)
    axB.set_title("(b) Filtered posterior $P(S_t=k\\,|\\,\\mathcal{F}_t)$ — return HMM",
                  loc="left", fontsize=9, fontweight="bold", pad=4)

    # --- Panel C: funding rate -------------------------------------------
    axC = fig.add_subplot(gs[2], sharex=axA)
    axC.plot(times, funding, color="#444", lw=0.6, alpha=0.9)
    axC.axhline(0, color="black", lw=0.5)
    axC.axhline(FLOOR_BPS_PER_HOUR, color="#B07A00", lw=0.7, ls=":")
    axC.fill_between(times, 0, funding, where=funding > 0,
                     color="#2E86AB", alpha=0.25, lw=0)
    axC.fill_between(times, 0, funding, where=funding < 0,
                     color="#C73E1D", alpha=0.25, lw=0)
    axC.set_ylabel("Funding rate (bps / hour)")
    axC.set_title(f"(c) Hourly funding rate, BTC perp "
                  f"(dotted: {FLOOR_BPS_PER_HOUR} bps/h interest floor)",
                  loc="left", fontsize=9, fontweight="bold", pad=4)
    # cap visualisation y-range to 0.5/99.5 percentile to keep readability,
    # with headroom above the floor line so the annotation never sits on data
    lo, hi = np.percentile(funding, [0.5, 99.5])
    pad = (hi - lo) * 0.1
    axC.set_ylim(lo - pad, max(hi, FLOOR_BPS_PER_HOUR) + 2 * pad)

    # --- Panel D: funding regime stripe ----------------------------------
    axD = fig.add_subplot(gs[3], sharex=axA)
    fund_cmap = ["#1B998B", "#E84855"]  # calm / stressed
    safe_post = np.where(np.isnan(fund_post), 0.5, fund_post)  # padding -> grey
    rgbf = np.zeros((len(times), 3))
    for k in range(2):
        c = np.array(to_rgba(fund_cmap[k])[:3])
        rgbf += safe_post[:, k:k + 1] * c[None, :]
    rgbf = np.clip(rgbf, 0, 1)
    axD.imshow(rgbf[None, :, :], aspect="auto",
               extent=[mdates.date2num(times[0]), mdates.date2num(times[-1]), 0, 1])
    axD.set_yticks([])
    axD.set_ylabel("Funding\nregime", rotation=0, ha="right", va="center", fontsize=9)
    axD.set_title("(d) Filtered posterior — MS-AR(1) on premium "
                  "(teal = calm, red = stressed)",
                  loc="left", fontsize=9, fontweight="bold", pad=4)

    # --- Panel E: rolling realised vol -----------------------------------
    axE = fig.add_subplot(gs[4], sharex=axA)
    rv = (
        pd.Series(panel["log_return"].values ** 2)
        .rolling(24).mean().pow(0.5).values * np.sqrt(24 * 365) * 100  # annualised %
    )
    # Regime-implied conditional vol: within-regime term only, filtered state.
    hmm = ret_res[K]["model"]
    cond_var = (post * (hmm.sigma ** 2)[None, :]).sum(axis=1)
    cond_vol_h = np.sqrt(cond_var) * np.sqrt(24 * 365) * 100
    cond_vol_smooth = pd.Series(cond_vol_h).rolling(24, min_periods=1).mean().values
    axE.plot(times, rv, color="#222", lw=0.7, label="24h realised vol", alpha=0.85)
    axE.plot(times, cond_vol_smooth, color="#C73E1D", lw=1.3,
             label="Regime-implied vol (within-regime, 24h-smoothed)")
    axE.set_ylabel("Annualised vol (%)")
    axE.set_xlabel("Date (UTC)")
    axE.legend(loc="upper right", fontsize=8, frameon=False)
    axE.set_title("(e) Realised vs. regime-implied volatility",
                  loc="left", fontsize=9, fontweight="bold", pad=4)

    for ax in (axA, axB, axC, axD, axE):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        if ax is not axE:
            plt.setp(ax.get_xticklabels(), visible=False)

    _save_figure(fig, "fig1_regime_dashboard")
    plt.close(fig)


# ----------------------------------------------------------------------
#     fig2_distributions — regime-conditional distributions
# ----------------------------------------------------------------------
def figure_regime_distributions(panel: pd.DataFrame, ret_res: dict, K: int = 3):
    r = panel["log_return"].values * 1e4  # bps
    hmm = ret_res[K]["model"]
    cmap = REGIME_COLORS_3 if K == 3 else REGIME_COLORS_2

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # Left: empirical density vs fitted mixture
    ax = axes[0]
    ax.hist(r, bins=140, density=True, alpha=0.35, color="#666", edgecolor="none")
    grid = np.linspace(r.min(), r.max(), 800)
    pi_stat = hmm.stationary_distribution()
    total = np.zeros_like(grid)
    from scipy.stats import norm
    for k in range(K):
        comp = pi_stat[k] * norm.pdf(grid, loc=hmm.mu[k] * 1e4, scale=hmm.sigma[k] * 1e4)
        label = (fr"Reg. {k+1}: $\mu={hmm.mu[k]*1e4:.2f}$, "
                 fr"$\sigma={hmm.sigma[k]*1e4:.1f}$ bps")
        ax.plot(grid, comp, lw=1.2, color=cmap[k], label=label)
        total += comp
    ax.plot(grid, total, "k--", lw=1.0, label="Mixture (stationary)")
    ax.set_xlim(np.percentile(r, 0.5), np.percentile(r, 99.5))
    ax.set_xlabel("Hourly log-return (bps)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Stationary mixture vs. empirical distribution "
                 "(x-axis clipped to 0.5/99.5 pct)", loc="left", fontsize=9.5, pad=6)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)   # headroom so the legend clears the peak
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # Right: transition matrix as heatmap (sequential colormap)
    ax = axes[1]
    A = hmm.A
    im = ax.imshow(A, cmap=PROB_CMAP, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center",
                    color=_annotation_color(PROB_CMAP, A[i, j]), fontsize=10)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    labels = [f"Reg. {k+1}" for k in range(K)]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.grid(False)
    ax.set_title("(b) Estimated transition matrix $\\hat A$",
                 loc="left", fontsize=9.5, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save_figure(fig, "fig2_distributions")
    plt.close(fig)


# ----------------------------------------------------------------------
#   fig3_funding_dynamics — funding regime: phase plot + half-life
# ----------------------------------------------------------------------
def figure_funding_dynamics(panel: pd.DataFrame, fund_res: dict):
    y = panel["premium"].values * 1e4  # bps
    smoothed = fund_res["smoothed"]
    fund_cmap = ["#1B998B", "#E84855"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # (a) Premium phase plot coloured by regime
    ax = axes[0]
    valid = ~np.isnan(smoothed[:, 0])
    r_idx = np.argmax(np.where(np.isnan(smoothed), 0, smoothed), axis=1)
    for k in range(2):
        mask = (r_idx == k) & valid
        ax.scatter(y[:-1][mask[1:]], y[1:][mask[1:]],
                   s=4, c=fund_cmap[k], alpha=0.35,
                   label=f"Regime {k+1}")
    # AR(1) lines. statsmodels estimates the MEAN-ADJUSTED form
    #     y_t - mu_{S_t} = phi_{S_t} (y_{t-1} - mu_{S_{t-1}}) + sigma_{S_t} eps_t
    # so const[k] is the regime mean mu_k and, within regime k
    # (S_{t-1} = S_t = k), the fitted line is
    #     y_t = mu_k (1 - phi_k) + phi_k y_{t-1}
    # (using mu_k as an intercept puts a spurious vertical offset on the plot).
    mu = fund_res["mu_const"]
    phi = fund_res["phi"]
    grid = np.linspace(y.min(), y.max(), 200)
    for k in range(2):
        line = mu[k] * (1.0 - phi[k]) + phi[k] * grid
        ax.plot(grid, line, color=fund_cmap[k], lw=1.2,
                label=fr"AR(1) reg. {k+1}: $\phi={phi[k]:.3f}$, $\bar y={mu[k]:.2f}$")
    ax.plot(grid, grid, "k--", lw=0.6, alpha=0.5, label="$y_t = y_{t-1}$")
    ax.set_xlabel("premium$_{t-1}$ (bps)")
    ax.set_ylabel("premium$_{t}$ (bps)")
    ax.set_title("(a) Premium phase plot, coloured by smoothed regime",
                 loc="left", fontsize=9.5, pad=6)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    lo, hi = np.percentile(y, [0.5, 99.5])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # (b) Implied half-life of mean reversion per regime, with delta-method CIs
    ax = axes[1]
    half_lives = np.asarray([b["half_life_hours"] for b in fund_res["half_lives"]], dtype=float)
    ses = np.asarray(
        [b["half_life_se_delta"] if b["half_life_se_delta"] is not None else np.nan
         for b in fund_res["half_lives"]], dtype=float)
    from .msar import Z95
    err = Z95 * ses
    bars = ax.bar(["Regime 1\n(calm)", "Regime 2\n(stressed)"],
                  half_lives, color=fund_cmap, alpha=0.8, edgecolor="black", lw=0.5,
                  yerr=err, capsize=4, error_kw={"lw": 0.8, "ecolor": "#333"})
    y_top = max(1.0, float(np.nanmax(half_lives + np.nan_to_num(err))) * 1.30)
    for b, hl, e in zip(bars, half_lives, err):
        top = hl + (e if np.isfinite(e) else 0.0)
        ax.text(b.get_x() + b.get_width() / 2, top + 0.035 * y_top,
                f"{hl:.1f} h", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean-reversion half-life (hours)")
    ax.set_ylim(0, y_top)
    ax.set_title(r"(b) Implied half-life $\tau=\ln 0.5/\ln|\phi_k|$, "
                 "delta-method 95% CI", loc="left", fontsize=9.5, pad=6)
    fig.tight_layout()
    _save_figure(fig, "fig3_funding_dynamics")
    plt.close(fig)


# ----------------------------------------------------------------------
#                          LaTeX tables
# ----------------------------------------------------------------------
def write_tables(panel: pd.DataFrame, ret_res: dict, fund_res: dict):
    TAB.mkdir(parents=True, exist_ok=True)

    # table1_model_selection: model selection across K
    rows = []
    for K in (2, 3):
        rows.append({
            "K": K,
            "logL": ret_res[K]["ll"],
            "n_params": ret_res[K]["model"].n_params(),
            "AIC": ret_res[K]["aic"],
            "BIC": ret_res[K]["bic"],
        })
    df = pd.DataFrame(rows)
    tex = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "$K$ & $\\log\\mathcal{L}$ & \\# params & AIC & BIC \\\\\n"
        "\\midrule\n"
    )
    for _, row in df.iterrows():
        tex += (f"{int(row['K'])} & {row['logL']:.1f} & {int(row['n_params'])} & "
                f"{row['AIC']:.1f} & {row['BIC']:.1f} \\\\\n")
    tex += "\\bottomrule\n\\end{tabular}\n"
    (TAB / "table1_model_selection.tex").write_text(tex)

    # table2_regime_params: regime parameters for chosen K=3
    hmm = ret_res[3]["model"]
    durs = hmm.expected_durations()
    tex = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Regime & $\\hat\\mu_k$ (bps/h) & $\\hat\\sigma_k$ (bps/h) "
        "& Annualised $\\sigma$ & $\\hat A_{kk}$ & $\\mathbb{E}[\\tau_k]$ (h) \\\\\n"
        "\\midrule\n"
    )
    for k in range(3):
        ann_vol = hmm.sigma[k] * np.sqrt(24 * 365) * 100
        tex += (f"{k+1} & {hmm.mu[k]*1e4:.3f} & {hmm.sigma[k]*1e4:.2f} & "
                f"{ann_vol:.1f}\\% & {hmm.A[k,k]:.4f} & {durs[k]:.1f} \\\\\n")
    tex += "\\bottomrule\n\\end{tabular}\n"
    (TAB / "table2_regime_params.tex").write_text(tex)

    # table3_funding_params: funding MS-AR(1) parameters, WITH standard errors.
    # Note: mu_k is the regime MEAN (statsmodels' mean-adjusted parameterisation),
    # not an intercept.
    raw_of = {int(fund_res["order"][k]): name
              for k, name in enumerate(("calm", "stressed"))}
    tex = (
        "% Generated by regime_engine.analysis -- do not edit by hand.\n"
        "% Parameter names are statsmodels' own regime indices. On this fit:\n"
        f"%   regime 0 = {raw_of[0]}, regime 1 = {raw_of[1]} "
        "(regimes are named by sigma2 ascending).\n"
        "% const[k] is the regime MEAN (mean-adjusted parameterisation), "
        "not an intercept.\n"
        f"% MS-AR start-search seed: {fund_res['seed']}.\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Parameter & Estimate & Std.\\ err. & $z$ & 95\\% CI \\\\\n"
        "\\midrule\n"
    )
    for row in fund_res["param_table"]:
        se = "--" if row["se"] is None else f"{row['se']:.4f}"
        z = "--" if row["z"] is None else f"{row['z']:.2f}"
        if row["ci95_low"] is None or row["ci95_high"] is None:
            ci = "--"
        else:
            ci = f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        name = row["name"].replace("_", "\\_").replace("->", "$\\to$")
        tex += f"\\texttt{{{name}}} & {row['coef']:.4f} & {se} & {z} & {ci} \\\\\n"
    tex += "\\midrule\n"
    for block in fund_res["half_lives"]:
        lo, hi = block["half_life_ci95_delta"]
        tex += (f"$\\tau_{{\\text{{{block['regime']}}}}}$ (h) & "
                f"{block['half_life_hours']:.2f} & "
                f"{block['half_life_se_delta']:.2f} & -- & "
                f"[{lo:.2f}, {hi:.2f}] \\\\\n")
    ratio = fund_res["half_life_ratio"]
    if ratio["se_delta"] is not None:
        tex += (f"$\\tau_{{\\text{{calm}}}}/\\tau_{{\\text{{stressed}}}}$ & "
                f"{ratio['value']:.3f} & {ratio['se_delta']:.3f} & -- & "
                f"[{ratio['ci95_delta'][0]:.3f}, {ratio['ci95_delta'][1]:.3f}] \\\\\n")
    tex += "\\midrule\n"
    tex += (f"$\\log\\mathcal{{L}}$ / AIC / BIC / $T$ & {fund_res['llf']:.2f} & "
            f"{fund_res['aic']:.2f} & {fund_res['bic']:.2f} & {fund_res['nobs']} \\\\\n")
    tex += "\\bottomrule\n\\end{tabular}\n"
    (TAB / "table3_funding_params.tex").write_text(tex)

    # table4_transition_matrix: the full estimated transition matrix.
    A = ret_res[3]["model"].A
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "From $\\backslash$ To & Regime 1 & Regime 2 & Regime 3 \\\\\n"
        "\\midrule\n"
    )
    for i in range(3):
        tex += (f"Regime {i+1} & " + " & ".join(f"{A[i, j]:.4f}" for j in range(3))
                + " \\\\\n")
    tex += "\\bottomrule\n\\end{tabular}\n"
    (TAB / "table4_transition_matrix.tex").write_text(tex)

    print(f"[LaTeX] wrote 4 tables to {TAB}")


def write_summary(panel: pd.DataFrame, ret_res: dict, fund_res: dict,
                  panel_source: str = ""):
    r"""Persist a JSON of every number the LaTeX paper substitutes via \input."""
    hmm3 = ret_res[3]["model"]
    hmm2 = ret_res[2]["model"]
    T = len(panel)

    def _hmm_block(hmm, res):
        return {
            "K": int(hmm.K),
            "seed": int(res["seed"]),
            "n_params": int(hmm.n_params()),
            "mu_bps_per_hour": [float(m * 1e4) for m in hmm.mu],
            "sigma_bps_per_hour": [float(s * 1e4) for s in hmm.sigma],
            "annualised_vol_pct": [float(s * np.sqrt(24 * 365) * 100) for s in hmm.sigma],
            "A": [[float(hmm.A[i, j]) for j in range(hmm.K)] for i in range(hmm.K)],
            "diag_A": [float(hmm.A[k, k]) for k in range(hmm.K)],
            "initial_pi": [float(p) for p in hmm.pi],
            "expected_duration_hours": [float(d) for d in hmm.expected_durations()],
            "stationary": [float(p) for p in hmm.stationary_distribution()],
            "logL": float(hmm.log_likelihood_),
            "AIC": float(hmm.aic()),
            "BIC": float(hmm.bic(T)),
        }

    funding_bps = panel["funding_bps"].values
    premium_bps = panel["premium"].values * 1e4
    returns_bps = panel["log_return"].values * 1e4

    summary = {
        "n_obs": int(T),
        "start": str(panel["time"].iloc[0]),
        "end": str(panel["time"].iloc[-1]),
        "panel_source": panel_source,
        "panel_fingerprint": ret_res.get("panel_fingerprint", ""),
        "price_range_usd": [float(panel["c"].min()), float(panel["c"].max())],
        "funding_mean_bps_per_hour": float(np.mean(funding_bps)),
        "funding_std_bps_per_hour": float(np.std(funding_bps, ddof=1)),
        "funding_annualised_pct": float(np.mean(funding_bps) * 24 * 365 / 100),
        "clamp": {
            "floor_bps_per_hour": FLOOR_BPS_PER_HOUR,
            "share_on_floor": clamp_share(funding_bps),
            "n_on_floor": int(np.sum(np.abs(funding_bps - FLOOR_BPS_PER_HOUR) < 1e-3)),
            "definition": "mean(|funding_bps - 0.125| < 1e-3)",
        },
        "descriptives": {
            "log_return_bps_per_hour": describe_series(returns_bps),
            "premium_bps_8h_equivalent": describe_series(premium_bps),
            "funding_bps_per_hour": describe_series(funding_bps),
        },
        "model_selection": {
            str(K): {
                "logL": float(ret_res[K]["ll"]),
                "n_params": int(ret_res[K]["model"].n_params()),
                "AIC": float(ret_res[K]["aic"]),
                "BIC": float(ret_res[K]["bic"]),
            } for K in (2, 3)
        },
        "delta_BIC_K3_minus_K2": float(ret_res[3]["bic"] - ret_res[2]["bic"]),
        "return_K3": _hmm_block(hmm3, ret_res[3]),
        "return_K2": _hmm_block(hmm2, ret_res[2]),
        "funding_MSAR": {
            "seed": int(fund_res["seed"]),
            "regime_order": "sigma2 ascending (0 = calm, 1 = stressed)",
            "parameterisation": "mean-adjusted: const[k] is the regime MEAN, not an intercept",
            "mu_const": [float(x) for x in fund_res["mu_const"]],
            "phi": [float(x) for x in fund_res["phi"]],
            "sigma2": [float(x) for x in fund_res["sigma2"]],
            "sigma": [float(x) for x in fund_res["sigma"]],
            "half_life_hours": [float(x) for x in fund_res["half_life_hours"]],
            "half_lives": fund_res["half_lives"],
            "half_life_ratio": fund_res["half_life_ratio"],
            "transition_matrix": fund_res["transition_matrix"],
            "params": fund_res["param_table"],
            "llf": fund_res["llf"],
            "AIC": fund_res["aic"],
            "BIC": fund_res["bic"],
            "nobs": fund_res["nobs"],
        },
        "software": _software_versions(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {OUT / 'summary.json'}")
    return summary


def _software_versions() -> dict:
    import matplotlib as _mpl
    import scipy
    import statsmodels
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        # Markov-switching start-parameter search has moved across 0.14.x
        # point releases; the exact version is part of the reproduction recipe.
        "statsmodels": statsmodels.__version__,
        "matplotlib": _mpl.__version__,
    }


# ----------------------------------------------------------------------
def _fetch_panel(coin: str, lookback_days: int) -> pd.DataFrame:
    """Network fallback. Imported lazily so the offline path needs no `requests`."""
    from .fetch_data_legacy import get_dataset
    print(f"[warn] no --panel given; fetching {coin} from the live Hyperliquid API. "
          "candleSnapshot retains only ~5,000 recent candles, so this CANNOT "
          "reproduce the paper's 2 Oct 2025 - 28 Apr 2026 window.")
    return get_dataset(coin, lookback_days=lookback_days)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m regime_engine.analysis",
        description="Fit the return HMM and the funding MS-AR, write figures, "
                    "tables and output/summary.json.",
    )
    ap.add_argument("--panel", metavar="PATH",
                    help="hourly panel to analyse (.csv or .parquet). Default: "
                         "data/processed/panel_BTC_subsample.csv if present.")
    ap.add_argument("--out", metavar="DIR", default=str(DEFAULT_OUT),
                    help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--fetch", action="store_true",
                    help="ignore --panel and fetch from the live Hyperliquid API")
    ap.add_argument("--coin", default="BTC", help="coin for --fetch (default: BTC)")
    ap.add_argument("--lookback-days", type=int, default=365,
                    help="lookback for --fetch (default: 365)")
    ap.add_argument("--msar-seed", type=int, default=MSAR_SEED,
                    help=f"seed for the MS-AR start search (default: {MSAR_SEED})")
    ap.add_argument("--funding-figure-paper-window", action="store_true",
                    help="regenerate fig3 on the paper's 2 Oct 2025 - 28 Apr 2026 "
                         "premium window, the window of "
                         "paper/tables/table3_funding_params.tex (the funding series "
                         "is not retention-bound), and exit")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any cached HMM fit and refit from scratch")
    return ap


def regenerate_paper_window_funding_figure(seed: int = MSAR_SEED) -> dict:
    """Redraw fig3_funding_dynamics on the paper's estimation window.

    fig1_regime_dashboard and fig2_distributions are drawn by ``main`` on
    whatever panel it is given, which in practice is the shipped sub-sample: an
    empirical return histogram and a price dashboard both need hourly data, and
    the shipped hourly panel starts on 3 December 2025, not 2 October 2025.
    fig3_funding_dynamics is different -- it is a function of the premium series
    alone, which ships in full, so it can be drawn on the paper's own window.
    That is what this entry point is for.

    The fit uses the multi-start protocol, so the half-life bars show the
    *maximum* of the likelihood. Drawn at the inferior optimum, the same figure
    would display 42.9 h against 1.7 h -- the 24.6x asymmetry the paper
    identifies as an artifact of single-start estimation -- next to a table
    reporting 11.2x.
    """
    from .panel import load_paper_window_premium

    y = load_paper_window_premium()
    print(f"[fig3] paper window: T = {len(y):,} hourly premium observations")
    fund_res = fit_msar_premium(y, seed=seed, n_pad=len(y))
    panel = pd.DataFrame({"premium": y / 1e4})
    tau = fund_res["half_life_hours"]
    print(f"[fig3] logL {fund_res['llf']:.2f}   calm {tau[0]:.2f} h   "
          f"stressed {tau[1]:.2f} h   ratio {tau[0] / tau[1]:.2f}")
    figure_funding_dynamics(panel, fund_res)
    return fund_res


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    set_output_dir(args.out)

    if args.funding_figure_paper_window:
        regenerate_paper_window_funding_figure(seed=args.msar_seed)
        return None

    from .paths import DEFAULT_PANEL
    if args.fetch:
        panel = _fetch_panel(args.coin, args.lookback_days)
        panel_source = f"live API: {args.coin}, {args.lookback_days}d lookback"
    else:
        panel_path = pathlib.Path(args.panel) if args.panel else DEFAULT_PANEL
        if not panel_path.exists():
            raise SystemExit(
                f"panel not found: {panel_path}\n"
                "Pass --panel PATH, or --fetch to pull from the live API "
                "(which cannot reproduce the paper's window)."
            )
        panel = load_panel(panel_path)
        panel_source = str(panel_path)

    print(f"\nLoaded {len(panel)} hourly observations from {panel['time'].iloc[0]} "
          f"to {panel['time'].iloc[-1]}")
    print(f"Source: {panel_source}")
    print(f"Output: {OUT}\n")

    ret_res = fit_return_hmms(panel, use_cache=not args.no_cache)
    fund_res = fit_funding_ms_ar(panel, seed=args.msar_seed)

    print("\n[Plot] regime dashboard ...")
    figure_regime_dashboard(panel, ret_res, fund_res, K=3)
    print("[Plot] regime distributions ...")
    figure_regime_distributions(panel, ret_res, K=3)
    print("[Plot] funding dynamics ...")
    figure_funding_dynamics(panel, fund_res)

    print("\n[LaTeX] tables ...")
    write_tables(panel, ret_res, fund_res)
    summary = write_summary(panel, ret_res, fund_res, panel_source=panel_source)

    print("\n=== headline numbers ===")
    print(f"  K=3 logL   {summary['return_K3']['logL']:.4f}")
    print(f"  K=3 BIC    {summary['return_K3']['BIC']:.2f}")
    print(f"  clamp share {summary['clamp']['share_on_floor']:.4%}")
    print(f"  half-lives  {summary['funding_MSAR']['half_life_hours']}")
    return summary


if __name__ == "__main__":
    main()
