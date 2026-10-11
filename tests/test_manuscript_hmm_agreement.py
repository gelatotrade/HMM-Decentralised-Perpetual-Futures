"""The sub-period return-HMM fits behind the manuscript, pinned to the artifacts.

Every HMM fit behind these numbers runs each start to its tolerance and polishes
the selected one (GaussianHMM.fit). Appendix A and the tables kept by hand
(paper/tables/table5_robustness.tex and table6_diagnostics.tex, Tables C6 and
C3) quote the fits at printed precision, and Table C7 is the generated
table7_benchmarks.tex; a regeneration that moves one of them has to move the
manuscript with it. The numbers of Section 6.2 are checked in
test_manuscript_numbers.
"""
import json
import re

import pytest

from regime_engine.paths import DEFAULT_OUT, repo_root


def _load(name):
    path = DEFAULT_OUT / name
    if not path.exists():                       # pragma: no cover
        pytest.skip(f"{name} has not been generated")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def tex():
    return (repo_root() / "paper" / "paper.tex").read_text()


def _table(name):
    return (repo_root() / "paper" / "tables" / name).read_text()


def _pct(v, dec=1):
    return f"{v:.{dec}f}\\%"


# ---------------------------------------------------------------------------
#  every shipped fit converged
# ---------------------------------------------------------------------------
def test_every_shipped_hmm_fit_converged():
    s = _load("summary.json")
    for K in (2, 3):
        assert s[f"return_K{K}"]["converged"], K
        assert s[f"return_K{K}"]["convergence"]["all_starts_converged"], K
    r = _load("robustness.json")
    assert r["raw"]["converged"] and r["winsorised"]["converged"]
    m = _load("model_selection_extended.json")
    assert m["all_converged"]
    for rec in m["records"]:
        c = rec["convergence"]
        assert c["n_starts_converged"] == c["n_starts"], rec["K"]
        assert c["start_families"]["default"] == 40, rec["K"]
        assert c["polish"]["converged"], rec["K"]
        assert rec["default_protocol"]["converged"], rec["K"]
    assert _load("two_hourly_refit.json")["converged"]
    assert _load("diagnostics.json")["hmm_K3_converged"]
    b = _load("benchmarks.json")
    assert all(row["converged"] for row in b["rows"] if "HMM" in row["model"])


def test_the_text_says_the_fits_converge_from_the_starts_it_names(tex):
    assert ("each fit starts from about $200$ points: $40$ random starts drawn like the eight "
            "(seed $45 + K$), $160$ with state volatilities spread log-uniformly over $0.15$ to "
            "$3.5$ times the sample standard deviation (seed $20261009 + K$)") in tex
    m = _load("model_selection_extended.json")
    assert m["all_converged"]
    for r in m["records"]:
        fam = r["convergence"]["start_families"]
        assert (fam["default"], fam["spread"]) == (40, 160)
        assert 200 <= r["n_starts"] <= 215
    assert m["protocol"]["spread_vol_range_x_sd"] == [0.15, 3.5]
    assert (m["protocol"]["default_seed"], m["protocol"]["spread_seed"]) == ("45 + K",
                                                                            "20261009 + K")


def test_the_text_names_the_seeds_of_the_eight_starts(tex):
    """Eight random starts seeded 42 + K on the data: the K = 3 fits use 45, the K = 2 fit 44."""
    assert ("eight per fit; on the data seeded $42 + K$, that is $45$ for $K = 3$ and $44$ for "
            "$K = 2$, and on simulated paths one seed per path") in tex
    s = _load("summary.json")
    for K in (2, 3):
        conv = s[f"return_K{K}"]["convergence"]
        assert (conv["random_state"], conv["n_starts"]) == (42 + K, 8), K


# ---------------------------------------------------------------------------
#  the sub-period fits of Section 6.2
# ---------------------------------------------------------------------------
def test_sub_period_volatilities():
    v = _load("summary.json")["return_K3"]["annualised_vol_pct"]
    assert [round(x, 1) for x in v] == [12.8, 31.0, 87.6]
    w = _load("robustness.json")["winsorised"]["ann_vol_pct_unrounded"]
    assert round(w[2], 1) == 85.4


def test_sub_period_kernel_has_the_high_to_low_zero():
    A = _load("summary.json")["return_K3"]["A"]
    assert A[2][0] < 1e-8
    assert round(A[2][1], 2) == 0.26


def test_two_hourly_refit():
    d = _load("two_hourly_refit.json")
    assert [round(x, 1) for x in d["annualised_sigma_pct"]] == [11.0, 31.7, 88.9]
    assert round(100 * d["stationary"][2], 1) == 20.9


def test_more_than_three_states():
    recs = {r["K"]: r for r in _load("model_selection_extended.json")["records"]}
    bic = {K: round(recs[K]["BIC"], 1) for K in recs}
    assert bic == {2: -28356.8, 3: -28533.4, 4: -28557.0, 5: -28500.6}
    assert round(recs[3]["BIC"] - recs[4]["BIC"], 1) == 23.6
    assert round(recs[2]["BIC"] - recs[3]["BIC"], 1) == 176.6
    assert round(recs[5]["logL"] - recs[4]["logL"], 1) == 16.7


# ---------------------------------------------------------------------------
#  Tables C3, C6 and C7
# ---------------------------------------------------------------------------
def _hmm_tests():
    t = _load("diagnostics.json")["tests"]
    return t[1], t[2]          # filtered, smoothed


def _p2(p):
    return f"{p:.2f}"


def test_table_c3_hmm_rows():
    tab = _table("table6_diagnostics.tex")
    filt, smooth = _hmm_tests()

    def lb(b, block, i):
        r = b[block][i]
        return r["Q"], r["p"]

    q, p = lb(filt, "ljung_box_z", 0)
    q2, p2 = lb(filt, "ljung_box_z2", 0)
    qa, pa = lb(filt, "ljung_box_abs_z", 0)
    assert (f"filtered  & 10 & {q:.1f} [{_p2(p)}] & {q2:.1f} [{_p2(p2)}] & "
            f"{qa:.1f} [{pa:.3f}] \\\\") in tab
    q, p = lb(filt, "ljung_box_z", 1)
    q2, p2 = lb(filt, "ljung_box_z2", 1)
    qa, _ = lb(filt, "ljung_box_abs_z", 1)
    assert f"& 24 & {q:.1f} [{_p2(p)}] & {q2:.1f} [{_p2(p2)}] & {qa:.1f} [$<$0.0001]" in tab
    q, p = lb(smooth, "ljung_box_z", 0)
    q2, _ = lb(smooth, "ljung_box_z2", 0)
    qa, pa = lb(smooth, "ljung_box_abs_z", 0)
    assert (f"smoothed  & 10 & {q:.1f} [{_p2(p)}] & {q2:.1f} [$<$0.0001] & "
            f"{qa:.1f} [{pa:.4f}] \\\\") in tab
    q, p = lb(smooth, "ljung_box_z", 1)
    q2, _ = lb(smooth, "ljung_box_z2", 1)
    qa, _ = lb(smooth, "ljung_box_abs_z", 1)
    assert f"& 24 & {q:.1f} [{_p2(p)}] & {q2:.1f} [$<$0.0001] & {qa:.1f} [$<$0.0001]" in tab
    for name, b in (("filtered", filt), ("smoothed", smooth)):
        a, jb = b["arch_lm"], b["jarque_bera"]
        arch = f"{a['LM']:.1f} [{a['p']:.4f}]" if a["p"] >= 1e-4 else f"{a['LM']:.1f} [$<$0.0001]"
        jbp = f"{jb['p']:.4f}" if jb["p"] >= 1e-4 else "$<$0.0001"
        jbv = f"{jb['JB']:,.0f}".replace(",", "{,}")
        skew = f"$-{abs(jb['skew']):.2f}$" if jb["skew"] < 0 else f"{jb['skew']:.2f}"
        row = (f"\\HMM\\ $K{{=}}3$, {name}  & {arch} & {jbv} [{jbp}] & {skew} & "
               f"{jb['excess_kurtosis']:.2f} \\\\")
        assert row in tab, (row, tab)
    assert "31{,}194" in tab and "13 [0.0016]" in tab


def test_table_c6_rows():
    r = _load("robustness.json")
    tab = _table("table5_robustness.tex")
    for label, block in (("Raw ($T=3{,}507$)", "raw"),
                         ("Winsorised (extreme 0.1\\%)", "winsorised")):
        cells = " & ".join(_pct(v) for v in r[block]["ann_vol_pct_unrounded"])
        assert re.search(re.escape(label) + r"\s*& " + re.escape(cells), tab), (label, cells)
    assert r["winsor_n_clipped"] == 4
    assert r["raw"]["A_high_to_low"] == 0.0 == r["winsorised"]["A_high_to_low"]


def test_table_c7_is_the_generated_table():
    gen = (DEFAULT_OUT / "tables" / "table7_benchmarks.tex").read_text()
    assert _table("table7_benchmarks.tex") == gen
    rows = {row["model"]: row for row in _load("benchmarks.json")["rows"]}
    h3 = rows["Gaussian HMM, K=3"]
    assert f"{h3['logL']:,.1f}" == "14,323.8"
    assert f"{h3['BIC']:,.1f}" == "-28,533.4"


def test_figure_8_names_the_bound_it_draws(tex):
    from regime_engine import paper_figures as pf
    b = _load("a31_evidence.json")["profile_likelihood"]["upper_bound_one_sided_95"]
    assert f"{100 * b:.2f}%/h" in pf.high_low_label()
    assert "the bound on its arrow is the profile-likelihood bound of the hourly sub-period" in tex
