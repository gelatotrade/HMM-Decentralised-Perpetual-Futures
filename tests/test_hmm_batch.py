"""The batched Baum--Welch agrees with GaussianHMM and its extras do what they claim."""
import numpy as np
import pytest

from regime_engine import diagnostics as dg
from regime_engine import hmm_batch as hb
from regime_engine.hmm import GaussianHMM

A3 = np.array([[0.95, 0.04, 0.01],
               [0.03, 0.93, 0.04],
               [0.02, 0.18, 0.80]])
MU3 = np.array([0.0002, -0.0001, -0.0005])
SIG3 = np.array([0.0015, 0.0035, 0.0095])


def _chain(T=1500, seed=3, A=A3):
    x, _ = dg.simulate_gaussian_hmm(A, MU3, SIG3, T, np.random.default_rng(seed))
    return x


def _params(A=A3):
    return {"pi": dg._stationary(A), "A": A, "mu": MU3, "sigma": SIG3}


def _hmm(p):
    h = GaussianHMM(K=len(p["mu"]))
    h.pi, h.A, h.mu, h.sigma = (np.array(p[k], float) for k in ("pi", "A", "mu", "sigma"))
    return h


def test_e_step_matches_gaussian_hmm():
    x = _chain()
    p = _params()
    gamma, xi, G, ll = hb.e_step(x, hb.BatchParams.stack([p]))
    g2, xi2, ll2 = _hmm(p)._e_step(x)
    assert ll[0] == pytest.approx(ll2, abs=1e-8)
    np.testing.assert_allclose(gamma[0], g2, atol=1e-10)
    np.testing.assert_allclose(xi[0], xi2, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(xi[0], p["A"] * G[0], rtol=1e-12)
    # summed over t, the expected transitions out of i are the occupancy of i
    np.testing.assert_allclose(xi[0].sum(axis=1), g2[:-1].sum(axis=0), rtol=1e-9)


def test_m_step_matches_gaussian_hmm():
    x = _chain()
    p = _params()
    gamma, xi, _, _ = hb.e_step(x, hb.BatchParams.stack([p]))
    new = hb.m_step(x, gamma, xi)
    h = _hmm(p)
    g2, xi2, _ = h._e_step(x)
    h._m_step(x, g2, xi2)
    np.testing.assert_allclose(new.A[0], h.A, atol=1e-10)
    np.testing.assert_allclose(new.mu[0], h.mu, rtol=1e-9)
    np.testing.assert_allclose(new.sigma[0], h.sigma, rtol=1e-9)


def test_batch_members_are_independent():
    x1, x2 = _chain(seed=1), _chain(seed=2)
    p = hb.BatchParams.stack([_params(), _params()])
    _, _, _, ll = hb.e_step(np.array([x1, x2]), p)
    for x, v in ((x1, ll[0]), (x2, ll[1])):
        assert v == pytest.approx(_hmm(_params())._e_step(x)[2], abs=1e-8)


def _reference_multistart(x, K, n_starts, random_state, n_iter, tol=1e-6):
    """The multi-start loop GaussianHMM.fit ran before the protocol, in log space."""
    master = np.random.default_rng(random_state)
    best = None
    for _ in range(n_starts):
        h = GaussianHMM(K=K)
        h._init_random(x, np.random.default_rng(master.integers(0, 10**9)))
        prev = -np.inf
        for it in range(n_iter):
            gamma, xi, ll = h._e_step(x)
            h._m_step(x, gamma, xi)
            if ll - prev < tol and it > 5:
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, h.pi.copy(), h.A.copy(), h.mu.copy(), h.sigma.copy())
    ll, pi, A, mu, sigma = best
    o = np.argsort(sigma)
    return {"ll": ll, "A": A[np.ix_(o, o)], "mu": mu[o], "sigma": sigma[o]}


def test_multistart_fit_reproduces_the_log_space_multistart_loop():
    x = _chain(T=1200, seed=5)
    ref = _reference_multistart(x, 3, 4, 11, 60)
    got = hb.multistart_fit(x, 3, 4, 11, 60)
    b = got["best"]
    np.testing.assert_allclose(b["sigma"], ref["sigma"], rtol=1e-7)
    np.testing.assert_allclose(b["mu"], ref["mu"], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(b["A"], ref["A"], atol=1e-7)
    assert b["ll"] == pytest.approx(ref["ll"], abs=1e-6)


def test_gaussian_hmm_fit_without_polish_returns_the_selected_start():
    x = _chain(T=1200, seed=5)
    h = GaussianHMM(K=3, n_starts=4, n_iter=60, random_state=11, polish_tol=None).fit(x)
    b = hb.multistart_fit(x, 3, 4, 11, 60)["best"]
    np.testing.assert_allclose(h.sigma, b["sigma"], rtol=1e-12)
    np.testing.assert_allclose(h.A, b["A"], atol=1e-12)
    # the log-likelihood is that of the returned parameters
    assert h.log_likelihood_ == pytest.approx(h._e_step(x)[2], abs=1e-8)
    assert h.fit_report_["polish"] is None


def test_run_em_step_rule_stops_only_once_the_parameters_stop_moving():
    x = _chain(T=1200, seed=5)
    init = hb.multistart_fit(x, 3, 4, 11, 30)
    p = hb._start_params(x, 3, 4, 11).take([init["best"]["best_start"]])
    loose = hb.run_em(x, p, 20000, 1e-9, rule="abs")
    tight = hb.run_em(x, p, 20000, 1e-9, rule="abs", step_tol=1e-10)
    assert tight.stopped[0] and tight.last_step[0] < 1e-10
    assert tight.n_iter[0] >= loose.n_iter[0]
    assert abs(tight.last_delta[0]) < 1e-9
    # one more EM step from the returned parameters moves nothing
    g, xi, _, _ = hb.e_step(x, tight.params)
    assert hb.max_param_step(tight.params, hb.m_step(x, g, xi))[0] < 1e-10


def test_protocol_records_convergence_and_flags_a_capped_start():
    x = _chain(T=1200, seed=5)
    h = GaussianHMM(K=3, n_starts=4, random_state=11).fit(x)
    r = h.fit_report_
    assert h.converged_ and r["converged"] and r["all_starts_converged"]
    assert len(r["per_start_iterations"]) == len(r["per_start_converged"]) == 4
    assert all(n < r["iteration_cap"] for n in r["per_start_iterations"])
    assert r["polish"]["converged"]
    assert abs(r["polish"]["final_dlogL"]) < 1e-9
    assert r["polish"]["final_max_param_step"] < 1e-10
    assert r["logL"] == pytest.approx(h._e_step(x)[2], abs=1e-8)
    # a cap below what the starts need is reported, not hidden
    capped = GaussianHMM(K=3, n_starts=4, n_iter=8, random_state=11, polish_cap=5).fit(x)
    assert not capped.converged_
    assert not capped.fit_report_["all_starts_converged"]
    assert not capped.fit_report_["polish"]["converged"]


def test_polish_does_not_lower_the_likelihood_of_the_selected_start():
    x = _chain(T=1200, seed=5)
    ms = hb.multistart_fit(x, 3, 4, 11, 100_000)
    p = ms["_final_params"].take([ms["starts"].index(ms["best"]["best_start"])])
    h = GaussianHMM(K=3, n_starts=4, random_state=11).fit(x)
    assert h.log_likelihood_ >= hb.loglik(x, p)[0] - 1e-9


# ---------------------------------------------------------------------------
#  the shipped sub-period fit
# ---------------------------------------------------------------------------
def _shipped(name):
    import json

    from regime_engine.paths import DEFAULT_OUT
    path = DEFAULT_OUT / name
    if not path.exists():                       # pragma: no cover
        pytest.skip(f"{name} has not been generated")
    return json.loads(path.read_text())


def _sub_returns():
    import pandas as pd

    from regime_engine.paths import DEFAULT_PANEL
    return pd.read_csv(DEFAULT_PANEL)["log_return"].to_numpy(float)


def test_the_protocol_reproduces_summary_json_return_k3():
    s = _shipped("summary.json")["return_K3"]
    h = GaussianHMM(K=3, n_starts=8, random_state=45).fit(_sub_returns())
    np.testing.assert_allclose(h.sigma * 1e4, s["sigma_bps_per_hour"], rtol=1e-9)
    np.testing.assert_allclose(h.mu * 1e4, s["mu_bps_per_hour"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(h.A, s["A"], atol=1e-10)
    assert h.log_likelihood_ == pytest.approx(s["logL"], abs=1e-8)
    assert s["converged"] and s["convergence"]["converged"]


def test_the_150_iteration_cap_reproduces_the_archived_sub_period_fit():
    """The fit the earlier artifacts reported: the same starts stopped at 150 iterations."""
    old = _shipped("a31_evidence.json")["convergence"]["fits"]["analysis_K3"]["capped"]
    b = hb.multistart_fit(_sub_returns(), 3, 8, 45, 150)["best"]
    np.testing.assert_allclose(b["sigma"] * 1e4, old["sigma_bps"], rtol=1e-10)
    np.testing.assert_allclose(b["A"], old["A"], atol=1e-10)
    # 150 iterations left the high state at 87.68% annualised, the converged fit at 87.64%
    assert round(old["ann_sigma_pct"][2], 2) == 87.68


def test_snapshot_equals_a_capped_run_and_merge_equals_one_run():
    x = _chain(T=1000, seed=6)
    capped = hb.multistart_fit(x, 3, 6, 21, 25)
    long = hb.multistart_fit(x, 3, 6, 21, 400, snapshot_iters=(25,))
    np.testing.assert_allclose(long["snapshots"][25]["sigma"], capped["best"]["sigma"],
                               rtol=1e-12)
    parts = [hb.multistart_fit(x, 3, 6, 21, 400, starts=s) for s in ([0, 1, 2], [3, 4, 5])]
    for q in parts:
        q.pop("_final_params")
    merged = hb.merge_multistart(parts)
    assert merged["best"]["best_start"] == long["best"]["best_start"]
    np.testing.assert_allclose(merged["best"]["sigma"], long["best"]["sigma"], rtol=1e-12)
    assert merged["per_start"]["n_iter"] == long["per_start"]["n_iter"]


def test_em_is_monotone_and_constrained_cell_is_held():
    x = _chain()
    p = hb.BatchParams.stack([_params()])
    p.A = hb.move_cell_mass(p.A, (2, 0), [0.05])
    lls = []
    for _ in range(15):
        gamma, xi, _, ll = hb.e_step(x, p)
        lls.append(ll[0])
        p = hb.m_step(x, gamma, xi, fixed_cell=(2, 0), fixed_value=[0.05])
        assert p.A[0, 2, 0] == 0.05
        np.testing.assert_allclose(p.A[0].sum(axis=1), 1.0, rtol=1e-12)
    assert np.all(np.diff(lls) > -1e-9)


def test_move_cell_mass_keeps_rows_stochastic_and_proportions():
    A = hb.move_cell_mass(A3, (2, 0), 0.1)
    assert A[2, 0] == 0.1
    assert A[2].sum() == pytest.approx(1.0)
    assert A[2, 1] / A[2, 2] == pytest.approx(A3[2, 1] / A3[2, 2])
    np.testing.assert_array_equal(A[:2], A3[:2])


def test_score_matrix_is_the_derivative_of_the_loglik():
    x = _chain(T=800)
    p = _params()
    _, _, G, _ = hb.e_step(x, hb.BatchParams.stack([p]))
    eps = 1e-7
    for (i, j) in ((2, 0), (0, 1), (1, 1)):
        up, dn = np.array(A3), np.array(A3)
        up[i, j] += eps
        dn[i, j] -= eps
        f = [hb.loglik(x, hb.BatchParams.stack([{**p, "A": a}]))[0] for a in (up, dn)]
        assert (f[0] - f[1]) / (2 * eps) == pytest.approx(G[0, i, j], rel=1e-5)


def test_score_ratio_is_the_em_growth_factor_of_a_small_cell():
    x = _chain()
    p = hb.BatchParams.stack([_params()])
    p.A = hb.move_cell_mass(p.A, (2, 0), [0.0])
    res = hb.run_em(x, p, 2000, 1e-11, fixed_cell=(2, 0), fixed_value=[0.0], rule="abs")
    ratio = hb.boundary_score_ratio(x, res.params, (2, 0))[0]
    q = res.params.copy()
    q.A = hb.move_cell_mass(q.A, (2, 0), [1e-9])
    gamma, xi, _, _ = hb.e_step(x, q)
    nxt = hb.m_step(x, gamma, xi)
    assert nxt.A[0, 2, 0] / 1e-9 == pytest.approx(ratio, rel=1e-4)


def test_sort_by_sigma_permutes_consistently():
    perm = [2, 0, 1]
    p = _params()
    shuffled = {"pi": p["pi"][perm], "A": A3[np.ix_(perm, perm)], "mu": MU3[perm],
                "sigma": SIG3[perm]}
    s = hb.sort_by_sigma(hb.BatchParams.stack([shuffled])).member(0)
    np.testing.assert_allclose(s["A"], A3)
    np.testing.assert_allclose(s["sigma"], SIG3)
    np.testing.assert_allclose(s["mu"], MU3)
