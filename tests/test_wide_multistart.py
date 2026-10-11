"""The wide multi-start of model selection: its starting points and its search."""
import numpy as np
import pytest

from regime_engine import diagnostics as dg
from regime_engine import hmm_batch as hb

A3 = np.array([[0.95, 0.04, 0.01],
               [0.03, 0.93, 0.04],
               [0.02, 0.18, 0.80]])
MU3 = np.array([0.0002, -0.0001, -0.0005])
SIG3 = np.array([0.0015, 0.0035, 0.0095])


@pytest.fixture(scope="module")
def x():
    v, _ = dg.simulate_gaussian_hmm(A3, MU3, SIG3, 900, np.random.default_rng(7))
    return v


def _stochastic(p: hb.BatchParams):
    np.testing.assert_allclose(p.A.sum(axis=2), 1.0, atol=1e-12)
    np.testing.assert_allclose(p.pi.sum(axis=1), 1.0, atol=1e-12)
    assert (p.A >= 0).all() and (p.pi >= 0).all() and (p.sigma > 0).all()


class TestSpreadStarts:
    def test_shapes_and_stochastic(self, x):
        p = hb.spread_starts(x, 4, 30, random_state=1)
        assert (p.R, p.K) == (30, 4)
        _stochastic(p)

    def test_volatilities_cover_the_stated_range(self, x):
        lo, hi = hb.SPREAD_VOL_RANGE
        p = hb.spread_starts(x, 5, 400, random_state=2)
        f = p.sigma / x.std()
        assert f.min() >= lo - 1e-12 and f.max() <= hi + 1e-12
        # log-uniform: the log-factors fill the range, not only its middle
        assert np.quantile(np.log(f), 0.02) < np.log(lo) + 0.2
        assert np.quantile(np.log(f), 0.98) > np.log(hi) - 0.2

    def test_reproducible_from_the_seed(self, x):
        a = hb.spread_starts(x, 3, 5, random_state=9)
        b = hb.spread_starts(x, 3, 5, random_state=9)
        np.testing.assert_array_equal(a.sigma, b.sigma)
        np.testing.assert_array_equal(a.A, b.A)


class TestSplitStarts:
    def test_every_state_split_by_every_factor(self):
        parent = {"pi": dg._stationary(A3), "A": A3, "mu": MU3, "sigma": SIG3}
        p = hb.split_starts(parent, factors=(1.5, 2.0))
        assert (p.R, p.K) == (3 * 2, 4)
        _stochastic(p)

    def test_a_split_keeps_the_other_states_and_brackets_the_split_one(self):
        parent = {"pi": dg._stationary(A3), "A": A3, "mu": MU3, "sigma": SIG3}
        p = hb.split_starts(parent, factors=(2.0,))
        s = np.sort(p.sigma[1])                              # state 2 split by 2
        np.testing.assert_allclose(s, np.sort([SIG3[0], SIG3[2],
                                               SIG3[1] / 2.0, SIG3[1] * 2.0]))

    def test_a_split_leaves_the_likelihood_of_the_parent_order_unchanged(self, x):
        """With factor 1 the split model is the parent model written with K+1 states."""
        parent = {"pi": dg._stationary(A3), "A": A3, "mu": MU3, "sigma": SIG3}
        p = hb.split_starts(parent, factors=(1.0,))
        ll_parent = hb.loglik(x, hb.BatchParams.stack([parent]))[0]
        np.testing.assert_allclose(hb.loglik(x, p), ll_parent, rtol=0, atol=1e-8)


class TestWideMultistart:
    @pytest.fixture(scope="class")
    def fit(self, x):
        return hb.wide_multistart(x, 3, n_default=4, default_seed=11, n_spread=8,
                                  spread_seed=12, parent=None)

    def test_never_worse_than_the_default_protocol(self, x, fit):
        base = hb.fit_protocol(x, 3, 4, 11)
        assert fit["ll"] >= base["ll"] - 1e-6
        assert fit["default_protocol"]["ll"] == pytest.approx(base["ll"], abs=1e-6)

    def test_record_counts_starts_and_the_ones_that_reach_the_best(self, fit):
        rec = fit["record"]
        assert rec["n_starts"] == 12
        assert rec["start_families"] == {"default": 4, "spread": 8, "split": 0}
        assert 1 <= rec["n_starts_reaching_best"] <= 12
        assert sum(rec["n_reaching_best_by_family"].values()) == rec["n_starts_reaching_best"]
        assert rec["converged"]
        assert rec["polish"]["converged"]
        assert rec["logL"] == pytest.approx(fit["ll"])

    def test_returned_states_are_sorted_by_sigma(self, fit):
        s = np.asarray(fit["params"]["sigma"])
        assert (np.diff(s) > 0).all()

    def test_parent_adds_split_starts(self, x):
        parent = {"pi": dg._stationary(A3), "A": A3, "mu": MU3, "sigma": SIG3}
        out = hb.wide_multistart(x, 4, n_default=2, default_seed=3, n_spread=2,
                                 spread_seed=4, parent=parent, split_factors=(1.5,))
        assert out["record"]["start_families"] == {"default": 2, "spread": 2, "split": 3}
