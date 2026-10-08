"""The fit cache must not survive a change to the estimator.

``fit_return_hmms`` memoises the K=2 and K=3 fits on disk. A key built from a
SHA-256 of the panel alone is sufficient only if the estimator never changes.
If the label-switching rule or the multi-start protocol changes, such a key
would reload fits computed under the old settings, and every artifact
regenerated from them -- summary.json, the LaTeX tables, the figures -- would
reflect those settings (after a change of labelling rule, the old state order)
with no warning and no mismatch a reader could see.

The key therefore identifies the code as well as the data.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from regime_engine import analysis


@pytest.fixture
def tiny_panel():
    rng = np.random.default_rng(7)
    n = 400
    r = np.concatenate([rng.normal(0, 5e-4, n // 2), rng.normal(0, 3e-3, n // 2)])
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "log_return": r,
        "premium": np.zeros(n),
    })


def test_cache_key_covers_the_estimator_not_only_the_panel(tiny_panel, tmp_path):
    """Two different estimators on one panel must not share a cache entry."""
    analysis.set_output_dir(tmp_path)
    analysis.fit_return_hmms(tiny_panel, use_cache=True)
    written = sorted(p.name for p in (tmp_path / "cache").glob("return_hmms_*.pkl"))
    assert len(written) == 1, written

    from regime_engine.panel import panel_fingerprint
    key_before = analysis.fit_cache_key(tiny_panel)
    assert panel_fingerprint(tiny_panel) in key_before, key_before

    original = analysis.HMM_N_STARTS
    try:
        # A different number of random restarts is a different estimator: it can
        # land on a different local maximum, so its fits must not be interchangeable.
        analysis.HMM_N_STARTS = original + 1
        key_after = analysis.fit_cache_key(tiny_panel)
    finally:
        analysis.HMM_N_STARTS = original

    assert key_before != key_after, (
        "the cache key did not move when the estimator did; a stale fit "
        "would be reloaded silently"
    )
    assert analysis.fit_cache_key(tiny_panel) == key_before, "key must be stable"


def test_cache_round_trips_when_nothing_changed(tiny_panel, tmp_path):
    analysis.set_output_dir(tmp_path)
    first = analysis.fit_return_hmms(tiny_panel, use_cache=True)
    second = analysis.fit_return_hmms(tiny_panel, use_cache=True)
    assert np.isclose(first[3]["ll"], second[3]["ll"])
    assert np.allclose(first[3]["model"].sigma, second[3]["model"].sigma)


def test_a_cache_file_from_a_foreign_estimator_is_ignored(tiny_panel, tmp_path):
    """Plant a poisoned entry under a panel-only cache name; it must not be used."""
    analysis.set_output_dir(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    from regime_engine.panel import panel_fingerprint
    stale = cache_dir / f"return_hmms_{panel_fingerprint(tiny_panel)}.pkl"
    stale.write_bytes(pickle.dumps({"panel_fingerprint": "poisoned"}))

    res = analysis.fit_return_hmms(tiny_panel, use_cache=True)
    assert 3 in res, "the panel-only cache entry was loaded instead of refitting"
    assert np.all(np.diff(res[3]["model"].sigma) > 0)
