"""Hyperliquid funding-rate mechanics.

Single source of truth for the clamp arithmetic, so that the unit tests and the
analysis pipeline exercise the *same* code and a regression cannot slip through
a test-local re-implementation.

Hyperliquid publishes an 8h-equivalent funding rate built from the time-averaged
premium index ``P̄_t`` and a constant interest rate ``r = 1e-4`` per 8h::

    F_t = P̄_t + clamp(r - P̄_t, -5e-4, +5e-4)

and charges ``F_t / 8`` every hour.  Whenever ``|r - P̄_t| < 5e-4`` the clamp is
not binding, ``F_t = r`` exactly, and the realised hourly rate pins to the
interest floor of ``r / 8 = 0.125`` bps/hour.  The band is asymmetric in the
premium: it binds for ``P̄_t`` in ``(-4, +6)`` bps (8h-equivalent).

Because ``F_t`` is a *censored* transform of ``P̄_t`` -- uninformative inside the
band -- the modelling object is the exchange-published premium index itself,
which the ``fundingHistory`` endpoint returns in its ``premium`` field.

References
----------
https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding (accessed 2026-08)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "R_8H",
    "CLAMP",
    "FLOOR_BPS_PER_HOUR",
    "funding_8h",
    "hourly_funding_bps",
    "clamp_is_binding",
    "clamp_share",
]

R_8H = 1e-4               # interest rate, per 8 hours
CLAMP = 5e-4              # clamp half-width on (r - premium)
FLOOR_BPS_PER_HOUR = 0.125  # = (r / 8) * 1e4


def funding_8h(premium_bar, r: float = R_8H, clamp: float = CLAMP):
    """8h-equivalent funding rate implied by the premium index.

    Accepts a scalar or an array; returns the same shape.
    """
    p = np.asarray(premium_bar, dtype=float)
    out = p + np.clip(r - p, -clamp, clamp)
    return float(out) if out.ndim == 0 else out


def hourly_funding_bps(premium_bar, r: float = R_8H, clamp: float = CLAMP):
    """Realised hourly funding in basis points: ``F_t / 8`` expressed in bps."""
    f8 = funding_8h(premium_bar, r=r, clamp=clamp)
    return np.asarray(f8, dtype=float) / 8.0 * 1e4 if np.ndim(f8) else f8 / 8.0 * 1e4


def clamp_is_binding(funding_bps, tol: float = 1e-3) -> np.ndarray:
    """Boolean mask: hours whose realised funding sits on the interest floor."""
    f = np.asarray(funding_bps, dtype=float)
    return np.abs(f - FLOOR_BPS_PER_HOUR) < tol


def clamp_share(funding_bps, tol: float = 1e-3) -> float:
    """Fraction of hours pinned to the 0.125 bps/h interest floor.

    Note the tolerance.  This counts hours within ``tol`` of the floor, which on
    the paper window is 2,306 of 5,001 (46.11%).  The manuscript quotes the
    *exact* count instead, 2,300 (46.0%) -- the six-observation difference is
    values near the floor but not on it.  The exact set coincides observation
    for observation with the hours whose premium lies inside the clamp's
    identity band.  On the shipped Dec 2025 - Apr 2026 hourly sub-sample this
    function returns 35.4%.
    """
    return float(np.mean(clamp_is_binding(funding_bps, tol=tol)))
