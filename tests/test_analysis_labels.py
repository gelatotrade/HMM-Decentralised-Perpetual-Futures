"""Labels of the appendix figures drawn by regime_engine.analysis.

The return model's states are named as in the regime-parameter table ("State k",
by volatility), the premium model is named as the premium's, and numbers carry a
typographic minus rather than mathtext's binary one.
"""
import types

import numpy as np

from regime_engine import analysis as a


def _hmm(sigmas):
    return types.SimpleNamespace(K=len(sigmas), sigma=np.asarray(sigmas) / 1e4)


def test_state_labels_name_the_volatility_level():
    labels = a._regime_labels(_hmm([13.72, 33.09, 93.64]))
    assert labels[0].startswith("State 1 — low volatility")
    assert labels[2].startswith("State 3 — high volatility")
    assert "13.7 bps/h" in labels[0] and "93.6 bps/h" in labels[2]


def test_component_labels_use_a_typographic_minus():
    """Estimates carry hats, as in the dashboard; the numbers stay outside mathtext."""
    assert (a._component_label(0, -0.1116, 13.722)
            == "State 1: $\\hat\\mu$ = −0.11, $\\hat\\sigma$ = 13.7")
    assert (a._component_label(2, 0.4, 93.64)
            == "State 3: $\\hat\\mu$ = 0.40, $\\hat\\sigma$ = 93.6")


def test_math_is_set_in_the_serif_face():
    import matplotlib.pyplot as plt
    assert plt.rcParams["mathtext.fontset"] == "dejavuserif"


def test_the_dashboard_names_the_premium_model():
    assert a.PREMIUM_PANEL_LABEL == "Premium\nregime"
    assert a.FUNDING_AXIS_LABEL == "Funding rate (bps/h)"
