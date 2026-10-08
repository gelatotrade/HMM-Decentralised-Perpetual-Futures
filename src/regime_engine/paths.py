"""Canonical filesystem locations for the replication package.

Everything the pipeline writes goes to the *repository's* ``output/`` directory,
not to a directory inside the installed package.  Resolving the repo root by
walking up from this file keeps that true for a source checkout and for an
editable install alike; if the package is ever installed non-editably (so that
no repo is on disk) we fall back to the current working directory.
"""
from __future__ import annotations

import pathlib

__all__ = ["REPO_ROOT", "DEFAULT_OUT", "DEFAULT_PANEL", "DEFAULT_PREMIUM_FULL",
           "PAPER_WINDOW", "repo_root"]


def repo_root() -> pathlib.Path:
    """Return the repository root, or the CWD if the package is not in one."""
    here = pathlib.Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return pathlib.Path.cwd()


REPO_ROOT = repo_root()
DEFAULT_OUT = REPO_ROOT / "output"
DEFAULT_PANEL = REPO_ROOT / "data" / "processed" / "panel_BTC_subsample.csv"
DEFAULT_PREMIUM_FULL = REPO_ROOT / "data" / "interim" / "premium_BTC_full.csv"

# The estimation window of the paper, as a single definition. The funding rows of
# the summary-statistics table, the MS-AR parameter table and the funding figure
# are all computed on this slice; defining it once keeps the figure and the
# table beside it on the same fit.
PAPER_WINDOW = ("2025-10-02", "2026-04-28 08:00:00+00:00")
