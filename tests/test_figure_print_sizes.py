"""Text in the paper's figures prints at 7 pt or more at the width it is set at.

The size is read from the saved PDF itself: every font size its content
streams set (``/F<n> <size> Tf``), scaled by the ratio of the printed width to
the PDF's page width, which includes savefig's padding. The figures and their
widths are read from paper.tex, so a figure the paper sets narrower is checked
at that width.
"""
import pathlib
import re
import zlib

import pytest

from regime_engine.paths import REPO_ROOT

TEXT_WIDTH_CM = 15.8
MIN_PT = 7.0
FIGURES = REPO_ROOT / "paper" / "figures"


def pdf_font_sizes(path) -> tuple:
    """(page width in pt, every font size set in the content streams)."""
    raw = pathlib.Path(path).read_bytes()
    width = float(re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", raw).group(1))
    sizes = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            data = zlib.decompress(m.group(1))
        except zlib.error:
            continue
        sizes += [float(x) for x in re.findall(rb"/F\d+ ([\d.]+) Tf", data)]
    return width, sizes


def printed_min_pt(path, fraction: float = 1.0) -> float:
    width, sizes = pdf_font_sizes(path)
    return min(sizes) * fraction * TEXT_WIDTH_CM / 2.54 * 72 / width


def included_figures() -> list:
    """(file name, fraction of the text width) of every figure paper.tex includes;
    the fraction is None for a figure set at its natural size."""
    tex = (REPO_ROOT / "paper" / "paper.tex").read_text()
    out = []
    for opts, name in re.findall(r"\\includegraphics(?:\[([^\]]*)\])?\{figures/([^}]+)\}", tex):
        m = re.search(r"width=([\d.]*)\\textwidth", opts)
        out.append((name, (float(m.group(1)) if m.group(1) else 1.0) if m else None))
    return out


def test_the_paper_includes_its_figures():
    assert len(included_figures()) >= 10


@pytest.mark.parametrize("name,fraction", included_figures())
def test_text_prints_at_seven_points_where_the_paper_sets_it(name, fraction):
    pdf = FIGURES / name
    if fraction is None:                    # natural size: printed as drawn
        width, sizes = pdf_font_sizes(pdf)
        assert width <= TEXT_WIDTH_CM / 2.54 * 72 + 1, name
        assert min(sizes) >= MIN_PT, name
    else:
        assert printed_min_pt(pdf, fraction) >= MIN_PT, (name, fraction)


def test_the_reader_finds_the_sizes_matplotlib_set(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_title("title", fontsize=9)
    ax.tick_params(labelsize=6.5)
    fig.savefig(tmp_path / "f.pdf", bbox_inches="tight")
    plt.close(fig)
    width, sizes = pdf_font_sizes(tmp_path / "f.pdf")
    assert min(sizes) == 6.5 and max(sizes) == 9.0
    assert width < 4 * 72 + 1
