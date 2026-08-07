"""
Shared figure style and two plotting helpers that are easy to get subtly wrong.

`FigureSaver` numbers figures in the order they are produced, so a report can cite them
stably, and clears the directory first so a renumbered run does not leave the previous set
interleaved with the new one.

`threshold_coloured_hist` colours each bar by which side of a threshold it falls on, and
**refuses** to draw if the threshold is not exactly on a bin edge — a straddling bar would be
part-accepted and part-rejected and its colour would be a lie.

`label_grid_corner` labels one panel of a large grid. Labelling all 45 is unreadable;
labelling none means the reader has to guess the units.

Depends on: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#333333"
GOOD_GREEN = "#4C9A5B"
FLAGGED_GREY = "#B0B0B0"
SOFT_RED = "#E8736B"
MUTED = "#BBBBBB"


def use_house_style() -> None:
    """Applied once per notebook, so no plotting function has to repeat it."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "grid.color": "#999999",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.titlepad": 10,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelpad": 8,
        "axes.labelsize": 12,
        "axes.labelcolor": INK,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
    })


class FigureSaver:
    """Saves figures under stable, sortable names. Clears stale PNGs on construction."""

    def __init__(self, directory: Path, enabled: bool = True, clear: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled
        self.count = 0
        self.saved: list[Path] = []
        if enabled and clear and self.directory.is_dir():
            for stale in self.directory.glob("*.png"):
                stale.unlink()

    def __call__(self, fig, name: str) -> Path | None:
        if not self.enabled:
            plt.show()
            return None
        self.count += 1
        path = self.directory / f"{self.count:02d}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self.saved.append(path)
        plt.show()
        return path


def threshold_coloured_hist(
    ax, values, bins, threshold: float, good_below: bool = True,
    good_colour: str = GOOD_GREEN, bad_colour: str = FLAGGED_GREY,
):
    """Histogram whose bars are coloured by which side of `threshold` the bin lies on.

    Raises if `threshold` is not exactly a bin edge. Returns `(counts, edges, is_good)`.
    """
    bins = np.asarray(bins, dtype=float)
    if not np.any(np.isclose(bins, threshold)):
        raise ValueError(
            f"threshold {threshold} is not a bin edge of {bins[0]}..{bins[-1]}; "
            "a straddling bin cannot be honestly coloured"
        )
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    is_good = (edges[1:] <= threshold) if good_below else (edges[:-1] >= threshold)
    ax.bar(centres, counts, width=widths * 0.95,
           color=[good_colour if g else bad_colour for g in is_good],
           edgecolor="white", lw=0.4)
    return counts, edges, is_good


def label_grid_corner(axes, n_used: int, n_columns: int, x_label: str, y_label: str) -> None:
    """Label the bottom-left occupied panel of a grid of subplots."""
    if n_used == 0:
        return
    corner = axes[((n_used - 1) // n_columns) * n_columns]
    corner.set_xlabel(x_label, fontsize=9)
    corner.set_ylabel(y_label, fontsize=9)
