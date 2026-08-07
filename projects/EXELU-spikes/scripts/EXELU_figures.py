"""
Every figure in this project, and nothing but figures.

Each function takes data that has already been computed, draws one figure, and returns it.
None of them computes an analysis result — if a number appears on a plot it was passed in, so
a figure and a table can never disagree.

Colour is used for one thing at a time: teal is cortex, orange is hippocampus, soft red marks
a threshold or a stimulus transition, grey means "not enough data to interpret".

These are EXELU-specific: they know about cortex and hippocampus, about the theta channel, and
about this experiment's stimulus conditions. The reusable pieces — the house style, the figure
saver, the threshold-coloured histogram — live in the shared `plotstyle` module.
"""

from __future__ import annotations

# Importing EXELU_paths puts <framework>/scripts on sys.path, so the shared
# analysis modules below resolve wherever this file is imported from.
import EXELU_paths  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import waveforms as wf
from EXELU_config import FS, HILL_FP_THRESHOLD_PCT, Params
from correlograms import autocorrelogram, rebin
from entrainment import transition_response
from plotstyle import (
    FLAGGED_GREY, GOOD_GREEN, INK, MUTED, SOFT_RED,
    label_grid_corner, threshold_coloured_hist, use_house_style,  # noqa: F401
)
from psth import block_psth
import EXELU_records as records
import EXELU_regions as regions
from EXELU_regions import CORTEX
from EXELU_units import waveform_of, wide_acg

# ---- palette ----------------------------------------------------------------
CTX_COLOUR = "#008883"
HPC_COLOUR = "#C77D00"

CELL_TYPE_COLOUR = {
    wf.PRINCIPAL: "#3D5A80",
    wf.INTERNEURON: "#C1666B",
    wf.UNCLASSIFIED: "#999999",
}


def _region_colour(region: str) -> str:
    return CTX_COLOUR if region == CORTEX else HPC_COLOUR


def _session_suptitle(session, headline: str, subtitle: str = "") -> str:
    tail = f"\n{subtitle}" if subtitle else ""
    return f"{headline}\n{records.label(session)}{tail}"


# =============================================================================
# review point 2 — autocorrelograms with identity and cell type
# =============================================================================

def plot_acg_grid(session, store, units: pd.DataFrame, params: Params, n_columns: int = 6):
    """Short-window autocorrelogram per unit, with an inset waveform and the cell-type call.

    The refractory window is marked with two dashed lines and a 6 % fill rather than a solid
    block, so counts *inside* the window stay visible — a solid patch hides exactly the
    evidence the panel exists to show.
    """
    ordered = units.sort_values(["region", "depth_um"]).reset_index(drop=True)
    templates = wf.unwhiten_templates(session.array("templates"),
                                      session.array("whitening_mat_inv"))
    n_rows = int(np.ceil(len(ordered) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3.0 * n_columns, 2.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, unit) in zip(axes, ordered.iterrows()):
        centres, counts = autocorrelogram(
            store.after(int(unit.unit), params.settle_s),
            params.acg_bin_ms, params.acg_window_ms, FS, params.acg_max_spikes,
        )
        colour = CELL_TYPE_COLOUR.get(unit.get("cell_type", wf.UNCLASSIFIED), "#666666")
        ax.step(centres, counts, where="mid", color=colour, lw=1.0)
        ax.fill_between(centres, counts, step="mid", color=colour, alpha=0.20, lw=0)

        for x in (-params.refractory_ms, params.refractory_ms):
            ax.axvline(x, color=SOFT_RED, ls="--", lw=1.1, alpha=0.9, zorder=3)
        ax.axvspan(-params.refractory_ms, params.refractory_ms, color=SOFT_RED, alpha=0.06, lw=0)
        for lag in (-25, 25):
            ax.axvline(lag, color="#999999", ls=":", lw=0.7)

        ax.set_title(
            f"u{int(unit.unit)} · {unit.region} · {str(unit.get('cell_type', ''))[:4]}",
            fontsize=9, fontweight="bold", pad=3, color=colour,
        )
        ax.text(
            0.02, 0.96,
            f"contam {unit.contam_pct:.1f}%\nFR {unit.fr_hz:.1f} Hz\n"
            f"t2p {unit.get('trough_to_peak_ms', np.nan):.2f} ms\n{unit.depth_um:.0f} µm",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
            color="#DC2511" if unit.contam_pct > params.clean_contam_pct else "#4C7A2B",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5), zorder=5,
        )

        inset = ax.inset_axes([0.70, 0.62, 0.28, 0.34])
        wave = waveform_of(session, int(unit.unit), templates)
        inset.plot(wave, color=colour, lw=1.0)
        inset.set_xticks([]); inset.set_yticks([]); inset.grid(False)
        for spine in inset.spines.values():
            spine.set_visible(False)

        ax.set_xlim(-params.acg_window_ms, params.acg_window_ms)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)

    for ax in axes[len(ordered):]:
        ax.axis("off")
    label_grid_corner(axes, len(ordered), n_columns, "lag [ms]", "spike pairs")

    fig.suptitle(
        _session_suptitle(
            session,
            f"Autocorrelograms — {len(ordered)} units, {params.acg_bin_ms} ms bins, "
            f"±{params.acg_window_ms:.0f} ms",
            "soft red = ±2.5 ms refractory window · grey dotted = 25 ms (40 Hz) · "
            "inset = template on peak channel",
        ),
        fontsize=15, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    return fig


# =============================================================================
# review point 3 — extended autocorrelogram for theta
# =============================================================================

def plot_acg_theta_grid(session, store, units: pd.DataFrame, params: Params, n_columns: int = 6):
    """Long-window autocorrelogram, wide enough to show theta-period side-peaks.

    Theta is 4-12 Hz, a period of 83-250 ms. The shaded band is 100-140 ms, where a
    theta-modulated unit's first side-peak falls, and the hatched band at 50-70 ms is the
    corresponding trough; the theta index is built from exactly those two windows.
    """
    ordered = units.sort_values(["region", "depth_um"]).reset_index(drop=True)
    factor = max(int(round(params.acg_wide_display_bin_ms / params.acg_wide_bin_ms)), 1)
    n_rows = int(np.ceil(len(ordered) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3.0 * n_columns, 2.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, unit) in zip(axes, ordered.iterrows()):
        centres, counts = wide_acg(store, int(unit.unit), params, FS)
        centres, counts = rebin(centres, counts, factor)
        colour = CELL_TYPE_COLOUR.get(unit.get("cell_type", wf.UNCLASSIFIED), "#666666")
        ax.fill_between(centres, counts, step="mid", color=colour, alpha=0.25, lw=0)
        ax.step(centres, counts, where="mid", color=colour, lw=0.9)

        for sign in (-1, 1):
            ax.axvspan(sign * params.theta_peak_ms[0], sign * params.theta_peak_ms[1],
                       color=GOOD_GREEN, alpha=0.14, lw=0)
            ax.axvspan(sign * params.theta_trough_ms[0], sign * params.theta_trough_ms[1],
                       color=SOFT_RED, alpha=0.10, lw=0)

        ax.set_title(f"u{int(unit.unit)} · {unit.region}", fontsize=9, fontweight="bold",
                     pad=3, color=colour)
        ax.text(
            0.02, 0.96,
            f"θ index {unit.get('theta_index', np.nan):+.3f}\n"
            f"burst {unit.get('burst_index', np.nan):.2f}\nFR {unit.fr_hz:.1f} Hz",
            transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5), zorder=5,
        )
        ax.set_xlim(-params.acg_wide_window_ms, params.acg_wide_window_ms)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)

    for ax in axes[len(ordered):]:
        ax.axis("off")
    label_grid_corner(axes, len(ordered), n_columns, "lag [ms]", "spike pairs")

    fig.suptitle(
        _session_suptitle(
            session,
            f"Extended autocorrelograms — ±{params.acg_wide_window_ms:.0f} ms, "
            f"{params.acg_wide_display_bin_ms:.0f} ms bins",
            f"green = {params.theta_peak_ms[0]:.0f}–{params.theta_peak_ms[1]:.0f} ms theta peak · "
            f"red = {params.theta_trough_ms[0]:.0f}–{params.theta_trough_ms[1]:.0f} ms trough",
        ),
        fontsize=15, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    return fig


def plot_celltype_summary(session, units: pd.DataFrame, params: Params,
                          antimodes: np.ndarray | None = None):
    """The evidence behind the cell-type labels: width distribution, width vs rate, waveforms."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) width distribution and where the split falls
    width = units.trough_to_peak_ms.dropna()
    bins = np.arange(0, width.max() + 0.06, 0.05)
    axes[0].hist(width, bins=bins, color="#8C9EA8", edgecolor="white", lw=0.6)
    axes[0].axvline(params.t2p_split_ms, color=SOFT_RED, ls="--", lw=1.8,
                    label=f"split used: {params.t2p_split_ms:.2f} ms")
    antimodes = np.asarray([] if antimodes is None else antimodes, dtype=float)
    for i, dip in enumerate(antimodes):
        axes[0].axvline(dip, color="#3D5A80", ls=":", lw=1.6,
                        label=f"density dip: {dip:.2f} ms" if i == 0 else None)
    if antimodes.size == 0:
        axes[0].plot([], [], " ", label="no prominent density dip\n(distribution not separable)")
    axes[0].set_xlabel("trough-to-peak duration [ms]")
    axes[0].set_ylabel("units")
    axes[0].set_title("(a) waveform width")
    axes[0].legend(fontsize=8)

    # (b) width vs rate, the classical plane
    for label, group in units.groupby("cell_type"):
        axes[1].scatter(group.trough_to_peak_ms, group.fr_hz, s=60,
                        color=CELL_TYPE_COLOUR.get(label, "#999999"),
                        edgecolor="w", lw=0.6, label=f"{label} (n={len(group)})", zorder=3)
    inconsistent = units[~units.cell_type_consistent.fillna(False)]
    axes[1].scatter(inconsistent.trough_to_peak_ms, inconsistent.fr_hz, s=170,
                    facecolor="none", edgecolor="k", lw=1.0, zorder=2,
                    label="physiology disagrees")
    axes[1].axvline(params.t2p_split_ms, color=SOFT_RED, ls="--", lw=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("trough-to-peak duration [ms]")
    axes[1].set_ylabel("firing rate [Hz]")
    axes[1].set_title("(b) width vs rate")
    axes[1].legend(fontsize=8)

    # (c) mean template per class, peak-normalised
    templates = wf.unwhiten_templates(session.array("templates"),
                                      session.array("whitening_mat_inv"))
    for label, group in units.groupby("cell_type"):
        waves = []
        for unit_id in group.unit:
            wave = waveform_of(session, int(unit_id), templates)
            depth = abs(wave.min())
            if depth > 0:
                waves.append(wave / depth)
        if not waves:
            continue
        stacked = np.array(waves)
        time_ms = (np.arange(stacked.shape[1]) - stacked.shape[1] // 2) / 30.0
        mean = stacked.mean(0)
        axes[2].plot(time_ms, mean, color=CELL_TYPE_COLOUR.get(label, "#999999"), lw=2.0, label=label)
        axes[2].fill_between(time_ms, mean - stacked.std(0), mean + stacked.std(0),
                             color=CELL_TYPE_COLOUR.get(label, "#999999"), alpha=0.15, lw=0)
    axes[2].axhline(0, color="#999999", lw=0.8)
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("amplitude (normalised to trough)")
    axes[2].set_title("(c) mean template per class")
    axes[2].legend(fontsize=9)

    fig.suptitle(_session_suptitle(session, "Cell-type classification"), fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# =============================================================================
# review point 4 — contamination metric comparison
# =============================================================================

def plot_contamination_comparison(session, units: pd.DataFrame, params: Params,
                                  ks_threshold: float, hill_threshold: float = HILL_FP_THRESHOLD_PCT):
    """Kilosort's ContamPct against Hill's f_p, each judged at its own threshold.

    The two thresholds differ (20 % and 10 %) because the two metrics are different
    quantities: ContamPct is an unbounded density ratio, f_p is a bounded fraction of
    misassigned spikes. A single shared cut-off would be a category error.
    """
    good = units[units.label == "good"]
    mua = units[units.label == "mua"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    # ---- (a) Kilosort ContamPct ------------------------------------------
    top = float(np.nanmax(units.contam_pct)) if len(units) else 100.0
    bins_ks = np.arange(0, np.ceil(top / 5) * 5 + 5, 5.0)
    counts, _, is_good = threshold_coloured_hist(
        axes[0], units.contam_pct.dropna(), bins_ks, ks_threshold
    )
    axes[0].axvline(ks_threshold, color=SOFT_RED, ls="--", lw=1.8)
    axes[0].axvline(100, color="#555555", ls=":", lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.7, max(counts.max() * 2, 10))
    axes[0].set_xlabel("Kilosort ContamPct [%]  (density ratio, unbounded)")
    axes[0].set_ylabel("clusters (log scale)")
    axes[0].set_title(
        f"(a) ContamPct — green = accepted at {ks_threshold:.0f} %\n"
        f"{int(counts[is_good].sum())} of {int(counts.sum())} clusters accepted", fontsize=13
    )
    axes[0].text(102, axes[0].get_ylim()[1] * 0.3, "100 % =\nbaseline\ndensity",
                 fontsize=8, color="#555555")

    # ---- (b) Hill f_p ------------------------------------------------------
    bins_hill = np.arange(0, 52, 2.0)
    finite = units.hill_fp_pct.dropna()
    counts_h, _, is_good_h = threshold_coloured_hist(axes[1], finite, bins_hill, hill_threshold)
    axes[1].axvline(hill_threshold, color=SOFT_RED, ls="--", lw=1.8)

    n_saturated = int(units.hill_fp_pct.isna().sum())
    axes[1].bar(56, n_saturated, width=3.0, color=FLAGGED_GREY, hatch="///",
                edgecolor="#666666", lw=0.8)
    axes[1].axvline(52, color="#999999", lw=0.8)
    axes[1].text(56, max(n_saturated, 1) * 1.15, "undefined\n(saturated)",
                 ha="center", va="bottom", fontsize=8, color="#555555")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.7, max(max(counts_h.max(), n_saturated) * 2, 10))
    axes[1].set_xticks(list(np.arange(0, 51, 10)) + [56])
    axes[1].set_xticklabels([str(v) for v in np.arange(0, 51, 10)] + ["sat."])
    axes[1].set_xlabel(r"Hill 2011 $f_p$ [%]  (fraction, bounded at 50 %)")
    axes[1].set_ylabel("clusters (log scale)")
    axes[1].set_title(
        f"(b) $f_p$ — green = accepted at {hill_threshold:.0f} %\n"
        f"{int(counts_h[is_good_h].sum())} accepted, {n_saturated} saturated", fontsize=13
    )

    # ---- (c) do they agree, on the good units? ----------------------------
    saturated_row = 52.0
    plotted = good.copy()
    plotted["hill_plot"] = plotted.hill_fp_pct.fillna(saturated_row)
    plotted["ks_flagged"] = plotted.contam_pct > ks_threshold
    plotted["hill_flagged"] = (plotted.hill_fp_pct > hill_threshold) | plotted.hill_fp_pct.isna()

    axes[2].axhspan(saturated_row - 1.5, saturated_row + 4, color="#EEEEEE", zorder=0)
    for region in plotted.region.unique():
        subset = plotted[plotted.region == region]
        finite_mask = subset.hill_fp_pct.notna()
        axes[2].scatter(subset.loc[finite_mask, "contam_pct"], subset.loc[finite_mask, "hill_plot"],
                        s=55, color=_region_colour(region), edgecolor="w", lw=0.6,
                        label=region, zorder=3)
        axes[2].scatter(subset.loc[~finite_mask, "contam_pct"], subset.loc[~finite_mask, "hill_plot"],
                        s=70, marker="^", facecolor="none", edgecolor=_region_colour(region),
                        lw=1.2, zorder=3)
    axes[2].axvline(ks_threshold, color=SOFT_RED, ls="--", lw=1.2)
    axes[2].axhline(hill_threshold, color=SOFT_RED, ls="--", lw=1.2)
    axes[2].set_xlim(-1.5, max(ks_threshold * 1.4, float(good.contam_pct.max()) * 1.15))
    axes[2].set_ylim(-3, 58)

    disagree = int((plotted.ks_flagged != plotted.hill_flagged).sum())
    quadrants = (
        ((0.30, 0.08), int((~plotted.ks_flagged & ~plotted.hill_flagged).sum()), "both accept"),
        ((0.30, 0.72), int((~plotted.ks_flagged & plotted.hill_flagged).sum()), "KS accepts,\nHill rejects"),
        ((0.85, 0.72), int((plotted.ks_flagged & plotted.hill_flagged).sum()), "both reject"),
        ((0.85, 0.10), int((plotted.ks_flagged & ~plotted.hill_flagged).sum()), "KS rejects,\nHill accepts"),
    )
    for (fx, fy), n, label in quadrants:
        axes[2].text(fx, fy, f"{n}\n{label}", transform=axes[2].transAxes, ha="center",
                     va="center", fontsize=9, color="#444444",
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    axes[2].text(0.99, 0.92, "saturated — $f_p$ undefined", transform=axes[2].transAxes,
                 ha="right", fontsize=8, color="#666666")
    axes[2].set_xlabel("Kilosort ContamPct [%]")
    axes[2].set_ylabel(r"Hill 2011 $f_p$ [%]")
    axes[2].set_title(f"(c) `good` units at each metric's own threshold\n"
                      f"{disagree} of {len(plotted)} disagree", fontsize=13)
    axes[2].legend(fontsize=9, loc="center left")

    fig.suptitle(
        _session_suptitle(
            session,
            f"Contamination — {len(good)} good + {len(mua)} mua clusters",
            rf"Hill parameters $t_{{ref}}$ = {params.hill_t_ref_ms:.1f} ms, "
            rf"$t_{{cens}}$ = {params.hill_t_cens_ms:.1f} ms",
        ),
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# =============================================================================
# review point 6 — block PSTH
# =============================================================================

def plot_baseline_variability(session, spread: pd.DataFrame, window_s: float = 2.0):
    """Trial-to-trial variability of the pre-stimulus baseline — the case for point 6.

    If the coefficient of variation were near zero, per-trial standardisation would be
    pointless. This panel is the evidence that it is not.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for region in sorted(spread.region.unique()):
        subset = spread[spread.region == region]
        axes[0].scatter(subset.baseline_mean_hz, subset.baseline_sd_hz, s=45,
                        color=_region_colour(region), edgecolor="w", lw=0.5, label=region)
    grid = np.linspace(max(spread.baseline_mean_hz.min(), 1e-3), spread.baseline_mean_hz.max(), 100)
    axes[0].plot(grid, np.sqrt(grid / window_s), color="#555555", ls="--", lw=1.2,
                 label=f"Poisson expectation ({window_s:.0f} s window)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("mean baseline rate across trials [Hz]")
    axes[0].set_ylabel("SD of baseline rate across trials [Hz]")
    axes[0].set_title("baseline varies more than Poisson")
    axes[0].legend(fontsize=9)

    axes[1].hist(spread.baseline_cv.dropna(), bins=30, color="#8C9EA8", edgecolor="white", lw=0.5)
    axes[1].axvline(float(spread.baseline_cv.median()), color=SOFT_RED, ls="--", lw=1.6,
                    label=f"median CV = {spread.baseline_cv.median():.2f}")
    axes[1].set_xlabel("coefficient of variation of the per-trial baseline")
    axes[1].set_ylabel("unit × condition")
    axes[1].set_title("how much each trial's baseline moves")
    axes[1].legend(fontsize=9)

    fig.suptitle(_session_suptitle(session, "Why the block PSTH is standardised per trial"),
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_block_psth_heatmap(session, centres_by_condition: dict, z_by_condition: dict,
                            ordered: pd.DataFrame, n_ctx: int, params: Params,
                            v_max: float = 6.0, row_labels: bool = True):
    """Per-trial-standardised block PSTH, cortex above hippocampus."""
    conditions = list(z_by_condition)
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.2 * len(conditions), 7.5), sharey=True)
    axes = np.atleast_1d(axes)

    image = None
    for ax, condition in zip(axes, conditions):
        centres, z = centres_by_condition[condition], z_by_condition[condition]
        image = ax.imshow(z, aspect="auto", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vcenter=0, vmin=-v_max, vmax=v_max),
                          extent=[centres[0], centres[-1], z.shape[0] - 0.5, -0.5],
                          interpolation="nearest")
        ax.axvline(0, color="k", lw=1.2)
        ax.axvline(params.psth_post_s, color="k", lw=0.8, ls=":")
        ax.axhline(n_ctx - 0.5, color="k", lw=2.2)
        ax.set_title(condition, fontsize=13)
        ax.set_xlabel("time from block onset [s]")
        ax.grid(False)

    if row_labels:
        axes[0].set_yticks(np.arange(len(ordered)))
        axes[0].set_yticklabels(
            [f"u{u} · {f:.1f} Hz" for u, f in zip(ordered.unit, ordered.fr_hz)], fontsize=7
        )
        for label, region in zip(axes[0].get_yticklabels(), ordered.region):
            label.set_color(_region_colour(region))
            label.set_fontweight("bold")
    axes[0].set_ylabel("unit", fontsize=12)

    bar = fig.colorbar(image, ax=axes, fraction=0.014, pad=0.01)
    bar.set_label("z, per-trial baseline removed", fontsize=11)
    fig.suptitle(
        _session_suptitle(
            session,
            f"Block PSTH — {len(ordered)} units, {params.psth_bin_s * 1000:.0f} ms bins, "
            "standardised per trial",
            f"cortex above the black line (n={n_ctx}, teal), hippocampus below "
            f"(n={len(ordered) - n_ctx}, orange); descending firing rate within each",
        ),
        fontsize=15, fontweight="bold", y=1.02,
    )
    return fig


def plot_mua_depth_heatmap(session, centres_by_condition: dict, z_by_condition: dict,
                           ordered: pd.DataFrame, n_ctx: int, params: Params, v_max: float = 6.0):
    """Every individual multi-unit cluster, ordered shallow to deep, with landmarks."""
    conditions = list(z_by_condition)
    fig, axes = plt.subplots(1, len(conditions), figsize=(5.0 * len(conditions), 8.5), sharey=True)
    axes = np.atleast_1d(axes)
    landmarks = regions.landmarks(session)

    image = None
    for ax, condition in zip(axes, conditions):
        centres, z = centres_by_condition[condition], z_by_condition[condition]
        image = ax.imshow(z, aspect="auto", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vcenter=0, vmin=-v_max, vmax=v_max),
                          extent=[centres[0], centres[-1], z.shape[0] - 0.5, -0.5],
                          interpolation="nearest")
        ax.axvline(0, color="k", lw=1.2)
        ax.axhline(n_ctx - 0.5, color="#004E50", ls="--", lw=2.0)
        for name, depth in sorted(landmarks.items(), key=lambda kv: kv[1]):
            if name == "thetaCh":
                continue
            row = int(np.searchsorted(ordered.depth_um.values, depth))
            if 0 < row < len(ordered):
                ax.axhline(row - 0.5, color="#576838", ls="--", lw=1.0)
                if ax is axes[-1]:
                    ax.text(centres[-1] * 1.02, row - 0.5, f" {name.replace('Ch', '')}",
                            fontsize=11, va="center", color="#576838")
        if ax is axes[-1]:
            ax.text(centres[-1] * 1.02, n_ctx - 0.5, " theta", fontsize=11,
                    va="center", color="#004E50")
        ax.set_title(condition, fontsize=13)
        ax.set_xlabel("time from block onset [s]")
        ax.grid(False)

    step = max(len(ordered) // 12, 1)
    rows = list(range(0, len(ordered), step))
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([f"{ordered.depth_um.iloc[r]:.0f}" for r in rows], fontsize=9)
    for label, row in zip(axes[0].get_yticklabels(), rows):
        label.set_color(_region_colour(ordered.region.iloc[row]))
        label.set_fontweight("bold")
    axes[0].set_ylabel("depth [µm]  (rows equally spaced by rank, not by µm)", fontsize=11)

    bar = fig.colorbar(image, ax=axes, fraction=0.014, pad=0.04)
    bar.set_label("z, per-trial baseline removed", fontsize=11)
    fig.suptitle(
        _session_suptitle(
            session,
            f"Block PSTH — {len(ordered)} individual multi-unit clusters, shallow to deep",
            f"thick line = theta channel ({regions.boundary(session):.0f} µm) · dashed = other landmarks",
        ),
        fontsize=15, fontweight="bold", y=1.03,
    )
    return fig


# =============================================================================
# review point 7 — responders
# =============================================================================

def plot_responders(session, table: pd.DataFrame, params: Params):
    """Response size against depth, per condition, with responders ringed."""
    conditions = list(dict.fromkeys(table.condition))
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.9 * len(conditions), 6.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, condition in zip(axes, conditions):
        subset = table[table.condition == condition]
        for region in sorted(subset.region.unique()):
            group = subset[subset.region == region]
            ax.scatter(group.z_stim, group.depth_um, s=55, color=_region_colour(region),
                       edgecolor="w", lw=0.6, label=region, zorder=3)
        responders = subset[subset.responder]
        ax.scatter(responders.z_stim, responders.depth_um, s=170, facecolor="none",
                   edgecolor="k", lw=1.1, zorder=2,
                   label=f"responder (FDR<{params.responder_alpha})")
        ax.axvline(0, color="#555555", ls="-", lw=0.9)
        ax.axhline(regions.boundary(session), color="k", lw=1.5)
        ax.set_title(f"{condition}\n{int(subset.responder.sum())}/{len(subset)} responders", fontsize=12)
        ax.set_xlabel("mean z of stimulus rate\n(unit's own baseline trials as reference)")

    axes[0].invert_yaxis()
    axes[0].set_ylabel("depth [µm]   —   line = theta channel")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(_session_suptitle(session, "Responders by depth"), fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# =============================================================================
# review point 9 — transition-triggered PSTH
# =============================================================================

def _transition_response(spikes, trigger, params: Params) -> dict:
    """Adapter: unpack this project's `Trigger` into the shared function's plain arguments."""
    return transition_response(
        spikes, trigger.transitions, trigger.period_s, FS,
        cycles=params.tt_cycles, bin_ms=params.tt_bin_ms, smooth_bins=params.tt_smooth_bins,
    )


def _pooled_block_psth(spikes, trigger, params: Params) -> dict:
    """Adapter: same idea for the block PSTH."""
    return block_psth(
        spikes, trigger.block_onsets,
        params.psth_pre_s, params.psth_post_s, params.psth_bin_s, FS,
    )


def plot_transition_grid(session, store, ordered: pd.DataFrame,
                         trigger, params: Params, n_columns: int = 6):
    """One panel per unit: does it follow each individual screen transition?"""
    period_ms = trigger.period_s * 1000
    n_rows = int(np.ceil(len(ordered) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.9 * n_columns, 2.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, unit) in zip(axes, ordered.iterrows()):
        result = _transition_response(store.spikes(int(unit.unit)), trigger, params)
        reliable = result["n_spikes"] >= params.tt_min_spikes
        colour = _region_colour(unit.region) if reliable else MUTED

        ax.step(result["centres_s"] * 1000, result["rate_hz"], where="mid", color=colour, lw=1.0)
        ax.fill_between(result["centres_s"] * 1000, result["rate_hz"], step="mid",
                        color=colour, alpha=0.2, lw=0)
        for k in np.arange(-2, 3) * period_ms:
            ax.axvline(k, color=SOFT_RED, ls="--", lw=0.8, alpha=0.8)

        star = "*" if result["rayleigh_p"] < 0.05 else ""
        text = (
            f"MD {result['mod_depth']:.2f}{star}\nnorm {result['mod_depth_norm']:.2f}\n"
            f"PPC {result['ppc']:.4f}" if reliable
            else f"only {result['n_spikes']} spikes\nmetrics not shown"
        )
        ax.set_title(f"u{int(unit.unit)} · {unit.region}", fontsize=9, fontweight="bold", pad=3,
                     color=INK if reliable else "#999999")
        ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=6.5,
                color=INK if reliable else "#999999",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5), zorder=5)
        ax.set_xlim(result["centres_s"][0] * 1000, result["centres_s"][-1] * 1000)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)

    for ax in axes[len(ordered):]:
        ax.axis("off")
    label_grid_corner(axes, len(ordered), n_columns, "time from transition [ms]", "rate [Hz]")

    fig.suptitle(
        _session_suptitle(
            session,
            f"Transition-triggered PSTH — {trigger.name} "
            f"({trigger.n_transitions:,} transitions, {params.tt_bin_ms:.0f} ms bins, "
            f"{params.tt_cycles} cycles shown)",
            "soft red = transition times · MD = peak/trough of the ONE-cycle histogram "
            "(* Rayleigh p<0.05)",
        ),
        fontsize=15, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    return fig


def plot_locking_vs_depth(session, table: pd.DataFrame, params: Params):
    """PPC against depth, per condition — the figure the cross-region claim rests on."""
    conditions = list(dict.fromkeys(table.condition))
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.9 * len(conditions), 6.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, condition in zip(axes, conditions):
        subset = table[(table.condition == condition) & table.reliable]
        for region in sorted(subset.region.unique()):
            group = subset[subset.region == region]
            ax.scatter(group.ppc, group.depth_um, s=55, color=_region_colour(region),
                       edgecolor="w", lw=0.6, label=region, zorder=3)
        locked = subset[subset.locked]
        ax.scatter(locked.ppc, locked.depth_um, s=170, facecolor="none", edgecolor="k",
                   lw=1.1, zorder=2, label="Rayleigh FDR<0.05")
        ax.axvline(0, color="#555555", lw=0.9)
        ax.axhline(regions.boundary(session), color="k", lw=1.5)
        ax.set_title(f"{condition}\n{int(subset.locked.sum())}/{len(subset)} locked", fontsize=12)
        ax.set_xlabel("PPC (bias-free phase locking)")

    axes[0].invert_yaxis()
    axes[0].set_ylabel("depth [µm]   —   line = theta channel")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        _session_suptitle(session, "Phase locking against depth",
                          "PPC rather than vector strength: the two regions differ in firing "
                          "rate, and vector strength is biased by spike count"),
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_harmonic_control(session, table: pd.DataFrame):
    """Is a two-peaks-per-cycle response being cancelled by vector-strength geometry?"""
    conditions = list(dict.fromkeys(table.condition))
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.0 * len(conditions), 5.4))
    axes = np.atleast_1d(axes)

    for ax, condition in zip(axes, conditions):
        subset = table[(table.condition == condition) & table.reliable]
        for region in sorted(subset.region.unique()):
            group = subset[subset.region == region]
            ax.scatter(group.vs_f0, group.vs_2f0, s=55, color=_region_colour(region),
                       edgecolor="w", lw=0.6, label=region, zorder=3)
        locked = subset[subset.q_f0 < 0.05]
        ax.scatter(locked.vs_f0, locked.vs_2f0, s=160, facecolor="none", edgecolor="k",
                   lw=1.0, zorder=2, label="locked at $f_0$ (FDR<0.05)")
        only_2f0 = subset[subset.locked_at_2f0_only]
        ax.scatter(only_2f0.vs_f0, only_2f0.vs_2f0, s=200, marker="s", facecolor="none",
                   edgecolor=SOFT_RED, lw=1.6, zorder=4, label="locked at $2f_0$ ONLY")
        top = max(0.05, float(np.nanmax(subset[["vs_f0", "vs_2f0"]].to_numpy())) * 1.15)
        ax.plot([0, top], [0, top], color="#999999", ls="--", lw=1)
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        ax.set_xlabel("vector strength at fundamental $f_0$")
        ax.set_ylabel("vector strength at second harmonic $2f_0$")
        ax.set_title(f"{condition} — {int(subset.locked_at_2f0_only.sum())} locked at $2f_0$ only",
                     fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle(
        _session_suptitle(session, "Harmonic control",
                          "a unit above the diagonal AND locked only at $2f_0$ would mean a real "
                          "two-peak response is being cancelled"),
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# =============================================================================
# review point 8 — pooled multi-unit activity
# =============================================================================

def plot_pooled_block_psth(session, store, pooled: pd.DataFrame,
                           triggers: dict, regulation: pd.DataFrame, params: Params):
    """Pooled multi-unit block PSTH over [-2 s, +4 s], one row per region."""
    conditions = list(triggers)
    fig, axes = plt.subplots(len(pooled), len(conditions),
                             figsize=(3.9 * len(conditions), 4.0 * len(pooled)), sharex="col")
    axes = np.atleast_2d(axes)

    for column, condition in enumerate(conditions):
        for row, (_, pool) in enumerate(pooled.iterrows()):
            ax = axes[row, column]
            profile = _pooled_block_psth(store.spikes(int(pool.unit)), triggers[condition], params)
            colour = _region_colour(pool.region)
            centres = profile["centres_s"]

            ax.fill_between(centres, profile["rate_hz"], step="mid", color=colour, alpha=0.25, lw=0)
            ax.step(centres, profile["rate_hz"], where="mid", color=colour, lw=1.3)
            ax.axhline(profile["baseline_hz"], color="#555555", ls="--", lw=1.0)
            ax.axvspan(-params.psth_pre_s, 0, color="#BBBBBB", alpha=0.18, lw=0)
            ax.axvline(0, color="k", lw=1.0)
            ax.axvline(params.psth_post_s, color="k", lw=0.8, ls=":")

            summary = regulation[(regulation.condition == condition)
                                 & (regulation.region == pool.region)]
            if len(summary):
                # bracket access throughout: several of these names (pct_change) collide
                # with pandas Series methods and attribute access silently returns the method
                row_data = summary.iloc[0]
                ax.text(0.02, 0.96,
                        f"{row_data['pct_change']:+.1f}%   ({row_data['delta_hz']:+.2f} Hz)\n"
                        f"MI {row_data['mod_index']:+.3f}\npeak z {row_data['peak_z']:+.1f}\n"
                        f"p {row_data['p_wilcoxon']:.1e}",
                        transform=ax.transAxes, va="top", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5), zorder=5)
            if row == 0:
                ax.set_title(condition, fontsize=12)
            if column == 0:
                ax.set_ylabel(f"pooled {pool.region} MUA\nrate [Hz]", fontsize=11)
            if row == len(pooled) - 1:
                ax.set_xlabel("time from block onset [s]")

    fig.suptitle(
        _session_suptitle(
            session,
            f"Pooled multi-unit block PSTH, [−{params.psth_pre_s:.0f} s, +{params.psth_post_s:.0f} s]",
            "grey band = baseline window · dashed line = baseline rate",
        ),
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_pooled_transition(session, store, pooled: pd.DataFrame,
                           triggers: dict, params: Params):
    """Pooled multi-unit transition-triggered response, one row per region."""
    conditions = [name for name, t in triggers.items() if t.period_s is not None]
    fig, axes = plt.subplots(len(pooled), len(conditions),
                             figsize=(3.9 * len(conditions), 4.0 * len(pooled)))
    axes = np.atleast_2d(axes)

    for column, condition in enumerate(conditions):
        trigger = triggers[condition]
        period_ms = trigger.period_s * 1000
        for row, (_, pool) in enumerate(pooled.iterrows()):
            ax = axes[row, column]
            result = _transition_response(store.spikes(int(pool.unit)), trigger, params)
            colour = _region_colour(pool.region)
            ax.fill_between(result["centres_s"] * 1000, result["rate_hz"], step="mid",
                            color=colour, alpha=0.25, lw=0)
            ax.step(result["centres_s"] * 1000, result["rate_hz"], where="mid", color=colour, lw=1.3)
            for k in np.arange(-2, 3) * period_ms:
                ax.axvline(k, color=SOFT_RED, ls="--", lw=0.9, alpha=0.8)
            ax.text(0.02, 0.96,
                    f"MD {result['mod_depth']:.3f}\nnorm {result['mod_depth_norm']:.3f}\n"
                    f"PPC {result['ppc']:.5f}\nRayleigh p {result['rayleigh_p']:.1e}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5), zorder=5)
            ax.set_xlim(result["centres_s"][0] * 1000, result["centres_s"][-1] * 1000)
            if row == 0:
                ax.set_title(condition, fontsize=12)
            if column == 0:
                ax.set_ylabel(f"pooled {pool.region} MUA\nrate [Hz]", fontsize=11)
            if row == len(pooled) - 1:
                ax.set_xlabel("time from transition [ms]")

    fig.suptitle(
        _session_suptitle(
            session, "Pooled multi-unit, transition-triggered",
            "at N ≈ 10⁵ spikes a Rayleigh p-value reports sample size, not effect size — read PPC",
        ),
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    return fig
