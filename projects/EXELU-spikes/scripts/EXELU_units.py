"""
The unit table: one row per cluster, carrying everything the analyses need to know about it.

This is the glue between the shared, project-agnostic modules (`contamination`, `waveforms`,
`correlograms`) and this experiment's anatomy (`regions`). The shared modules compute the
numbers; this file decides which numbers, over which spikes, and how they join to a depth and
a region.

Contamination and firing rate are computed over the **whole** recording, including the
settling period, because that is what Kilosort did and the re-implementation has to be
comparable to the sorter's own file. The autocorrelograms behind the cell-type metrics use
only the settled portion, because those are properties of the unit's spiking and the settling
period is not representative of it.
"""

from __future__ import annotations

# Importing EXELU_paths puts <framework>/scripts on sys.path, so the shared
# analysis modules below resolve wherever this file is imported from.
import EXELU_paths  # noqa: F401

import numpy as np
import pandas as pd

import EXELU_regions as regions
from EXELU_config import Params
from contamination import hill_false_positive, ks_contam_pct
from correlograms import autocorrelogram
from spiketrains import SpikeStore
from waveforms import (
    classify, modal_template, peak_channel_waveform, rhythmicity_metrics,
    unwhiten_templates, waveform_metrics,
)


def wide_acg(store: SpikeStore, unit: int, params: Params, fs: float):
    """The long-window autocorrelogram used for both the theta figure and the indices."""
    return autocorrelogram(
        store.after(unit, params.settle_s),
        bin_ms=params.acg_wide_bin_ms,
        window_ms=params.acg_wide_window_ms,
        fs=fs,
        max_spikes=params.acg_max_spikes,
    )


def build_unit_table(session, store: SpikeStore, params: Params, fs: float) -> pd.DataFrame:
    """One row per cluster: label, depth, region, rate, and both contamination metrics."""
    boundary = regions.boundary(session)
    rows = []
    positions = session.array("spike_positions")
    clusters_by_spike = session.array("spike_clusters")
    for _, cluster in session.clusters.iterrows():
        unit = int(cluster["cluster_id"])
        spikes = store.spikes(unit)
        if spikes.size == 0:
            continue
        mask = clusters_by_spike == unit
        rows.append({
            "unit": unit,
            "label": cluster["KSLabel"],
            "contam_pct": float(cluster["ContamPct"]),
            "n_spikes": int(spikes.size),
            "fr_hz": spikes.size / session.duration_s,
            "depth_um": float(np.median(positions[mask, 1])),
            "contam_pct_recomputed": ks_contam_pct(spikes, fs),
            "hill_fp_pct": 100.0 * hill_false_positive(
                spikes, fs, params.hill_t_ref_ms, params.hill_t_cens_ms
            ),
        })

    table = pd.DataFrame(rows)
    table["region"] = regions.region_of_depth(table["depth_um"].values, boundary)
    table["near_boundary"] = regions.near_boundary(table["depth_um"].values, boundary)
    table = table.sort_values("depth_um").reset_index(drop=True)
    if not table.columns.is_unique:
        raise AssertionError("unit table has duplicate columns")
    return table


def select_clean(units: pd.DataFrame, params: Params) -> pd.DataFrame:
    """The analysis set: Kilosort `good`, contamination under the ceiling, enough spikes."""
    keep = (
        (units.label == "good")
        & (units.contam_pct <= params.clean_contam_pct)
        & (units.n_spikes >= params.min_spikes_unit)
    )
    return units[keep].sort_values(["region", "depth_um"]).reset_index(drop=True)


def add_cell_types(session, store: SpikeStore, units: pd.DataFrame,
                   params: Params, fs: float) -> pd.DataFrame:
    """Waveform and rhythmicity metrics, and a cell-type label, for every unit in `units`."""
    templates = unwhiten_templates(session.array("templates"),
                                   session.array("whitening_mat_inv"))

    empty_shape = dict.fromkeys(
        ["peak_channel", "trough_to_peak_ms", "half_width_ms", "peak_trough_ratio",
         "amplitude_au", "repolarises", "positive_dominant"], np.nan
    )

    rows = []
    for unit_id in units.unit.astype(int):
        template_index = modal_template(session.array("spike_templates"),
                                        session.array("spike_clusters"), unit_id)
        shape = (
            waveform_metrics(templates[template_index], fs)
            if template_index >= 0 else dict(empty_shape)
        )
        centres, counts = wide_acg(store, unit_id, params, fs)
        rows.append({
            "unit": unit_id,
            "template": template_index,
            **shape,
            **rhythmicity_metrics(
                centres, counts,
                burst_lag_ms=params.burst_index_lag_ms,
                burst_ref_ms=params.burst_index_ref_ms,
                theta_peak_ms=params.theta_peak_ms,
                theta_trough_ms=params.theta_trough_ms,
            ),
        })

    merged = units.merge(pd.DataFrame(rows), on="unit", how="left", validate="one_to_one")
    return classify(merged, t2p_split_ms=params.t2p_split_ms)


def waveform_of(session, unit: int, templates_unwhitened=None) -> np.ndarray:
    """The unit's template on its peak channel, for plotting."""
    if templates_unwhitened is None:
        templates_unwhitened = unwhiten_templates(session.array("templates"),
                                                  session.array("whitening_mat_inv"))
    return peak_channel_waveform(
        templates_unwhitened, session.array("spike_templates"),
        session.array("spike_clusters"), int(unit)
    )


def order_by_region_then_rate(units: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Cortex block above, hippocampus below, descending firing rate within each.

    Returns `(ordered_table, n_cortical)` so a figure can draw the divider without recounting.
    """
    ctx = units[units.region == regions.CORTEX].sort_values("fr_hz", ascending=False)
    hpc = units[units.region == regions.HIPPOCAMPUS].sort_values("fr_hz", ascending=False)
    return pd.concat([ctx, hpc]).reset_index(drop=True), len(ctx)


def attach_metadata(table: pd.DataFrame, units: pd.DataFrame,
                    columns=("label", "region", "depth_um", "fr_hz", "cell_type")) -> pd.DataFrame:
    """Merge unit metadata onto a result table returned by one of the shared modules.

    The shared modules deliberately return only what they compute — unit, condition and the
    statistics — so that they stay usable on any dataset. Region, depth and cell type are this
    project's business and are joined back on here.
    """
    available = [c for c in columns if c in units.columns]
    return table.merge(units[["unit", *available]], on="unit", how="left")
