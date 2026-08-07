"""
Waveform shape and spike-train rhythmicity: separating principal cells from interneurons.

**Why waveform width separates them.** An extracellular spike is the field produced by
transmembrane currents at the contact: a sharp negative **trough** (sodium influx, a current
sink) followed by a slower positive **rebound** (potassium efflux, a source). The interval
between them is a direct readout of repolarisation speed. Fast-spiking interneurons express
Kv3-family potassium channels with unusually fast kinetics — which is what lets them sustain
high rates — so their trough-to-peak interval is short, typically under ~0.45 ms. Pyramidal
cells repolarise more slowly. This is a property of the membrane, not of behavioural state,
which is why it is the primary criterion. `Barthó et al. 2004, J Neurophysiol 92:600`

**Two supporting measurements from the autocorrelogram**, independent of the waveform:

* **burst index** = mean ACG at 3-5 ms / mean ACG at 200-300 ms. Above 1 means the unit
  fires in bursts. `Royer et al. 2012, Nat Neurosci 15:769`
* **theta modulation index** = (peak at 100-140 ms - trough at 50-70 ms) / their sum.
  Bounded in [-1, 1] and independent of firing rate. Same source.

Note what the theta index is **not**: it is rhythmicity of the unit's *own spike train*, not
phase-locking to a measured theta oscillation. That needs an LFP.

**Two waveform shapes the width rule cannot describe**, both flagged explicitly rather than
silently labelled:

* *non-repolarising* — the waveform never returns above baseline after the trough, so the
  trough-to-peak interval is not a repolarisation time and the rule is undefined;
* *positive-dominant* — the positive lobe exceeds the trough, which usually means the contact
  is not near the soma (an axonal spike, or the return current seen from a distance).

Depends on: numpy, pandas, correlograms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from correlograms import band_mean

PRINCIPAL = "principal"
INTERNEURON = "interneuron"
UNCLASSIFIED = "unclassified"

MIN_BAND_COUNTS = 20
"""Minimum spike pairs a lag band must hold before an index built from it means anything. A
0.1 Hz unit contributes about five pairs to the 200-300 ms shoulder over a 40-minute
recording, and dividing by that produces burst indices in the hundreds — an artefact of an
empty denominator, not a bursty neuron. Below this, the index is NaN."""


# =============================================================================
# waveform
# =============================================================================

def unwhiten_templates(templates: np.ndarray, whitening_mat_inv: np.ndarray) -> np.ndarray:
    """Kilosort templates in data space: `templates @ whitening_mat_inv`, as Phy does it.

    Kilosort spatially whitens before sorting and the templates inherit the transform. The
    whitening matrix is usually near-diagonal, so the effect on a single channel's time
    course is small — but it is not zero, and it can change *which channel is the peak
    channel*, which then changes the measured width.
    """
    return np.einsum("tsc,cd->tsd", templates, whitening_mat_inv)


def modal_template(spike_templates: np.ndarray, spike_clusters: np.ndarray, unit: int) -> int:
    """Which template a cluster's spikes mostly matched.

    Usually cluster id and template id coincide, but they diverge after manual curation, so
    take the modal template of the cluster's own spikes rather than trusting the id.
    """
    matched = spike_templates[spike_clusters == unit]
    if matched.size == 0:
        return -1
    values, counts = np.unique(matched, return_counts=True)
    return int(values[counts.argmax()])


def peak_channel(template: np.ndarray) -> int:
    """Channel with the largest peak-to-peak amplitude.

    Not the largest absolute value: some templates are positive-going on some channels, and
    max|V| would select a positive lobe on the wrong contact.
    """
    return int(np.argmax(template.max(axis=0) - template.min(axis=0)))


def waveform_metrics(template: np.ndarray, fs: float) -> dict:
    """Shape metrics from one template of shape `(n_samples, n_channels)`."""
    peak_to_peak = template.max(axis=0) - template.min(axis=0)
    channel = peak_channel(template)
    wave = template[:, channel].astype(np.float64)

    trough_index = int(np.argmin(wave))
    trough = float(wave[trough_index])

    # search for the peak *after* the trough: that is what makes this a repolarisation
    # measurement rather than "the distance between the two biggest excursions"
    after = wave[trough_index:]
    peak_index = trough_index + int(np.argmax(after)) if after.size > 1 else trough_index
    peak = float(wave[peak_index])

    half = trough / 2.0
    left = trough_index
    while left > 0 and wave[left] < half:
        left -= 1
    right = trough_index
    while right < wave.size - 1 and wave[right] < half:
        right += 1

    return {
        "peak_channel": channel,
        "trough_to_peak_ms": (peak_index - trough_index) / fs * 1000.0,
        "half_width_ms": (right - left) / fs * 1000.0,
        "peak_trough_ratio": peak / abs(trough) if trough != 0 else np.nan,
        "amplitude_au": float(peak_to_peak[channel]),
        "repolarises": bool(peak > 0),
        "positive_dominant": bool(wave.max() > abs(wave.min())),
    }


def peak_channel_waveform(
    templates_unwhitened: np.ndarray, spike_templates: np.ndarray,
    spike_clusters: np.ndarray, unit: int,
) -> np.ndarray:
    """The unit's template on its peak channel, for plotting."""
    index = modal_template(spike_templates, spike_clusters, unit)
    if index < 0:
        return np.zeros(templates_unwhitened.shape[1])
    template = templates_unwhitened[index]
    return template[:, peak_channel(template)]


# =============================================================================
# rhythmicity
# =============================================================================

def rhythmicity_metrics(
    centres_ms: np.ndarray,
    counts: np.ndarray,
    burst_lag_ms: tuple[float, float] = (3.0, 5.0),
    burst_ref_ms: tuple[float, float] = (200.0, 300.0),
    theta_peak_ms: tuple[float, float] = (100.0, 140.0),
    theta_trough_ms: tuple[float, float] = (50.0, 70.0),
) -> dict:
    """Burst index and theta modulation index from a long-window autocorrelogram.

    Both are NaN when the bands they are built from are too sparse to support them, rather
    than a large number produced by an almost-empty denominator.
    """
    burst_num, burst_num_n = band_mean(centres_ms, counts, *burst_lag_ms)
    burst_den, burst_den_n = band_mean(centres_ms, counts, *burst_ref_ms)
    peak, peak_n = band_mean(centres_ms, counts, *theta_peak_ms)
    trough, trough_n = band_mean(centres_ms, counts, *theta_trough_ms)

    burst_ok = burst_den > 0 and burst_den_n >= MIN_BAND_COUNTS and burst_num_n >= 1
    theta_ok = (peak + trough) > 0 and peak_n >= MIN_BAND_COUNTS and trough_n >= MIN_BAND_COUNTS

    return {
        "burst_index": burst_num / burst_den if burst_ok else np.nan,
        "theta_index": (peak - trough) / (peak + trough) if theta_ok else np.nan,
        "rhythmicity_reliable": bool(burst_ok and theta_ok),
    }


# =============================================================================
# classification
# =============================================================================

def kde_antimodes(
    values: np.ndarray,
    lo: float = 0.05,
    hi: float = 1.35,
    bandwidth: float = 0.05,
    min_prominence: float = 0.75,
) -> np.ndarray:
    """Every prominent dip in a kernel density estimate of `values`, sorted.

    Diagnostic only. It answers "is this distribution actually separable, and where?" without
    letting the answer set the classification threshold — a per-session fitted threshold makes
    the label mean something different in every session, which destroys cross-session pooling.

    A local minimum is not enough; a noisy tail produces plenty. A dip is returned only if its
    density is below `min_prominence` times the smaller of the two peaks flanking it. An empty
    array — "not visibly separable" — is a perfectly good answer.

    Read the result with care: a dip is not automatically *the* interneuron/principal
    boundary. A distribution can be separable on some other axis entirely.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return np.array([])

    grid = np.linspace(lo, hi, 500)
    density = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2).sum(axis=1)

    interior = np.arange(1, grid.size - 1)
    is_minimum = (density[interior] < density[interior - 1]) & (density[interior] < density[interior + 1])

    dips = []
    for index in interior[is_minimum]:
        flank = min(density[:index].max(), density[index + 1:].max())
        if flank > 0 and density[index] < min_prominence * flank:
            dips.append(float(grid[index]))
    return np.array(sorted(dips))


def classify(
    table: pd.DataFrame,
    t2p_split_ms: float = 0.45,
    slow_hz: float = 2.0,
    fast_hz: float = 15.0,
    bursty: float = 1.5,
) -> pd.DataFrame:
    """Add `cell_type` and `cell_type_consistent` to a table carrying the metrics.

    `cell_type` comes from waveform width alone, with one exception: a unit whose waveform
    never returns above baseline after the trough is `unclassified`, because for it the
    trough-to-peak interval is not a repolarisation time and the rule is not merely uncertain
    but undefined.

    `cell_type_consistent` is False in three cases, none of which cause a relabel — they are
    marked so a reader can see how much of the classification is being taken on trust:
    a narrow-waveform unit that fires slowly *and* bursts; a wide-waveform unit firing fast;
    a positive-dominant waveform.

    Firing rate and burst index deliberately do **not** vote. They depend on behavioural
    state and on recording length, so folding them into the label would make the label
    state-dependent and would hide the disagreements that are scientifically interesting.
    """
    table = table.copy()
    width = table["trough_to_peak_ms"]
    repolarises = table.get("repolarises", pd.Series(True, index=table.index)).fillna(False).astype(bool)

    table["cell_type"] = np.where(
        width.isna() | ~repolarises, UNCLASSIFIED,
        np.where(width < t2p_split_ms, INTERNEURON, PRINCIPAL),
    )

    narrow_but_slow_and_bursty = (
        (table.cell_type == INTERNEURON) & (table.fr_hz < slow_hz) & (table.burst_index > bursty)
    )
    wide_but_fast = (table.cell_type == PRINCIPAL) & (table.fr_hz > fast_hz)
    atypical = table.get("positive_dominant", pd.Series(False, index=table.index)).fillna(False).astype(bool)

    table["cell_type_consistent"] = ~(narrow_but_slow_and_bursty | wide_but_fast | atypical)
    table.loc[table.cell_type == UNCLASSIFIED, "cell_type_consistent"] = False
    return table
