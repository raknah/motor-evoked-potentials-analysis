"""
Does a unit fire at a consistent point within each stimulus cycle?

A different question from `psth.py`. That one asks *did the stimulus change the rate?*; this
asks *does the unit follow each individual cycle?* They are independent, and the bin width
alone forces the separation: one 25 ms bin **is** one 40 Hz cycle, so a 25 ms histogram is
structurally incapable of showing 40 Hz following.

**Three layers, in increasing order of how much they can be trusted.**

1. **The display histogram.** Every transition is a trigger; spikes are gathered over a few
   cycles at fine resolution. For the eye only. The windows *overlap* — triggers are one cycle
   apart and the window is several cycles wide — so each spike is counted several times and
   the histogram is periodic **by construction**. No number is measured from it.

2. **The one-cycle histogram.** Each spike is assigned to the single transition preceding it
   and expressed as a phase in [0, 2 pi). Every spike is counted exactly once. Peak and trough
   are read off this, after a circular boxcar.

   **Never fold a display window modulo the period.** If the window is a non-integer number of
   cycles, some phases are covered more often than others and a perfectly flat train reads a
   large modulation depth from geometry alone. A 3.5-cycle window folds to a 4:3 step and
   reads 1.33.

3. **The phase statistics.** Computed from per-spike phases directly, with no histogram and
   no peak picking, so no binning or smoothing choice can influence them.

       VS  = |sum exp(i theta)| / N                       in [0, 1]
       PPC = (|sum exp(i theta)|^2 - N) / (N (N - 1))     unbiased by N

   VS is biased upward at low spike count: under the null, E[|sum|^2] = N, so a unit with 200
   spikes reads VS ~ 0.07 from nothing at all. **Use PPC for any comparison between
   populations that differ in firing rate** — otherwise the comparison is partly a firing-rate
   comparison. PPC subtracts exactly that N, which is why it can go slightly negative; that is
   the estimator being unbiased, not an error.
   `Vinck et al. 2010, NeuroImage 51:112`

**The known blind spot.** VS reads about zero for clean n:1 locking — a unit firing twice per
cycle at opposite phases has its two phase vectors cancel. `harmonic_table` tests for it. The
diagnostic is *not* "which harmonic is larger on average" (for an unmodulated unit that is a
coin flip carrying no information) but whether any unit is locked at 2f0 while **not** locked
at f0.

Depends on: numpy, pandas, spiketrains, statstools.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from spiketrains import gather
from statstools import benjamini_hochberg


# =============================================================================
# phase
# =============================================================================

def cycle_phases(
    spikes: np.ndarray, transitions: np.ndarray, period_s: float, fs: float, harmonic: int = 1
) -> np.ndarray:
    """Phase in [0, 2 pi) of each spike within its stimulus cycle.

    A spike is kept only if it falls in [transition, transition + period) for the transition
    immediately preceding it. That does two things at once: it restricts the analysis to
    in-block spikes without a separate mask, and it guarantees **each spike is used exactly
    once**.

    `harmonic > 1` folds within period/harmonic instead, which is how the n:1 blind spot is
    tested.
    """
    spikes = np.sort(np.asarray(spikes))
    transitions = np.asarray(transitions, dtype=np.int64)
    if spikes.size == 0 or transitions.size == 0:
        return np.empty(0)

    preceding = np.searchsorted(transitions, spikes, side="right") - 1
    valid = preceding >= 0
    elapsed = np.full(spikes.shape, np.inf)
    elapsed[valid] = (spikes[valid] - transitions[preceding[valid]]) / fs
    valid &= (elapsed >= 0) & (elapsed < period_s)

    folded = period_s / harmonic
    return 2 * np.pi * (elapsed[valid] % folded) / folded


def phase_statistics(theta: np.ndarray) -> dict:
    """Vector strength, PPC, Rayleigh p and preferred phase from per-spike phases.

    The Rayleigh p-value uses the standard closed form, accurate for N above ~50 and
    conservative below. Note that at N ~ 1e5 — which pooled multi-unit trains reach — a
    p-value is a statement about **sample size**, not effect size. Read PPC, not p.
    """
    n = int(theta.size)
    if n < 2:
        return {"n_spikes": n, "vs": np.nan, "ppc": np.nan,
                "rayleigh_p": 1.0, "pref_phase_rad": np.nan}
    resultant = np.sum(np.exp(1j * theta))
    R = float(np.abs(resultant))
    return {
        "n_spikes": n,
        "vs": R / n,
        "ppc": (R ** 2 - n) / (n * (n - 1)),
        "rayleigh_p": float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))),
        "pref_phase_rad": float(np.angle(resultant) % (2 * np.pi)),
    }


# =============================================================================
# modulation depth
# =============================================================================

def cycle_histogram(
    theta: np.ndarray, period_s: float, n_transitions: int, bin_s: float, smooth_bins: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-cycle rate histogram, plus its smoothed version.

    Returns `(bin_edges_s, rate_hz, smoothed_hz)`. Each transition contributes exactly one
    cycle, so dividing by `bin_s * n_transitions` gives a rate directly comparable with a
    block PSTH.

    Smoothing is a **circular** boxcar, because phase 0 and phase 2 pi are the same place and
    a linear filter would corrupt both ends of the cycle — which is exactly where a
    transition-locked response sits. Without any smoothing, a single noisy bin sets the peak
    and a single empty bin sets the trough. Keep the kernel narrow: it is a low-pass filter
    and attenuates real structure as well as noise.
    """
    n_bins = int(round(period_s / bin_s))
    edges = np.arange(n_bins + 1) * bin_s
    counts, _ = np.histogram(theta / (2 * np.pi) * period_s, bins=edges)
    rate = counts / (bin_s * max(n_transitions, 1))

    width = max(int(smooth_bins), 1)
    kernel = np.ones(width) / width
    pad = width // 2
    padded = np.r_[rate[-pad:], rate, rate[:pad]] if pad else rate
    smoothed = np.convolve(padded, kernel, mode="same")
    smoothed = smoothed[pad:pad + n_bins] if pad else smoothed
    return edges, rate, smoothed


def modulation_depth(smoothed: np.ndarray, vs: float) -> dict:
    """Three modulation depths from the same cycle histogram, deliberately not one.

    * `mod_depth`         = peak / trough. Diverges as the trough approaches zero and is
                            biased upward at low spike count, because peak and trough are
                            **order statistics** — sampling noise alone pushes the maximum up
                            and the minimum down, so the ratio exceeds 1 with no modulation at
                            all. Roughly 1 + 3.5*sqrt(n_bins / (3 N)). **Not comparable across
                            units of different firing rate.**
    * `mod_depth_norm`    = (peak - trough) / (peak + trough), bounded in [0, 1]. Same
                            ingredients, no divergence.
    * `mod_depth_fourier` = 2 x vector strength. The amplitude of the first Fourier harmonic
                            of the cycle histogram divided by its mean — the classical
                            definition of modulation depth — which for a point process is
                            exactly twice the vector strength. No peak picking, so no
                            order-statistic bias. Equals `mod_depth_norm` for a perfectly
                            sinusoidal response and is smaller for a sharp one.
    """
    peak, trough = float(smoothed.max()), float(smoothed.min())
    return {
        "fr_peak_hz": peak,
        "fr_trough_hz": trough,
        "mod_depth": peak / trough if trough > 0 else np.inf,
        "mod_depth_norm": (peak - trough) / (peak + trough) if (peak + trough) > 0 else np.nan,
        "mod_depth_fourier": 2.0 * vs if np.isfinite(vs) else np.nan,
    }


def transition_response(
    spikes: np.ndarray, transitions: np.ndarray, period_s: float, fs: float,
    cycles: float = 3.5, bin_ms: float = 1.0, smooth_bins: int = 3,
) -> dict:
    """Everything for one unit and one rhythmic condition."""
    half_s = cycles / 2 * period_s
    bin_s = bin_ms / 1000.0
    n_transitions = len(transitions)

    relative, _ = gather(spikes, transitions, int(half_s * fs), int(half_s * fs))
    edges = np.arange(-half_s, half_s + bin_s / 2, bin_s)
    display_counts, _ = np.histogram(relative / fs, bins=edges)

    theta = cycle_phases(spikes, transitions, period_s, fs)
    stats = phase_statistics(theta)
    _, cycle_rate, smoothed = cycle_histogram(theta, period_s, n_transitions, bin_s, smooth_bins)

    return {
        "centres_s": (edges[:-1] + edges[1:]) / 2,
        "rate_hz": display_counts / (bin_s * max(n_transitions, 1)),
        "cycle_rate_hz": cycle_rate,
        "cycle_smoothed_hz": smoothed,
        "n_transitions": n_transitions,
        **stats,
        **modulation_depth(smoothed, stats["vs"]),
    }


def transition_table(
    spikes_by_unit: Mapping[int, np.ndarray],
    transitions_by_condition: Mapping[str, np.ndarray],
    period_by_condition: Mapping[str, float],
    fs: float,
    cycles: float = 3.5,
    bin_ms: float = 1.0,
    smooth_bins: int = 3,
    min_spikes: int = 200,
) -> pd.DataFrame:
    """One row per unit x rhythmic condition. Merge metadata on `unit` afterwards."""
    rows = []
    for condition, transitions in transitions_by_condition.items():
        period = period_by_condition[condition]
        for unit, spikes in spikes_by_unit.items():
            result = transition_response(spikes, transitions, period, fs, cycles, bin_ms, smooth_bins)
            rows.append({
                "unit": int(unit),
                "condition": condition,
                "n_cycle_spikes": result["n_spikes"],
                "fr_peak_hz": result["fr_peak_hz"],
                "fr_trough_hz": result["fr_trough_hz"],
                "mod_depth": result["mod_depth"],
                "mod_depth_norm": result["mod_depth_norm"],
                "mod_depth_fourier": result["mod_depth_fourier"],
                "vs": result["vs"],
                "ppc": result["ppc"],
                "rayleigh_p": result["rayleigh_p"],
                "pref_phase_deg": np.degrees(result["pref_phase_rad"]),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["reliable"] = table.n_cycle_spikes >= min_spikes
    table["rayleigh_q"] = np.nan
    for condition in table.condition.unique():
        mask = table.condition == condition
        table.loc[mask, "rayleigh_q"] = benjamini_hochberg(table.loc[mask, "rayleigh_p"].values)
    table["locked"] = table.reliable & (table.rayleigh_q < 0.05)
    return table


def harmonic_table(
    spikes_by_unit: Mapping[int, np.ndarray],
    transitions_by_condition: Mapping[str, np.ndarray],
    period_by_condition: Mapping[str, float],
    fs: float,
    min_spikes: int = 200,
) -> pd.DataFrame:
    """Phase statistics at the fundamental and the second harmonic.

    `locked_at_2f0_only` is the whole point: it is the only signature of vector-strength
    cancellation by an n:1 response.
    """
    rows = []
    for condition, transitions in transitions_by_condition.items():
        period = period_by_condition[condition]
        for unit, spikes in spikes_by_unit.items():
            row = {"unit": int(unit), "condition": condition}
            for harmonic, tag in ((1, "f0"), (2, "2f0")):
                stats = phase_statistics(cycle_phases(spikes, transitions, period, fs, harmonic))
                row[f"vs_{tag}"] = stats["vs"]
                row[f"ppc_{tag}"] = stats["ppc"]
                row[f"p_{tag}"] = stats["rayleigh_p"]
                row["n_spikes"] = stats["n_spikes"]
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for condition in table.condition.unique():
        mask = table.condition == condition
        for tag in ("f0", "2f0"):
            table.loc[mask, f"q_{tag}"] = benjamini_hochberg(table.loc[mask, f"p_{tag}"].values)
    table["reliable"] = table.n_spikes >= min_spikes
    table["locked_at_2f0_only"] = (table.q_2f0 < 0.05) & (table.q_f0 >= 0.05) & table.reliable
    return table
