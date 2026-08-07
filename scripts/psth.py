"""
Peri-stimulus time histograms, including per-trial standardisation.

A PSTH aligns spikes to every stimulus onset, bins them, and divides by
(bin width x number of trials) to get a firing rate in spikes per second.

**Why standardise per trial.** The animal is not in the same state on every trial. When it is
moving, cortical firing is elevated *before* the stimulus arrives as well as during it.
Averaging raw trials and only then subtracting one grand baseline leaves that trial-to-trial
offset inside the average, so a session with more movement looks like a session with more
stimulus drive. Removing each trial's own baseline first eliminates the offset, and only the
within-trial change survives.

**Which flavour, and why not the literal one.** Dividing each trial by its own baseline
standard deviation is unusable for sparse units: a 0.5 Hz unit fires 0 or 1 spikes in a 2 s
baseline, so the per-trial SD is 0 on most trials and the z-score is undefined. What is
implemented instead:

    r[t, b]   rate of the unit in trial t, bin b
    m[t]      = mean over baseline bins of r[t, .]        one number per trial
    d[t, b]   = r[t, b] - m[t]                            trial offset removed
    a[b]      = mean over trials of d[t, b]               the standardised PSTH
    sigma     = SD over baseline bins of a[.], floored at the Poisson expectation
    z[b]      = a[b] / sigma

Each trial contributes its own baseline (the part that matters); the scale comes from one
pooled estimate (the part that is unstable per trial).

**One caveat, stated because it is real and small.** Subtracting a trial's own baseline mean
makes that trial's baseline bins sum to zero, so they acquire a slight negative correlation.
With 80 baseline bins the induced correlation is of order 1/80 — negligible against counting
noise, but it does mean `sigma` is very slightly underestimated.

Depends on: numpy, pandas, spiketrains, statstools.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from spiketrains import trial_bin_counts
from statstools import poisson_rate_sd


def block_psth(
    spikes: np.ndarray, onsets: np.ndarray, pre_s: float, post_s: float,
    bin_s: float, fs: float,
) -> dict:
    """Raw and per-trial-standardised PSTH for one unit and one set of onsets.

    Returns a dict with:
        centres_s          bin centres, seconds relative to onset
        rate_hz            conventional trial-averaged rate (no per-trial correction)
        centred_hz         a[b] above — trial-averaged after removing each trial's baseline
        z                  centred_hz / sigma
        sigma_hz           the scale actually used
        baseline_hz        grand mean baseline rate
        trial_baseline_hz  m[t], one per trial — the evidence for whether this is needed
        n_trials
    """
    centres, counts = trial_bin_counts(spikes, onsets, pre_s, post_s, bin_s, fs)
    rate = counts / bin_s                                  # (n_trials, n_bins)
    baseline_bins = centres < 0

    trial_baseline = rate[:, baseline_bins].mean(axis=1, keepdims=True)   # m[t]
    centred = rate - trial_baseline                                       # d[t, b]
    mean_centred = centred.mean(axis=0)                                   # a[b]

    empirical_sd = float(mean_centred[baseline_bins].std())
    floor_sd = poisson_rate_sd(float(rate[:, baseline_bins].mean()), bin_s, len(onsets))
    sigma = max(empirical_sd, floor_sd)

    return {
        "centres_s": centres,
        "rate_hz": rate.mean(axis=0),
        "centred_hz": mean_centred,
        "z": mean_centred / sigma if sigma > 0 else np.zeros_like(mean_centred),
        "sigma_hz": sigma,
        "baseline_hz": float(rate[:, baseline_bins].mean()),
        "trial_baseline_hz": trial_baseline.ravel(),
        "n_trials": len(onsets),
    }


def block_psth_matrix(
    spikes_by_unit: Mapping[int, np.ndarray], unit_order, onsets: np.ndarray,
    pre_s: float, post_s: float, bin_s: float, fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack `block_psth` over units, in the order given.

    Returns `(centres_s, z_matrix, rate_matrix, unit_ids)`. The caller controls the row order,
    so a figure's y-axis needs no separate bookkeeping.
    """
    centres = None
    z_rows, rate_rows, unit_ids = [], [], []
    for unit in unit_order:
        result = block_psth(spikes_by_unit[int(unit)], onsets, pre_s, post_s, bin_s, fs)
        centres = result["centres_s"]
        z_rows.append(result["z"])
        rate_rows.append(result["rate_hz"])
        unit_ids.append(int(unit))
    return centres, np.array(z_rows), np.array(rate_rows), np.array(unit_ids)


def trial_baseline_spread(
    spikes_by_unit: Mapping[int, np.ndarray], onsets_by_condition: Mapping[str, np.ndarray],
    pre_s: float, post_s: float, bin_s: float, fs: float,
) -> pd.DataFrame:
    """How much the pre-stimulus baseline actually varies from trial to trial.

    This is the evidence for or against per-trial standardisation being necessary. If the
    coefficient of variation of m[t] is small, the correction changes nothing; if it is large,
    it is doing real work.
    """
    rows = []
    for condition, onsets in onsets_by_condition.items():
        for unit, spikes in spikes_by_unit.items():
            baselines = block_psth(spikes, onsets, pre_s, post_s, bin_s, fs)["trial_baseline_hz"]
            mean = float(baselines.mean())
            rows.append({
                "unit": int(unit),
                "condition": condition,
                "baseline_mean_hz": mean,
                "baseline_sd_hz": float(baselines.std()),
                "baseline_cv": float(baselines.std() / mean) if mean > 0 else np.nan,
            })
    return pd.DataFrame(rows)
