"""
Which units changed their firing rate when the stimulus came on.

Per unit and per condition:

1. **Trial-wise rates.** Spikes in the window before onset and the window during the
   stimulus, for each trial. **Rates, not counts** — the two windows usually differ in
   length, and a paired test on raw counts would report a difference that is pure window
   length, with a vanishing p-value, for every unit in the dataset.

2. **z-scored against the unit's own baseline trials.** The reference distribution is that
   unit's own pre-stimulus trials, so a 30 Hz interneuron and a 0.3 Hz granule cell land on
   the same scale and "z = 2" means *two of this unit's own baseline standard deviations*.

   The alternative — standardising against the population of other units — assumes most units
   do not respond (their SD is inflated by the very effect being looked for) and returns
   2-5 % "responders" from pure noise, because any finite sample has a tail.

3. **Paired test, pre versus post, within condition.** Wilcoxon signed-rank: paired because
   the same trial supplies both numbers, non-parametric because trial-wise counts are neither
   normal nor equal-variance.

4. **Benjamini-Hochberg across units**, within condition. The family is "all units, this
   stimulus" — that is the question actually being asked.

5. **Sign-flip permutation** as confirmation, since it assumes strictly less.

A rate change is **not** entrainment: a unit can double its rate with no temporal
relationship to the stimulus cycle, and can lock perfectly while its mean rate falls. See
`entrainment.py`.

Depends on: numpy, pandas, spiketrains, statstools.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from spiketrains import window_counts
from statstools import benjamini_hochberg, paired_wilcoxon, poisson_rate_sd, sign_flip_permutation


def trial_rates(
    spikes: np.ndarray, onsets: np.ndarray, pre_s: float, stim_s: float, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial firing rate before and during the stimulus, in Hz."""
    pre = window_counts(spikes, onsets, -pre_s, 0.0, fs) / pre_s
    post = window_counts(spikes, onsets, 0.0, stim_s, fs) / stim_s
    return pre, post


def z_against_baseline(
    pre_rates: np.ndarray, post_rates: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Standardise both windows against the *pre* distribution of the same unit.

    Returns `(z_pre, z_post, mu, sigma)`. Sigma is floored at the Poisson expectation for a
    window of `window_s` seconds, so a unit whose baseline happens to be constant across
    trials does not get an infinite z.
    """
    mu = float(np.mean(pre_rates))
    sigma = max(
        float(np.std(pre_rates, ddof=1)) if pre_rates.size > 1 else 0.0,
        poisson_rate_sd(mu, window_s, 1),
    )
    return (pre_rates - mu) / sigma, (post_rates - mu) / sigma, mu, sigma


def responder_table(
    spikes_by_unit: Mapping[int, np.ndarray],
    onsets_by_condition: Mapping[str, np.ndarray],
    pre_s: float,
    stim_s: float,
    fs: float,
    alpha: float = 0.05,
    n_permutations: int = 10_000,
) -> pd.DataFrame:
    """One row per unit x condition, with the responder decision and its ingredients.

    Returns only the quantities this module can compute. Region, depth, cell type and any
    other metadata are the caller's business — merge them on `unit` afterwards, so this
    function stays usable on any dataset.
    """
    rows = []
    for condition, onsets in onsets_by_condition.items():
        for unit, spikes in spikes_by_unit.items():
            pre, post = trial_rates(spikes, onsets, pre_s, stim_s, fs)
            _, z_post, _, sigma = z_against_baseline(pre, post, pre_s)
            fr_pre, fr_post = float(pre.mean()), float(post.mean())

            rows.append({
                "unit": int(unit),
                "condition": condition,
                "n_trials": len(pre),
                "fr_pre_hz": fr_pre,
                "fr_stim_hz": fr_post,
                "baseline_sd_hz": sigma,
                "z_stim": float(z_post.mean()),
                "z_stim_sem": float(z_post.std(ddof=1) / np.sqrt(z_post.size)) if z_post.size > 1 else np.nan,
                "delta_hz": fr_post - fr_pre,
                "mod_index": ((fr_post - fr_pre) / (fr_post + fr_pre)) if (fr_post + fr_pre) > 0 else np.nan,
                "pct_change": (100.0 * (fr_post - fr_pre) / fr_pre) if fr_pre > 0 else np.nan,
                "p_wilcoxon": paired_wilcoxon(post, pre),
                "p_permutation": sign_flip_permutation(post, pre, n_permutations),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["q_wilcoxon"] = np.nan
    table["z_pct_population"] = np.nan
    for condition in table.condition.unique():
        mask = table.condition == condition
        table.loc[mask, "q_wilcoxon"] = benjamini_hochberg(table.loc[mask, "p_wilcoxon"].values)
        # the population-relative rule, retained for comparison only — see the module header
        pct = table.loc[mask, "pct_change"]
        table.loc[mask, "z_pct_population"] = (pct - pct.mean()) / pct.std()

    table["responder"] = table.q_wilcoxon < alpha
    table["direction"] = np.where(table.delta_hz > 0, "up", "down")
    table.loc[~table.responder, "direction"] = "none"
    return table


def test_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """Do the Wilcoxon and the permutation test agree? If not, the Wilcoxon is suspect."""
    usable = table[table.p_permutation.notna()]
    if usable.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "unit_x_condition_cells": len(usable),
        "both_significant_raw": int(((usable.p_wilcoxon < 0.05) & (usable.p_permutation < 0.05)).sum()),
        "wilcoxon_only": int(((usable.p_wilcoxon < 0.05) & (usable.p_permutation >= 0.05)).sum()),
        "permutation_only": int(((usable.p_wilcoxon >= 0.05) & (usable.p_permutation < 0.05)).sum()),
        "spearman_r": float(usable.p_wilcoxon.corr(usable.p_permutation, method="spearman")),
    }])
