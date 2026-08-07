"""
Guards: run estimators on data whose answer is known before trusting them on data whose
answer is not.

**The modulation-depth estimator, against three synthetic spike trains.** This exists because
a previous version folded a 3.5-cycle window modulo the stimulus period. 3.5 is not a whole
number, so phases below 0.75 of a cycle were covered four times and phases above it three
times: a perfectly flat spike train folded to a 4:3 step and read a modulation depth of 1.33
from geometry alone. It was caught only because the pooled multi-unit trace read ~1.35 for
*every* condition, including one where nothing is locked. The test below would have caught it
immediately.

    flat        homogeneous Poisson, no locking      -> MD ~ 1,  VS ~ 0
    locked      von Mises concentrated at one phase  -> MD >> 1, VS high
    two-peak    two peaks half a cycle apart         -> VS ~ 0 but VS at 2f0 high

The third is not redundant: it is the vector-strength blind spot, and it is why the harmonic
control exists.

**The Kilosort contamination re-implementation, against Kilosort's own output file.** If
`ks_contam_pct` does not reproduce `cluster_ContamPct.tsv` to within its rounding, it is
measuring something else and every statement about "Kilosort's threshold" is wrong.
"""

from __future__ import annotations

# Importing EXELU_paths puts <framework>/scripts on sys.path, so the shared
# analysis modules below resolve wherever this file is imported from.
import EXELU_paths  # noqa: F401

import numpy as np
import pandas as pd

from EXELU_config import FS, Params
from contamination import ks_contam_pct
from entrainment import cycle_histogram, cycle_phases, modulation_depth, phase_statistics


def modulation_depth_guard(
    params: Params | None = None,
    period_s: float = 1 / 40,
    n_cycles: int = 9_600,
    spikes_per_cycle: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run three known spike trains through the real estimator and report what it says."""
    params = params or Params()
    rng = np.random.default_rng(seed)
    transitions = (np.arange(n_cycles) * period_s * FS).astype(np.int64)
    n_spikes = int(n_cycles * spikes_per_cycle)
    cycle_index = rng.integers(0, n_cycles, size=n_spikes)
    bin_s = params.tt_bin_ms / 1000.0

    trains = {
        "flat (Poisson, unmodulated)": rng.uniform(0, 2 * np.pi, n_spikes),
        "locked (von Mises, kappa=4)": rng.vonmises(mu=np.pi, kappa=4.0, size=n_spikes) % (2 * np.pi),
        "two peaks, half a cycle apart": (
            rng.vonmises(mu=0.0, kappa=8.0, size=n_spikes) + np.pi * rng.integers(0, 2, n_spikes)
        ) % (2 * np.pi),
    }

    rows = []
    for name, phases in trains.items():
        offsets = phases / (2 * np.pi) * period_s * FS
        spikes = np.sort((transitions[cycle_index] + offsets).astype(np.int64))

        theta = cycle_phases(spikes, transitions, period_s, FS)
        stats = phase_statistics(theta)
        _, _, smoothed = cycle_histogram(theta, period_s, n_cycles, bin_s, params.tt_smooth_bins)
        depth = modulation_depth(smoothed, stats["vs"])
        harmonic = phase_statistics(cycle_phases(spikes, transitions, period_s, FS, harmonic=2))

        rows.append({
            "train": name,
            "n_spikes": stats["n_spikes"],
            "mod_depth": round(depth["mod_depth"], 3),
            "mod_depth_norm": round(depth["mod_depth_norm"], 3),
            "vs_f0": round(stats["vs"], 4),
            "vs_2f0": round(harmonic["vs"], 4),
            "ppc_f0": round(stats["ppc"], 5),
        })

    table = pd.DataFrame(rows)

    flat = table.iloc[0]
    if flat.mod_depth > 1.25:
        raise AssertionError(
            f"an unmodulated train reads modulation depth {flat.mod_depth:.3f}. The estimator is "
            "manufacturing structure — check that the cycle histogram is not folding a "
            "non-integer number of cycles."
        )
    if flat.vs_f0 > 0.05:
        raise AssertionError(f"an unmodulated train reads vector strength {flat.vs_f0:.4f}")
    locked = table.iloc[1]
    if locked.vs_f0 < 0.3:
        raise AssertionError(f"a strongly locked train reads vector strength only {locked.vs_f0:.4f}")
    two_peak = table.iloc[2]
    if not (two_peak.vs_f0 < 0.1 < two_peak.vs_2f0):
        raise AssertionError(
            "the two-peak train should cancel at the fundamental and survive at the second "
            f"harmonic, but read VS_f0={two_peak.vs_f0:.4f}, VS_2f0={two_peak.vs_2f0:.4f}"
        )
    return table


def contamination_guard(session, store, tolerance_pct: float = 0.06) -> pd.DataFrame:
    """Check `ks_contam_pct` against Kilosort's own `cluster_ContamPct.tsv`.

    Kilosort writes the file to one decimal place, so agreement is asserted to just over half
    of that rounding step.
    """
    rows = []
    for _, cluster in session.clusters.iterrows():
        unit = int(cluster["cluster_id"])
        spikes = store.spikes(unit)
        if spikes.size < 11:
            continue
        rows.append({
            "unit": unit,
            "kilosort_file": float(cluster["ContamPct"]),
            "reimplemented": ks_contam_pct(spikes, FS),
        })

    table = pd.DataFrame(rows)
    table["abs_error"] = (table.kilosort_file - table.reimplemented).abs()
    worst = float(table.abs_error.max())
    if worst > tolerance_pct:
        raise AssertionError(
            f"the ContamPct re-implementation differs from Kilosort's file by up to "
            f"{worst:.3f} percentage points — it is not the same quantity"
        )
    return table
