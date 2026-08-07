"""
Per-unit spike access, and the windowing primitives every peri-event analysis is built on.

**Why a store rather than a filter.** The naive way to get unit 42's spikes is
`spike_times[spike_clusters == 42]`, which scans the whole array. With hundreds of clusters
times several conditions times several analyses that scan runs thousands of times and
dominates runtime. Bucketing once, up front, turns every later lookup into a dictionary hit.

**Pooled units.** A pooled train (say "all cortical multi-unit spikes as one train") is
registered under a *negative* id and returned like any other unit, so downstream code needs
no special case.

**The primitives.**

`gather` answers: *for every trigger, which spikes fell near it, and how far away?* That is
the only question a PSTH, an autocorrelogram or a phase estimate ever asks.

`window_counts` answers: *how many spikes did each trigger get in this window?* — one number
per trial, which is what a paired statistical test needs.

`trial_bin_counts` returns the whole trial x bin matrix rather than collapsing it, which is
what makes per-trial normalisation and trial-wise statistics possible at all.

None of them loops over triggers in Python. The trick in `gather` is **ragged range
expansion**: `searchsorted` gives the first and last spike index for every trigger, and those
variable-length ranges become one flat index array via `repeat` and `cumsum`.

Times are **sample indices** throughout; `fs` is passed explicitly wherever seconds and
samples meet, so this module never assumes a sampling rate.

Depends on: numpy.
"""

from __future__ import annotations

import numpy as np


class SpikeStore:
    """Per-unit spike times, bucketed once. Negative ids are pooled virtual units."""

    def __init__(self, spike_times: np.ndarray, spike_clusters: np.ndarray, fs: float):
        order = np.argsort(spike_clusters, kind="stable")
        unique, counts = np.unique(spike_clusters[order], return_counts=True)
        edges = np.r_[0, np.cumsum(counts)]
        self.fs = float(fs)
        self._by_unit: dict[int, np.ndarray] = {
            int(unit): np.sort(spike_times[order[edges[i]:edges[i + 1]]])
            for i, unit in enumerate(unique)
        }
        self._pools: dict[int, list[int]] = {}

    def __contains__(self, unit: int) -> bool:
        return int(unit) in self._by_unit

    @property
    def unit_ids(self) -> list[int]:
        return sorted(u for u in self._by_unit if u >= 0)

    def spikes(self, unit: int) -> np.ndarray:
        """Sorted sample indices for one unit. Pooled ids are built on first use."""
        unit = int(unit)
        if unit in self._by_unit:
            return self._by_unit[unit]
        if unit in self._pools:
            self._by_unit[unit] = np.sort(
                np.concatenate([self._by_unit[int(m)] for m in self._pools[unit]])
            )
            return self._by_unit[unit]
        return np.empty(0, dtype=np.int64)

    def register_pool(self, pool_id: int, members: list[int]) -> int:
        """Register a virtual unit that is the union of `members`. `pool_id` must be < 0."""
        if pool_id >= 0:
            raise ValueError("pooled unit ids must be negative so they cannot collide with clusters")
        self._pools[int(pool_id)] = [int(m) for m in members]
        self._by_unit.pop(int(pool_id), None)      # drop any stale cached union
        return pool_id

    def after(self, unit: int, t_s: float) -> np.ndarray:
        """This unit's spikes from `t_s` seconds onwards — e.g. to drop a settling period."""
        spikes = self.spikes(unit)
        return spikes[spikes >= t_s * self.fs]

    def as_dict(self, unit_ids) -> dict[int, np.ndarray]:
        """`{unit_id: spikes}` for the given ids — the input format the generic tables want."""
        return {int(u): self.spikes(int(u)) for u in unit_ids}


# =============================================================================
# primitives
# =============================================================================

def gather(
    spikes: np.ndarray, triggers: np.ndarray, pre_samples: int, post_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spike times relative to each trigger, for spikes inside [-pre, +post], in samples.

    Returns `(relative_samples, n_per_trigger)`. `relative_samples` is flat and ordered by
    trigger, so `np.split(relative, np.cumsum(n_per_trigger)[:-1])` recovers the per-trigger
    lists when they are actually needed (usually they are not).

    When triggers are closer together than the window, a spike is returned more than once,
    under more than one trigger. That is correct for a peri-event histogram — the rate axis
    divides by the trigger count — but it means adjacent parts of the resulting histogram are
    **not statistically independent**.
    """
    spikes = np.asarray(spikes)
    triggers = np.asarray(triggers, dtype=np.int64)
    lo = np.searchsorted(spikes, triggers - pre_samples, side="left")
    hi = np.searchsorted(spikes, triggers + post_samples, side="right")
    n_per_trigger = hi - lo
    total = int(n_per_trigger.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), n_per_trigger
    starts = np.repeat(lo, n_per_trigger)
    offsets = np.arange(total) - np.repeat(np.cumsum(n_per_trigger) - n_per_trigger, n_per_trigger)
    relative = spikes[starts + offsets] - np.repeat(triggers, n_per_trigger)
    return relative, n_per_trigger


def window_counts(
    spikes: np.ndarray, triggers: np.ndarray, lo_s: float, hi_s: float, fs: float
) -> np.ndarray:
    """Spike count per trigger in the half-open window [lo_s, hi_s) relative to it.

    Half-open so that back-to-back windows partition time without double-counting a spike
    that lands exactly on the boundary.
    """
    spikes = np.asarray(spikes)
    triggers = np.asarray(triggers, dtype=np.int64)
    return (
        np.searchsorted(spikes, triggers + int(hi_s * fs), side="left")
        - np.searchsorted(spikes, triggers + int(lo_s * fs), side="left")
    )


def trial_bin_counts(
    spikes: np.ndarray, triggers: np.ndarray, pre_s: float, post_s: float,
    bin_s: float, fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The trial x bin count matrix underlying a PSTH.

    Returns `(bin_centres_s, counts)` with `counts` of shape `(n_triggers, n_bins)`.
    The conventional trial-averaged PSTH is `counts.mean(0) / bin_s`.
    """
    relative, n_per_trigger = gather(spikes, triggers, int(pre_s * fs), int(post_s * fs))
    edges = np.arange(-pre_s, post_s + bin_s / 2, bin_s)
    n_bins = edges.size - 1
    counts = np.zeros((len(triggers), n_bins), dtype=np.int64)

    if relative.size:
        trial_index = np.repeat(np.arange(len(triggers)), n_per_trigger)
        bin_index = np.floor((relative / fs + pre_s) / bin_s).astype(np.int64)
        inside = (bin_index >= 0) & (bin_index < n_bins)
        np.add.at(counts, (trial_index[inside], bin_index[inside]), 1)

    centres = (edges[:-1] + edges[1:]) / 2
    return centres, counts
