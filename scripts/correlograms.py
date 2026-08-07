"""
Autocorrelograms.

Histogram of the time differences between every pair of spikes from one unit. Formally
C(tau) = sum over i != j of delta[tau - (t_i - t_j)]. Read at two window widths it answers
two unrelated questions:

* At +/- 50 ms with 0.5 ms bins, it is a **quality check**. A real neuron cannot fire twice
  within its absolute refractory period (~2.5 ms), so a genuine single unit's
  autocorrelogram must fall to near zero at short lags. A filled-in centre means the cluster
  contains spikes from more than one neuron.
* At +/- 500 ms with 5 ms bins, it is a **rhythmicity readout**. A unit firing in
  theta-locked bursts shows side-peaks at one theta period, 83-250 ms. The narrow window
  cannot show even one of them.

The computation is **chunked** over spikes so peak memory stays bounded whatever the firing
rate: a wide window on a high-rate train can produce hundreds of millions of pairs.

Depends on: numpy.
"""

from __future__ import annotations

import numpy as np

MAX_PAIRS_PER_CHUNK = 20_000_000
"""Memory ceiling for the expanded pair list. Lower it on a small machine; it changes speed,
never the result."""


def pair_lag_histogram(
    times_ms: np.ndarray, window_ms: float, bin_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of all pairwise lags within +/- window, self-pairs removed.

    Returns `(bin_centres_ms, counts)`. Input is in **milliseconds**.
    """
    times_ms = np.sort(np.asarray(times_ms, dtype=np.float64))
    edges = np.arange(-window_ms, window_ms + bin_ms / 2, bin_ms)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    n_total = times_ms.size
    if n_total < 2:
        return (edges[:-1] + edges[1:]) / 2, counts

    lo = np.searchsorted(times_ms, times_ms - window_ms, side="left")
    hi = np.searchsorted(times_ms, times_ms + window_ms, side="right")
    neighbours = hi - lo

    cumulative = np.cumsum(neighbours)
    chunk_edges = [0]
    while chunk_edges[-1] < n_total:
        base = cumulative[chunk_edges[-1] - 1] if chunk_edges[-1] > 0 else 0
        end = int(np.searchsorted(cumulative, base + MAX_PAIRS_PER_CHUNK, side="right"))
        chunk_edges.append(max(end, chunk_edges[-1] + 1))

    for start, stop in zip(chunk_edges[:-1], chunk_edges[1:]):
        n_chunk = neighbours[start:stop]
        total = int(n_chunk.sum())
        if total == 0:
            continue
        starts = np.repeat(lo[start:stop], n_chunk)
        offsets = np.arange(total) - np.repeat(np.cumsum(n_chunk) - n_chunk, n_chunk)
        lags = times_ms[starts + offsets] - np.repeat(times_ms[start:stop], n_chunk)
        counts += np.histogram(lags, bins=edges)[0]

    # every spike pairs with itself exactly once, and all such pairs land in the lag-0 bin
    zero_bin = int(np.floor(window_ms / bin_ms))
    counts[zero_bin] -= n_total
    return (edges[:-1] + edges[1:]) / 2, counts


def all_pair_lags_s(times_s: np.ndarray, window_s: float) -> np.ndarray:
    """Every pairwise lag within +/- window, **self-pairs included**. Seconds in and out.

    Kept separate from `pair_lag_histogram` because Kilosort's contamination metric measures
    its shoulder densities *before* removing the self-pairs, so it needs the raw list.
    """
    lo = np.searchsorted(times_s, times_s - window_s, side="left")
    hi = np.searchsorted(times_s, times_s + window_s, side="right")
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        return np.empty(0)
    starts = np.repeat(lo, n)
    offsets = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    return times_s[starts + offsets] - np.repeat(times_s, n)


def autocorrelogram(
    spike_samples: np.ndarray,
    bin_ms: float,
    window_ms: float,
    fs: float,
    max_spikes: int | None = None,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelogram of one unit. Returns `(bin_centres_ms, counts)`.

    If `max_spikes` is set and exceeded, spikes are uniformly subsampled first. Uniform
    subsampling leaves the *shape* unchanged and scales counts by the sampling fraction
    squared, so only relative structure should ever be read off the result.
    """
    times_ms = np.sort(np.asarray(spike_samples)) / fs * 1000.0
    if max_spikes is not None and times_ms.size > max_spikes:
        rng = np.random.default_rng(rng_seed)
        times_ms = np.sort(rng.choice(times_ms, size=max_spikes, replace=False))
    return pair_lag_histogram(times_ms, window_ms, bin_ms)


def rebin(centres: np.ndarray, counts: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Coarsen a histogram by an integer factor. For display only."""
    usable = (counts.size // factor) * factor
    return (
        centres[:usable].reshape(-1, factor).mean(1),
        counts[:usable].reshape(-1, factor).sum(1),
    )


def band_mean(centres_ms: np.ndarray, counts: np.ndarray, lo_ms: float, hi_ms: float) -> tuple[float, int]:
    """Mean and total count over a symmetric lag band |tau| in [lo, hi]."""
    mask = (np.abs(centres_ms) >= lo_ms) & (np.abs(centres_ms) <= hi_ms)
    if not mask.any():
        return np.nan, 0
    return float(counts[mask].mean()), int(counts[mask].sum())
