"""
Cluster contamination: how many of a cluster's spikes came from a different neuron.

A real neuron cannot fire twice within its refractory period (~2.5 ms). Spikes at short lags
therefore came from somewhere else, and counting them is how cluster quality is measured.

Two established ways to count them. They are **not the same quantity** and must not be
compared at a shared threshold:

|  | `ks_contam_pct` | `hill_false_positive` |
|---|---|---|
| what it is | refractory-window spike **density** / the unit's own baseline density | estimated **fraction of spikes** that do not belong |
| range | 0 % to unbounded | 0 to 0.5, then undefined |
| 100 % means | refractory window as full as baseline | — |
| above 100 % | common; it means the unit **bursts** | impossible |

Kilosort's metric is re-implemented from `kilosort/CCG.py` rather than taken from a library
because no library computes Kilosort's R12. Always check a re-implementation against the
sorter's own `cluster_ContamPct.tsv` before believing it is the same number.

Depends on: numpy, correlograms.
"""

from __future__ import annotations

import numpy as np

from correlograms import all_pair_lags_s


def ks_contam_pct(
    spike_samples: np.ndarray, fs: float, n_bins: int = 500, bin_s: float = 1e-3
) -> float:
    """Kilosort 4's `ContamPct`, re-implemented from `kilosort/CCG.py`.

        ContamPct = 100 * R12
        R12 = min over i in 1..10 ms of R_i  /  max(R00, R01)
        R_i = K(|tau| <= i) / (2 i dt N^2 / T)

    with K the 1 ms-binned autocorrelogram (lag-0 bin excluded), R_i the observed
    refractory-window count divided by what a homogeneous Poisson train of the same overall
    rate would give — this ratio is the firing-rate correction — and R00, R01 the same
    density measured in the far (250-500 ms) and near (10-50 ms) shoulders. Dividing by their
    maximum re-references the number to the unit's own baseline rather than to Poisson.
    """
    times_s = np.sort(np.asarray(spike_samples)) / fs
    n = times_s.size
    if n < 11:
        return np.nan
    window_s = n_bins * bin_s
    duration_s = float(times_s.max() - times_s.min())

    # Kilosort's exact indexing: 2*n_bins+1 bins centred on lag 0, self-pairs INCLUDED at
    # this stage (removed below, after the shoulder densities have been measured).
    lags = np.round(all_pair_lags_s(times_s, window_s) / bin_s).astype(np.int64) + n_bins
    lags = lags[(lags >= 0) & (lags <= 2 * n_bins)]
    K = np.bincount(lags, minlength=2 * n_bins + 1).astype(np.float64)

    far = np.hstack((np.arange(1, n_bins // 2), np.arange(3 * n_bins // 2, 2 * n_bins)))
    near_left = np.arange(n_bins - 50, n_bins - 10)
    near_right = np.arange(n_bins + 10, n_bins + 50)

    def density(index: np.ndarray) -> float:
        return K[index].sum() / (len(index) * bin_s * n * n / duration_s)

    R00 = density(far)
    R01 = max(density(near_left), density(near_right))

    zero = K[n_bins]
    K[n_bins] = 0.0
    R = np.array([
        K[np.arange(n_bins - i, n_bins + i + 1)].sum() / (2 * i * bin_s * n * n / duration_s)
        for i in range(1, 11)
    ])
    K[n_bins] = zero
    return float(100.0 * R.min() / (1e-10 + max(R00, R01)))


def hill_false_positive(
    spike_samples: np.ndarray, fs: float, t_ref_ms: float = 2.0, t_cens_ms: float = 0.5
) -> float:
    """Hill et al. (2011) *J Neurosci* 31:8699 estimated false-positive fraction.

        f_p = 0.5 * (1 - sqrt(1 - 2 N_viol T / (N^2 (t_ref - t_cens))))

    `t_ref` is the refractory period assumed by the estimator — deliberately shorter than the
    period drawn on a quality-control figure, because a generous window inflates f_p.
    `t_cens` is the sorter's dead time, during which a second spike cannot be detected, so
    violations shorter than it are invisible and must be excluded from the expected count.

    Returns NaN when the discriminant is negative — that is, when violations are denser than
    *any* contamination fraction can account for. Those units are **saturated, not clean**,
    and must be displayed as such rather than silently dropped.
    """
    times_s = np.sort(np.asarray(spike_samples)) / fs
    n = times_s.size
    if n < 2:
        return np.nan
    duration_s = float(times_s.max() - times_s.min())
    t_ref, t_cens = t_ref_ms / 1000.0, t_cens_ms / 1000.0
    isi = np.diff(times_s)
    n_violations = int(np.sum((isi > t_cens) & (isi <= t_ref)))
    discriminant = 1.0 - (2.0 * n_violations * duration_s) / (n * n * (t_ref - t_cens))
    if discriminant < 0:
        return np.nan
    return float(0.5 * (1.0 - np.sqrt(discriminant)))
