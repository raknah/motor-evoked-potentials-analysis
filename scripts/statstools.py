"""
Statistics used across spike-train analyses. Nothing project-specific.

**Multiple comparisons.** Every analysis tests dozens of units at once. At alpha = 0.05,
testing 45 units produces two false positives on average even if nothing is happening. The
Benjamini-Hochberg procedure controls the *false discovery rate* — the expected proportion
of the rejections that are false — which is the right guarantee when the question is "which
units responded", not "did any unit respond".

**Paired comparison of two windows.** Pre and post counts come from the same trial of the
same unit, so they are paired. Trial-wise spike counts are neither normal nor homoscedastic,
so the signed-rank test is the default rather than a t-test.

**Sign-flip permutation.** The signed-rank test still assumes the null distribution of
differences is symmetric about zero. A permutation test assumes only *exchangeability of the
sign of each paired difference under the null*, which is exactly what "the stimulus did
nothing" means. Use it to confirm the Wilcoxon rather than to replace it: a permutation
p-value cannot go below 1/(n_permutations + 1), so it cannot be FDR-corrected across many
units without a very large permutation count.

Depends on: numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values). Same length and order as the input."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(n)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(adjusted, 1.0)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples.

    Returns 1.0 when every pair is tied, which is the correct answer (no evidence) and avoids
    scipy raising on a degenerate input. `zero_method="zsplit"` keeps zero differences in the
    ranking rather than discarding them: discarding shrinks n and inflates significance,
    which matters because ties are common for low-rate units.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or np.all(a == b):
        return 1.0
    return float(sstats.wilcoxon(a, b, zero_method="zsplit").pvalue)


def sign_flip_permutation(
    a: np.ndarray, b: np.ndarray, n_permutations: int = 10_000, seed: int = 0
) -> float:
    """Two-sided p-value for `mean(a - b) == 0` under random sign flips of each difference.

    The +1 in numerator and denominator includes the observed arrangement in its own null,
    which is what makes the test exact rather than anti-conservative at small counts
    (Phipson & Smyth 2010, *Stat Appl Genet Mol Biol* 9:39).

    Returns NaN if `n_permutations` is 0.
    """
    if n_permutations <= 0:
        return np.nan
    differences = np.asarray(a, float) - np.asarray(b, float)
    if differences.size == 0 or np.all(differences == 0):
        return 1.0
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, differences.size))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n_permutations + 1))


def poisson_rate_sd(mean_rate_hz: float, bin_s: float, n_trials: int) -> float:
    """Standard deviation of a binned rate estimate if the spike train were Poisson.

    A count in one bin of one trial has variance equal to its mean — the defining property of
    the Poisson distribution — so a rate estimated from `n_trials` bins of width `bin_s` has
    SD sqrt(mu / (bin_s * n_trials)).

    Use as a *floor* on empirical baseline SD. A near-silent unit whose baseline bins happen
    to be almost identical otherwise gets a tiny SD and its z-scores explode on counting noise
    alone. The floor only ever raises SD, so it can only ever be conservative.
    """
    return float(np.sqrt(max(mean_rate_hz, 1e-12) / (bin_s * max(n_trials, 1))))
