"""
Every number this project's analysis depends on, in one place.

Two kinds of number live here and they are kept apart on purpose.

**Constants** (module level, UPPER_CASE) are properties of the *hardware* or of the *sorter*.
They are not choices. `FS` is 30 kHz because the amplifier sampled at 30 kHz.

**Parameters** (fields of `Params`) are analysis *choices*. Every one could defensibly be
something else, so every one is visible at the call site and can be overridden without
editing this file::

    params = Params()                                 # the defaults documented below
    params = replace(params, psth_bin_s=0.010)        # one thing changed, the rest stated

Nothing in this file executes anything, and nothing else in this project defines a magic
number. The shared modules in `../../../scripts/` take these values as arguments — they are
deliberately ignorant of this project, which is what makes them reusable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace  # noqa: F401  (replace re-exported for callers)

# =============================================================================
# CONSTANTS — properties of the recording hardware and of Kilosort, not choices
# =============================================================================

FS = 30_000
"""Sampling rate in Hz. Every raw time in the dataset is a sample index; divide by this once,
at the boundary, and work in seconds thereafter."""

CHANNEL_PITCH_UM = 20.0
"""Centre-to-centre spacing of adjacent probe contacts, in micrometres. Verified at load time
against `channel_positions.npy`; the loader raises if the file disagrees."""

KS_CONTAM_THRESHOLD_PCT = 20.0
"""The contamination threshold Kilosort 4 itself uses to assign the `good` label.

Determined empirically rather than taken on faith: across all 15 sessions the largest
`ContamPct` carried by any `good` cluster is 17.7-20.0 %, and the smallest carried by any
`mua` cluster is 20.0 %. So `good` implies `ContamPct < 20 %`.

The converse does NOT hold — `mua` clusters with `ContamPct = 0 %` exist in 14 of 15 sessions,
because Kilosort also rejects on spike count and amplitude. Treat 20 % as *Kilosort's
contamination threshold*, not as *Kilosort's whole labelling rule*."""

HILL_FP_THRESHOLD_PCT = 10.0
"""Conventional acceptance threshold for the Hill et al. (2011) false-positive fraction f_p,
as a percentage. A different number from `KS_CONTAM_THRESHOLD_PCT` because it is a different
quantity — f_p is a bounded fraction of misassigned spikes, ContamPct is an unbounded density
ratio. Comparing the two at a single shared cut-off would be a category error; that is the
whole point of the comparison figure."""


# =============================================================================
# PARAMETERS — analysis choices
# =============================================================================

@dataclass(frozen=True)
class Params:
    """Analysis choices. Frozen so a parameter cannot be mutated halfway through a run."""

    # ---- exclusions ---------------------------------------------------------
    settle_s: float = 300.0
    """Seconds discarded at the start while the probe settles. The first stimulus block is at
    ~370 s, so this never touches a stimulus response; it changes only the autocorrelogram and
    firing-rate numbers. Still unconfirmed by the professor."""

    excluded_conditions: tuple[str, ...] = ("phase_reversing_2",)
    """Analog channels dropped everywhere. `phase_reversing_2` is the opposite edge polarity
    of the same physical grating reversal as `phase_reversing_1` (identical 160 transitions
    per block, offset by 12.5 ms), so including both double-counts one stimulus."""

    # ---- unit inclusion -----------------------------------------------------
    min_spikes_unit: int = 100
    """Below this many spikes a unit cannot support a firing-rate comparison, let alone a
    phase estimate."""

    clean_contam_pct: float = KS_CONTAM_THRESHOLD_PCT
    """Contamination ceiling for the analysis set. Defaults to Kilosort's own threshold, which
    makes the filter redundant with the `good` label by design — the set is then exactly "what
    Kilosort called a single unit". Lower it to 10.0 for a stricter set; the notebook prints
    both counts so the choice stays visible."""

    # ---- autocorrelogram, short window (refractory / quality control) -------
    acg_bin_ms: float = 0.5
    """At or below 0.5 ms: the refractory dip is ~2.5 ms wide, and a bin wider than about a
    fifth of that smears the dip into the surrounding counts."""

    acg_window_ms: float = 50.0
    """+/- 50 ms. Wide enough to show the 25 ms side-peaks a 40 Hz-periodic unit would produce,
    narrow enough that the refractory dip is still legible."""

    refractory_ms: float = 2.5
    """Absolute refractory period assumed for mouse cortex. Drawn on every panel as the window
    a true single unit must fall to near zero inside."""

    # ---- autocorrelogram, long window (theta / cell type) -------------------
    acg_wide_window_ms: float = 500.0
    """+/- 500 ms. Theta is 4-12 Hz, a period of 83-250 ms, so half a second shows three to six
    theta side-peaks. The +/- 50 ms window cannot show even one."""

    acg_wide_bin_ms: float = 1.0
    """Computed at 1 ms; the display is re-binned. The fine underlying bin is kept because the
    burst index is read at 3-5 ms lag."""

    acg_wide_display_bin_ms: float = 5.0
    """Bin width for *plotting* the long autocorrelogram. At 1 ms the theta side-peaks are
    buried in counting noise; 5 ms is well under a fifth of the shortest theta period."""

    acg_max_spikes: int = 200_000
    """Above this, spikes are uniformly subsampled before the all-pairs lag computation.
    Subsampling leaves the shape unchanged and scales the counts; it guards against a pooled
    train exhausting memory."""

    # ---- Hill et al. (2011) false-positive estimate -------------------------
    hill_t_ref_ms: float = 2.0
    """Refractory period assumed by the Hill estimator. Deliberately shorter than
    `refractory_ms`: the estimator counts violations strictly inside the refractory period,
    and a generous window inflates f_p."""

    hill_t_cens_ms: float = 0.5
    """Censored period — the sorter's dead time, during which a second spike cannot be
    detected, so violations shorter than this are invisible and must be excluded from the
    expected count."""

    # ---- cell type ----------------------------------------------------------
    t2p_split_ms: float = 0.45
    """Trough-to-peak duration separating narrow-waveform (putative fast-spiking interneuron)
    from wide-waveform (putative principal cell). 0.4-0.5 ms is the standard extracellular
    convention; `waveforms.kde_antimodes` recomputes the separable points from this session's
    own distribution so the default can be checked rather than trusted."""

    burst_index_lag_ms: tuple[float, float] = (3.0, 5.0)
    """Short-lag window for the burst index numerator (Royer et al. 2012)."""

    burst_index_ref_ms: tuple[float, float] = (200.0, 300.0)
    """Far-shoulder window for the burst index denominator — outside any post-spike dynamics,
    so it estimates the unit's own baseline density."""

    theta_peak_ms: tuple[float, float] = (100.0, 140.0)
    """Lag window in which a theta-modulated unit's first autocorrelogram side-peak falls."""

    theta_trough_ms: tuple[float, float] = (50.0, 70.0)
    """Lag window of the corresponding trough, half a theta cycle."""

    # ---- block PSTH ---------------------------------------------------------
    psth_pre_s: float = 2.0
    """Baseline window before block onset. Asserted at load time to contain no other
    condition's stimulus."""

    psth_post_s: float = 4.0
    """Blocks are exactly 4 s long."""

    psth_bin_s: float = 0.025
    """25 ms. Resolves the *rate envelope* and nothing faster. One 25 ms bin is exactly one
    40 Hz cycle, so this histogram is structurally incapable of showing 40 Hz following —
    which is why the transition-triggered analysis exists separately, at 1 ms."""

    # ---- responder metric ---------------------------------------------------
    responder_alpha: float = 0.05
    """False-discovery rate for the Benjamini-Hochberg correction across units."""

    responder_n_permutations: int = 10_000
    """Sign-flip permutations for the assumption-free confirmation of the Wilcoxon p-value.
    Set to 0 to skip."""

    responder_z_legacy: float = 2.0
    """The professor's original rule: |z of the population-relative % change| > 2. Kept as a
    comparison column so the change in definition is auditable, not as the headline."""

    # ---- transition-triggered PSTH and modulation depth ---------------------
    tt_cycles: float = 3.5
    """Display width of the transition-triggered histogram, in stimulus cycles. Display only:
    the metrics come from a single cycle. See methods/modulation-depth-calculation.md for why
    folding this window would be an error."""

    tt_bin_ms: float = 1.0
    """25 bins per cycle at 40 Hz — fine enough to resolve within-cycle structure."""

    tt_min_spikes: int = 200
    """Below this many in-cycle spikes, peak and trough are dominated by counting noise and the
    modulation depth is not reported."""

    tt_smooth_bins: int = 3
    """Width of the circular boxcar applied to the one-cycle histogram before peak and trough
    are read off. Without it a single noisy bin sets the value."""

    # ---- static condition ---------------------------------------------------
    static_win_s: float = 2.0
    """Symmetric before/during window for the static grating, if the symmetric comparison is
    wanted instead of the uniform 2 s pre / 4 s stimulus."""

    # ---- pooled MUA ---------------------------------------------------------
    pool_include_good: bool = False
    """Whether the pooled per-region train includes `good` clusters as well as `mua` ones.
    False matches the conventional meaning of MUA; True gives a truer population rate. Exposed
    because the choice changes the pooled magnitude and should not be silent."""
