"""
Preprocessing for continuous data. Operates on an `ephyslink.SessionOE`.

Each function takes a session, transforms `continuous` (or `epochs`), records what it did in
the session's history, and returns the session — so a pipeline is a chain and the finished
object still knows how it was made::

    import preprocessing as pp

    session = pp.detrend(session)
    session = pp.bandpass(session, 0.1, 300, notch_hz=[50, 100])
    session = pp.rereference(session, "average")
    session = pp.downsample(session, target_fs=1000)
    session = pp.epoch_at(session, session.events["ttl1_rising"], pre_ms=50, post_ms=200)
    session = pp.reject_artifacts(session, threshold=500)
    session.save("preprocessed/2026-01-01.h5")

**They mutate the session.** That is deliberate — the whole point of carrying one object is
not to accumulate copies of a multi-gigabyte array. Copy first if you want the original:
`before = session.continuous.copy()`.

---

### What was dropped from the old `openephysextract.preprocess`, and why

Ported here: detrend, bandpass/notch, re-reference, surface Laplacian, downsample, epoch,
event-triggered epoch, artifact rejection, standardise. Each is checked against a synthetic
signal with a known answer in `selftest_preprocessing.py`.

**Not ported: ASR, ICA removal, EOG regression, channel interpolation, and the automatic
bad-channel detector.** Each is a heuristic whose output cannot be checked without ground
truth, and the old implementations were untested. Porting them unverified would move code
whose correctness nobody has established into a place that implies it has been. They remain
in git history if they are wanted back; better still, `mne` implements all five and is
maintained by people who test them.

**Torch was removed.** Every step the old code ran through torch — detrend, filtering,
epoching — is a scipy or numpy operation that torch was converting to and from anyway. It
bought nothing and cost a heavyweight dependency.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import butter, decimate, detrend as _detrend, filtfilt, iirnotch, sosfiltfilt

CONTINUOUS = "continuous"
EPOCHS = "epochs"


def _active(session) -> tuple[str, np.ndarray]:
    """The array a step should act on: epochs once they exist, otherwise the trace."""
    name = EPOCHS if EPOCHS in session.arrays else CONTINUOUS
    if name not in session.arrays:
        raise KeyError("session has neither 'epochs' nor 'continuous'")
    return name, session.arrays[name]


# =============================================================================
# whole-trace operations
# =============================================================================

def detrend(session, kind: str = "linear"):
    """Remove a linear (or constant) trend from each channel.

    Drift dominates the low-frequency end of most recordings and will otherwise leak through
    a filter's transition band.
    """
    name, values = _active(session)
    axis = session.axis(name, "sample")
    session.arrays[name] = np.ascontiguousarray(
        _detrend(values.astype(np.float64), axis=axis, type=kind), dtype=np.float32
    )
    return session.log("detrend", array=name, kind=kind)


def bandpass(
    session, low_hz: float | None = None, high_hz: float | None = None,
    order: int = 4, notch_hz: Sequence[float] = (), notch_q: float = 30.0,
):
    """Zero-phase Butterworth bandpass, plus optional notches.

    **Zero-phase** (`filtfilt`/`sosfiltfilt`) because a causal filter delays the signal by a
    frequency-dependent amount, and a latency measured through one is the filter's latency as
    much as the brain's. The cost is that it is non-causal — it cannot be used online, and it
    smears a sharp transient symmetrically in both directions.

    Second-order sections for the bandpass: a 4th-order or higher band filter expressed as
    transfer-function coefficients is numerically unstable at the low ratios of cutoff to
    sampling rate that LFP work uses.
    """
    name, values = _active(session)
    axis = session.axis(name, "sample")
    data = values.astype(np.float64)
    nyquist = 0.5 * session.fs_hz

    for frequency in notch_hz:
        if frequency >= nyquist:
            raise ValueError(f"notch at {frequency} Hz is above Nyquist ({nyquist} Hz)")
        b, a = iirnotch(frequency, Q=notch_q, fs=session.fs_hz)
        data = filtfilt(b, a, data, axis=axis)

    if low_hz is not None or high_hz is not None:
        low = (low_hz or 0.0) / nyquist
        high = min(high_hz or nyquist * 0.99, nyquist * 0.99) / nyquist
        if low <= 0:
            sos = butter(order, high, btype="low", output="sos")
        elif high >= 1:
            sos = butter(order, low, btype="high", output="sos")
        else:
            if not low < high:
                raise ValueError(f"invalid band {low_hz}–{high_hz} Hz at fs={session.fs_hz}")
            sos = butter(order, [low, high], btype="band", output="sos")
        data = sosfiltfilt(sos, data, axis=axis)

    session.arrays[name] = np.ascontiguousarray(data, dtype=np.float32)
    return session.log("bandpass", array=name, low_hz=low_hz, high_hz=high_hz,
                       order=order, notch_hz=list(notch_hz))


def rereference(session, method: str = "average", pairs: Sequence[tuple[int, int]] = ()):
    """Re-reference. `average` subtracts the mean across channels; `bipolar` takes differences.

    Average referencing assumes the channels sample the field roughly symmetrically. On a
    linear probe crossing a laminar boundary that assumption is poor, and the common average
    then injects one region's signal into the other with the opposite sign.
    """
    name, values = _active(session)
    channel_axis = session.axis(name, "channel")

    if method == "average":
        session.arrays[name] = (values - values.mean(axis=channel_axis, keepdims=True)).astype(np.float32)
    elif method == "bipolar":
        if not pairs:
            raise ValueError("bipolar re-referencing needs `pairs`")
        moved = np.moveaxis(values, channel_axis, 0)
        stacked = np.stack([moved[i] - moved[j] for i, j in pairs])
        session.arrays[name] = np.moveaxis(stacked, 0, channel_axis).astype(np.float32)
        session.meta["bipolar_pairs"] = [list(p) for p in pairs]
    else:
        raise ValueError(f"unknown method {method!r}; use 'average' or 'bipolar'")

    return session.log("rereference", array=name, method=method, n_pairs=len(pairs))


def surface_laplacian(session, neighbours: dict[int, list[int]]):
    """Subtract each channel's neighbours' mean from it — a discrete spatial second derivative.

    Sharpens spatial resolution by removing the broadly distributed component. A channel with
    no neighbours listed is left alone rather than silently zeroed.
    """
    name, values = _active(session)
    channel_axis = session.axis(name, "channel")
    moved = np.moveaxis(values, channel_axis, 0)
    out = moved.copy()
    for channel, neighbour_list in neighbours.items():
        if neighbour_list:
            out[channel] = moved[channel] - moved[list(neighbour_list)].mean(axis=0)
    session.arrays[name] = np.moveaxis(out, 0, channel_axis).astype(np.float32)
    return session.log("surface_laplacian", array=name, n_channels=len(neighbours))


def downsample(session, target_fs: float):
    """Decimate to approximately `target_fs`, anti-aliased.

    The factor is `floor(fs / target_fs)`, so the result is at or above the target rather than
    below it, and the achieved rate is written back to `session.fs_hz` — decimating and then
    leaving the old rate in place is how sample indices silently stop meaning anything.

    `scipy.signal.decimate` low-pass filters first. Plain striding would fold everything above
    the new Nyquist back into the band you keep, and it would look like signal.
    """
    factor = int(max(1, session.fs_hz // target_fs))
    if factor == 1:
        return session.log("downsample", factor=1, note="already at or below target")

    name, values = _active(session)
    axis = session.axis(name, "sample")
    session.arrays[name] = np.ascontiguousarray(
        decimate(values.astype(np.float64), factor, axis=axis, ftype="fir", zero_phase=True),
        dtype=np.float32,
    )
    if "sample_numbers" in session.arrays:
        session.arrays["sample_numbers"] = session.arrays["sample_numbers"][::factor]
    for event_name, samples in session.events.items():
        session.events[event_name] = samples // factor

    previous = session.fs_hz
    session.fs_hz = previous / factor
    return session.log("downsample", array=name, factor=factor,
                       from_fs=previous, to_fs=session.fs_hz)


# =============================================================================
# epoching
# =============================================================================

def epoch(session, frame: int, stride: int, baseline_ms: float = 0.0):
    """Cut `continuous` into overlapping windows → `epochs` (channel, sample, epoch).

    `frame` and `stride` are in samples. With `baseline_ms > 0`, each epoch has the mean of
    its own first `baseline_ms` subtracted.
    """
    values = session.array(CONTINUOUS)
    channel_axis = session.axis(CONTINUOUS, "channel")
    data = np.moveaxis(values, channel_axis, 0)              # (channel, sample)
    n_channels, n_samples = data.shape

    starts = np.arange(0, n_samples - frame + 1, stride)
    if starts.size == 0:
        raise ValueError(f"frame={frame} is longer than the recording ({n_samples} samples)")

    epochs = np.stack([data[:, s:s + frame] for s in starts], axis=2)   # (ch, frame, epoch)
    epochs = _baseline_correct(epochs, baseline_ms, session.fs_hz)

    session.add_array(EPOCHS, epochs.astype(np.float32),
                      dims=["channel", "sample", "epoch"],
                      units=session.array_meta.get(CONTINUOUS, {}).get("units"))
    session.add_array("epoch_onsets", starts.astype(np.int64), dims=["epoch"],
                      description="sample index of each epoch's first sample")
    return session.log("epoch", frame=frame, stride=stride, baseline_ms=baseline_ms,
                       n_epochs=int(starts.size))


def epoch_at(session, event_samples, pre_ms: float, post_ms: float, baseline_ms: float = 0.0):
    """Cut epochs around event times → `epochs` (channel, sample, epoch).

    Events too close to either end are dropped rather than zero-padded, and how many were
    dropped is logged — a silently zero-padded epoch reads as a real response of zero.
    """
    values = session.array(CONTINUOUS)
    channel_axis = session.axis(CONTINUOUS, "channel")
    data = np.moveaxis(values, channel_axis, 0)
    n_samples = data.shape[1]

    pre = int(round(pre_ms * session.fs_hz / 1000))
    post = int(round(post_ms * session.fs_hz / 1000))
    events = np.asarray(event_samples, dtype=np.int64).ravel()

    usable = events[(events - pre >= 0) & (events + post <= n_samples)]
    if usable.size == 0:
        raise ValueError(
            f"no event has {pre_ms} ms before and {post_ms} ms after it inside the recording"
        )

    epochs = np.stack([data[:, e - pre:e + post] for e in usable], axis=2)
    epochs = _baseline_correct(epochs, baseline_ms or pre_ms, session.fs_hz)

    session.add_array(EPOCHS, epochs.astype(np.float32), dims=["channel", "sample", "epoch"])
    session.add_array("epoch_onsets", usable, dims=["epoch"],
                      description="sample index of the event each epoch is aligned to")
    session.meta["epoch_window_ms"] = [-pre_ms, post_ms]
    return session.log("epoch_at", pre_ms=pre_ms, post_ms=post_ms, baseline_ms=baseline_ms,
                       n_events=int(events.size), n_kept=int(usable.size),
                       n_dropped_at_edges=int(events.size - usable.size))


def _baseline_correct(epochs: np.ndarray, baseline_ms: float, fs_hz: float) -> np.ndarray:
    """Subtract each epoch's own leading baseline. Epochs are (channel, sample, epoch)."""
    if baseline_ms <= 0:
        return epochs
    n = max(1, int(round(baseline_ms * fs_hz / 1000)))
    return epochs - epochs[:, :n, :].mean(axis=1, keepdims=True)


def reject_artifacts(session, threshold: float, consensus: float = 0.6):
    """Drop epochs where too many channels exceed a peak-to-peak `threshold`.

    An epoch is dropped when the fraction of channels over threshold is at least `consensus`.
    Requiring agreement across channels is what distinguishes a movement artefact, which hits
    everything at once, from one bad channel, which should be dealt with as a channel.
    """
    epochs = session.array(EPOCHS)
    sample_axis = session.axis(EPOCHS, "sample")
    channel_axis = session.axis(EPOCHS, "channel")
    epoch_axis = session.axis(EPOCHS, "epoch")

    peak_to_peak = epochs.max(axis=sample_axis) - epochs.min(axis=sample_axis)
    over = (peak_to_peak > threshold)
    channel_dim = 0 if channel_axis < sample_axis else 1
    fraction = over.mean(axis=channel_dim)
    keep = fraction < consensus

    if not keep.any():
        # Writing an empty session is worse than failing: downstream code sees a valid file
        # with nothing in it and reports zero of everything. In a batch run the caller catches
        # this and records the session as failed, which is the honest outcome.
        median_ptp = float(np.median(peak_to_peak))
        raise ValueError(
            f"every one of the {keep.size} epochs was rejected at threshold={threshold:g}. "
            f"The median peak-to-peak amplitude is {median_ptp:.0f} — the threshold is "
            "probably in the wrong units, or the data is not in the units you think."
        )

    session.arrays[EPOCHS] = np.take(epochs, np.flatnonzero(keep), axis=epoch_axis)
    if "epoch_onsets" in session.arrays:
        session.arrays["epoch_onsets"] = session.arrays["epoch_onsets"][keep]

    return session.log("reject_artifacts", threshold=threshold, consensus=consensus,
                       n_before=int(keep.size), n_kept=int(keep.sum()))


# =============================================================================
# scaling
# =============================================================================

def standardize(session, method: str = "zscore", per_epoch: bool = True):
    """Rescale each channel. `zscore`, `minmax`, or `robust` (median and IQR).

    `robust` is the one to reach for when artefacts survive rejection: a single large
    transient sets the standard deviation and the range, but barely moves the median or the
    interquartile range.
    """
    name, values = _active(session)
    data = values.astype(np.float64)

    if name == EPOCHS and per_epoch:
        axis = session.axis(name, "sample")
    else:
        axis = session.axis(name, "sample")

    if method == "zscore":
        centre = data.mean(axis=axis, keepdims=True)
        scale = data.std(axis=axis, keepdims=True)
    elif method == "robust":
        centre = np.median(data, axis=axis, keepdims=True)
        q1, q3 = (np.percentile(data, p, axis=axis, keepdims=True) for p in (25, 75))
        scale = q3 - q1
    elif method == "minmax":
        centre = data.min(axis=axis, keepdims=True)
        scale = data.max(axis=axis, keepdims=True) - centre
    else:
        raise ValueError(f"unknown method {method!r}; use 'zscore', 'robust' or 'minmax'")

    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)      # a flat channel stays flat
    session.arrays[name] = ((data - centre) / scale).astype(np.float32)
    session.array_meta.setdefault(name, {})["units"] = f"{method} units"
    return session.log("standardize", array=name, method=method, per_epoch=per_epoch)
