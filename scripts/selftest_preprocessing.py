#!/usr/bin/env python3
"""
Checks every preprocessing step against a synthetic signal whose answer is known.

    python scripts/selftest_preprocessing.py

A preprocessing function that runs without error is not a preprocessing function that works.
Each test below constructs data where the correct output can be written down in advance —
a 10 Hz sine plus a 60 Hz sine plus a linear drift — and asserts that.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import preprocessing as pp  # noqa: E402
from ephyslink import SessionOE  # noqa: E402

FS = 1000.0
DURATION_S = 20.0
N_CHANNELS = 4


def build() -> SessionOE:
    """4 channels of 10 Hz + 60 Hz + drift, with a known common-mode component on top."""
    t = np.arange(int(FS * DURATION_S)) / FS
    signal = np.stack([
        100 * np.sin(2 * np.pi * 10 * t)          # the band of interest
        + 50 * np.sin(2 * np.pi * 60 * t)         # line noise
        + 20 * t                                   # drift
        + 30 * np.sin(2 * np.pi * 3 * t)          # common mode, identical on every channel
        for _ in range(N_CHANNELS)
    ]).astype(np.float32)
    session = SessionOE(id="synthetic", fs_hz=FS, experiment="preprocessing selftest")
    session.add_array("continuous", signal, dims=["channel", "sample"], units="µV")
    return session


def power_at(signal: np.ndarray, fs: float, frequency: float) -> float:
    """Amplitude of one frequency component, in the signal's own units.

    Normalised by length, so a value computed before downsampling is comparable with one
    computed after — a raw `|rfft|` is not, because its magnitude scales with the number of
    samples, and comparing the two would make any decimation look like attenuation.
    """
    spectrum = np.abs(np.fft.rfft(signal)) * 2 / signal.size
    freqs = np.fft.rfftfreq(signal.size, 1 / fs)
    return float(spectrum[np.argmin(np.abs(freqs - frequency))])


def ok(name: str, condition: bool) -> None:
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name


def main() -> int:
    # --- detrend ---------------------------------------------------------
    s = pp.detrend(build())
    ok("detrend removes the linear drift (channel mean ≈ 0)",
       abs(float(s.continuous.mean())) < 1e-3)

    # --- bandpass + notch ------------------------------------------------
    s = pp.bandpass(build(), 5, 40, notch_hz=[60])
    trace = s.continuous[0]
    ok("bandpass keeps 10 Hz", power_at(trace, FS, 10) > 0.5 * power_at(build().continuous[0], FS, 10))
    ok("bandpass + notch removes 60 Hz (>40 dB down)",
       power_at(trace, FS, 60) < power_at(build().continuous[0], FS, 60) / 100)
    # Zero phase means the 10 Hz component comes out at the phase it went in at. Measured on a
    # single clean tone, because the phase of a *composite* signal's peak is not a filter
    # property. Contrasted with a causal filter, which is what the delay would look like.
    t = np.arange(int(FS * DURATION_S)) / FS
    tone = np.stack([100 * np.sin(2 * np.pi * 10 * t)] * N_CHANNELS).astype(np.float32)
    pure = SessionOE(id="tone", fs_hz=FS)
    pure.add_array("continuous", tone.copy(), dims=["channel", "sample"])

    def phase_at(signal, frequency):
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(signal.size, 1 / FS)
        return float(np.angle(spectrum[np.argmin(np.abs(freqs - frequency))]))

    filtered = pp.bandpass(pure, 5, 40).continuous[0]
    from scipy.signal import butter as _butter, sosfilt as _sosfilt
    causal = _sosfilt(_butter(4, [5 / (FS / 2), 40 / (FS / 2)], btype="band", output="sos"), tone[0])

    shift_zero_phase = abs(phase_at(filtered, 10) - phase_at(tone[0], 10))
    shift_causal = abs(phase_at(causal, 10) - phase_at(tone[0], 10))
    ok(f"zero-phase leaves 10 Hz unshifted ({shift_zero_phase:.4f} rad) where a causal "
       f"filter would delay it ({shift_causal:.3f} rad)",
       shift_zero_phase < 0.01 and shift_causal > 0.1)

    # --- rereference -----------------------------------------------------
    s = build()
    s.arrays["continuous"][0] += 200 * np.sin(2 * np.pi * 25 * np.arange(s.n_samples) / FS)
    s = pp.rereference(s, "average")
    ok("average reference removes the component common to all channels",
       power_at(s.continuous[1], FS, 3) < 0.05 * power_at(build().continuous[1], FS, 3))

    s = pp.rereference(build(), "bipolar", pairs=[(0, 1), (2, 3)])
    ok("bipolar reference gives one channel per pair, and identical channels cancel",
       s.continuous.shape[0] == 2 and np.allclose(s.continuous, 0, atol=1e-3))

    # --- surface laplacian ------------------------------------------------
    s = pp.surface_laplacian(build(), {0: [1, 2], 3: []})
    ok("laplacian cancels identical neighbours and leaves an unlisted channel alone",
       np.allclose(s.continuous[0], 0, atol=1e-3)
       and np.allclose(s.continuous[3], build().continuous[3], atol=1e-3))

    # --- downsample -------------------------------------------------------
    s = build()
    s.add_events("stim", np.array([1000, 5000], dtype=np.int64))
    s = pp.downsample(s, target_fs=250)
    ok(f"downsample updates fs to {s.fs_hz:g} Hz and the array length together",
       s.fs_hz == 250.0 and s.continuous.shape[1] == int(FS * DURATION_S) // 4)
    ok("downsample rescales event sample indices too",
       np.array_equal(s.events["stim"], [250, 1250]))
    ok("anti-aliasing: 60 Hz survives below the new 125 Hz Nyquist",
       power_at(s.continuous[0], s.fs_hz, 60) > 0.5 * power_at(build().continuous[0], FS, 60))

    # a component above the new Nyquist must be removed, not folded back
    t = np.arange(int(FS * DURATION_S)) / FS
    alias = build()
    alias.arrays["continuous"] = np.stack(
        [200 * np.sin(2 * np.pi * 200 * t)] * N_CHANNELS).astype(np.float32)
    alias = pp.downsample(alias, target_fs=250)
    ok("anti-aliasing: a 200 Hz component is removed, not folded to 50 Hz",
       power_at(alias.continuous[0], alias.fs_hz, 50) < 0.02 * 200.0)

    # --- epoching ---------------------------------------------------------
    s = pp.epoch(build(), frame=500, stride=500)
    ok("epoch produces (channel, sample, epoch)",
       s.arrays["epochs"].shape == (N_CHANNELS, 500, 40)
       and s.dims["epochs"] == ["channel", "sample", "epoch"])

    s = build()
    events = np.array([2000, 5000, 8000, 19_990], dtype=np.int64)   # the last is too close
    s = pp.epoch_at(s, events, pre_ms=100, post_ms=200)
    ok("epoch_at drops events too close to the edge instead of zero-padding",
       s.arrays["epochs"].shape == (N_CHANNELS, 300, 3)
       and s.history[-1]["params"]["n_dropped_at_edges"] == 1)
    ok("epoch_at baseline-corrects against the pre-event window",
       abs(float(s.arrays["epochs"][:, :100, :].mean())) < 1e-3)

    # --- artifact rejection ----------------------------------------------
    # A *transient*, not an offset: rejection is on peak-to-peak, and adding a constant to a
    # whole epoch leaves peak-to-peak untouched. Getting that wrong the first time is exactly
    # the kind of thing this file exists to catch.
    s = pp.epoch(build(), frame=500, stride=500)
    s.arrays["epochs"][:, 100, 5] = 10_000       # every channel — a movement artefact
    s.arrays["epochs"][0, 100, 7] = 10_000       # one channel — a bad channel, keep the epoch
    s = pp.reject_artifacts(s, threshold=5_000, consensus=0.6)
    ok("artifact rejection drops the all-channel epoch and keeps the one-channel one",
       s.arrays["epochs"].shape[2] == 39 and s.history[-1]["params"]["n_kept"] == 39)

    # --- standardize ------------------------------------------------------
    s = pp.standardize(pp.detrend(build()), "zscore")
    ok("zscore gives unit variance per channel",
       np.allclose(s.continuous.std(axis=1), 1.0, atol=1e-3))

    s = build()
    s.arrays["continuous"][0, 0] = 1e6           # one enormous sample
    robust = pp.standardize(s, "robust").continuous[0]
    zscore = pp.standardize(build(), "zscore").continuous[0]
    ok("robust scaling is barely moved by a single outlier, unlike zscore",
       abs(robust[1:].std() - zscore[1:].std()) / zscore[1:].std() > 0.1)

    s = pp.standardize(build(), "minmax")
    ok("minmax maps each channel onto [0, 1]",
       np.allclose(s.continuous.min(axis=1), 0, atol=1e-5)
       and np.allclose(s.continuous.max(axis=1), 1, atol=1e-5))

    # --- the whole chain, and it still saves ------------------------------
    import tempfile
    s = build()
    s.add_events("stim", np.array([2000, 5000, 8000], dtype=np.int64))
    s = pp.detrend(s)
    s = pp.bandpass(s, 1, 100, notch_hz=[60])
    s = pp.rereference(s, "average")
    s = pp.downsample(s, target_fs=250)
    s = pp.epoch_at(s, s.events["stim"], pre_ms=100, post_ms=200)
    s = pp.reject_artifacts(s, threshold=1e9)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chain.h5"
        s.save(path)
        from ephyslink import Session
        back = Session.load(path)
    ok(f"a full chain round-trips, carrying its {len(back.history)}-step history",
       len(back.history) == 6 and back.fs_hz == 250.0
       and back.arrays["epochs"].shape == s.arrays["epochs"].shape)

    print("\nAll preprocessing checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
