"""
Loading an Open Ephys binary recording into a `SessionOE`.

Reads the format directly — no `open-ephys-python-tools` dependency. The binary format is
simple enough that a reader is shorter than the wrapper around someone else's, and it means
one less package that can break.

What Open Ephys writes::

    <recording>/Record Node 1NN/experimentN/recordingM/
        structure.oebin                    JSON: streams, channels, sample rate, bit_volts
        continuous/<stream>/continuous.dat int16, interleaved (sample, channel)
        continuous/<stream>/sample_numbers.npy   hardware sample index per sample
        continuous/<stream>/timestamps.npy       seconds per sample
        events/<stream>/TTL/sample_numbers.npy   event times
        events/<stream>/TTL/states.npy           +n rising on line n, -n falling

**`continuous.dat` is not loaded whole.** An hour of 138 channels at 30 kHz is 28 GB. It is
memory-mapped, and only the channels and time window you ask for are materialised. A request
larger than `max_gb` raises rather than swapping the machine to death — pass `max_gb=None` if
you really mean it.

**Decimation is anti-aliased.** `decimate=n` runs `scipy.signal.decimate`, which low-pass
filters first. Plain striding would fold everything above the new Nyquist back into the band
you keep, and it would look like real signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .session import SessionOE

INT16_BYTES = 2


def find_recording(path: str | Path) -> Path:
    """The directory containing `structure.oebin`, given anything above it."""
    path = Path(path)
    if (path / "structure.oebin").is_file():
        return path
    candidates = sorted(path.glob("**/structure.oebin"))
    if not candidates:
        raise FileNotFoundError(f"no structure.oebin at or below {path}")
    if len(candidates) > 1:
        raise ValueError(
            f"{path} contains {len(candidates)} recordings:\n  "
            + "\n  ".join(str(c.parent.relative_to(path)) for c in candidates)
            + "\nPass the one you want."
        )
    return candidates[0].parent


def read_structure(recording_dir: Path) -> dict:
    return json.loads((Path(recording_dir) / "structure.oebin").read_text())


def describe_recording(path: str | Path) -> str:
    """What streams and channels a recording holds, without reading any samples."""
    recording = find_recording(path)
    structure = read_structure(recording)
    lines = [str(recording), ""]
    for i, stream in enumerate(structure.get("continuous", [])):
        dat = recording / "continuous" / stream["folder_name"] / "continuous.dat"
        n_channels = int(stream["num_channels"])
        rate = float(stream["sample_rate"])
        lines.append(f"[{i}] {stream['folder_name'].rstrip('/')}")

        if dat.is_file():
            size = dat.stat().st_size
            n_samples = size // (INT16_BYTES * n_channels)
            lines.append(
                f"     {n_channels} channels · {n_samples:,} samples · {rate:g} Hz · "
                f"{n_samples / rate / 60:.1f} min · {size / 1e9:.2f} GB on disk"
            )
        else:
            lines.append(f"     {n_channels} channels · {rate:g} Hz · continuous.dat MISSING")

        names = [c.get("channel_name", "?") for c in stream.get("channels", [])]
        if names:
            lines.append(f"     {', '.join(names[:8])}{' …' if len(names) > 8 else ''}")
    return "\n".join(lines)


def memmap_continuous(path: str | Path, stream: int | str = 0) -> tuple[np.memmap, dict]:
    """A read-only `(sample, channel)` view of `continuous.dat`, and its stream metadata.

    Nothing is read from disk until the array is indexed. Use this when the recording is too
    large to load, or when only a slice is needed.
    """
    recording = find_recording(path)
    structure = read_structure(recording)
    info = _select_stream(structure, stream)
    folder = recording / "continuous" / info["folder_name"]
    n_channels = int(info["num_channels"])
    data = np.memmap(folder / "continuous.dat", dtype=np.int16, mode="r")
    return data.reshape(-1, n_channels), info


def load_openephys(
    path: str | Path,
    stream: int | str = 0,
    channels: list[int] | None = None,
    start_s: float = 0.0,
    stop_s: float | None = None,
    decimate: int = 1,
    to_microvolts: bool = True,
    max_gb: float | None = 4.0,
    session_id: str | None = None,
    experiment: str = "",
    meta: dict | None = None,
) -> SessionOE:
    """Load a recording into memory as `(channel, sample)`.

    `channels` are indices into the stream's channel list, in the order you want them kept.
    `decimate` is an integer factor applied with an anti-aliasing filter.
    """
    recording = find_recording(path)
    structure = read_structure(recording)
    info = _select_stream(structure, stream)
    folder = recording / "continuous" / info["folder_name"]

    raw, _ = memmap_continuous(recording, stream)
    fs = float(info["sample_rate"])
    n_samples_total, n_channels_total = raw.shape

    first = int(round(start_s * fs))
    last = n_samples_total if stop_s is None else min(int(round(stop_s * fs)), n_samples_total)
    if not 0 <= first < last:
        raise ValueError(f"empty time window: start_s={start_s}, stop_s={stop_s}")

    keep = list(range(n_channels_total)) if channels is None else [int(c) for c in channels]
    bad = [c for c in keep if not 0 <= c < n_channels_total]
    if bad:
        raise IndexError(f"channels {bad} are outside 0..{n_channels_total - 1}")

    estimated_gb = (last - first) * len(keep) * 4 / 1e9      # float32 output
    if max_gb is not None and estimated_gb > max_gb:
        raise MemoryError(
            f"that selection is {estimated_gb:.1f} GB in memory ({len(keep)} channels × "
            f"{last - first:,} samples). Narrow it with `channels=`, `start_s`/`stop_s` or "
            f"`decimate=`, use `memmap_continuous()` instead, or pass max_gb=None."
        )

    block = np.asarray(raw[first:last, keep], dtype=np.float32)     # (sample, channel)
    if to_microvolts:
        bit_volts = np.array(
            [float(info["channels"][c].get("bit_volts", 1.0)) for c in keep], dtype=np.float32
        )
        block *= bit_volts

    continuous = np.ascontiguousarray(block.T)                      # (channel, sample)
    del block

    if decimate > 1:
        from scipy.signal import decimate as _decimate
        continuous = np.ascontiguousarray(
            _decimate(continuous, int(decimate), axis=1, ftype="fir", zero_phase=True),
            dtype=np.float32,
        )
        fs = fs / int(decimate)

    session = SessionOE(
        id=session_id or recording.parents[2].name,
        fs_hz=fs,
        experiment=experiment,
        source=str(recording),
        meta={
            "channel_names": [info["channels"][c].get("channel_name", f"CH{c}") for c in keep],
            "source_channels": keep,
            "stream": info["folder_name"].rstrip("/"),
            "original_fs_hz": float(info["sample_rate"]),
            "start_s": start_s,
            **(meta or {}),
        },
    )
    session.add_array("continuous", continuous, dims=["channel", "sample"],
                      units="µV" if to_microvolts else "int16 units")

    sample_numbers = folder / "sample_numbers.npy"
    if sample_numbers.is_file():
        numbers = np.load(sample_numbers, mmap_mode="r")[first:last:decimate]
        session.add_array("sample_numbers", np.asarray(numbers, dtype=np.int64),
                          dims=["sample"],
                          description="hardware sample index, for aligning to events")

    for name, samples in _read_ttl_events(recording, first, last, decimate).items():
        session.add_events(name, samples)

    session.log("load_openephys", stream=info["folder_name"], channels=keep,
                start_s=start_s, stop_s=stop_s, decimate=decimate, fs_hz=fs)
    session.validate()
    return session.mark_source()


def _select_stream(structure: dict, stream: int | str) -> dict:
    streams = structure.get("continuous", [])
    if not streams:
        raise ValueError("structure.oebin lists no continuous streams")
    if isinstance(stream, int):
        return streams[stream]
    matches = [s for s in streams if stream in s["folder_name"]]
    if len(matches) != 1:
        raise ValueError(
            f"{'no' if not matches else 'more than one'} stream matches {stream!r}. "
            f"Available: {[s['folder_name'].rstrip('/') for s in streams]}"
        )
    return matches[0]


def _read_ttl_events(recording: Path, first: int, last: int, decimate: int) -> dict[str, np.ndarray]:
    """TTL edges per line, as sample indices relative to the loaded window.

    Open Ephys stores `states` as +n for a rising edge on line n and -n for a falling edge, so
    the two are separated here rather than left for the caller to rediscover.
    """
    events: dict[str, np.ndarray] = {}
    for ttl_dir in sorted((recording / "events").glob("**/TTL")):
        sample_file, state_file = ttl_dir / "sample_numbers.npy", ttl_dir / "states.npy"
        if not (sample_file.is_file() and state_file.is_file()):
            continue
        samples = np.load(sample_file).astype(np.int64)
        states = np.load(state_file).astype(np.int64)

        # sample_numbers are hardware indices; rebase onto the continuous stream's first sample
        continuous_start = _first_sample_number(recording)
        if continuous_start is not None:
            samples = samples - continuous_start

        inside = (samples >= first) & (samples < last)
        samples, states = samples[inside] - first, states[inside]
        if decimate > 1:
            samples = samples // decimate

        for line in np.unique(np.abs(states)):
            if line == 0:
                continue
            events[f"ttl{line}_rising"] = samples[states == line]
            events[f"ttl{line}_falling"] = samples[states == -line]
    return {k: v for k, v in events.items() if v.size}


def _first_sample_number(recording: Path) -> int | None:
    for file in sorted((recording / "continuous").glob("*/sample_numbers.npy")):
        return int(np.load(file, mmap_mode="r")[0])
    return None
