#!/usr/bin/env python3
"""
Extract Open Ephys recordings, preprocess them, and save one ephyslink file per session.

    python scripts/simple_extract.py --dry-run     # list what would be processed
    python scripts/simple_extract.py

Each output file is a complete session: the preprocessed trace, the epochs, the events, the
metadata and the list of steps that produced it. Julia reads it with the same axis order —
see modules/FORMAT.md.

Edit the settings block below, or override any of it on the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "modules"))
sys.path.insert(0, str(_ROOT / "scripts"))

import preprocessing as pp  # noqa: E402
from ephyslink import describe_recording, find_recording, load_openephys  # noqa: E402

# ================================
# SETTINGS
# ================================
SOURCE_FOLDER = Path("/Volumes/STORAGE 1.0/UNIC Research/5xFAD Resting State")
OUTPUT_FOLDER = Path("~/Documents/Research/UNIC Research/5xFAD Resting State/sessions").expanduser()
EXPERIMENT = "5xFAD Resting State"

CHANNELS: list[int] | None = None      # None keeps every channel
TARGET_FS = 250.0                      # Hz, after anti-aliased decimation
BANDPASS = (0.5, 100.0)                # Hz
NOTCH_HZ = [50.0, 100.0]               # mains and its first harmonic
EPOCH_S = 2.0                          # sliding-window epoch length
EPOCH_STRIDE_S = 2.0                   # no overlap
ARTIFACT_UV = 1_000.0                  # peak-to-peak rejection threshold


def process(recording_dir: Path, output_dir: Path) -> Path:
    session = load_openephys(
        recording_dir,
        channels=CHANNELS,
        experiment=EXPERIMENT,
        max_gb=8.0,
    )
    session = pp.detrend(session)
    session = pp.bandpass(session, *BANDPASS, notch_hz=NOTCH_HZ)
    session = pp.rereference(session, "average")
    session = pp.downsample(session, target_fs=TARGET_FS)
    session = pp.epoch(
        session,
        frame=int(EPOCH_S * session.fs_hz),
        stride=int(EPOCH_STRIDE_S * session.fs_hz),
    )
    session = pp.reject_artifacts(session, threshold=ARTIFACT_UV)
    return session.save(output_dir / f"{session.id}.h5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_FOLDER)
    parser.add_argument("--output", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    if not arguments.source.is_dir():
        print(f"source folder not found: {arguments.source}")
        return 1

    recordings = sorted(p for p in arguments.source.iterdir() if p.is_dir())
    if arguments.limit:
        recordings = recordings[: arguments.limit]

    print(f"source : {arguments.source}")
    print(f"output : {arguments.output}")
    print(f"found  : {len(recordings)} recording folders\n")

    if arguments.dry_run:
        for recording in recordings:
            try:
                print(describe_recording(find_recording(recording)), "\n")
            except (FileNotFoundError, ValueError) as error:
                print(f"{recording.name}: {error}\n")
        return 0

    failures = []
    for recording in recordings:
        try:
            path = process(recording, arguments.output)
            print(f"  ✓ {recording.name} → {path.name}")
        except Exception as error:                      # noqa: BLE001 — one bad session
            failures.append((recording.name, error))    # must not stop the batch
            print(f"  ✗ {recording.name}: {type(error).__name__}: {error}")

    print(f"\n{len(recordings) - len(failures)}/{len(recordings)} sessions written")
    for name, error in failures:
        print(f"  failed: {name} — {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
