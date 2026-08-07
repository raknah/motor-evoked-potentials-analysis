"""
Turning this experiment's analog TTL channels into trial onsets and stimulus transitions.

Every later analysis is anchored to a time t = 0. Getting those anchors wrong is the most
expensive kind of error available, because nothing downstream can detect it. Three traps in
this dataset, all handled explicitly:

1. **Isolated events before the protocol starts.** Single transitions at 283-348 s, seconds
   apart, before the real protocol at ~370 s — almost certainly a calibration sequence. They
   are removed by requiring a block to contain *exactly* the expected number of transitions.

2. **Static events are ON/OFF pairs, not onsets.** Each static trial logs an ON, then an OFF
   4.001 s later. Treating all of them as onsets folds stimulus-*offset* responses into the
   stimulus-*onset* histogram. An ON is therefore defined as an event whose partner follows
   4.000-4.002 s later. The tolerance must stay tight: when two static trials abut, the gap
   between *trials* is 4.03 s, so a loose tolerance re-introduces the error.

3. **Contact bounce.** Each static OFF is followed ~6 ms later by a duplicate edge. Those
   bounces have no partner 4 s later so the ON rule already rejects them, but they are removed
   first anyway so that any *count* of static events is right and the rule does not quietly
   depend on that coincidence.

A fourth thing is checked rather than assumed: conditions interleave, so one condition's
pre-stimulus baseline could land inside another condition's block. `check_baseline_clear`
asserts it does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from EXELU_config import FS, Params

STATIC = "static"

PERIODIC_SPEC: dict[str, dict] = {
    "flicker_40Hz":      {"transitions_per_block": 160, "period_s": 1 / 40},
    "flicker_20Hz":      {"transitions_per_block": 80,  "period_s": 1 / 20},
    "phase_reversing_1": {"transitions_per_block": 160, "period_s": 1 / 40},
    "phase_reversing_2": {"transitions_per_block": 160, "period_s": 1 / 40},
}
"""Expected structure of each rhythmic condition. A 4 s block at 40 Hz contains 160 logged
transitions. `phase_reversing_2` is normally excluded — see `Params.excluded_conditions`."""

BLOCK_GAP_MS = 200.0
"""Any inter-event gap longer than this starts a new block. Comfortably longer than the 50 ms
period of the slowest rhythmic condition and far shorter than the inter-block interval."""

STATIC_DURATION_S = 4.0
STATIC_TOLERANCE_S = 0.01
STATIC_DEBOUNCE_MS = 50.0


@dataclass
class Trigger:
    """One condition's anchor times, all in sample indices."""

    name: str
    transitions: np.ndarray     # every within-block stimulus transition
    block_onsets: np.ndarray    # first transition of each block
    block_offsets: np.ndarray   # last transition of each block
    period_s: float | None      # None for static, which has no within-block rhythm
    transitions_per_block: int
    dropped_block_sizes: np.ndarray

    @property
    def n_blocks(self) -> int:
        return self.block_onsets.size

    @property
    def n_transitions(self) -> int:
        return self.transitions.size


def read_analog_events(session_path) -> dict[str, np.ndarray]:
    """The stimulus TTLs from `analog_events.mat`, one sorted array of samples per channel.

    Separate from `build_triggers` because reading the file and interpreting the edges are
    different jobs: this is what the acquisition wrote, `build_triggers` is what the protocol
    means. Kept here rather than in a loader module so that everything to do with this
    experiment's stimulus timing is in one file.
    """
    import scipy.io as sio

    matlab = sio.loadmat(Path(session_path) / "analog_events.mat")
    names = [str(entry[0]) for entry in matlab["analog_ch_names"][0]]
    return {
        name: np.sort(np.asarray(matlab["analogEvents"][0][i]).ravel().astype(np.int64))
        for i, name in enumerate(names)
    }


def debounce(samples: np.ndarray, min_gap_ms: float) -> np.ndarray:
    """Drop events closer than `min_gap_ms` to the previously kept event.

    These are gaps between events on a stimulus TTL line — contact bounce — not inter-spike
    intervals. Nothing physiological is being discarded.
    """
    times = np.sort(np.asarray(samples).ravel().astype(np.int64))
    if times.size == 0:
        return times
    keep = [0]
    for i in range(1, times.size):
        if (times[i] - times[keep[-1]]) / FS * 1000.0 > min_gap_ms:
            keep.append(i)
    return times[keep]


def split_into_blocks(
    samples: np.ndarray, gap_ms: float, expected_n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a periodic transition train into blocks, keeping only complete ones.

    Returns `(times, start_indices, end_indices, dropped_lengths)`. A block whose length
    differs from `expected_n` is dropped and its length reported, which is how the calibration
    singletons are removed without hard-coding their timestamps.
    """
    times = np.sort(np.asarray(samples).ravel().astype(np.int64))
    gaps_ms = np.diff(times) / FS * 1000.0
    starts = np.r_[0, np.where(gaps_ms > gap_ms)[0] + 1]
    ends = np.r_[starts[1:], times.size]
    complete = (ends - starts) == expected_n
    return times, starts[complete], ends[complete], (ends - starts)[~complete]


def build_triggers(events: dict[str, np.ndarray], params: Params) -> dict[str, Trigger]:
    """All conditions' anchors, with `params.excluded_conditions` removed."""
    triggers: dict[str, Trigger] = {}

    for name, spec in PERIODIC_SPEC.items():
        if name in params.excluded_conditions or name not in events:
            continue
        times, starts, ends, dropped = split_into_blocks(
            events[name], BLOCK_GAP_MS, spec["transitions_per_block"]
        )
        keep = (
            np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
            if starts.size else np.empty(0, dtype=np.int64)
        )
        triggers[name] = Trigger(
            name=name,
            transitions=times[keep],
            block_onsets=times[starts],
            block_offsets=times[ends - 1],
            period_s=spec["period_s"],
            transitions_per_block=spec["transitions_per_block"],
            dropped_block_sizes=dropped,
        )

    if STATIC not in params.excluded_conditions and STATIC in events:
        clean = debounce(events[STATIC], STATIC_DEBOUNCE_MS)
        gaps_s = np.diff(clean) / FS
        is_onset = np.where(np.abs(gaps_s - STATIC_DURATION_S) < STATIC_TOLERANCE_S)[0]
        triggers[STATIC] = Trigger(
            name=STATIC,
            transitions=clean[is_onset],
            block_onsets=clean[is_onset],
            block_offsets=clean[is_onset + 1],
            period_s=None,
            transitions_per_block=1,
            dropped_block_sizes=np.array([]),
        )

    return triggers


def flicker_conditions(triggers: dict[str, Trigger]) -> list[str]:
    """Conditions with a within-block rhythm, i.e. everything except static."""
    return [name for name, t in triggers.items() if t.period_s is not None]


def onsets_by_condition(triggers: dict[str, Trigger]) -> dict[str, np.ndarray]:
    """`{condition: block onsets}` — the input format the shared analysis modules want."""
    return {name: t.block_onsets for name, t in triggers.items()}


def transitions_by_condition(triggers: dict[str, Trigger]) -> dict[str, np.ndarray]:
    """`{condition: transitions}` for the rhythmic conditions only."""
    return {name: t.transitions for name, t in triggers.items() if t.period_s is not None}


def periods_by_condition(triggers: dict[str, Trigger]) -> dict[str, float]:
    """`{condition: stimulus period in seconds}` for the rhythmic conditions only."""
    return {name: t.period_s for name, t in triggers.items() if t.period_s is not None}


def summarise(triggers: dict[str, Trigger]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "condition": t.name,
            "blocks": t.n_blocks,
            "transitions": t.n_transitions,
            "block_duration_s": round(float(np.median((t.block_offsets - t.block_onsets) / FS)), 3),
            "incomplete_blocks_dropped": len(t.dropped_block_sizes),
        }
        for t in triggers.values()
    ])


def check_baseline_clear(triggers: dict[str, Trigger], pre_s: float) -> float:
    """Assert that no block onset has another condition's stimulus inside its baseline.

    Conditions interleave, so this is a real risk rather than a formality. Returns the
    smallest stimulus-free gap found, in seconds; raises if it is shorter than `pre_s`.
    """
    all_stimulus_times = np.sort(np.concatenate(
        [t.transitions for t in triggers.values()] + [t.block_offsets for t in triggers.values()]
    ))
    worst = np.inf
    for trigger in triggers.values():
        for onset in trigger.block_onsets:
            before = all_stimulus_times[all_stimulus_times < onset]
            if before.size:
                worst = min(worst, float((onset - before[-1]) / FS))
    if worst < pre_s:
        raise AssertionError(
            f"baseline window of {pre_s:.2f} s is not stimulus-free: the tightest gap before a "
            f"block onset is {worst:.3f} s"
        )
    return worst
