"""
Loading a Kilosort 4 output directory into a `SessionKS`.

What Kilosort writes, and what each file is::

    spike_times.npy        sample index of every spike                     (N,)
    spike_clusters.npy     which cluster each spike belongs to             (N,)
    spike_positions.npy    estimated (x, depth) of every spike             (N, 2)
    spike_templates.npy    which template each spike matched               (N,)
    amplitudes.npy         fitted amplitude of every spike                 (N,)
    templates.npy          the templates       (n_templates, n_samples, n_channels)
    templates_ind.npy      channel index of each template column
    whitening_mat_inv.npy  undoes Kilosort's spatial whitening
    channel_positions.npy  (x, depth) of every contact                     (n_ch, 2)
    channel_map.npy        recording-channel index of each sorted channel
    cluster_group.tsv      the good / mua label per cluster
    cluster_ContamPct.tsv  the contamination estimate per cluster
    params.py              sample rate, dtype, path to the raw binary

Times on disk are **sample indices**. They stay sample indices in the session; divide by
`fs_hz` once, at the point of use.

The sampling rate is read from `params.py` rather than assumed, and the caller can override
it — if the two disagree that is worth knowing about, so the mismatch is reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .session import SessionKS

ARRAY_FILES = {
    "spike_times":       ["spike"],
    "spike_clusters":    ["spike"],
    "spike_positions":   ["spike", "axis"],
    "spike_templates":   ["spike"],
    "amplitudes":        ["spike"],
    "templates":         ["template", "sample", "channel"],
    "templates_ind":     ["template", "channel"],
    "whitening_mat_inv": ["channel", "channel_out"],
    "whitening_mat":     ["channel", "channel_out"],
    "channel_positions": ["channel", "axis"],
    "channel_map":       ["channel"],
}
"""Which files to read and what their axes mean. A file absent from the directory is skipped,
so this works on a partial export; a file present but not listed here is ignored, which keeps
`amplitudes`-sized arrays out of the session unless they are wanted."""

REQUIRED = ("spike_times", "spike_clusters")


def read_params(kilosort_dir: Path) -> dict:
    """Parse Kilosort's `params.py` without executing it as a module."""
    values: dict[str, object] = {}
    path = Path(kilosort_dir) / "params.py"
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, raw = line.partition("=")
        raw = raw.strip()
        try:
            values[key.strip()] = eval(raw, {"__builtins__": {}}, {})   # literals only
        except Exception:
            values[key.strip()] = raw.strip("'\"")
    return values


def load_kilosort(
    path: str | Path,
    session_id: str | None = None,
    experiment: str = "",
    fs_hz: float | None = None,
    arrays: dict[str, list[str]] | None = None,
    meta: dict | None = None,
) -> SessionKS:
    """Load a Kilosort output directory.

    `path` may be the directory containing `spike_times.npy`, or its parent — a recording
    folder with a `kilosort4/` subdirectory is the common case and is found automatically.
    """
    path = Path(path)
    kilosort_dir = _find_kilosort_dir(path)
    session_id = session_id or (path.name if kilosort_dir != path else path.parent.name)

    params = read_params(kilosort_dir)
    file_fs = float(params.get("sample_rate", 0) or 0)
    if fs_hz is None:
        if not file_fs:
            raise ValueError(
                f"no sample_rate in {kilosort_dir / 'params.py'} and no fs_hz given — "
                "the sampling rate cannot be guessed"
            )
        fs_hz = file_fs
    elif file_fs and abs(file_fs - fs_hz) > 1e-6:
        raise ValueError(
            f"fs_hz={fs_hz} was given but params.py says {file_fs}. One of them is wrong, "
            "and every time in this session depends on which."
        )

    session = SessionKS(
        id=session_id, fs_hz=fs_hz, experiment=experiment, source=str(kilosort_dir),
        meta={"kilosort_params": params, **(meta or {})},
    )

    wanted = arrays or ARRAY_FILES
    for name, dims in wanted.items():
        file = kilosort_dir / f"{name}.npy"
        if not file.is_file():
            if name in REQUIRED:
                raise FileNotFoundError(f"{file} is missing; this is not a Kilosort output directory")
            continue
        values = np.load(file)
        if name == "spike_times":
            values = values.astype(np.int64)
        # a (N, 1) column that should be (N,) — Kilosort has written both over the years
        if len(dims) == 1 and values.ndim == 2 and values.shape[1] == 1:
            values = values.ravel()
        session.add_array(name, values, dims=dims[: values.ndim])

    clusters = _read_cluster_tables(kilosort_dir)
    if clusters is not None:
        session.add_table("clusters", clusters)

    session.log("load_kilosort", directory=str(kilosort_dir), fs_hz=fs_hz)
    session.validate()
    # everything added after this point is analysis output, not raw data — which is what
    # lets export_results() strip the raw data without being told what it was
    return session.mark_source()


def _find_kilosort_dir(path: Path) -> Path:
    """Accept either the Kilosort directory itself or a recording folder containing one."""
    if (path / "spike_times.npy").is_file():
        return path
    candidates = sorted(p for p in path.glob("*") if (p / "spike_times.npy").is_file())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"no spike_times.npy in {path} or in any immediate subdirectory"
        )
    raise ValueError(
        f"{path} contains more than one Kilosort output: "
        f"{[c.name for c in candidates]}. Pass the one you want."
    )


def _read_cluster_tables(kilosort_dir: Path) -> pd.DataFrame | None:
    """Merge every `cluster_*.tsv` on `cluster_id` into one table."""
    files = sorted(kilosort_dir.glob("cluster_*.tsv"))
    if not files:
        return None
    merged: pd.DataFrame | None = None
    for file in files:
        frame = pd.read_csv(file, sep="\t")
        if "cluster_id" not in frame.columns:
            continue
        # cluster_group.tsv and cluster_KSLabel.tsv hold the same column under two names
        merged = frame if merged is None else merged.merge(
            frame, on="cluster_id", how="outer", suffixes=("", f"_{file.stem}")
        )
    return merged.sort_values("cluster_id").reset_index(drop=True) if merged is not None else None
