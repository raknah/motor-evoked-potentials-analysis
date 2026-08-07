"""
Reading and writing the ephyslink HDF5 file. The format contract is `../FORMAT.md`.

Only this module knows what the file looks like. `session.py` knows what a session *is*, the
loaders know what the instruments produce, and neither of them opens an HDF5 file.

The one rule worth repeating here, because getting it wrong is silent: h5py works in C order
and HDF5.jl works in Fortran order, so an array written from Python with shape (a, b, c) is
read by Julia as (c, b, a). That is deterministic and needs no inspection of the data. Python
therefore stores arrays as-is; the Julia side does the reversal. Every array also carries a
`dims` attribute naming its axes, so no reader ever has to guess from the shape — which is
exactly what the previous implementation did, and why it silently transposed real data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FORMAT_NAME = "ephyslink"
FORMAT_VERSION = 1

ROOT_ATTRS = ("format", "format_version", "kind", "id", "experiment", "fs_hz",
              "source", "created", "meta_json", "history_json")


class _JsonEncoder(json.JSONEncoder):
    """Metadata routinely contains numpy scalars and arrays that json refuses by default."""

    def default(self, o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (Path, datetime)):
            return str(o)
        return super().default(o)


def _finite_only(obj: Any) -> Any:
    """Replace NaN and +/-Inf with null, recursively, before serialising.

    `json.dumps` happily emits bare `NaN` and `Infinity`. **Those are not valid JSON** — they
    are a Python extension, and a strict parser (Julia's JSON3 among them) rejects the whole
    document. The old code would then have fallen back to an empty dict and the metadata would
    have vanished on the Julia side *silently*.

    NaN reaching this point means "not computed" — an empty group's median, a test that could
    not run — and JSON `null` carries exactly that meaning in both languages. The conversion is
    therefore lossless in intent even though it is lossy in type.
    """
    if isinstance(obj, dict):
        return {k: _finite_only(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_only(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def _dumps(obj: Any) -> str:
    # allow_nan=False so that anything _finite_only missed raises here rather than producing a
    # file that Python reads and Julia silently cannot
    return json.dumps(_finite_only(obj), cls=_JsonEncoder, allow_nan=False)


def _loads(text: Any, fallback):
    if text in (None, ""):
        return fallback
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _as_str(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


# =============================================================================
# writing
# =============================================================================

def write_session(path: str | Path, session, compression: str | None = "gzip") -> Path:
    """Write a Session to `path`, overwriting. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT_NAME
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["kind"] = session.kind
        f.attrs["id"] = str(session.id)
        f.attrs["experiment"] = str(session.experiment or "")
        f.attrs["fs_hz"] = float(session.fs_hz)
        f.attrs["source"] = str(session.source or "")
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.attrs["meta_json"] = _dumps(session.meta)
        f.attrs["history_json"] = _dumps(session.history)

        if session.arrays:
            group = f.create_group("arrays")
            for name, array in session.arrays.items():
                values = np.ascontiguousarray(array)
                # A bool array becomes an HDF5 *enum*, which HDF5.jl does not read back as
                # Bool. Only plain integer and float types are portable, so bools are stored
                # as int8 with the original dtype recorded for Python to restore.
                original_dtype = None
                if values.dtype == bool:
                    values, original_dtype = values.astype(np.int8), "bool"
                if values.dtype.kind in "OUS":
                    raise TypeError(
                        f"array '{name}' has dtype {values.dtype}; only numeric arrays can be "
                        "stored. Put strings in a table column instead."
                    )
                # compression on a scalar or empty dataset is an error in h5py
                use = compression if values.ndim > 0 and values.size > 0 else None
                dataset = group.create_dataset(name, data=values, compression=use)
                dataset.attrs["dims"] = _axis_names(session, name, values.ndim)
                if original_dtype:
                    dataset.attrs["original_dtype"] = original_dtype
                for key in ("units", "description"):
                    value = session.array_meta.get(name, {}).get(key)
                    if value is not None:
                        dataset.attrs[key] = str(value)

        if session.tables:
            group = f.create_group("tables")
            for name, frame in session.tables.items():
                _write_table(group, name, frame, compression)

        if session.events:
            group = f.create_group("events")
            for name, samples in session.events.items():
                values = np.asarray(samples, dtype=np.int64).ravel()
                group.create_dataset(
                    name, data=values, compression=compression if values.size else None
                )

    return path


def _axis_names(session, array_name: str, ndim: int) -> list[str]:
    """Axis labels for an array, falling back to positional names if none were declared."""
    declared = session.dims.get(array_name)
    if declared and len(declared) == ndim:
        return [str(d) for d in declared]
    return [f"axis{i}" for i in range(ndim)]


def _write_table(parent: h5py.Group, name: str, frame, compression: str | None) -> None:
    """One dataset per column. See FORMAT.md §2 for why, rather than a compound dataset."""
    group = parent.create_group(name)
    group.attrs["columns"] = [str(c) for c in frame.columns]
    group.attrs["n_rows"] = int(len(frame))

    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype == object or values.dtype.kind in "USO":
            # variable-length UTF-8 is the one string encoding both h5py and HDF5.jl agree on
            values = np.array(["" if v is None else str(v) for v in values],
                              dtype=h5py.string_dtype(encoding="utf-8"))
        elif values.dtype == bool:
            values = values.astype(np.int8)
        group.create_dataset(
            str(column), data=values, compression=compression if len(values) else None
        )


# =============================================================================
# reading
# =============================================================================

def read_session_dict(path: str | Path) -> dict:
    """Read a file into plain dicts. `session.py` turns this into the right Session class.

    Kept separate so the format can be inspected without constructing anything, which is what
    you want when a file will not load.
    """
    import pandas as pd

    path = Path(path)
    with h5py.File(path, "r") as f:
        found = _as_str(f.attrs.get("format", ""))
        if found != FORMAT_NAME:
            raise ValueError(
                f"{path} is not an ephyslink file (root attribute format={found!r}). "
                "Files written by the old openephysextract code are not readable — "
                "re-extract from source."
            )
        version = int(f.attrs.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise ValueError(
                f"{path} is format version {version}; this code understands up to "
                f"{FORMAT_VERSION}. Update ephyslink."
            )

        result = {
            "kind": _as_str(f.attrs["kind"]),
            "id": _as_str(f.attrs["id"]),
            "experiment": _as_str(f.attrs.get("experiment", "")),
            "fs_hz": float(f.attrs["fs_hz"]),
            "source": _as_str(f.attrs.get("source", "")),
            "created": _as_str(f.attrs.get("created", "")),
            "meta": _loads(f.attrs.get("meta_json"), {}),
            "history": _loads(f.attrs.get("history_json"), []),
            "arrays": {},
            "dims": {},
            "array_meta": {},
            "tables": {},
            "events": {},
        }

        for name, dataset in f.get("arrays", {}).items():
            values = dataset[()]
            if _as_str(dataset.attrs.get("original_dtype", "")) == "bool":
                values = values.astype(bool)
            result["arrays"][name] = values
            result["dims"][name] = [_as_str(d) for d in dataset.attrs.get("dims", [])]
            extra = {k: _as_str(dataset.attrs[k]) for k in ("units", "description")
                     if k in dataset.attrs}
            if extra:
                result["array_meta"][name] = extra

        for name, group in f.get("tables", {}).items():
            columns = [_as_str(c) for c in group.attrs["columns"]]
            data = {}
            for column in columns:
                values = group[column][()]
                if values.dtype.kind in "OS":
                    values = np.array([_as_str(v) for v in values])
                data[column] = values
            result["tables"][name] = pd.DataFrame(data, columns=columns)

        for name, dataset in f.get("events", {}).items():
            result["events"][name] = dataset[()].astype(np.int64)

    return result


def describe_file(path: str | Path) -> str:
    """What is actually in a file, without constructing a Session. For when loading fails."""
    path = Path(path)
    lines = [f"{path}", ""]
    with h5py.File(path, "r") as f:
        lines.append("root attributes")
        for key in sorted(f.attrs):
            value = _as_str(f.attrs[key])
            lines.append(f"  {key:<16} {value[:100]}{'…' if len(value) > 100 else ''}")
        for section in ("arrays", "tables", "events"):
            if section not in f:
                continue
            lines.append(f"\n{section}")
            for name, item in f[section].items():
                if isinstance(item, h5py.Dataset):
                    dims = [_as_str(d) for d in item.attrs.get("dims", [])]
                    label = f"  [{', '.join(dims)}]" if dims else ""
                    lines.append(f"  {name:<22} {str(item.shape):<20} {item.dtype}{label}")
                else:
                    columns = [_as_str(c) for c in item.attrs.get("columns", [])]
                    lines.append(f"  {name:<22} {item.attrs.get('n_rows', '?')} rows × "
                                 f"{len(columns)} columns: {', '.join(columns)}")
    return "\n".join(lines)
