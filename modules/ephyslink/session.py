"""
The Session object: one recording, and everything the pipeline attaches to it.

The point of this class is that a pipeline stage should not produce a new file. It should
take a session, add its output to it, and hand the same object on. Whatever is in `arrays`,
`tables` and `events` at the end is what `save()` writes, and `load()` gives back — in Python
or in Julia, with the same axis order and the same names.

    session = load_kilosort("…/2026-04-14_11-46-55")
    session.add_table("units", unit_table)
    session.add_array("psth", psth_matrix, dims=["unit", "bin"])
    session.log("responder test", alpha=0.05)
    session.save("…/analysis/2026-04-14.h5")

Two typed views over one file format:

* `SessionOE` — continuous traces. LFP, MEP, EEG. The data is `(channel, sample)`, and after
  epoching `(channel, sample, epoch)`.
* `SessionKS` — spike-sorted output. Spike times, clusters, templates, and a cluster table.

They differ only in convenience accessors and in what `validate()` insists on; the file they
write is the same format, distinguished by the `kind` attribute. Adding a third kind means
adding a subclass, not a new format.

**Axis order is never inferred.** Every array is stored with a `dims` attribute naming its
axes. `session.axis("continuous", "channel")` gives the position of an axis by name, so
analysis code does not have to hard-code 0 or 1 and does not break when an array is
transposed upstream.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import format as fmt


class Session:
    """Base class. Use `SessionOE` or `SessionKS`; this holds what they share."""

    kind = "base"

    def __init__(
        self,
        id: str,
        fs_hz: float,
        experiment: str = "",
        source: str | Path = "",
        arrays: dict[str, np.ndarray] | None = None,
        dims: dict[str, list[str]] | None = None,
        tables: dict[str, pd.DataFrame] | None = None,
        events: dict[str, np.ndarray] | None = None,
        meta: dict[str, Any] | None = None,
        history: list[dict] | None = None,
        array_meta: dict[str, dict] | None = None,
    ):
        self.id = str(id)
        self.fs_hz = float(fs_hz)
        self.experiment = str(experiment or "")
        self.source = str(source or "")
        self.arrays: dict[str, np.ndarray] = dict(arrays or {})
        self.dims: dict[str, list[str]] = dict(dims or {})
        self.tables: dict[str, pd.DataFrame] = dict(tables or {})
        self.events: dict[str, np.ndarray] = dict(events or {})
        self.meta: dict[str, Any] = dict(meta or {})
        self.history: list[dict] = list(history or [])
        self.array_meta: dict[str, dict] = dict(array_meta or {})

    # ---- results ----------------------------------------------------------
    #
    # `results` is a free-form dict for the session-level numbers an analysis produces —
    # counts, medians, test outcomes. It is stored inside `meta`, so it round-trips through
    # the file and into Julia with no extra machinery and no format change.
    #
    # Derived *tables* do not go here. They go in `tables` like any other table; what makes
    # them results is that they were not there when the session was loaded. See
    # `mark_source()` and `export_results()`.

    @property
    def results(self) -> dict[str, Any]:
        """Session-level numbers, accumulated as the analysis runs.

            session.results["n_clean_units"] = 45
            session.results["median_ppc_ctx_20hz"] = 0.0277
        """
        return self.meta.setdefault("results", {})

    def mark_source(self) -> "Session":
        """Record what is in the session *now* as raw input rather than analysis output.

        Called by the loaders once they have finished. Everything added afterwards is, by
        definition, something the analysis produced — which is what lets `export_results`
        strip the raw data without being told what the raw data was.
        """
        self.meta["source_arrays"] = sorted(self.arrays)
        self.meta["source_tables"] = sorted(self.tables)
        self.meta["source_events"] = sorted(self.events)
        return self

    def derived_tables(self) -> list[str]:
        """Tables added since `mark_source()`. Everything if the source was never marked."""
        source = set(self.meta.get("source_tables", []))
        return [name for name in self.tables if name not in source]

    def derived_arrays(self) -> list[str]:
        source = set(self.meta.get("source_arrays", []))
        return [name for name in self.arrays if name not in source]

    def export_results(
        self,
        path: str | Path,
        include_arrays: list[str] | None = None,
        drop_meta: tuple[str, ...] = ("kilosort_params", "source_arrays",
                                      "source_tables", "source_events"),
    ) -> Path:
        """Write a compact results-only session: no spikes, no traces, no templates.

        Keeps every table the analysis added, at full per-row resolution, plus `results`,
        the metadata and the full history. A comparison across sessions then needs only these
        files — for the EXELU probe session that is 859 KB against 44.6 MB with the raw data in
        it, a factor of 52.

        Arrays are excluded by default: they are the bulky intermediates (PSTH matrices,
        cycle histograms) and are re-derivable from the raw session. Name the ones you want
        in `include_arrays`.

        The result is an ordinary ephyslink file with `kind="RESULTS"`, so it loads with the
        same `Session.load` and the same Julia reader.
        """
        derived = self.derived_tables()
        if not derived and not self.results:
            raise ValueError(
                f"session '{self.id}' has no derived tables and an empty results dict — "
                "there is nothing to export. Attach analysis output with add_table() and "
                "session.results[...] first."
            )

        meta = {k: v for k, v in self.meta.items() if k not in drop_meta}
        meta["exported_from"] = self.kind
        meta["source_tables_dropped"] = sorted(set(self.tables) - set(derived))

        export = SessionResults(
            id=self.id, fs_hz=self.fs_hz, experiment=self.experiment, source=self.source,
            tables={name: self.tables[name] for name in derived},
            meta=meta, history=list(self.history),
        )
        for name in include_arrays or []:
            export.add_array(name, self.arrays[name], dims=self.dims[name])

        export.log("export_results", tables=derived, arrays=list(include_arrays or []),
                   n_results=len(self.results))
        return fmt.write_session(path, export)

    # ---- attaching things -------------------------------------------------

    def add_array(
        self, name: str, values, dims: Iterable[str],
        units: str | None = None, description: str | None = None,
    ) -> "Session":
        """Attach an array. `dims` is required — an unlabelled axis is how sessions get
        silently transposed, so the format does not allow one."""
        values = np.asarray(values)
        dims = [str(d) for d in dims]
        if len(dims) != values.ndim:
            raise ValueError(
                f"array '{name}' has {values.ndim} axes but {len(dims)} names were given: {dims}"
            )
        if values.dtype.kind not in "fiub":
            # Caught here rather than at save time, so the error points at the line that made
            # the mistake. Strings and object arrays have no portable HDF5 representation —
            # h5py and HDF5.jl disagree — so they belong in a table column.
            raise TypeError(
                f"array '{name}' has dtype {values.dtype}; arrays must be numeric or boolean. "
                "Strings and mixed types belong in a table: session.add_table(...)."
            )
        self.arrays[name] = values
        self.dims[name] = dims
        extra = {k: v for k, v in (("units", units), ("description", description)) if v}
        if extra:
            self.array_meta[name] = extra
        return self

    def add_table(self, name: str, frame: pd.DataFrame) -> "Session":
        self.tables[name] = frame.reset_index(drop=True)
        return self

    def add_events(self, name: str, samples) -> "Session":
        self.events[name] = np.asarray(samples, dtype=np.int64).ravel()
        return self

    def log(self, step: str, **params) -> "Session":
        """Record a pipeline step. Saved with the session, so a file explains itself."""
        self.history.append({
            "step": str(step),
            "params": params,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        return self

    # ---- reading things ---------------------------------------------------

    def array(self, name: str) -> np.ndarray:
        if name not in self.arrays:
            raise KeyError(f"no array '{name}'. Available: {sorted(self.arrays)}")
        return self.arrays[name]

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"no table '{name}'. Available: {sorted(self.tables)}")
        return self.tables[name]

    def axis(self, array_name: str, axis_name: str) -> int:
        """Position of a named axis. Use this instead of hard-coding 0 or 1."""
        names = self.dims.get(array_name, [])
        if axis_name not in names:
            raise KeyError(f"array '{array_name}' has axes {names}, not '{axis_name}'")
        return names.index(axis_name)

    def n_along(self, array_name: str, axis_name: str) -> int:
        """Length of a named axis."""
        return int(self.arrays[array_name].shape[self.axis(array_name, axis_name)])

    def seconds(self, samples) -> np.ndarray:
        """Sample indices to seconds. The one place the sampling rate is divided by."""
        return np.asarray(samples, dtype=np.float64) / self.fs_hz

    # ---- input / output ---------------------------------------------------

    def save(self, path: str | Path, compression: str | None = "gzip") -> Path:
        self.validate()
        return fmt.write_session(path, self, compression=compression)

    @staticmethod
    def load(path: str | Path) -> "Session":
        """Load any ephyslink file, returning the subclass its `kind` attribute names."""
        data = fmt.read_session_dict(path)
        cls = {"OE": SessionOE, "KS": SessionKS, "RESULTS": SessionResults}.get(
            data["kind"], Session)
        session = cls(
            id=data["id"], fs_hz=data["fs_hz"], experiment=data["experiment"],
            source=data["source"], arrays=data["arrays"], dims=data["dims"],
            tables=data["tables"], events=data["events"], meta=data["meta"],
            history=data["history"], array_meta=data["array_meta"],
        )
        session.meta.setdefault("created", data["created"])
        return session

    def validate(self) -> None:
        """Checks that would otherwise fail silently, or fail later in Julia."""
        for name, values in self.arrays.items():
            declared = self.dims.get(name)
            if not declared or len(declared) != values.ndim:
                raise ValueError(
                    f"array '{name}' has shape {values.shape} but dims {declared}. Every array "
                    "must name its axes — an unlabelled axis is how a session gets silently "
                    "transposed between languages."
                )
        if self.fs_hz <= 0:
            raise ValueError(f"fs_hz is {self.fs_hz}")

    # ---- niceties ---------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{type(self).__name__}('{self.id}'", f"fs={self.fs_hz:g} Hz"]
        if self.arrays:
            parts.append(f"{len(self.arrays)} arrays")
        if self.tables:
            parts.append(f"{len(self.tables)} tables")
        if self.events:
            parts.append(f"{len(self.events)} event channels")
        return ", ".join(parts) + ")"

    def summary(self) -> str:
        """Everything in the session, in the order it would be written."""
        lines = [f"{type(self).__name__}  {self.id}",
                 f"  experiment  {self.experiment or '—'}",
                 f"  fs          {self.fs_hz:g} Hz",
                 f"  source      {self.source or '—'}"]
        if self.arrays:
            lines.append("  arrays")
            for name, values in self.arrays.items():
                dims = ", ".join(self.dims.get(name, []))
                lines.append(f"    {name:<20} {str(values.shape):<22} {values.dtype}  [{dims}]")
        if self.tables:
            lines.append("  tables")
            for name, frame in self.tables.items():
                lines.append(f"    {name:<20} {len(frame)} rows × {len(frame.columns)} columns")
        if self.events:
            lines.append("  events")
            for name, samples in self.events.items():
                lines.append(f"    {name:<20} {samples.size} events")
        if self.meta:
            lines.append(f"  meta        {', '.join(sorted(self.meta))}")
        if self.history:
            lines.append("  history")
            for entry in self.history:
                lines.append(f"    {entry['time'][:19]}  {entry['step']}")
        return "\n".join(lines)


# =============================================================================

class SessionResults(Session):
    """What an analysis produced, with the raw data stripped out.

    Produced by `Session.export_results()`, never by a loader. Holds the derived tables at
    full per-row resolution, the `results` dict of session-level numbers, the metadata and the
    history — everything a cross-session comparison needs and nothing it does not.

    It is an ordinary ephyslink file, so Julia reads it with the same code.
    """

    kind = "RESULTS"

    def validate(self) -> None:
        super().validate()
        if not self.tables and not self.results:
            raise ValueError(f"results session '{self.id}' is empty")


class SessionOE(Session):
    """Continuous data — LFP, MEP, EEG. `continuous` is (channel, sample)."""

    kind = "OE"

    @property
    def continuous(self) -> np.ndarray:
        return self.array("continuous")

    @property
    def n_channels(self) -> int:
        return self.n_along("continuous", "channel")

    @property
    def n_samples(self) -> int:
        return self.n_along("continuous", "sample")

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs_hz

    @property
    def channel_names(self) -> list[str]:
        return list(self.meta.get("channel_names", []))

    @property
    def times_s(self) -> np.ndarray:
        """Time axis aligned to `continuous`, in seconds."""
        return np.arange(self.n_samples, dtype=np.float64) / self.fs_hz

    def validate(self) -> None:
        super().validate()
        if "continuous" in self.arrays:
            names = self.dims["continuous"]
            if names[:2] != ["channel", "sample"]:
                raise ValueError(
                    f"SessionOE 'continuous' must be (channel, sample); its dims are {names}. "
                    "Transpose it before attaching rather than relabelling the axes."
                )


class SessionKS(Session):
    """Spike-sorted output. Spikes are sample indices; divide by `fs_hz` once, at the edge."""

    kind = "KS"

    @property
    def spike_times(self) -> np.ndarray:
        return self.array("spike_times")

    @property
    def spike_clusters(self) -> np.ndarray:
        return self.array("spike_clusters")

    @property
    def clusters(self) -> pd.DataFrame:
        return self.table("clusters")

    @property
    def n_spikes(self) -> int:
        return int(self.spike_times.size)

    @property
    def duration_s(self) -> float:
        times = self.spike_times
        return float(times.max() - times.min()) / self.fs_hz if times.size else 0.0

    def spikes_of(self, cluster: int) -> np.ndarray:
        """One cluster's spike times, in samples.

        Linear in the number of spikes. For repeated access across many clusters, bucket once
        with `scripts/spiketrains.py::SpikeStore` instead — this is the convenience path, not
        the fast one.
        """
        return self.spike_times[self.spike_clusters == int(cluster)]

    def validate(self) -> None:
        super().validate()
        n = self.arrays.get("spike_times", np.empty(0)).size
        for name in ("spike_clusters", "spike_positions", "spike_templates"):
            if name in self.arrays and len(self.arrays[name]) != n:
                raise ValueError(
                    f"'{name}' has {len(self.arrays[name])} rows but spike_times has {n}"
                )
        if "spike_times" in self.arrays and not np.all(np.diff(self.spike_times) >= 0):
            raise ValueError("spike_times is not sorted; every searchsorted downstream assumes it is")
