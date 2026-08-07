"""
Reading results-only sessions back, for comparison across sessions.

`Session.export_results()` writes one small file per session. This reads a directory of them
and stacks each named table into one DataFrame with a `session` column, which is the shape a
cross-session comparison actually wants::

    from ephyslink import read_results

    results = read_results("output/_results")
    results.tables["responders"]        # every unit x condition from every session
    results.summary                     # one row per session, from each session's results dict

No raw data is touched: these files contain no spikes, traces or templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .session import Session


@dataclass
class ResultSet:
    """Every session's results, stacked."""

    sessions: dict[str, Session] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        shapes = ", ".join(f"{n}={len(t)}" for n, t in self.tables.items())
        return f"ResultSet({len(self.sessions)} sessions; {shapes})"


def read_results(paths, pattern: str = "*.h5") -> ResultSet:
    """Load result files from a directory, a list of paths, or a single path."""
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        files = sorted(root.glob(pattern)) if root.is_dir() else [root]
    else:
        files = [Path(p) for p in paths]
    if not files:
        raise FileNotFoundError(f"no result files matching {pattern} in {paths}")

    sessions: dict[str, Session] = {}
    stacked: dict[str, list[pd.DataFrame]] = {}
    summary_rows = []

    for file in files:
        session = Session.load(file)
        if session.kind != "RESULTS":
            raise ValueError(
                f"{file} has kind={session.kind!r}, not 'RESULTS'. read_results is for files "
                "written by export_results(); use Session.load for a full session."
            )
        sessions[session.id] = session

        for name, frame in session.tables.items():
            tagged = frame.copy()
            if "session" in tagged.columns:
                # a table that already identifies its session — check it agrees rather than
                # overwriting, because a mismatch means a table was attached to the wrong one
                mismatched = set(tagged["session"].unique()) - {session.id}
                if mismatched:
                    raise ValueError(
                        f"{file}: table '{name}' has a `session` column naming {mismatched}, "
                        f"but the file is session '{session.id}'"
                    )
            else:
                # session first, so a stacked table reads like a table of sessions
                tagged.insert(0, "session", session.id)
            stacked.setdefault(name, []).append(tagged)

        summary_rows.append({
            "session": session.id,
            "experiment": session.experiment,
            **{k: v for k, v in session.meta.items()
               if isinstance(v, (int, float, str, bool)) and k != "results"},
            **{k: v for k, v in session.results.items()
               if isinstance(v, (int, float, str, bool))},
        })

    return ResultSet(
        sessions=sessions,
        tables={n: pd.concat(f, ignore_index=True) for n, f in stacked.items()},
        summary=pd.DataFrame(summary_rows).set_index("session", drop=False),
    )
