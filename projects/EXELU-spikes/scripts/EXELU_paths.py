"""
Where things are on disk, and how the two script folders get onto the import path.

Three trees:

* the **framework root** — `neuroelectrophysiology/`, holding `scripts/` (the shared,
  project-agnostic analysis code) and `projects/`. Located by walking up until a directory
  containing both is found, so everything works from any working directory.
* the **project** — `projects/EXELU-spikes/`, holding this file, the notebook, the log and
  everything the analysis writes.
* the **data root** — the read-only recordings. Not in the repository (tens of GB), so its
  location is machine-dependent and resolved from a candidate list or `EXELU_DATA_ROOT`.

`add_script_paths()` puts both script folders on `sys.path`. Call it before importing any
analysis module; the notebook and every runnable script do exactly that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_NAME = "EXELU-spikes"

DATA_ROOT_CANDIDATES: tuple[str, ...] = (
    "/Users/fomo/Local Data/SpikesAugust2026",
    "/sessions/gallant-sleepy-volta/mnt/SpikesAugust2026",
)
"""Searched in order; the first that exists wins. Add to this list rather than editing code
when the data moves. `EXELU_DATA_ROOT` overrides everything."""


_THIS_FILE = Path(__file__).resolve()
"""This file is at `<framework>/projects/EXELU-spikes/scripts/paths.py`, so every other
location is a fixed number of steps from it. Anchoring on the module's own location rather
than on the current working directory means the notebook, a script, and a batch job all
resolve identically, whatever directory they were launched from."""


def project_root() -> Path:
    """`<framework>/projects/EXELU-spikes/` — this project."""
    return _THIS_FILE.parents[1]


def framework_root() -> Path:
    """`neuroelectrophysiology/` — the directory holding both `scripts/` and `projects/`."""
    root = _THIS_FILE.parents[3]
    if not ((root / "scripts").is_dir() and (root / "projects").is_dir()):
        raise FileNotFoundError(
            f"expected {root} to contain scripts/ and projects/. This project has been moved "
            "out of the framework tree; the shared analysis modules cannot be found."
        )
    return root


def add_script_paths() -> tuple[Path, Path]:
    """Put the shared scripts, the project scripts and `modules/` on `sys.path`.

    `modules/` carries `ephyslink`, which is a package rather than a loose script, so it goes
    on the path as a directory to import *from*.

    Project scripts first, so a project module shadows a shared one of the same name should
    that ever be needed. Returns the shared and project directories, so a caller can print
    them and see exactly where its imports came from.
    """
    root = framework_root()
    shared, project, modules = root / "scripts", project_root() / "scripts", root / "modules"
    for directory in (modules, shared, project):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return shared, project


def data_root() -> Path:
    """Directory holding one subdirectory per recording session."""
    override = os.environ.get("EXELU_DATA_ROOT")
    candidates = (override, *DATA_ROOT_CANDIDATES) if override else DATA_ROOT_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError(
        "no data root found. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def session_names(root: Path | None = None) -> list[str]:
    """Every session directory under the data root, sorted, hidden entries excluded."""
    root = root or data_root()
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))


def session_dir(name: str, root: Path | None = None) -> Path:
    path = (root or data_root()) / name
    if not path.is_dir():
        raise FileNotFoundError(f"no session directory {path}")
    return path


def mice_records_path(project: Path | None = None) -> Path:
    """The animal metadata table. Accepts `mice-records` with or without a .csv suffix."""
    project = project or project_root()
    for name in ("mice-records.csv", "mice-records"):
        candidate = project / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no mice-records[.csv] in {project}")


def output_dir(session_name: str, project: Path | None = None) -> Path:
    """Per-session output directory, created if absent.

    Always under the project root, never under the current working directory — the notebook
    lives in `notebooks/` and a cwd-relative path would scatter output into it.
    """
    path = (project or project_root()) / "output" / session_name
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(session_name: str, project: Path | None = None) -> Path:
    return output_dir(session_name, project) / "figures"


def log_dir(project: Path | None = None) -> Path:
    return (project or project_root()) / "log"


def results_dir(project: Path | None = None) -> Path:
    """Where the results-only session files go, one per session.

    These are what a cross-session comparison reads: derived tables and summary numbers, no
    spikes and no traces. `ephyslink.read_results` points here.
    """
    path = (project or project_root()) / "output" / "_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_path(session_id: str, project: Path | None = None) -> Path:
    return results_dir(project) / f"{session_id}.h5"


def session_file(session_id: str, project: Path | None = None) -> Path:
    """Where a full working session (raw data plus everything attached) is saved."""
    return output_dir(session_id, project) / f"{session_id}.h5"


def cross_session_dir(project: Path | None = None) -> Path:
    """Where per-session summary rows accumulate so they can be compared across sessions."""
    path = (project or project_root()) / "output" / "_across_sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Run on import, deliberately. Every other EXELU_* module begins with `import EXELU_paths`,
# so importing any one of them makes the shared `<framework>/scripts` folder resolvable —
# there is no separate setup step to forget, and no module that only works from the notebook.
add_script_paths()
