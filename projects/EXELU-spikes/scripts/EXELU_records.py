"""
Animal metadata: reading `mice-records`, and writing it into `session_particulars.txt`.

`mice-records` is the hand-maintained table of which animal each recording came from.
`session_particulars.txt` is the per-session file that sits next to the recording and
already holds the anatomical landmark channels. Point 1 of the review is to get
`Animal_ID`, `genotype` and `comments` into the second file, so that a session on disk is
self-describing and the analysis never has to reach back to a spreadsheet.

The two are joined on the recording's directory name, which appears in `mice-records` as
the `filebase` column.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

import EXELU_paths as paths

PARTICULARS_FILENAME = "session_particulars.txt"

MICE_RECORD_FIELDS = ("Animal_ID", "genotype", "comments")
"""The three fields written into session_particulars.txt. Deliberately a short list:
everything else in mice-records is either derivable, provisional, or free text that would
not survive a key=value file."""


# =============================================================================
# reading
# =============================================================================

def load_mice_records(path: Path | None = None) -> pd.DataFrame:
    """The animal table, one row per recording, blank separator rows removed.

    Column names are used verbatim from the file except for whitespace stripping, so a
    future column appears without a code change.
    """
    path = path or paths.mice_records_path()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
    df = df[df["filebase"] != ""].reset_index(drop=True)
    if df["filebase"].duplicated().any():
        duplicated = df.loc[df["filebase"].duplicated(), "filebase"].tolist()
        raise ValueError(f"mice-records has duplicate filebase rows: {duplicated}")
    return df


def record_for_session(session_name: str, records: pd.DataFrame | None = None) -> dict[str, str]:
    """The mice-records row for one session, as a plain dict. Empty dict if absent."""
    records = load_mice_records() if records is None else records
    match = records[records["filebase"] == session_name]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def parse_particulars(path: Path) -> tuple[dict[str, str | None], list[str]]:
    """Read a session_particulars.txt.

    Returns `(values, raw_lines)`. `values` maps key -> string, or None where the key is
    present but blank. `raw_lines` is the file verbatim, so a rewrite can preserve
    ordering, comments and the `DICT:` lines the old pipeline appended.
    """
    raw_lines = path.read_text().splitlines() if path.is_file() else []
    values: dict[str, str | None] = {}
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("DICT:") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip() or None
    return values, raw_lines


# =============================================================================
# writing  (review point 1)
# =============================================================================

def _sanitise(value: str) -> str:
    """A value has to survive a one-line `key=value` file, so newlines become spaces."""
    return " ".join(str(value).split())


def update_particulars_file(
    session_path: Path,
    record: dict[str, str],
    fields: tuple[str, ...] = MICE_RECORD_FIELDS,
    backup: bool = True,
    dry_run: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Write `fields` from `record` into this session's session_particulars.txt.

    Idempotent by construction: a key that is already present is *replaced in place*,
    never appended a second time, so running this repeatedly converges. Keys not in
    `fields` are untouched, and `DICT:` lines are preserved in their original position.

    A `.bak` copy is made the first time only, so the pristine original is never
    overwritten by a later run.

    Returns `(new_lines, written)` where `written` is what was actually set.
    """
    path = session_path / PARTICULARS_FILENAME
    _, raw_lines = parse_particulars(path)

    written = {
        field: _sanitise(record[field])
        for field in fields
        if record.get(field, "") not in ("", None)
    }

    new_lines: list[str] = []
    seen: set[str] = set()
    for line in raw_lines:
        key = line.strip().partition("=")[0].strip()
        if key in written:
            new_lines.append(f"{key}={written[key]}")
            seen.add(key)
        else:
            new_lines.append(line)

    # Anything not already present is inserted before the first DICT: line, so the
    # human-readable key=value block stays together at the top of the file.
    missing = [f"{k}={v}" for k, v in written.items() if k not in seen]
    if missing:
        first_dict = next(
            (i for i, l in enumerate(new_lines) if l.strip().startswith("DICT:")),
            len(new_lines),
        )
        new_lines = new_lines[:first_dict] + missing + new_lines[first_dict:]

    if not dry_run:
        if backup and path.is_file() and not path.with_suffix(".txt.bak").is_file():
            shutil.copy2(path, path.with_suffix(".txt.bak"))
        path.write_text("\n".join(new_lines) + "\n")

    return new_lines, written


# =============================================================================
# who this recording is, asked of an ephyslink SessionKS
#
# The particulars file is authoritative — `update_particulars` writes the animal identity into
# it, so a session on disk is self-describing. `mice-records` is the fallback for a session
# whose particulars have not been updated.
# =============================================================================

def _identity(session, field: str, default: str = "") -> str:
    particulars = session.meta.get("particulars", {}) or {}
    record = session.meta.get("mice_record", {}) or {}
    return particulars.get(field) or record.get(field, "") or default


def animal_id(session) -> str:
    return _identity(session, "Animal_ID", "unknown")


def genotype(session) -> str:
    return _identity(session, "genotype", "unknown")


def comments(session) -> str:
    return _identity(session, "comments")


def label(session) -> str:
    """One line of identity for a figure title: which session, which animal, which genotype."""
    return f"{session.id}  ·  {animal_id(session)}  ·  {genotype(session)}"


def describe(session) -> str:
    """The header printed at the top of the notebook."""
    import EXELU_regions as regions

    depth = session.array("spike_positions")[:, 1]
    return "\n".join([
        "=" * 92,
        f"SESSION   {session.id}",
        f"ANIMAL    {animal_id(session)}   genotype {genotype(session)}",
        f"COMMENTS  {comments(session) or '(none recorded)'}",
        "=" * 92,
        f"duration            {session.duration_s / 60:.1f} min",
        f"spikes              {session.n_spikes:,}",
        f"clusters            {dict(session.clusters['KSLabel'].value_counts())}",
        f"spike depth range   {depth.min():.0f} – {depth.max():.0f} µm",
        f"landmarks (µm)      { {k: round(v) for k, v in regions.landmarks(session).items()} }",
        f"CTX/HPC boundary    {regions.boundary(session):.0f} µm  (thetaCh "
        f"{regions.particulars(session).get('thetaCh')}, 1-based)",
        f"analog channels     {list(session.events)}",
        f"events per channel  { {k: v.size for k, v in session.events.items()} }",
    ])


def update_all_particulars(
    data_root: Path | None = None,
    records: pd.DataFrame | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Apply `update_particulars_file` to every session directory. Returns a report."""
    data_root = data_root or paths.data_root()
    records = load_mice_records() if records is None else records

    rows = []
    for name in paths.session_names(data_root):
        record = record_for_session(name, records)
        if not record:
            rows.append({"session": name, "status": "NOT IN mice-records", **dict.fromkeys(MICE_RECORD_FIELDS, "")})
            continue
        _, written = update_particulars_file(data_root / name, record, dry_run=dry_run)
        rows.append({
            "session": name,
            "status": "would write" if dry_run else "written",
            **{f: written.get(f, "") for f in MICE_RECORD_FIELDS},
        })
    return pd.DataFrame(rows)
