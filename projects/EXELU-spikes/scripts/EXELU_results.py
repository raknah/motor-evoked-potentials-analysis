"""
The end of the pipeline: turn a finished session into the numbers that get compared.

The analysis attaches its tables to the session as it goes — `units`, `responders`,
`entrainment`, `harmonic_control`, `pooled_mua_regulation`. `summarise()` then reduces those
to a flat dict of session-level numbers, and `export()` writes a results-only file: the
tables at full per-row resolution plus that dict, with every spike, template and waveform
stripped out.

For this session that is 859 KB against 44.6 MB with the raw data in it — a factor of 52,
which is what makes a fifteen-session comparison a thing you can hold in memory.

    session.results.update(EXELU_results.summarise(session))
    EXELU_results.export(session)

Read them back with `ephyslink.read_results("output/_results")`.

**Every number in the summary must be a scalar.** It ends up as one cell of one row of the
cross-session table, and a list or an array there turns that column into an object column
that nothing will aggregate.
"""

from __future__ import annotations

# Importing EXELU_paths puts <framework>/scripts on sys.path, so the shared
# analysis modules below resolve wherever this file is imported from.
import EXELU_paths  # noqa: F401

from pathlib import Path

import numpy as np
import pandas as pd

import EXELU_paths as paths
import EXELU_records as records
import EXELU_regions as regions

RESULT_TABLES = ("units", "responders", "entrainment", "harmonic_control",
                 "pooled_mua_regulation", "baseline_spread")
"""The tables the notebook is expected to have attached. Missing ones are skipped rather than
raising: a partial run should still be able to export what it did produce."""


def _median(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].median()) if len(frame) else float("nan")


def summarise(session) -> dict:
    """Flatten the session's tables into the scalars a cross-session comparison compares.

    Names are `<metric>_<region>_<condition>`, lowercase, so the resulting columns sort into
    coherent groups and can be selected with a prefix.
    """
    summary: dict[str, object] = {
        "animal_id": records.animal_id(session),
        "genotype": records.genotype(session),
        "duration_min": round(session.duration_s / 60, 2),
        "n_spikes": int(session.n_spikes),
        "boundary_um": round(regions.boundary(session), 1),
        "n_clusters_good": int((session.clusters.KSLabel == "good").sum()),
        "n_clusters_mua": int((session.clusters.KSLabel == "mua").sum()),
    }

    units = session.tables.get("units")
    if units is not None:
        summary["n_clean"] = len(units)
        for region, group in units.groupby("region"):
            summary[f"n_clean_{region.lower()}"] = len(group)
        for cell_type, group in units.groupby("cell_type"):
            summary[f"n_{cell_type}"] = len(group)
        summary["n_celltype_flagged"] = int((~units.cell_type_consistent).sum())

    responders = session.tables.get("responders")
    if responders is not None:
        for (condition, region), group in responders.groupby(["condition", "region"]):
            tag = f"{condition.lower()}_{region.lower()}"
            summary[f"n_responders_{tag}"] = int(group.responder.sum())
            summary[f"n_tested_{tag}"] = len(group)
            summary[f"median_z_stim_{tag}"] = round(_median(group, "z_stim"), 5)
            summary[f"median_pct_change_{tag}"] = round(_median(group, "pct_change"), 3)

    entrainment = session.tables.get("entrainment")
    if entrainment is not None:
        reliable = entrainment[entrainment.reliable]
        for (condition, region), group in reliable.groupby(["condition", "region"]):
            tag = f"{condition.lower()}_{region.lower()}"
            summary[f"n_locked_{tag}"] = int(group.locked.sum())
            summary[f"n_reliable_{tag}"] = len(group)
            summary[f"median_ppc_{tag}"] = round(_median(group, "ppc"), 6)
            summary[f"median_mod_depth_norm_{tag}"] = round(_median(group, "mod_depth_norm"), 4)

    harmonics = session.tables.get("harmonic_control")
    if harmonics is not None:
        summary["n_locked_at_2f0_only"] = int(harmonics.locked_at_2f0_only.sum())

    pooled = session.tables.get("pooled_mua_regulation")
    if pooled is not None:
        for _, row in pooled.iterrows():
            tag = f"{row['condition'].lower()}_{row['region'].lower()}"
            summary[f"pooled_pct_change_{tag}"] = round(float(row["pct_change"]), 3)
            summary[f"pooled_peak_z_{tag}"] = round(float(row["peak_z"]), 3)
            summary[f"pooled_p_{tag}"] = float(row["p_wilcoxon"])

    spread = session.tables.get("baseline_spread")
    if spread is not None:
        summary["median_baseline_cv"] = round(_median(spread, "baseline_cv"), 3)

    non_scalar = {k: type(v).__name__ for k, v in summary.items()
                  if not isinstance(v, (int, float, str, bool, np.integer, np.floating))}
    if non_scalar:
        raise TypeError(
            f"these summary entries are not scalars and would break the cross-session table: "
            f"{non_scalar}"
        )
    return summary


def export(session, path: str | Path | None = None) -> Path:
    """Write the results-only file for this session.

    Defaults to `output/_results/<session id>.h5`, which is where `read_results` looks.
    """
    path = Path(path) if path else paths.results_path(session.id)
    return session.export_results(path)


def check_exported(session, path: str | Path) -> str:
    """Reload what was written and confirm the raw data really is gone.

    An export that silently kept the spike arrays would be 44 MB and nobody would notice until
    the fifteen-session comparison would not fit in memory.
    """
    from ephyslink import Session

    reloaded = Session.load(path)
    size_kb = Path(path).stat().st_size / 1e3
    leaked = sorted(set(reloaded.arrays) & set(session.meta.get("source_arrays", [])))
    if leaked:
        raise AssertionError(f"the results file still contains raw arrays: {leaked}")

    rows = {name: len(frame) for name, frame in reloaded.tables.items()}
    return (f"{Path(path).name}  {size_kb:.0f} KB  ·  "
            f"{len(reloaded.tables)} tables {rows}  ·  "
            f"{len(reloaded.results)} summary numbers  ·  "
            f"{len(reloaded.history)} steps  ·  no raw arrays")
