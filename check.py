#!/usr/bin/env python3
"""
One command that checks everything.

    python check.py

Runs the Python test suites, then drives Julia itself — you do not have to run anything by
hand, chain files together, or know which mode to pass. Exit code 0 means every check passed.
Anything else prints what failed and where.

    python check.py --quick     skip the slow real-data checks
    python check.py --verbose   print every individual check, not just failures

If Julia is not on your PATH, set JULIA=/path/to/julia. The cross-language half is the point
of this file, so a missing Julia is a failure, not a skip.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

JULIA_PROJECT = ROOT / "modules" / "EphysLink.jl"
BRIDGE = JULIA_PROJECT / "test" / "bridge.jl"
CHAIN = JULIA_PROJECT / "test" / "chain.jl"
BENCH = JULIA_PROJECT / "test" / "bench.jl"
JULIA_SUITE = JULIA_PROJECT / "test" / "roundtrip.jl"


# =============================================================================
# reporting
# =============================================================================

class Report:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.passed = 0
        self.failures: list[tuple[str, str]] = []
        self.section = ""

    def start(self, name: str) -> None:
        self.section = name
        print(f"\n{name}")
        print("─" * len(name))

    def check(self, description: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            if self.verbose:
                print(f"  ok    {description}")
        else:
            self.failures.append((f"{self.section} · {description}", detail))
            print(f"  FAIL  {description}")
            if detail:
                print(f"        {detail}")
        return bool(condition)

    def note(self, text: str) -> None:
        print(f"  ·     {text}")

    def summary(self) -> int:
        print()
        if not self.failures:
            print(f"ALL {self.passed} CHECKS PASSED")
            return 0
        print(f"{len(self.failures)} FAILED, {self.passed} passed\n")
        for name, detail in self.failures:
            print(f"  {name}")
            if detail:
                print(f"      {detail}")
        return 1


# =============================================================================
# the arrays every check runs over
#
# Chosen to break things, not to look tidy:
#   - non-square, so an axis reversal changes the shape and cannot hide
#   - SQUARE, where a reversal does NOT change the shape, so only values catch it
#   - singleton axes, where several shapes look alike
#   - rank 1 through 5
#   - a zero-length axis
# =============================================================================

CASES: list[tuple[str, tuple[int, ...], list[str]]] = [
    ("vector",            (500,),          ["spike"]),
    ("rank2_wide",        (4, 7),          ["channel", "sample"]),
    ("rank2_tall",        (7, 4),          ["sample", "channel"]),
    ("rank2_square",      (5, 5),          ["channel", "sample"]),
    ("rank2_singleton_a", (1, 9),          ["channel", "sample"]),
    ("rank2_singleton_b", (9, 1),          ["channel", "sample"]),
    ("rank3",             (3, 5, 11),      ["channel", "sample", "epoch"]),
    ("rank3_singleton",   (1, 6, 1),       ["channel", "sample", "epoch"]),
    ("rank4",             (2, 3, 5, 7),    ["channel", "sample", "epoch", "band"]),
    ("rank5",             (2, 3, 2, 5, 3), ["a", "b", "c", "d", "e"]),
    ("zero_axis",         (3, 0),          ["channel", "sample"]),
]

DTYPES = ["float32", "float64", "int8", "int16", "int32", "int64",
          "uint8", "uint16", "uint32", "bool"]


def build_case_array(shape: tuple[int, ...], dtype: str = "float32") -> np.ndarray:
    """Distinct value per element, so any reordering is visible."""
    n = int(np.prod(shape)) if shape else 0
    if dtype == "bool":
        return (np.arange(n) % 3 == 0).reshape(shape)
    values = np.arange(n)
    if dtype.startswith("uint"):
        values = values % 200
    elif dtype in ("int8",):
        values = (values % 200) - 100
    elif dtype in ("int16", "int32", "int64"):
        values = values - n // 2
    return values.astype(dtype).reshape(shape)


def build_fixture():
    """The session every cross-language check is run on."""
    from ephyslink import SessionKS

    session = SessionKS(id="check_fixture", fs_hz=30_000.0,
                        experiment="ephyslink check · µV é 日本", source="synthetic")
    for name, shape, dims in CASES:
        session.add_array(name, build_case_array(shape), dims=dims)
    for dtype in DTYPES:
        session.add_array(f"dtype_{dtype}", build_case_array((3, 4), dtype),
                          dims=["channel", "sample"])

    session.add_table("clusters", pd.DataFrame({
        "cluster_id": np.arange(6, dtype=np.int32),
        "KSLabel": ["good", "mua", "good", "mua", "good", "mua"],
        "unicode": ["µV", "", "日本", "a b", "x,y", "quote\"s"],
        "ContamPct": np.array([0.0, 1.5, 100.0, 320.25, -1.0, 0.125]),
        "flag": np.array([True, False, True, True, False, False]),
    }))
    session.add_table("empty_table", pd.DataFrame({"a": np.array([], dtype=np.int64),
                                                   "b": np.array([], dtype=float)}))
    session.add_events("flicker_40Hz", np.arange(0, 10_000, 750, dtype=np.int64))
    session.add_events("empty_events", np.array([], dtype=np.int64))
    session.meta.update({
        "animal": "XU20",
        "genotype": "5xFAD",
        "nested": {"landmarks": {"thetaCh": 54, "CA1Ch": 69}, "bad_channels": []},
        "missing": None,
        "not_computed": float("nan"),
        "unicode": "µV é 日本",
        "true": True,
        "big_int": 2**40,
    })
    session.log("build_fixture", seed=0)
    return session


# =============================================================================
# 1 · python round-trip
# =============================================================================

def python_roundtrip(report: Report, tmp: Path) -> None:
    from ephyslink import Session

    report.start("1 · Python round-trip — every shape, every dtype")
    original = build_fixture()
    path = original.save(tmp / "fixture.h5")
    back = Session.load(path)

    report.check("kind, id, fs and experiment survive",
                 (back.kind, back.id, back.fs_hz, back.experiment)
                 == (original.kind, original.id, original.fs_hz, original.experiment))

    report.check("same set of arrays", set(back.arrays) == set(original.arrays),
                 f"missing {set(original.arrays) - set(back.arrays)}")

    for name, values in original.arrays.items():
        other = back.arrays[name]
        report.check(f"array '{name}' shape {values.shape} preserved",
                     other.shape == values.shape, f"got {other.shape}")
        report.check(f"array '{name}' values identical", np.array_equal(other, values))
        report.check(f"array '{name}' dtype {values.dtype} preserved",
                     other.dtype == values.dtype, f"got {other.dtype}")
        report.check(f"array '{name}' dims preserved", back.dims[name] == original.dims[name])

    for name, frame in original.tables.items():
        other = back.tables[name]
        report.check(f"table '{name}' column order preserved",
                     list(other.columns) == list(frame.columns))
        report.check(f"table '{name}' row count preserved", len(other) == len(frame))
        for column in frame.columns:
            left, right = frame[column].to_numpy(), other[column].to_numpy()
            same = (np.allclose(left.astype(float), right.astype(float))
                    if left.dtype.kind in "fiub"
                    else np.array_equal(left.astype(str), right.astype(str)))
            report.check(f"table '{name}'.{column} values preserved", same)

    for name, samples in original.events.items():
        report.check(f"events '{name}' preserved", np.array_equal(back.events[name], samples))

    report.check("nested metadata preserved", back.meta["nested"] == original.meta["nested"])
    report.check("unicode metadata preserved", back.meta["unicode"] == "µV é 日本")
    report.check("NaN metadata became null, not a crash", back.meta["not_computed"] is None)
    report.check("large integer metadata preserved", back.meta["big_int"] == 2**40)
    report.check("history preserved", len(back.history) == len(original.history))


# =============================================================================
# 2 · the on-disk contract, checked with raw h5py
#
# A round trip can pass while the file is wrong, if both readers are wrong in the same way.
# This section opens the file with plain h5py and checks it against FORMAT.md directly.
# =============================================================================

def on_disk_contract(report: Report, path: Path, label: str) -> None:
    import h5py

    report.start(f"2 · On-disk contract — {label}")
    def attr(key, default=""):
        """h5py returns str for its own variable-length strings and bytes for the
        fixed-length ones HDF5.jl writes. Normalise, as the library itself does."""
        value = f.attrs.get(key, default)
        return value.decode("utf-8") if isinstance(value, bytes) else value

    with h5py.File(path, "r") as f:
        report.check("root says format=ephyslink", attr("format") == "ephyslink",
                     f"got {f.attrs.get('format')!r}")
        report.check("string attributes are readable whichever language wrote them",
                     isinstance(attr("id"), str) and attr("id") != "",
                     "a fixed-length ASCII attribute must still decode to a usable string")
        report.check("root carries a format_version", "format_version" in f.attrs)
        for key in ("kind", "id", "fs_hz", "meta_json", "history_json"):
            report.check(f"root attribute '{key}' present", key in f.attrs)

        report.check("meta_json is strictly valid JSON (no bare NaN/Infinity)",
                     _strict_json(attr("meta_json")),
                     "Julia's JSON3 rejects bare NaN and would silently lose all metadata")
        report.check("history_json is strictly valid JSON", _strict_json(attr("history_json")))

        for name, dataset in f.get("arrays", {}).items():
            dims = [d.decode() if isinstance(d, bytes) else str(d)
                    for d in dataset.attrs.get("dims", [])]
            report.check(f"'{name}' declares one axis name per axis",
                         len(dims) == dataset.ndim,
                         f"shape {dataset.shape}, dims {dims}")
            report.check(f"'{name}' on-disk dtype is portable ({dataset.dtype})",
                         dataset.dtype.kind in "fiu",
                         "HDF5 enums and object types do not cross into Julia cleanly")

        for name, group in f.get("tables", {}).items():
            columns = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in group.attrs["columns"]]
            report.check(f"table '{name}' records its column order",
                         set(columns) == set(group.keys()))
            lengths = {len(group[c]) for c in columns}
            report.check(f"table '{name}' columns are all the same length", len(lengths) <= 1,
                         f"lengths {lengths}")


def _strict_json(text) -> bool:
    if isinstance(text, bytes):
        text = text.decode()
    try:
        json.loads(text, parse_constant=_reject)
        return True
    except Exception:
        return False


def _reject(name):
    raise ValueError(f"non-standard JSON constant {name}")


# =============================================================================
# 3 · guards — the ways a mistake should be caught rather than stored
# =============================================================================

def guards(report: Report, tmp: Path) -> None:
    from ephyslink import Session, SessionOE

    report.start("3 · Guards — mistakes are refused, not stored")

    session = SessionOE(id="g", fs_hz=1000.0)
    report.check("an array with the wrong number of axis names is refused",
                 _raises(lambda: session.add_array("bad", np.zeros((3, 4)), dims=["one"])))

    session.arrays["sneaky"] = np.zeros((3, 4))
    report.check("validate() catches an array with no dims at all",
                 _raises(session.validate))
    del session.arrays["sneaky"]

    session.add_array("continuous", np.zeros((7, 4), np.float32), dims=["sample", "channel"])
    report.check("SessionOE refuses a transposed 'continuous'", _raises(session.validate))

    good = SessionOE(id="g2", fs_hz=1000.0)
    report.check("a string array is refused at add_array(), where the mistake is",
                 _raises(lambda: good.add_array("s", np.array(["a", "b"]), dims=["x"])))
    report.check("an object array is refused too",
                 _raises(lambda: good.add_array("o", np.array([1, "a"], dtype=object), dims=["x"])))
    report.check("a bool array IS allowed (it is stored as int8 and restored)",
                 not _raises(lambda: good.add_array("flags", np.array([True, False]), dims=["x"])))

    fixture = build_fixture()
    report.check("axis() finds an axis by name", fixture.axis("rank4", "epoch") == 2)
    report.check("axis() refuses an axis that does not exist",
                 _raises(lambda: fixture.axis("rank4", "frequency")))
    report.check("n_along() agrees with the shape",
                 all(fixture.n_along(n, d) == fixture.arrays[n].shape[i]
                     for n, dims in fixture.dims.items() for i, d in enumerate(dims)))

    import h5py
    legacy = tmp / "legacy.h5"
    with h5py.File(legacy, "w") as f:
        f.create_dataset("raw", data=np.zeros((6, 100), np.float32))
        f.attrs["session"] = "2023-08-25_15-32-14"
    report.check("a file from the old openephysextract code is refused, with a reason",
                 _raises(lambda: Session.load(legacy)))


def _raises(call) -> bool:
    try:
        call()
    except Exception:
        return True
    return False


# =============================================================================
# 4 · results export
# =============================================================================

def results_export(report: Report, tmp: Path) -> None:
    from ephyslink import Session, read_results

    report.start("4 · Results export — raw data out, derived data in")
    session = build_fixture()
    session.mark_source()
    session.add_table("responders", pd.DataFrame(
        {"unit": [1, 2, 3], "condition": ["a", "a", "b"], "q": [0.01, 0.5, 0.2]}))
    session.add_array("psth", np.zeros((3, 10), np.float32), dims=["unit", "bin"])
    session.results.update({"n_responders": 1, "median_ppc": 0.0277,
                            "not_computed": float("nan")})

    path = session.export_results(tmp / "results" / "check_fixture.h5")
    loaded = Session.load(path)

    report.check("kind is RESULTS", loaded.kind == "RESULTS")
    report.check("only the derived table is kept", set(loaded.tables) == {"responders"},
                 f"got {sorted(loaded.tables)}")
    report.check("no arrays leak into the results file", not loaded.arrays,
                 f"got {sorted(loaded.arrays)}")
    report.check("summary numbers survive", loaded.results["n_responders"] == 1)
    report.check("a NaN summary number becomes null rather than breaking the file",
                 loaded.results["not_computed"] is None)
    report.check("metadata survives", loaded.meta["animal"] == "XU20")
    report.check("naming an array explicitly does include it",
                 "psth" in Session.load(
                     session.export_results(tmp / "with_array.h5",
                                            include_arrays=["psth"])).arrays)

    bare = build_fixture()
    bare.mark_source()
    report.check("a session with nothing derived refuses to export an empty file",
                 _raises(lambda: bare.export_results(tmp / "empty.h5")))

    stacked = read_results(tmp / "results")
    report.check("read_results stacks tables and adds a session column",
                 list(stacked.tables["responders"].columns)[0] == "session")
    report.check("read_results builds one summary row per session",
                 stacked.summary.loc["check_fixture", "n_responders"] == 1)


# =============================================================================
# 5 · Julia — the cross-language half
# =============================================================================

def find_julia(report: Report) -> str | None:
    candidate = os.environ.get("JULIA") or shutil.which("julia")
    if candidate and Path(candidate).exists():
        return candidate
    for guess in ("/usr/local/bin/julia", "/opt/homebrew/bin/julia",
                  Path.home() / ".juliaup" / "bin" / "julia"):
        if Path(guess).exists():
            return str(guess)
    return None


def run_julia(julia: str, args: list[str], report: Report, what: str) -> str | None:
    try:
        done = subprocess.run([julia, f"--project={JULIA_PROJECT}"] + args,
                              capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        report.check(f"{what} finished", False, "timed out after 900 s")
        return None
    if done.returncode != 0:
        combined = (done.stdout + "\n" + done.stderr).strip().splitlines()
        # keep the lines that name a failing test, plus the tail for context
        interesting = [l for l in combined
                       if any(k in l for k in ("Test Failed", "Error During Test",
                                               "Expression:", "Evaluated:", "ERROR"))]
        tail = (interesting[:30] or combined[-30:])
        report.check(f"{what} succeeded", False, "\n        ".join(tail))
        return None
    return done.stdout


def julia_checks(report: Report, tmp: Path, fixture_path: Path,
                 quick: bool = False) -> None:
    from ephyslink import Session

    report.start("5 · Julia — cross-language")
    julia = find_julia(report)
    if not report.check("Julia was found", julia is not None,
                        "install Julia, or set JULIA=/path/to/julia. The cross-language "
                        "half is the point of this file, so this counts as a failure."):
        return

    report.note(f"using {julia}")
    if run_julia(julia, ["-e", "using Pkg; Pkg.instantiate()"], report,
                 "Julia dependencies installed") is None:
        return

    outdir = tmp / "julia"
    if run_julia(julia, [str(BRIDGE), str(fixture_path), str(outdir)], report,
                 "the Julia bridge ran") is None:
        return

    original = Session.load(fixture_path)
    report_path = outdir / "report.json"
    if not report.check("Julia wrote its report", report_path.is_file()):
        return
    seen = json.loads(report_path.read_text())
    report.note(f"Julia {seen['julia_version']}")

    # --- what Julia saw in memory -------------------------------------------
    report.check("Julia read the session identity correctly",
                 (seen["kind"], seen["id"], seen["fs_hz"])
                 == (original.kind, original.id, original.fs_hz))
    report.check("Julia read unicode in the experiment field",
                 seen["experiment"] == original.experiment, f"got {seen['experiment']!r}")
    report.check("Julia read the nested metadata",
                 seen["meta"]["nested"]["landmarks"]["thetaCh"] == 54)
    report.check("Julia read a null where Python wrote NaN",
                 seen["meta"].get("not_computed") is None)

    for name, values in original.arrays.items():
        got = seen["arrays"].get(name)
        if not report.check(f"Julia has array '{name}'", got is not None):
            continue
        expected_shape = list(values.shape)
        report.check(f"'{name}' matched mode gives Python's shape {tuple(expected_shape)}",
                     got["matched_shape"] == expected_shape,
                     f"Julia reported {got['matched_shape']}")
        report.check(f"'{name}' native mode gives the reversed shape",
                     got["native_shape"] == expected_shape[::-1],
                     f"Julia reported {got['native_shape']}")
        report.check(f"'{name}' matched dims equal Python's", got["matched_dims"] == original.dims[name])
        report.check(f"'{name}' native dims are Python's reversed",
                     got["native_dims"] == original.dims[name][::-1])
        report.check(f"'{name}' axis name to length pairing holds in Julia",
                     all(got["n_along"][d] == values.shape[i]
                         for i, d in enumerate(original.dims[name])))
        # the real test: values, in a defined order, element by element
        report.check(f"'{name}' VALUES identical in Julia, element by element",
                     np.allclose(np.asarray(got["c_order"], dtype=float),
                                 values.ravel(order="C").astype(float), equal_nan=True),
                     "shape can match while the contents are scrambled; this catches that")

    for name, frame in original.tables.items():
        got = seen["tables"].get(name)
        if not report.check(f"Julia has table '{name}'", got is not None):
            continue
        report.check(f"table '{name}' column order preserved in Julia",
                     got["columns"] == list(frame.columns), f"Julia reported {got['columns']}")
        for column in frame.columns:
            left = frame[column].to_numpy()
            right = np.asarray(got["values"][column])
            same = (np.allclose(left.astype(float), right.astype(float))
                    if left.dtype.kind in "fiub"
                    else np.array_equal(left.astype(str), right.astype(str)))
            report.check(f"table '{name}'.{column} values identical in Julia", same)

    for name, samples in original.events.items():
        report.check(f"events '{name}' identical in Julia",
                     np.array_equal(np.asarray(seen["events"][name], dtype=np.int64), samples))

    # --- what Julia wrote back ----------------------------------------------
    for written in ("from_matched.h5", "from_native.h5"):
        path = outdir / written
        if not report.check(f"Julia wrote {written}", path.is_file()):
            continue
        back = Session.load(path)
        for name, values in original.arrays.items():
            report.check(f"{written}: '{name}' round-trips shape and values",
                         back.arrays[name].shape == values.shape
                         and np.array_equal(back.arrays[name].astype(float),
                                            values.astype(float)),
                         f"got shape {back.arrays[name].shape}")
            report.check(f"{written}: '{name}' dims round-trip",
                         back.dims[name] == original.dims[name])
        on_disk_contract(report, path, f"written by Julia ({written})")
        report.start("5 · Julia — cross-language (continued)")

    added_path = outdir / "with_added.h5"
    if report.check("Julia wrote with_added.h5", added_path.is_file()):
        added = Session.load(added_path)
        expected = np.arange(1, 61, dtype=np.float32).reshape((3, 4, 5), order="F")
        report.check("an array BUILT in Julia arrives with the right shape",
                     added.arrays["julia_built"].shape == (5, 4, 3),
                     f"got {added.arrays['julia_built'].shape}")
        report.check("an array BUILT in Julia arrives with reversed dims",
                     added.dims["julia_built"] == ["c", "b", "a"],
                     f"got {added.dims['julia_built']}")
        report.check("an array BUILT in Julia arrives with the right values",
                     np.array_equal(added.arrays["julia_built"],
                                    np.transpose(expected, (2, 1, 0))))

    # --- Julia's own suite ---------------------------------------------------
    output = run_julia(julia, [str(JULIA_SUITE), str(fixture_path)], report,
                       "Julia's own test suite passed")
    if output:
        for line in output.strip().splitlines():
            if "Test Summary" in line or "EphysLink" in line:
                report.note(line.strip())

    julia_chains(report, julia, tmp, fixture_path)
    if not quick:
        julia_benchmark(report, julia, tmp)
    else:
        report.start("5c · Julia performance")
        report.note("skipped by --quick")



# =============================================================================
# 5b · every variant of the back and forth
#
# One round trip can pass while both sides are wrong in cancelling ways. This walks the tree
# of layout-mode sequences — Julia reads each file in BOTH modes and writes both back, Python
# copies everything in between — and compares every file at every depth against the ORIGINAL.
#
# After three rounds that is 2 + 4 + 8 = 14 Julia-written files covering every mode sequence
# up to depth three, with a Python hop between each. Three Julia launches, not fourteen.
# =============================================================================

def julia_chains(report: Report, julia: str, tmp: Path, fixture_path: Path,
                 rounds: int = 3) -> None:
    from ephyslink import Session

    report.start("5b · Every variant of the Python↔Julia back and forth")
    original = Session.load(fixture_path)

    current = tmp / "chain" / "round0"
    current.mkdir(parents=True)
    shutil.copy(fixture_path, current / "seed.h5")

    total = 0
    for round_number in range(1, rounds + 1):
        julia_out = tmp / "chain" / f"round{round_number}_julia"
        if run_julia(julia, [str(CHAIN), str(current), str(julia_out)], report,
                     f"Julia chain round {round_number}") is None:
            return

        chain_report = json.loads((julia_out / "chain_report.json").read_text())
        python_out = tmp / "chain" / f"round{round_number}_python"
        python_out.mkdir(parents=True)

        for file in sorted(julia_out.glob("*.h5")):
            path_label = file.stem                       # e.g. seed__N__M
            back = Session.load(file)
            total += 1

            ok_arrays = all(
                back.arrays[name].shape == values.shape
                and np.array_equal(back.arrays[name].astype(float), values.astype(float))
                for name, values in original.arrays.items()
            )
            report.check(f"[{path_label}] every array matches the original",
                         ok_arrays and set(back.arrays) == set(original.arrays),
                         _first_array_mismatch(original, back))
            report.check(f"[{path_label}] every dims list matches the original",
                         back.dims == original.dims)
            report.check(f"[{path_label}] tables and events survive",
                         all(list(back.tables[n].columns) == list(f.columns)
                             for n, f in original.tables.items())
                         and all(np.array_equal(back.events[n], e)
                                 for n, e in original.events.items()))
            report.check(f"[{path_label}] metadata survives",
                         back.meta.get("nested") == original.meta.get("nested"))

            seen = chain_report.get(file.name, {})
            for name, info in seen.get("arrays", {}).items():
                expected = original.arrays[name].shape
                wanted = list(expected) if seen["mode"] == "M" else list(expected)[::-1]
                report.check(f"[{path_label}] Julia held '{name}' as {tuple(wanted)} in memory",
                             info["shape"] == wanted, f"Julia reported {info['shape']}")
                report.check(f"[{path_label}] '{name}' axis names still pair with lengths",
                             all(info["n_along"][d] == expected[i] for i, d in
                                 enumerate(original.dims[name])))

            # the Python hop between rounds
            back.save(python_out / f"{path_label}.h5")

        current = python_out

    report.note(f"{total} Julia-written files across {rounds} rounds, all compared to the "
                f"original")


def _first_array_mismatch(original, back) -> str:
    for name, values in original.arrays.items():
        if name not in back.arrays:
            return f"'{name}' missing"
        got = back.arrays[name]
        if got.shape != values.shape:
            return f"'{name}' shape {got.shape} != {values.shape}"
        if not np.array_equal(got.astype(float), values.astype(float)):
            return f"'{name}' values differ"
    return ""


# =============================================================================
# 5c · is the layout actually paying for itself, and does it cost seamlessness?
# =============================================================================

def julia_benchmark(report: Report, julia: str, tmp: Path,
                    n_channels: int = 64, n_samples: int = 500_000) -> None:
    from ephyslink import SessionOE

    report.start("5c · Julia performance — optimised, and still seamless")

    rng = np.random.default_rng(0)
    trace = rng.standard_normal((n_channels, n_samples), dtype=np.float32)
    session = SessionOE(id="bench", fs_hz=30_000.0)
    session.add_array("continuous", trace, dims=["channel", "sample"], units="µV")
    path = session.save(tmp / "bench.h5", compression=None)
    report.note(f"{n_channels} channels × {n_samples:,} samples "
                f"({trace.nbytes / 1e6:.0f} MB)")

    out = tmp / "bench.json"
    if run_julia(julia, [str(BENCH), str(path), str(out)], report,
                 "the Julia benchmark ran") is None:
        return
    result = json.loads(out.read_text())
    seconds = result["seconds"]
    report.note(f"Julia {result['julia_version']}, {result['threads']} thread(s), "
                f"best of {result['repeats']} runs with a collection before each")

    # ---- seamlessness: the same code, the same answers ----------------------
    report.check("both layouts report the same channel count",
                 result["n_channels"]["native"] == result["n_channels"]["matched"] == n_channels)
    report.check("both layouts report the same sample count",
                 result["n_samples"]["native"] == result["n_samples"]["matched"] == n_samples)
    # The two layouts sum the SAME values in the SAME order, but Julia's `sum` blocks a
    # contiguous view differently from a strided one, so the float32 additions associate
    # differently and the last bits differ. That is reduction-order noise, not a layout error:
    # a real layout error changes the answer by O(the data), not by O(eps · Σ|x|).
    # Bound: eps32 · log2(n) · Σ|x|, with a factor of 10 for headroom.
    abs_sum = float(np.abs(trace).sum())
    fp_tolerance = 10 * np.finfo(np.float32).eps * np.log2(max(n_samples, 2)) * abs_sum
    sum_gap = abs(result["value_sum"]["native"] - result["value_sum"]["matched"])
    report.note(f"{'sum agreement':<20} native and matched differ by {sum_gap:.2e} "
                f"(float32 noise bound {fp_tolerance:.2e})")
    report.check("the same per-channel loop gives the same SUM in both layouts, "
                 "to float32 accuracy",
                 sum_gap < fp_tolerance,
                 f"native {result['value_sum']['native']} vs "
                 f"matched {result['value_sum']['matched']} — gap {sum_gap:.3e} exceeds the "
                 f"float32 reduction-order bound, so the layout is not transparent")
    # The decisive seamlessness check: PER-CHANNEL numbers. The grand total above is
    # invariant to how samples are grouped into channels, so it cannot detect a layout error;
    # these can, and they are also checked against Python's own per-channel sums.
    native_sums = np.asarray(result["channel_sums"]["native"], dtype=float)
    matched_sums = np.asarray(result["channel_sums"]["matched"], dtype=float)
    python_sums = trace.astype(np.float64).sum(axis=1)
    per_channel_bound = fp_tolerance / max(n_channels, 1)
    report.check("PER-CHANNEL sums agree between the two layouts",
                 native_sums.shape == python_sums.shape
                 and np.allclose(native_sums, matched_sums, atol=per_channel_bound),
                 "the two layouts group samples into channels differently — this is the "
                 "check the grand total cannot make")
    report.check("PER-CHANNEL sums agree with Python's",
                 np.allclose(native_sums, python_sums, atol=per_channel_bound),
                 f"worst channel differs by "
                 f"{np.max(np.abs(native_sums - python_sums)) if native_sums.shape == python_sums.shape else 'n/a'}")

    report.check("the same per-channel loop gives the same RANGE in both layouts",
                 abs(result["value_extrema"]["native"]
                     - result["value_extrema"]["matched"]) < fp_tolerance,
                 "max and min involve no summation, so these should agree exactly")
    report.check("the axis named 'channel' is found in both, at its own position",
                 result["axis_of_channel"]["native"] == 2
                 and result["axis_of_channel"]["matched"] == 1,
                 f"got {result['axis_of_channel']}")
    exact = float(trace.astype(np.float64).sum())
    report.check("Python's total agrees with Julia's, to float32 summation accuracy",
                 abs(result["value_sum"]["native"] - exact) < fp_tolerance,
                 f"Julia {result['value_sum']['native']:.3f} vs Python {exact:.3f}, "
                 f"bound {fp_tolerance:.3f}")

    # ---- optimisation: the mechanism, then the effect -----------------------
    report.check("under the default layout a channel is CONTIGUOUS in memory",
                 result["channel_stride"]["native"] == 1,
                 f"stride {result['channel_stride']['native']}; the whole point of the "
                 "default is that this is 1")
    report.check("under matched a channel is strided by the channel count",
                 result["channel_stride"]["matched"] == n_channels,
                 f"stride {result['channel_stride']['matched']}")

    for label, native_key, matched_key in (
        ("per-channel sum", "per_channel_sum_native", "per_channel_sum_matched"),
        ("per-channel range", "per_channel_extrema_native", "per_channel_extrema_matched"),
    ):
        fast, slow = seconds[native_key], seconds[matched_key]
        ratio = slow / fast if fast > 0 else float("inf")
        report.note(f"{label:<20} native {fast * 1000:8.1f} ms   "
                    f"matched {slow * 1000:8.1f} ms   = {ratio:5.1f}× ")
        report.check(f"{label}: the default layout is not slower",
                     ratio >= 0.9,
                     f"native was {1 / ratio:.1f}× SLOWER — the default is wrong for this "
                     "machine and should be reconsidered")

    read_ratio = (seconds["read_matched"] / seconds["read_native"]
                  if seconds["read_native"] > 0 else float("inf"))
    report.note(f"{'read':<20} native {seconds['read_native'] * 1000:8.1f} ms   "
                f"matched {seconds['read_matched'] * 1000:8.1f} ms   = {read_ratio:5.1f}× ")
    report.note(f"{'permutedims copy':<20} {seconds['permutedims_copy'] * 1000:8.1f} ms "
                f"— what matched mode pays on every read")
    report.check("the default read is not slower than the copying one",
                 read_ratio >= 0.9,
                 f"native read was {1 / read_ratio:.1f}× slower, which should be impossible: "
                 "it does strictly less work")


# =============================================================================
# 6 · real data, if it is on this machine
# =============================================================================

def real_data(report: Report, tmp: Path) -> None:
    from ephyslink import Session, load_kilosort

    report.start("6 · Real data")
    try:
        sys.path.insert(0, str(ROOT / "projects" / "EXELU-spikes" / "scripts"))
        import EXELU_paths as paths
        root = paths.data_root()
        names = paths.session_names(root)
    except Exception as error:
        report.note(f"no recordings on this machine ({type(error).__name__}); skipped")
        return

    name = names[-1]
    session = load_kilosort(paths.session_dir(name, root), session_id=name)
    report.note(f"{name}: {session.n_spikes:,} spikes, {len(session.arrays)} arrays")

    report.check("spike_times is int64 and sorted",
                 session.array("spike_times").dtype == np.int64
                 and np.all(np.diff(session.array("spike_times")) >= 0))
    report.check("every array declares its axes",
                 all(len(session.dims[n]) == a.ndim for n, a in session.arrays.items()))
    report.check("the sampling rate came from params.py, not a default",
                 session.fs_hz == 30_000.0)

    path = session.save(tmp / "real.h5")
    back = Session.load(path)
    report.check("a real session round-trips byte for byte",
                 all(np.array_equal(back.arrays[n], a) for n, a in session.arrays.items()))
    report.check("a real session's dims round-trip", back.dims == session.dims)
    report.note(f"{path.stat().st_size / 1e6:.1f} MB written and re-read")


# =============================================================================
# 7 · the other suites in this repository
# =============================================================================

def other_suites(report: Report) -> None:
    report.start("7 · Other suites")
    for script in (ROOT / "scripts" / "selftest_preprocessing.py",):
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        report.check(f"{script.name} passed", done.returncode == 0,
                     (done.stdout + done.stderr).strip().splitlines()[-5:] and
                     "\n        ".join((done.stdout + done.stderr).strip().splitlines()[-5:]))


# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the real-data checks")
    parser.add_argument("--verbose", action="store_true", help="print every check")
    arguments = parser.parse_args()

    report = Report(arguments.verbose)
    print(f"ephyslink check · {ROOT}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        try:
            python_roundtrip(report, tmp)
            on_disk_contract(report, tmp / "fixture.h5", "written by Python")
            guards(report, tmp)
            results_export(report, tmp)
            julia_checks(report, tmp, tmp / "fixture.h5", quick=arguments.quick)
            if not arguments.quick:
                real_data(report, tmp)
            other_suites(report)
        except Exception:
            report.check("the check script itself ran to completion", False,
                         traceback.format_exc().replace("\n", "\n        "))

    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
