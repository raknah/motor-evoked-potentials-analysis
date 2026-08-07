#!/usr/bin/env python3
"""
Self-test for the Python side of ephyslink, and the fixture generator for the Julia side.

    python modules/ephyslink/selftest.py                 # Python round-trip
    python modules/ephyslink/selftest.py --write-fixture /tmp/ephyslink_fixture.h5

The fixture contains arrays of rank 1, 2 and 3 with deliberately *non-square* shapes, so a
transposition cannot hide. That is the whole point: the previous implementation guessed axis
order from shape, and a square-ish test array would have let it pass.

To check both languages agree, run in order:

    python modules/ephyslink/selftest.py --write-fixture /tmp/fixture_py.h5
    julia --project=modules/EphysLink.jl modules/EphysLink.jl/test/roundtrip.jl /tmp/fixture_py.h5
    python modules/ephyslink/selftest.py --check-fixture /tmp/fixture_jl.h5
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ephyslink import Session, SessionKS, SessionOE, describe_file  # noqa: E402

# Non-square on every axis, so a transpose changes the shape and cannot pass unnoticed.
SHAPE_2D = (4, 7)
SHAPE_3D = (3, 5, 11)
SHAPE_4D = (2, 3, 5, 7)


def build_fixture() -> SessionKS:
    rng = np.random.default_rng(0)
    session = SessionKS(id="fixture", fs_hz=30_000.0, experiment="ephyslink selftest",
                        source="synthetic")

    session.add_array("spike_times", np.sort(rng.integers(0, 10**7, 500)).astype(np.int64),
                      dims=["spike"])
    session.add_array("spike_clusters", rng.integers(0, 9, 500).astype(np.int32),
                      dims=["spike"])
    session.add_array("matrix2d", np.arange(np.prod(SHAPE_2D), dtype=np.float32).reshape(SHAPE_2D),
                      dims=["channel", "sample"], units="µV", description="rank-2 probe")
    session.add_array("matrix3d", np.arange(np.prod(SHAPE_3D), dtype=np.float32).reshape(SHAPE_3D),
                      dims=["channel", "sample", "epoch"])
    # rank 4 as well: the reversal is general, so the test of it should be too
    session.add_array("matrix4d", np.arange(np.prod(SHAPE_4D), dtype=np.float32).reshape(SHAPE_4D),
                      dims=["channel", "sample", "epoch", "band"])

    session.add_table("clusters", pd.DataFrame({
        "cluster_id": np.arange(9, dtype=np.int32),
        "KSLabel": ["good", "mua"] * 4 + ["good"],
        "ContamPct": np.round(rng.uniform(0, 200, 9), 2),
    }))
    session.add_events("flicker_40Hz", np.arange(0, 10_000, 750, dtype=np.int64))
    session.meta.update({
        "animal": "XU20",
        "genotype": "5xFAD",
        "nested": {"landmarks": {"thetaCh": 54, "CA1Ch": 69}, "bad_channels": []},
        "missing": None,
    })
    session.log("build_fixture", seed=0)
    return session


def _assert_equal(a: Session, b: Session, context: str) -> None:
    assert a.kind == b.kind, f"{context}: kind {a.kind} != {b.kind}"
    assert a.id == b.id, f"{context}: id"
    assert a.fs_hz == b.fs_hz, f"{context}: fs_hz"

    assert set(a.arrays) == set(b.arrays), (
        f"{context}: arrays differ — only in first {set(a.arrays) - set(b.arrays)}, "
        f"only in second {set(b.arrays) - set(a.arrays)}"
    )
    for name, values in a.arrays.items():
        other = b.arrays[name]
        assert values.shape == other.shape, (
            f"{context}: array '{name}' shape {values.shape} != {other.shape}. "
            "If these are reverses of each other, the C-order/Fortran-order reversal was "
            "applied an odd number of times."
        )
        assert np.array_equal(values, other), f"{context}: array '{name}' values differ"
        assert a.dims[name] == b.dims[name], (
            f"{context}: array '{name}' dims {a.dims[name]} != {b.dims[name]}"
        )

    assert set(a.tables) == set(b.tables), f"{context}: tables differ"
    for name, frame in a.tables.items():
        other = b.tables[name]
        assert list(frame.columns) == list(other.columns), (
            f"{context}: table '{name}' column order {list(frame.columns)} != {list(other.columns)}"
        )
        for column in frame.columns:
            left, right = frame[column].to_numpy(), other[column].to_numpy()
            if left.dtype.kind in "fc":
                assert np.allclose(left, right), f"{context}: table '{name}'.{column}"
            else:
                assert np.array_equal(left.astype(str), right.astype(str)), \
                    f"{context}: table '{name}'.{column}"

    assert set(a.events) == set(b.events), f"{context}: events differ"
    for name, samples in a.events.items():
        assert np.array_equal(samples, b.events[name]), f"{context}: events '{name}'"

    assert a.meta.get("animal") == b.meta.get("animal"), f"{context}: meta"
    assert a.meta.get("nested") == b.meta.get("nested"), f"{context}: nested meta"


def python_roundtrip() -> None:
    original = build_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.h5"
        original.save(path)
        reloaded = Session.load(path)
        _assert_equal(original, reloaded, "python round-trip")
        assert isinstance(reloaded, SessionKS), "kind attribute did not select SessionKS"
    print("PASS  python round-trip: arrays, dims, tables, events, nested metadata")


def kind_dispatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        oe = SessionOE(id="oe", fs_hz=1000.0)
        oe.add_array("continuous", np.zeros((4, 7), np.float32), dims=["channel", "sample"])
        path = Path(tmp) / "oe.h5"
        oe.save(path)
        assert isinstance(Session.load(path), SessionOE)
        assert Session.load(path).n_channels == 4
    print("PASS  kind dispatch: an OE file loads as SessionOE with working accessors")


def guards() -> None:
    session = SessionOE(id="x", fs_hz=1000.0)
    try:
        session.add_array("bad", np.zeros((3, 4)), dims=["only_one"])
    except ValueError:
        pass
    else:
        raise AssertionError("an array with the wrong number of axis names was accepted")

    session.arrays["sneaky"] = np.zeros((3, 4))          # bypass add_array
    try:
        session.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("validate() accepted an array with no dims")

    del session.arrays["sneaky"]
    session.add_array("continuous", np.zeros((7, 4), np.float32), dims=["sample", "channel"])
    try:
        session.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("SessionOE accepted a transposed 'continuous'")
    print("PASS  guards: unlabelled axes and a transposed 'continuous' are both rejected")


def axis_lookup() -> None:
    session = build_fixture()
    assert session.axis("matrix3d", "channel") == 0
    assert session.axis("matrix3d", "epoch") == 2
    assert session.n_along("matrix3d", "epoch") == SHAPE_3D[2]
    try:
        session.axis("matrix3d", "frequency")
    except KeyError:
        pass
    else:
        raise AssertionError("axis() invented an axis that does not exist")
    print("PASS  axis lookup by name, and a clear error for an axis that is not there")


def results_export() -> None:
    """The export must keep every derived table and drop every source array."""
    session = build_fixture()
    session.mark_source()                       # everything so far is raw input

    session.add_table("responders", pd.DataFrame({
        "unit": [1, 2, 3], "condition": ["a", "a", "b"], "q": [0.01, 0.5, 0.2],
    }))
    session.add_array("psth", np.zeros((3, 10), np.float32), dims=["unit", "bin"])
    session.results["n_responders"] = 1
    session.results["median_ppc"] = 0.0277

    with tempfile.TemporaryDirectory() as tmp:
        path = session.export_results(Path(tmp) / "results.h5")
        loaded = Session.load(path)

        assert loaded.kind == "RESULTS", loaded.kind
        assert set(loaded.tables) == {"responders"}, (
            f"expected only the derived table, got {sorted(loaded.tables)} — a source table "
            "leaked into the results file"
        )
        assert not loaded.arrays, f"arrays should be excluded by default, got {sorted(loaded.arrays)}"
        assert loaded.results["n_responders"] == 1
        assert loaded.results["median_ppc"] == 0.0277
        assert loaded.meta["animal"] == "XU20", "metadata did not survive the export"
        assert len(loaded.tables["responders"]) == 3

        # opting an array in must work
        path2 = session.export_results(Path(tmp) / "with_array.h5", include_arrays=["psth"])
        assert Session.load(path2).arrays["psth"].shape == (3, 10)

        # and a session with nothing derived must say so rather than write an empty file
        bare = build_fixture(); bare.mark_source()
        try:
            bare.export_results(Path(tmp) / "empty.h5")
        except ValueError:
            pass
        else:
            raise AssertionError("exported a results file with no results in it")

        # read_results stacks across sessions
        from ephyslink import read_results
        stacked = read_results(Path(tmp), pattern="results.h5")
        assert list(stacked.tables["responders"].columns)[0] == "session"
        assert stacked.summary.loc["fixture", "n_responders"] == 1
    print("PASS  results export: derived tables kept, source arrays dropped, summary stacked")


def julia_layout_spec() -> None:
    """Exercise the dimension rule against a numpy model of what HDF5.jl does.

    This is a **specification** test, not a test of `format.jl` — it cannot catch a typo in the
    Julia source. What it does catch is a change to the *rule* that is made on one side and not
    the other, and it runs without Julia installed, so it runs everywhere.

    The invariant, in one line: **`dims[i]` describes `size(A, i)` in whichever language is
    holding the array.** Every path below must preserve it.
    """
    def hdf5jl_read(buffer, python_shape, dtype=np.float32):
        """HDF5.jl: the same bytes, shape reversed, Fortran order."""
        return np.frombuffer(buffer, dtype=dtype).reshape(python_shape[::-1], order="F")

    def reverse_axes(a):
        return a if a.ndim <= 1 else np.transpose(a, axes=range(a.ndim)[::-1])

    def julia_read(buffer, python_shape, dims, native):
        a = hdf5jl_read(buffer, python_shape)
        return (a, list(reversed(dims))) if native else (reverse_axes(a), list(dims))

    def julia_write(a, dims, native):
        stored = a if native else reverse_axes(a)
        return (np.asfortranarray(stored).tobytes(order="F"),
                stored.shape[::-1],
                list(reversed(dims)) if native else list(dims))

    cases = [((4, 7), ["channel", "sample"]),
             (SHAPE_3D, ["channel", "sample", "epoch"]),
             (SHAPE_4D, ["channel", "sample", "epoch", "band"]),
             ((500,), ["spike"])]

    for shape, dims in cases:
        python = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        buffer = python.tobytes(order="C")

        matched, matched_dims = julia_read(buffer, shape, dims, native=False)
        assert matched.shape == shape and matched_dims == dims, f"matched read of {shape}"
        assert np.array_equal(matched, python), f"matched read changed values, {shape}"

        native, native_dims = julia_read(buffer, shape, dims, native=True)
        assert native.shape == shape[::-1] and native_dims == dims[::-1], f"native read of {shape}"
        for name in dims:
            assert native.shape[native_dims.index(name)] == python.shape[dims.index(name)], (
                f"native read broke the name-to-length pairing for '{name}' in {shape}"
            )

        for source, source_dims, flag in ((matched, matched_dims, False), (native, native_dims, True)):
            buf, back_shape, back_dims = julia_write(source, source_dims, flag)
            assert back_shape == shape and back_dims == dims, (
                f"julia write (native={flag}) of {shape} gave python {back_shape}, {back_dims}"
            )
            assert np.array_equal(
                np.frombuffer(buf, np.float32).reshape(back_shape, order="C"), python
            ), f"julia write (native={flag}) changed values, {shape}"

    # the path nothing tested before: an array BUILT in Julia inside a native session
    built = np.arange(int(np.prod(SHAPE_3D)), dtype=np.float32).reshape(SHAPE_3D, order="F")
    buf, python_shape, labels = julia_write(built, ["p", "q", "r"], native=True)
    assert python_shape == SHAPE_3D[::-1] and labels == ["r", "q", "p"]
    again, again_dims = julia_read(buf, python_shape, labels, native=True)
    assert np.array_equal(again, built) and again_dims == ["p", "q", "r"]

    print("PASS  dimension rule holds for ranks 1-4, both directions, both layout modes,")
    print("      including an array built in Julia inside a natively-read session")


def legacy_rejected() -> None:
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=np.zeros((6, 100), np.float32))
            f.attrs["session"] = "2023-08-25_15-32-14"
        try:
            Session.load(path)
        except ValueError as error:
            assert "not an ephyslink file" in str(error)
        else:
            raise AssertionError("an old openephysextract file was accepted")
    print("PASS  an old openephysextract file is refused with a message that says why")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixture", type=Path, default=None)
    parser.add_argument("--check-fixture", type=Path, default=None)
    arguments = parser.parse_args()

    if arguments.write_fixture:
        path = build_fixture().save(arguments.write_fixture)
        print(f"fixture written to {path}\n")
        print(describe_file(path))
        return 0

    if arguments.check_fixture:
        expected = build_fixture()
        actual = Session.load(arguments.check_fixture)
        _assert_equal(expected, actual, f"julia → python ({arguments.check_fixture})")
        print(f"PASS  {arguments.check_fixture} written by Julia matches the Python fixture")
        return 0

    python_roundtrip()
    kind_dispatch()
    guards()
    axis_lookup()
    results_export()
    julia_layout_spec()
    legacy_rejected()
    print("\nAll Python-side checks passed. For the cross-language check, see the header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
