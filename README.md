# Neuroelectrophysiology

Analysis of Open Ephys recordings — LFP, MEPs, EEG, and spike-sorted data — in Python and
Julia, sharing one session object and one file format.

## Layout

**New here? Start with [`modules/QUICKSTART.md`](modules/QUICKSTART.md)** — what the session
object is, how to use it from either language, and which language to reach for when.

```
modules/
  QUICKSTART.md       start here
  ephyslink/          Python: the Session object and its HDF5 format
  EphysLink.jl/       Julia: the same object, the same format
  FORMAT.md           the contract both implementations follow

scripts/              shared analysis code, reusable on any dataset
  preprocessing.py      filtering, re-referencing, epoching, artifact rejection
  spiketrains.py        per-unit spike access, PSTH windowing primitives
  psth.py               peri-stimulus histograms, per-trial standardisation
  correlograms.py       autocorrelograms
  contamination.py      Kilosort ContamPct, Hill f_p
  waveforms.py          waveform shape, burst/theta index, cell type
  entrainment.py        phase locking, vector strength, PPC, modulation depth
  responders.py         which units changed rate
  statstools.py         FDR, paired tests, permutation
  plotstyle.py          shared figure style
  simple_extract.py     batch: Open Ephys → preprocessed session files

projects/<name>/      one directory per study
  scripts/              that study's specifics, prefixed with the project name
  notebooks/            the worked analysis
  log/                  decisions, errors found and fixed, open questions
```

The split is by **reusability**. A PSTH is a PSTH, so `scripts/psth.py` is shared. A
cortex/hippocampus boundary defined by a theta channel belongs to one experiment, so
`projects/EXELU-spikes/scripts/EXELU_regions.py` is not.

Nothing here is an installable package. Everything is plain `.py` and `.jl` files you can
open and read. Imports resolve via `sys.path`, set up by the project's `*_paths.py`.

## The session object

One HDF5 file holds a recording and everything an analysis attaches to it, readable and
writable from both languages with the same axis order.

```python
from ephyslink import load_kilosort, Session

session = load_kilosort("/data/2026-04-14_11-46-55")
session.add_table("units", unit_table)
session.add_array("psth", matrix, dims=["unit", "bin"])
session.log("responder test", alpha=0.05)
session.save("analysis/2026-04-14.h5")
```

```julia
using EphysLink

s = read_session("analysis/2026-04-14.h5")
s.arrays["psth"]              # same axis order as Python
axis(s, "psth", "unit")       # 1-based position of a named axis
add_array!(s, "spectrum", S, ["unit", "frequency"])
write_session("analysis/2026-04-14.h5", s)
```

Two types: `SessionOE` for continuous data, `SessionKS` for spike-sorted output. Same file
format, distinguished by an attribute.

**Axis order is never guessed.** h5py works in C order and HDF5.jl in Fortran order, so the
same bytes appear with reversed axes — deterministically, with no need to inspect the data.
Each reader reverses once, and every array carries a `dims` attribute naming its axes. See
`modules/FORMAT.md` for the full contract.

## Setup

```bash
pip install numpy scipy pandas matplotlib h5py
julia --project=modules/EphysLink.jl -e 'using Pkg; Pkg.instantiate()'
```

Point Julia at the module with `--project=modules/EphysLink.jl`, or
`push!(LOAD_PATH, "modules/EphysLink.jl/src")`.

## Tests

One command. It runs the Python suites, then drives Julia itself — nothing to chain by hand.

```bash
python check.py
```

Exit code 0 means every check passed. `--quick` skips the real-data checks; `--verbose` prints
every individual check rather than only failures. If Julia is not on your PATH, set
`JULIA=/path/to/julia`.

What it covers:

| section | what it proves |
|---|---|
| 1 Python round-trip | every rank 1–5, square and singleton shapes, a zero-length axis, all ten dtypes, unicode, NaN, empty tables and events |
| 2 On-disk contract | opens the file with raw h5py and checks it against `FORMAT.md` — catches a file that is wrong even when both readers agree |
| 3 Guards | wrong axis count, missing dims, transposed `continuous`, string arrays, legacy files — all refused |
| 4 Results export | derived tables kept, raw arrays dropped, stacking across sessions |
| 5 Julia | Julia's in-memory view compared **element by element in a defined order**, both layout modes, files Julia wrote read back in Python, and an array *built* in Julia |
| 5b Every variant | the tree of layout-mode sequences to depth three — 14 Julia-written files, a Python hop between each round, all compared against the **original** |
| 5c Performance | measures in real Julia that the default layout is faster **and** that the same code gives the same answers in both layouts |
| 6 Real data | a real Kilosort session round-trips, if the recordings are on this machine |

Sections 5b and 5c are the ones that matter.

**5b** exists because a single round trip passes even when both sides are wrong in cancelling
ways. Walking every sequence of layout modes, comparing each file back to the original at every
depth, does not.

**5c** exists because "optimised for Julia" is a claim, and a claim should be measured on the
machine it is claimed about. It reports the actual timings — per-channel reduction under each
layout, the read, the `permutedims` copy — and fails if the default is slower. It also runs the
*same* per-channel loop under both layouts and fails if the answers differ, which is what
"seamless" has to mean.

## Projects

| project | what it is |
|---|---|
| `EXELU-spikes` | 40 Hz visual flicker, spike-sorted probe from visual cortex to dentate gyrus, 5xFAD and WT. See its `log/` |
| `5xFAD-Resting-State` | resting-state EEG |
| `5xFAD-MEPs` | motor evoked potentials |
| `EXELU` | the MATLAB acquisition side |
