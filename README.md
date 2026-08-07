# Neuroelectrophysiology

Analysis of Open Ephys recordings — LFP, MEPs, EEG, and spike-sorted data — in Python and
Julia, sharing one session object and one file format.

## Layout

```
modules/
  ephyslink/          Python: the Session object and its HDF5 format
  EphysLink.jl/       Julia: the same object, the same format
  ephyslink/FORMAT.md the contract both implementations follow

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
`modules/ephyslink/FORMAT.md` for the full contract.

## Setup

```bash
pip install numpy scipy pandas matplotlib h5py
julia --project=modules/EphysLink.jl -e 'using Pkg; Pkg.instantiate()'
```

Point Julia at the module with `--project=modules/EphysLink.jl`, or
`push!(LOAD_PATH, "modules/EphysLink.jl/src")`.

## Tests

Run these after changing anything in `modules/` or `scripts/`:

```bash
python modules/ephyslink/selftest.py            # session I/O, round-trip, guards
python scripts/selftest_preprocessing.py        # every preprocessing step vs a known answer

# cross-language, run in order
python modules/ephyslink/selftest.py --write-fixture /tmp/fixture_py.h5
julia --project=modules/EphysLink.jl modules/EphysLink.jl/test/roundtrip.jl /tmp/fixture_py.h5
python modules/ephyslink/selftest.py --check-fixture /tmp/fixture_py_jl.h5
```

The cross-language test is the one that matters. A change to one language's reader that is
not mirrored in the other is the failure mode this design exists to prevent.

## Projects

| project | what it is |
|---|---|
| `EXELU-spikes` | 40 Hz visual flicker, spike-sorted probe from visual cortex to dentate gyrus, 5xFAD and WT. See its `log/` |
| `5xFAD-Resting-State` | resting-state EEG |
| `5xFAD-MEPs` | motor evoked potentials |
| `EXELU` | the MATLAB acquisition side |
