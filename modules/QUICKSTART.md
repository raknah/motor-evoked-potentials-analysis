# ephyslink — quick start

One recording. One file. Both languages. No conversion step, no "which script made this
`.npy` again", no transposing something and hoping.

```python
session = load_kilosort("/data/2026-04-14_11-46-55")   # Python
session.add_table("units", my_results)
session.save("analysis/session.h5")
```
```julia
s = read_session("analysis/session.h5")                # Julia, same file, same names
s.tables["units"]
```

That's the whole idea. The rest of this page is the useful detail.

---

## Why bother

Most ephys analysis dies of the same thing: a directory of forty files whose relationship to
each other lives only in your head. `spikes_clean_v3_FINAL.npy`. A `.mat` someone exported in
2023. A pickle that only opens in the environment that made it.

A session file is the opposite. It holds the data **and** everything you have worked out
about it — tables, arrays, events, metadata, and a log of the steps that produced them. Six
months later the file tells you what it is. And because both languages read it natively, you
stop choosing a language for the project and start choosing one per *task*.

---

## Setup, once

```bash
pip install numpy scipy pandas matplotlib h5py
julia --project=modules/EphysLink.jl -e 'using Pkg; Pkg.instantiate()'
python check.py          # ~1250 checks, both languages. Should end "ALL CHECKS PASSED"
```

In Python, `modules/` has to be importable. Inside this repo the project scripts do it for
you; standalone:

```python
import sys; sys.path.insert(0, "modules")
from ephyslink import load_kilosort, load_openephys, Session
```

In Julia, point at the package:

```bash
julia --project=modules/EphysLink.jl
```
```julia
using EphysLink
```

---

## The five-minute tour (Python)

```python
from ephyslink import load_kilosort

session = load_kilosort("/data/2026-04-14_11-46-55")
print(session.summary())
```
```
SessionKS  2026-04-14_11-46-55
  fs          30000 Hz
  arrays
    spike_times      (3755190,)      int64    [spike]
    spike_clusters   (3755190,)      int32    [spike]
    templates        (207, 61, 130)  float32  [template, sample, channel]
    …
  tables
    clusters         207 rows × 5 columns
```

Two flavours, same file format:

| | holds | you get |
|---|---|---|
| `SessionKS` | spike-sorted output | `spike_times`, `spike_clusters`, `clusters`, `spikes_of(id)` |
| `SessionOE` | continuous traces (LFP, MEP, EEG) | `continuous`, `n_channels`, `times_s` |

Attach whatever your analysis produces. It travels with the session:

```python
session.add_table("units", unit_table)                      # a DataFrame
session.add_array("psth", matrix, dims=["unit", "bin"])      # an array + its axis names
session.add_events("flicker_40Hz", onset_samples)            # sample indices
session.results["median_ppc_ctx"] = 0.028                    # one number
session.log("responder test", alpha=0.05)                    # what you just did

session.save("analysis/session.h5")
```

Later, or in Julia, or on another machine:

```python
session = Session.load("analysis/session.h5")
session.tables["units"]
session.history          # every step, with its parameters and a timestamp
```

### Loading from raw

```python
load_kilosort("/data/session")                    # finds kilosort4/ automatically
load_openephys("/data/session", decimate=120)     # anti-aliased, → 250 Hz
load_openephys(path, channels=[0, 4, 8], start_s=60, stop_s=120)
```

`continuous.dat` is memory-mapped, never slurped — an hour of 138 channels at 30 kHz is 28 GB.
Ask for more than 4 GB and it stops you rather than swapping your laptop to death.

---

## The same thing in Julia

```julia
using EphysLink

s = read_session("analysis/session.h5")
overview(s) |> print

s.tables["units"]
n_channels(s)

for trace in each_channel(s)          # one channel at a time, zero copy
    # …
end

add_array!(s, "spectrum", S, ["unit", "frequency"])
log!(s, "spectral analysis"; nw = 3)
write_session("analysis/session.h5", s)
```

Python reads that back unchanged. Genuinely — there is a test that does exactly this, three
times in a row, in every combination.

---

## The one concept worth understanding

Ask for axes **by name**, not by number.

```python
session.axis("psth", "unit")        # Python: 0
session.n_along("psth", "bin")      # 240
```
```julia
axis(s, "psth", "unit")             # Julia: 2 (and 1-based)
n_along(s, "psth", "bin")           # 240 — the same number
```

The index differs between languages. **The name never does.** Every array carries its axis
names, so code written with `axis` / `n_along` / `channel` / `each_channel` reads the same in
both and cannot be broken by a transpose.

Why the index differs at all: Python and Julia disagree about which end of an array is the
"fast" one, so the same bytes look reversed. Handling that is the library's job — it is
deterministic, not a guess, and it is tested to death. You just never hard-code a `0` or a
`1`, and it stops being your problem.

**The practical upshot:** in Julia a channel comes back as a contiguous run of memory. That is
free — it is the same bytes Python had — and it is why per-channel work is fast rather than
crawling. `check.py` prints the measured difference on your machine; it is large, and it is
the reason the default is what it is.

---

## Which language for which job

This is the actual payoff, so it is worth being blunt.

### Reach for Python when

- **The library already exists.** `scipy.signal`, `statsmodels`, `scikit-learn`, `spikeinterface`,
  `mne`. Reimplementing a Wilcoxon test or an IIR filter to feel clever is a bad trade.
- **You are exploring.** Notebook, `pandas`, a plot, change one thing, plot again. Nothing
  beats it for the first hour with a new dataset.
- **The work is table-shaped.** `groupby`, joins, tidy wrangling. `DataFrames.jl` is good;
  pandas has twenty years of other people's edge cases already solved.
- **The heavy lifting happens inside numpy anyway.** An FFT or a matrix multiply is C either
  way. Python's overhead is per-*call*, and a vectorised call has almost none.

### Reach for Julia when

- **You have to write the loop yourself.** This is the big one. Spike-train algorithms —
  pairwise phase consistency, cross-correlograms, ISI statistics, template matching — often
  cannot be vectorised without building an enormous intermediate array. In Python you choose
  between a slow loop and a memory blow-up. In Julia the loop *is* the fast version.
- **It is memory-bound and per-channel.** Filtering, reductions, sliding windows over long
  traces. Julia's layout puts each channel contiguous in memory; the difference is not 20 %,
  it is a multiple.
- **You want to iterate on the algorithm, not on the vectorisation.** Writing `for spike in
  spikes` and having it run at C speed changes what you are willing to try.
- **You need real parallelism.** Threads that actually use your cores, without the GIL and
  without pickling data into subprocesses.

### A shape that works well

Load and clean in Python → hand the session to Julia for the inner-loop-heavy computation →
hand it back to Python for statistics and figures. The session is the handoff, and the handoff
costs nothing but a file write.

```
Python   load_kilosort, preprocessing, unit table, QC
   ↓     session.save(...)
Julia    the expensive per-spike / per-channel computation
   ↓     write_session(...)
Python   statistics, plots, the results export
```

### And the honest part

For most of what you will do day to day, **either is fine and Python is more convenient**.
Julia earns its place on the specific step that is slow, not across a whole project. The point
of this tool is that you no longer have to decide up front — you can move one step across the
line when it starts to hurt, and move it back if it does not help.

---

## Recipes

**Continue an analysis you left last week**
```python
session = Session.load("analysis/session.h5")
print(session.history)              # what you already did, and with what parameters
```

**Ship results without shipping 44 GB of spikes**
```python
session.export_results("results/2026-04-14.h5")     # ~1 MB: derived tables + summary numbers
```
It knows what to drop because the loader recorded what it produced — anything you added
afterwards is a result by definition.

**Compare across sessions**
```python
from ephyslink import read_results
comparison = read_results("output/_results")
comparison.tables["responders"]     # every unit from every session, stacked
comparison.summary                  # one row per session
```
No raw data touched. Fifteen of these fit in memory; fifteen full sessions do not.

**Preprocess a continuous recording**
```python
import preprocessing as pp                       # scripts/preprocessing.py
s = pp.bandpass(load_openephys(path), 0.5, 300, notch_hz=[50])
s = pp.rereference(s, "average")
s = pp.downsample(s, target_fs=1000)
s = pp.epoch_at(s, s.events["ttl1_rising"], pre_ms=50, post_ms=200)
s.save("preprocessed/session.h5")
```
Each step records itself in `history`, so the file explains how it was made.

**Look inside a file that will not load**
```python
from ephyslink import describe_file
print(describe_file("suspicious.h5"))
```

---

## Things that will bite you once

- **Arrays need axis names.** `add_array("psth", m, dims=["unit", "bin"])`. This is not
  bureaucracy — an unlabelled axis is exactly how data gets silently transposed.
- **Strings go in tables, not arrays.** Arrays are numeric or boolean. Text belongs in a
  DataFrame column, which round-trips cleanly; a string array does not.
- **Saving twice to one path overwrites.** Give pipeline stages different filenames if you
  want to keep both.
- **`NaN` in metadata becomes `null`.** Valid JSON has no `NaN`, and silently losing the whole
  metadata block on the Julia side would be much worse.
- **Old `.h5` files from `openephysextract` will not open.** They mixed axis conventions
  within a single file. Re-extract from source.
- **Julia's indices are 1-based and its axis order is reversed.** Use `axis`, `n_along`,
  `channel` and you will not notice.

---

## Where things live

| | |
|---|---|
| `modules/ephyslink/` | the Python side |
| `modules/EphysLink.jl/` | the Julia side |
| `modules/FORMAT.md` | the file format, if you ever need to open one by hand |
| `scripts/` | shared analysis — PSTHs, correlograms, phase locking, preprocessing, stats |
| `check.py` | run it after changing anything in either language |

Every module starts with a plain-English header explaining what it is for and why it works
the way it does. Start there rather than at the code.
