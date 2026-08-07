# The ephyslink file format

One HDF5 file holds one recording session and everything an analysis pipeline attaches to
it. Python and Julia both read and write it, and a file written by either must be
byte-for-byte equivalent to one written by the other.

This document is the contract. If the Python and the Julia implementations ever disagree,
this file is right and both are wrong.

---

## 1 · The dimension rule — read this before anything else

This is what broke the previous version, so it is stated first and precisely.

**HDF5 has no concept of row-major or column-major.** It stores a flat byte sequence plus a
shape. What differs is how each language *interprets* that sequence:

- `h5py` writes and reads **C order** (row-major): the last axis varies fastest.
- `HDF5.jl` writes and reads **Fortran order** (column-major): the first axis varies fastest.

The consequence is exact, deterministic, and requires no inspection of the data:

> An array written from Python with logical shape `(a, b, c)` is read by Julia with shape
> `(c, b, a)`. The axes are **reversed**. Always. Every time.

So the rule is:

| direction | what happens |
|---|---|
| Python writes | store the array as-is |
| Julia reads | `permutedims(A, ndims(A):-1:1)` |
| Julia writes | `permutedims(A, ndims(A):-1:1)`, then store |
| Python reads | take the array as-is |

After that, **an array has the same logical axis order in both languages**. A continuous
trace is `(channel, sample)` in Python and `(channel, sample)` in Julia.

The old implementation instead guessed:

```julia
if nrows > ncols && nrows > 1000
    transpose(data)      # "it's probably (samples, channels)"
end
```

That is a coin flip whenever the two axes are of comparable size, and it silently
transposed real data. **Never infer axis order from array shape.** Every array in this
format carries a `dims` attribute naming its axes, in Python logical order, and every
reader is expected to trust that attribute rather than the shape.

### The performance escape hatch

Matching Python's axis order costs Julia a `permutedims` copy on read, and leaves the array
laid out against the grain of column-major memory. For a large continuous recording that
matters. Both readers therefore accept `native=true`, which skips the permute and returns
the reversed axes — with the `dims` attribute reversed to match, so the array stays
self-describing. Default is `native=false`: correctness and cross-language sameness first.

---

## 2 · Layout

```
/                                   root
    attrs:
        format          "ephyslink"                    identifies the file
        format_version  1                              integer
        kind            "OE" | "KS"                     which Session subclass
        id              e.g. "2026-04-14_11-46-55"      session identifier
        experiment      free text
        fs_hz           float                           sampling rate
        source          original path the data came from
        created         ISO-8601 UTC timestamp
        meta_json       JSON object   — arbitrary metadata
        history_json    JSON array    — one entry per pipeline step

/arrays/<name>                      one dataset per array
    attrs:
        dims            list of axis names, in PYTHON logical order
        units           optional, free text
        description     optional, free text

/tables/<name>/<column>             one dataset per column
    group attrs:
        columns         ordered list of column names
        n_rows          integer

/events/<name>                      int64 dataset of sample indices
```

Everything except the root attributes is optional. A session with no tables has no
`/tables` group.

### Why tables are stored column by column

HDF5 compound (record) datasets exist, but h5py and HDF5.jl disagree about string handling
inside them, which is the single most common source of "it loads in Python but not in
Julia". One dataset per column sidesteps it entirely: each column is a plain 1-D array of a
primitive type, and variable-length UTF-8 strings round-trip cleanly in both libraries. The
`columns` attribute preserves the ordering that a column-per-dataset layout would otherwise
lose.

### Why metadata is a JSON string, not HDF5 attributes

HDF5 attributes cannot hold nested structures, `None`/`nothing`, or heterogeneous lists.
Metadata in practice is a nested dict with missing values in it. One JSON string is
readable from both languages with a single well-tested library each, and survives nesting.

---

## 3 · Conventional array names

Nothing enforces these — `arrays` is an open namespace, which is the point: an analysis step
attaches its output to the session and it travels with it. But loaders produce these names,
and code that expects them should find them.

### `kind = "OE"` — continuous data (LFP, MEP, EEG)

| array | dims | dtype | meaning |
|---|---|---|---|
| `continuous` | `["channel", "sample"]` | float32 | the trace, in microvolts |
| `epochs` | `["channel", "sample", "epoch"]` | float32 | after epoching |
| `channel_positions` | `["channel", "axis"]` | float32 | contact geometry, µm |
| `sample_numbers` | `["sample"]` | int64 | hardware sample index, for alignment |

### `kind = "KS"` — spike-sorted data

| array | dims | dtype | meaning |
|---|---|---|---|
| `spike_times` | `["spike"]` | int64 | sample index of each spike, sorted |
| `spike_clusters` | `["spike"]` | int32 | which cluster each spike belongs to |
| `spike_positions` | `["spike", "axis"]` | float32 | estimated position, µm |
| `spike_templates` | `["spike"]` | int32 | which template each spike matched |
| `templates` | `["template", "sample", "channel"]` | float32 | the templates |
| `whitening_mat_inv` | `["channel", "channel_out"]` | float32 | undoes Kilosort's whitening |
| `channel_positions` | `["channel", "axis"]` | float32 | contact geometry, µm |
| `channel_map` | `["channel"]` | int32 | recording-channel index of each sorted channel |

with `tables["clusters"]` holding `cluster_id`, `KSLabel`, `ContamPct`.

---

## 4 · What this format is not

- **Not a replacement for the raw data.** A 130-channel hour-long recording at 30 kHz is
  ~28 GB; that stays in Open Ephys's `continuous.dat` and is memory-mapped, not copied in
  here. `SessionOE` is for data you have already reduced — downsampled LFP, epoched MEPs.
- **Not versioned storage.** Saving twice to the same path overwrites. Give pipeline stages
  different filenames if you want to keep both.
- **Not lazy.** `read_session` loads every array into memory. For the sizes this format is
  meant for that is the right trade; if it stops being, add a lazy accessor rather than
  changing the layout.

---

## 5 · Checking an implementation

`selftest.py` writes a file containing arrays of every rank, a table with string and numeric
columns, nested metadata and events, reads it back, and asserts equality. `test/roundtrip.jl`
does the same from the Julia side and additionally reads the Python-written file and writes
one for Python to read back.

Run both after any change to either implementation. A change to one language's reader that
is not mirrored in the other is the failure mode this format exists to prevent.
