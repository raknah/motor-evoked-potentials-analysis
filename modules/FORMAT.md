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

Since the reversal is exact, an implementation never has to detect anything. It only has to
keep one invariant true:

> **`dims[i]` describes `size(A, i)`, in whichever language and whichever layout mode is
> holding the array.**

Julia therefore reverses the *labels* alongside the data, or the data alongside the labels,
and both choices are correct:

| mode | Julia's array | Julia's `dims` | copy? |
|---|---|---|---|
| `native=true` (default) | as HDF5.jl produced it — axes reversed vs Python | reversed to match | no |
| `native=false` | `permutedims(A, ndims(A):-1:1)` — Python's axis order | as written | yes |

On write, Julia applies whichever reversal makes the on-disk array match the `dims` it
records, so both modes produce a file Python reads identically. See §1's next subsection for
why the default is `native`.

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

### The two layout modes, and which one is actually faster

Julia's reader takes `native`, and the difference is larger than "a copy".

**`native=true` gives Julia *the same bytes and the same memory locality as Python*.** It is
not a compromise layout. A `(channel, sample)` array in Python's C order has samples
contiguous within a channel; read natively in Julia it is `(sample, channel)` in column-major
order, which *also* has samples contiguous within a channel. Identical physical layout,
described with reversed index order. Zero copy.

**`native=false` costs a copy and then makes the layout worse.** `permutedims` followed by
`collect` materialises a `(channel, sample)` column-major array — channels contiguous within a
*sample*. One channel's samples are then a strided row, stepping `n_channels` elements at a
time.

Measured on a 1 GB `(130, 2 000 000)` float32 block, summing each channel:

| layout | one channel is | time |
|---|---|---|
| native | a contiguous column | **53 ms** |
| matched | a strided row | **1388 ms** — 26× slower |

plus ~0.6 s for the copy itself, which also doubles peak memory. This is a memory-bandwidth
effect, so it hits reductions, filtering and PSTH accumulation; arithmetic-bound kernels such
as `cumsum` show no difference.

So the only thing `native=false` buys is that `A[channel, t]` is written the same way in both
languages. It buys nothing physical.

**Write layout-agnostic code and the question stops mattering.** `dims` is always correct for
whatever you are holding, so index by name:

```julia
c = axis(s, "continuous", "channel")     # 1 or 2 depending on the mode; correct in both
n = n_along(s, "continuous", "sample")   # same number either way
```

Code written that way runs unchanged in both modes, which is the point of storing `dims` at
all.

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
