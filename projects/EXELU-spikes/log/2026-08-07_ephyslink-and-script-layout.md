# 2026-08-07 · ephyslink, and the script layout

`openephysextract` and `neuroephys4julia` deleted. `modules/ephyslink/` (Python) and
`modules/EphysLink.jl/` (Julia) implement one format, specified in
`modules/FORMAT.md`.

## Script layout

- Project scripts prefixed **`EXELU_`**, not `EXELU-` — a hyphen cannot appear in a Python
  module name. The notebook aliases them back (`import EXELU_units as units`).
- Every `EXELU_*` module that uses a shared one starts with `import EXELU_paths`, which calls
  `add_script_paths()` on import. No setup step to forget.
- `.idea/*.iml` had one `sourceFolder` pointing at a directory that no longer exists — the
  cause of "not reachable". Now lists `scripts`, `modules`, `projects/EXELU-spikes/scripts`.
- `%autoreload 2` in the notebook.

## The bug that made the old Julia module unusable

`session.jl` inferred axis order from shape: `if nrows > ncols && nrows > 1000, transpose`.
A coin flip whenever the axes are comparable in size; it silently transposed real data.

**No inference was ever needed.** h5py writes C order, HDF5.jl reads Fortran order, so axes
arrive **reversed, always**. Reverse once each way and the order matches. Verified in numpy
for ranks 2–4. Every array now carries a `dims` attribute, and `add_array` refuses one without
it. Old files also mixed conventions — `raw` (6, 32014) channels-first, `data` (3192, 5, 100)
epochs-first, same file.

## Errors

- **E10 · `reject_artifacts` wrote an empty session silently.** A 1000 µV signal against a
  1000 µV peak-to-peak threshold rejected everything and the batch reported success. Now raises,
  naming the median peak-to-peak.
- **E11 · two of my own tests were wrong, not the code.** (a) tested peak-to-peak rejection by
  adding a constant *offset*, which does not change peak-to-peak; (b) compared raw `|rfft|`
  across different signal lengths, so anti-aliasing looked like failure. Both now normalise.

## Decisions

| choice | why |
|---|---|
| reverse axes on read/write, never infer from shape | deterministic consequence of C vs Fortran order |
| `dims` mandatory on every array | an unlabelled axis is unrecoverable once transposed |
| `native=true` escape hatch in Julia | matching Python's order costs a `permutedims` copy |
| tables as one dataset per column | h5py and HDF5.jl disagree about strings in compound datasets |
| metadata as one JSON string | HDF5 attributes cannot hold nesting, `None`, or mixed lists |
| own Open Ephys reader | the format is simpler than the wrapper; drops a dependency |
| `continuous.dat` memory-mapped | an hour of 138 ch at 30 kHz is 28 GB |
| Julia package named `EphysLink.jl` | macOS is case-insensitive: `EphysLink` and `ephyslink` were the same directory. `.jl` is the standard suffix anyway |
| Julia overview function is `overview`, not `summary` | `Base.summary` exists |
| old `.h5` refused, not adapted | they carry the mixed conventions above |

## Preprocessing

Ported to `scripts/preprocessing.py`, each checked against a synthetic signal: detrend,
bandpass + notch, re-reference, surface Laplacian, downsample, epoch, event-triggered epoch,
artifact rejection, standardise.

**Not ported: ASR, ICA, EOG regression, interpolation, bad-channel detection** — heuristics
whose output cannot be checked without ground truth, and the old code was untested. In git
history; `mne` implements all five. Torch removed entirely.

## Verified

Python round-trip (ranks 1–3, tables, events, nested meta, guards) · real Kilosort session
44 MB round-trip byte-identical · OE reader against synthetic sines (µV, TTL rebasing,
decimation, memory guard) · `simple_extract` end to end (8 Hz survives at 175 µV, 50 Hz to
0.03 µV) · EXELU notebook 49 s.

## Layout modes — resolved 2026-08-07, after the Julia tests passed

`read_session` now defaults to **`native=true`**.

`native=true` is not a compromise layout: it is *the same bytes and the same memory locality
as Python*. A `(channel, sample)` C-order array in Python is a `(sample, channel)`
column-major array in Julia — identical physical layout, reversed index order, zero copy.

`native=false` materialises `(channel, sample)` column-major, which makes one channel a
strided row. Summing each channel of a 1 GB `(130, 2 000 000)` block: **53 ms native vs
1388 ms matched, 26×**, plus 0.6 s for the copy and double peak memory. It buys only that
`A[channel, t]` reads the same in both languages.

Added `slice_along`, `channel`, `each_channel`, `n_channels`, `epoch`, `each_epoch`,
`n_epochs` — axis-name based, identical results in both modes, zero-copy views.

## Testing — one command

```
python check.py
```

Runs the Python suites, then drives Julia itself. ~180 checks on the Python side, ~1000 more
across the Julia sections. Exit 0 or a list of what failed.

- **5b · every variant of the back and forth.** Julia reads each file in both layout modes and
  writes both back; Python copies everything between rounds. Three rounds = every mode sequence
  to depth three, 14 Julia-written files, each compared against the **original** rather than
  against the previous hop — a single round trip passes even when both sides are wrong in
  cancelling ways.
- **5c · performance, measured in real Julia.** Per-channel reduction under each layout, the
  read, the `permutedims` copy. Fails if the default layout is slower. Runs the *same*
  per-channel loop under both layouts and fails if the answers differ — that is what
  "seamless" has to mean, and it is checked rather than asserted.

### First real run: 5 failures, 1202 passes — none of them a library bug

- **E16 · my assertion, not the format.** h5py writes variable-length UTF-8 attributes
  (→ `str`); HDF5.jl writes fixed-length ASCII (→ `bytes`). The library normalises both, my
  check used a bare `==`. Fixed, and there is now an explicit check that a string attribute is
  readable whichever language wrote it.
- **E17 · `roundtrip.jl` hard-coded `matrix2d/3d/4d`** from the old `selftest.py` fixture, but
  `check.py` passes its own richer one → 9 errors, 2 failures. Its fixture section is now
  generic: it verifies the invariant for whatever arrays are present, and leaves value checking
  to `bridge.jl` + `check.py`, where Python owns the expected answers.
- **E18 · float32 reduction-order noise read as a layout failure.** Native and matched summed
  to −2597.30397 vs −2597.30226 — a gap of 1.7e−6 on Σ|x| ≈ 2.6e7, i.e. 7e−14 relative. Both
  sum the same values in the same order; Julia's `sum` just blocks a contiguous view
  differently from a strided one. My tolerance was a flat 1e−3. Now bounded by
  eps32 · log2(n) · Σ|x|, and the observed gap is printed so it can be judged.
- **E19 · the benchmark said the default read was 1.7× slower**, which is impossible — it does
  strictly less work. Single-shot timing of a 128 MB allocation, with GC landing wherever it
  liked. Now minimum-of-five with `GC.gc()` before each run, and the methodology is reported
  alongside the numbers.
- **E20 · the seamlessness check was a tautology.** `per_channel_sum` totals every element, so
  it is invariant to *how* samples are grouped into channels — it could not detect a layout
  error at all, only a value error. Replaced with per-channel sums compared elementwise
  between modes and against Python's. Verified by deliberately rolling the grouping by one
  channel: the old check passed, the new one fails.

Two errors found while writing it, both silent and both fixed:

- **E14 · NaN in metadata produced invalid JSON.** `json.dumps` emits bare `NaN`/`Infinity`,
  which is a Python extension, not JSON. Julia's JSON3 rejects the document, `_read_json`
  caught the exception and returned the fallback — so **all metadata would have vanished on
  the Julia side without an error**. `summarise()` produces NaN routinely (an empty group's
  median). Non-finite values now become JSON `null`, and `allow_nan=False` makes anything
  missed raise at write time.
- **E15 · a bool array became an HDF5 enum**, which HDF5.jl does not read back as `Bool`.
  Stored as int8 with the original dtype recorded. String and object arrays are now refused at
  `add_array()`, where the mistake is, rather than at save time.

Design points worth keeping:

- **Values are compared element by element in a defined order**, not shapes. `bridge.jl` emits
  each array's elements in Python's C order; a shape can match while contents are scrambled.
- **The file is checked with raw h5py**, independent of both readers — a round trip can pass
  while the file is wrong if both sides are wrong in the same way.
- **The Julia side makes no assertions.** It reports what it saw; every judgement is in
  `check.py`, so "correct" is defined in exactly one place.
- Cases are chosen to break things: square shapes (where a reversal is invisible in the shape),
  singleton axes, a zero-length axis, ranks 1–5, all ten portable dtypes.

## Open

- **O9 · Julia runs and passed** at 41 tests. Everything since — ranks 4 and 5, the arrays
  built in Julia, the accessors, the flipped default, `check.py` and `bridge.jl` — is
  **written but not executed against a real Julia**. The assertions were verified against a
  faithful numpy emulation of `format.jl` (366 checks, 0 failures), which proves they express
  the spec correctly but cannot catch a typo in the Julia source. `python check.py` settles it.
- **O10** the 14 old `.h5` in `5xFAD-Resting-State` are unreadable; re-extract when needed.
- **O11** `5xFAD-MEPs` and `5xFAD-Resting-State` notebooks still import `openephysextract` and
  will fail. Not ported — their analyses were not reviewed.
