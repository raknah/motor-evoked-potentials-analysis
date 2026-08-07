"""
    EphysLink

The Julia half of `ephyslink`: one session object, two languages, one file.

    using EphysLink

    s = read_session("analysis/2026-04-14.h5")
    overview(s) |> print

    s.arrays["psth"]                    # zero-copy, Julia's own memory layout
    axis(s, "psth", "unit")             # 1-based position of a named axis
    for trace in each_channel(s)        # layout-independent, contiguous by default
        ...
    end
    add_array!(s, "spectrum", S, ["unit", "frequency"])
    log!(s, "spectral analysis"; nw = 3)
    write_session("analysis/2026-04-14.h5", s)   # Python reads it back unchanged

New to this? `modules/QUICKSTART.md` is the tour.

The file format is specified in `../../FORMAT.md`, and the Python implementation is
`modules/ephyslink/`. If the two ever disagree, `FORMAT.md` is right and both are wrong.

The one thing to know: h5py works in C order and HDF5.jl works in Fortran order, so the same
bytes appear with reversed axes. `read_session` reverses them back, so an array has the same
logical axis order in both languages. It does **not** guess from array shape — the previous
implementation did, and silently transposed real data. Every array carries a `dims` attribute
naming its axes; trust that.

`read_session` defaults to `native=true`: zero copy, and *the same bytes and memory locality
as Python* — a `(channel, sample)` C-order array in Python is a `(sample, channel)`
column-major array here, which is the same thing. Pass `native=false` to get Python's literal
index order instead; it costs a `permutedims` copy and measured 26× slower on per-channel
reductions, and buys only that `A[channel, t]` reads the same in both languages.

Either way `dims` describes the array you are holding, so index by name — `axis`, `n_along`,
`channel`, `each_channel` — and the layout stops mattering.
"""
module EphysLink

using Dates
using HDF5
using JSON3

include("session.jl")
include("format.jl")

export Session,
       read_session, write_session, describe_file,
       add_array!, add_table!, add_events!, log!,
       array, table, axis, n_along, seconds, validate, overview,
       slice_along, continuous, n_channels, channel, each_channel, n_samples,
       n_epochs, epoch, each_epoch,
       spike_times, spike_clusters, n_spikes, spikes_of,
       duration_s,
       FORMAT_NAME, FORMAT_VERSION

end # module
