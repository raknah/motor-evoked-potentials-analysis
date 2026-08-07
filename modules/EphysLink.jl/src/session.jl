# The Session object in Julia. Mirrors modules/ephyslink/session.py field for field.
#
# One mutable struct rather than a type hierarchy: `kind` says whether it came from continuous
# data ("OE") or a spike sorter ("KS"), and the accessors below check it. A Julia subtype per
# kind would buy nothing here — the file format is identical, and dispatch on a string field
# is enough for two cases.

using Dates

"""
    Session

One recording plus everything an analysis pipeline has attached to it.

Fields:

- `kind`          `"OE"` (continuous) or `"KS"` (spike-sorted)
- `id`            session identifier
- `experiment`    free text
- `fs_hz`         sampling rate
- `source`        where the data originally came from
- `arrays`        name → array. Axis order matches Python unless `native` is true
- `dims`          name → axis names, describing `arrays` as they currently are
- `array_meta`    name → units / description
- `tables`        name → (column name → vector)
- `table_columns` name → column order, which a Dict does not preserve
- `events`        name → sample indices
- `meta`          arbitrary metadata
- `history`       one entry per pipeline step
- `native`        true if arrays are in HDF5-native (reversed) orientation
"""
mutable struct Session
    kind::String
    id::String
    experiment::String
    fs_hz::Float64
    source::String

    arrays::Dict{String,Any}
    dims::Dict{String,Vector{String}}
    array_meta::Dict{String,Dict{String,String}}

    tables::Dict{String,Dict{String,Any}}
    table_columns::Dict{String,Vector{String}}

    events::Dict{String,Vector{Int64}}

    meta::Any
    history::Any

    native::Bool
end

"""
    Session(kind, id, fs_hz; experiment="", source="")

An empty session, for building one up in Julia rather than loading it.
"""
Session(kind::AbstractString, id::AbstractString, fs_hz::Real;
        experiment::AbstractString = "", source::AbstractString = "") =
    Session(String(kind), String(id), String(experiment), Float64(fs_hz), String(source),
            Dict{String,Any}(), Dict{String,Vector{String}}(),
            Dict{String,Dict{String,String}}(),
            Dict{String,Dict{String,Any}}(), Dict{String,Vector{String}}(),
            Dict{String,Vector{Int64}}(),
            Dict{String,Any}(), Any[], false)


# ---- attaching things -------------------------------------------------------

"""
    add_array!(s, name, A, dims; units=nothing, description=nothing)

Attach an array. `dims` is required — an unlabelled axis is how a session gets silently
transposed between languages, so the format does not allow one.
"""
function add_array!(s::Session, name::AbstractString, A::AbstractArray,
                    dims::AbstractVector{<:AbstractString};
                    units = nothing, description = nothing)
    length(dims) == ndims(A) || error(
        "array \"$(name)\" has $(ndims(A)) axes but $(length(dims)) names were given: $(dims)"
    )
    s.arrays[String(name)] = A
    s.dims[String(name)] = String.(collect(dims))
    extra = Dict{String,String}()
    units === nothing || (extra["units"] = String(units))
    description === nothing || (extra["description"] = String(description))
    isempty(extra) || (s.array_meta[String(name)] = extra)
    return s
end

"""
    add_table!(s, name, columns; order=collect(keys(columns)))

Attach a table as a `Dict` of column name → vector. `order` fixes the column order, which a
Dict does not preserve and which Python's DataFrame will otherwise show scrambled.
"""
function add_table!(s::Session, name::AbstractString, columns::AbstractDict;
                    order::AbstractVector{<:AbstractString} = collect(keys(columns)))
    lengths = unique(length(v) for v in values(columns))
    length(lengths) == 1 || error("table \"$(name)\" has columns of differing length: $(lengths)")
    s.tables[String(name)] = Dict{String,Any}(String(k) => v for (k, v) in columns)
    s.table_columns[String(name)] = String.(collect(order))
    return s
end

add_events!(s::Session, name::AbstractString, samples) =
    (s.events[String(name)] = Vector{Int64}(samples); s)

"""
    log!(s, step; params...)

Record a pipeline step. Saved with the session, so a file explains itself.
"""
function log!(s::Session, step::AbstractString; params...)
    push!(s.history, Dict{String,Any}(
        "step" => String(step),
        "params" => Dict{String,Any}(String(k) => v for (k, v) in params),
        "time" => string(Dates.now(Dates.UTC)),
    ))
    return s
end


# ---- reading things ---------------------------------------------------------

function array(s::Session, name::AbstractString)
    haskey(s.arrays, name) ||
        error("no array \"$(name)\". Available: $(sort(collect(keys(s.arrays))))")
    return s.arrays[name]
end

function table(s::Session, name::AbstractString)
    haskey(s.tables, name) ||
        error("no table \"$(name)\". Available: $(sort(collect(keys(s.tables))))")
    return s.tables[name]
end

"""
    axis(s, array_name, axis_name) -> Int

Position of a named axis, **1-based** as Julia expects. Use this instead of hard-coding an
index: it is the difference between code that survives a transposed array and code that does
not.
"""
function axis(s::Session, array_name::AbstractString, axis_name::AbstractString)
    names = get(s.dims, array_name, String[])
    index = findfirst(==(axis_name), names)
    index === nothing &&
        error("array \"$(array_name)\" has axes $(names), not \"$(axis_name)\"")
    return index
end

n_along(s::Session, array_name::AbstractString, axis_name::AbstractString) =
    size(array(s, array_name), axis(s, array_name, axis_name))

"""
    seconds(s, samples)

Sample indices to seconds. The one place the sampling rate is divided by.
"""
seconds(s::Session, samples) = samples ./ s.fs_hz


# ---- kind-specific convenience ---------------------------------------------

_require(s::Session, kind::AbstractString, what::AbstractString) =
    s.kind == kind || error("$(what) is only defined for kind=\"$(kind)\" sessions; this is \"$(s.kind)\"")

"""
    slice_along(s, array_name, axis_name, i) -> SubArray

A zero-copy view with the named axis fixed at `i`, whatever position that axis occupies.

This is what makes code layout-independent. `slice_along(s, "continuous", "channel", 3)` is
one channel's samples in both `native` and matched mode, with no `if` at the call site and no
hard-coded index.

It is a *view*: nothing is copied, and writing through it writes into the session.

Speed, stated because it is the whole reason `native` defaults to true: in native mode the
returned view of a channel is a **contiguous column**. In matched mode it is a strided row, so
it is correct but slow. Same code, same answer, different memory behaviour.
"""
function slice_along(s::Session, array_name::AbstractString,
                     axis_name::AbstractString, i::Integer)
    A = array(s, array_name)
    k = axis(s, array_name, axis_name)
    return view(A, ntuple(d -> d == k ? i : Colon(), ndims(A))...)
end

continuous(s::Session) = (_require(s, "OE", "continuous"); array(s, "continuous"))

"""
    n_channels(s; array="continuous")
    channel(s, i; array="continuous")
    each_channel(s; array="continuous")

One channel's data, by number, regardless of layout mode.

    for trace in each_channel(s)
        # `trace` is a view of one channel's samples, contiguous under the default layout
    end
"""
n_channels(s::Session; array::AbstractString = "continuous") = n_along(s, array, "channel")
channel(s::Session, i::Integer; array::AbstractString = "continuous") =
    slice_along(s, array, "channel", i)
each_channel(s::Session; array::AbstractString = "continuous") =
    (channel(s, i; array = array) for i in 1:n_channels(s; array = array))

n_samples(s::Session; array::AbstractString = "continuous") = n_along(s, array, "sample")

"""
    n_epochs(s; array="epochs")
    epoch(s, i; array="epochs")
    each_epoch(s; array="epochs")

One epoch, by number. Same idea as `channel`: the axis is found by name, so the code does not
change when the array is stored the other way round.
"""
n_epochs(s::Session; array::AbstractString = "epochs") = n_along(s, array, "epoch")
epoch(s::Session, i::Integer; array::AbstractString = "epochs") =
    slice_along(s, array, "epoch", i)
each_epoch(s::Session; array::AbstractString = "epochs") =
    (epoch(s, i; array = array) for i in 1:n_epochs(s; array = array))

spike_times(s::Session) = (_require(s, "KS", "spike_times"); array(s, "spike_times"))
spike_clusters(s::Session) = (_require(s, "KS", "spike_clusters"); array(s, "spike_clusters"))
n_spikes(s::Session) = length(spike_times(s))

"""
    spikes_of(s, cluster)

One cluster's spike times, in samples. Linear in the number of spikes — for repeated access
across many clusters, bucket once instead.
"""
spikes_of(s::Session, cluster::Integer) = spike_times(s)[spike_clusters(s) .== cluster]

function duration_s(s::Session)
    if s.kind == "OE"
        return n_samples(s) / s.fs_hz
    else
        t = spike_times(s)
        return isempty(t) ? 0.0 : (maximum(t) - minimum(t)) / s.fs_hz
    end
end


# ---- validation and display -------------------------------------------------

function validate(s::Session)
    for (name, A) in s.arrays
        labels = get(s.dims, name, String[])
        length(labels) == ndims(A) || error(
            "array \"$(name)\" has $(ndims(A)) axes but dims $(labels). Every array must " *
            "name its axes — an unlabelled axis is how a session gets silently transposed " *
            "between languages."
        )
    end
    s.fs_hz > 0 || error("fs_hz is $(s.fs_hz)")
    if s.kind == "KS" && haskey(s.arrays, "spike_times")
        t = s.arrays["spike_times"]
        issorted(t) || error("spike_times is not sorted; every searchsorted downstream assumes it is")
    end
    return s
end

function Base.show(io::IO, s::Session)
    parts = ["Session(\"$(s.id)\"", "kind=$(s.kind)", "fs=$(s.fs_hz) Hz"]
    isempty(s.arrays) || push!(parts, "$(length(s.arrays)) arrays")
    isempty(s.tables) || push!(parts, "$(length(s.tables)) tables")
    isempty(s.events) || push!(parts, "$(length(s.events)) event channels")
    print(io, join(parts, ", "), ")")
end

"""
    overview(s) -> String

Everything in the session, in the order it would be written.

Named `overview` rather than `summary` deliberately: `Base.summary` exists, and a method that
shadows it would break unrelated code in any session that does `using EphysLink`.
"""
function overview(s::Session)
    io = IOBuffer()
    println(io, "Session  ", s.id, "  [", s.kind, "]")
    println(io, "  experiment  ", isempty(s.experiment) ? "—" : s.experiment)
    println(io, "  fs          ", s.fs_hz, " Hz")
    println(io, "  source      ", isempty(s.source) ? "—" : s.source)
    s.native && println(io, "  layout      NATIVE (axes reversed relative to Python)")
    if !isempty(s.arrays)
        println(io, "  arrays")
        for name in sort(collect(keys(s.arrays)))
            A = s.arrays[name]
            println(io, "    ", rpad(name, 20), rpad(string(size(A)), 22), eltype(A),
                    "  [", join(get(s.dims, name, String[]), ", "), "]")
        end
    end
    if !isempty(s.tables)
        println(io, "  tables")
        for name in sort(collect(keys(s.tables)))
            columns = s.table_columns[name]
            println(io, "    ", rpad(name, 20), length(s.tables[name][first(columns)]),
                    " rows × ", length(columns), " columns")
        end
    end
    if !isempty(s.events)
        println(io, "  events")
        for name in sort(collect(keys(s.events)))
            println(io, "    ", rpad(name, 20), length(s.events[name]), " events")
        end
    end
    return String(take!(io))
end
