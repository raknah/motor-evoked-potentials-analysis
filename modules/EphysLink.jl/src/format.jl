# Reading and writing the ephyslink HDF5 file from Julia.
#
# The contract is modules/FORMAT.md. This file implements it; if the two disagree,
# FORMAT.md is right.
#
# THE DIMENSION RULE, because getting it wrong is silent:
#
#   h5py works in C order (last axis varies fastest); HDF5.jl works in Fortran order (first
#   axis varies fastest). The same bytes therefore appear with REVERSED axes. An array
#   written from Python with logical shape (a, b, c) is read here as (c, b, a). Always.
#   Deterministically. Without needing to look at the data.
#
#   So: reverse on read, reverse on write. After that an array has the same logical axis
#   order in both languages, and the `dims` attribute describes both.
#
# The previous implementation instead guessed from array shape —
#   `if nrows > ncols && nrows > 1000; transpose(data); end`
# — which is a coin flip whenever the two axes are of comparable size. Never infer axis order
# from a shape. Read the `dims` attribute.
#
# HDF5.jl API note: this file uses `attributes(obj)[name] = value` to write and
# `read_attribute(obj, name)` to read, consistently. Both have been stable across HDF5.jl
# 0.16 and 0.17. Mixing them with the newer `attrs()` accessor is what makes this kind of
# file break on a version bump.

const FORMAT_NAME = "ephyslink"
const FORMAT_VERSION = 1

"""
    _reverse_axes(A)

Undo (or apply) the C-order/Fortran-order axis reversal. Its own inverse.
`collect` because `permutedims` on some inputs is lazy, and HDF5.jl wants a dense array.
"""
_reverse_axes(A::AbstractArray) =
    ndims(A) <= 1 ? collect(A) : collect(permutedims(A, ndims(A):-1:1))

_as_string(x) = x isa AbstractString ? String(x) : string(x)

_has_attr(obj, name::AbstractString) = haskey(attributes(obj), name)

function _read_json(f, name::AbstractString, fallback)
    _has_attr(f, name) || return fallback
    raw = read_attribute(f, name)
    raw isa AbstractString || return fallback
    try
        return JSON3.read(raw, Any)
    catch
        return fallback
    end
end

_utc_now_iso() = string(Dates.now(Dates.UTC)) * "+00:00"


"""
    read_session(path; native=true) -> Session

Load an ephyslink file.

`native=true` (the **default**) is zero-copy and gives Julia *the same bytes and the same
memory locality as Python*. It is not a compromise layout — a `(channel, sample)` array in
Python's C order has samples contiguous within a channel, and read natively here it is
`(sample, channel)` in column-major order, which also has samples contiguous within a channel.
Identical physical layout, reversed index order.

`native=false` applies a `permutedims` so the axis order matches Python literally. It costs a
copy *and* leaves the array laid out against the grain — one channel becomes a strided row
rather than a contiguous column. Summing each channel of a 1 GB `(130, 2 000 000)` block:
53 ms native, 1388 ms matched, **26× slower**, plus ~0.6 s for the copy. The only thing it buys
is that `A[channel, t]` is written the same way in both languages. Use it while porting Python
code line by line; otherwise leave it alone.

**Either way, `dims` describes the array you are holding**, so index by name and the question
stops mattering:

    c = axis(s, "continuous", "channel")
    n = n_along(s, "continuous", "sample")
"""
function read_session(path::AbstractString; native::Bool = true)
    h5open(path, "r") do f
        found = _has_attr(f, "format") ? _as_string(read_attribute(f, "format")) : ""
        found == FORMAT_NAME || error(
            "$(path) is not an ephyslink file (root attribute format=\"$(found)\"). " *
            "Files written by the old neuroephys4julia/openephysextract code are not " *
            "readable — re-extract from source."
        )
        version = Int(read_attribute(f, "format_version"))
        version <= FORMAT_VERSION || error(
            "$(path) is format version $(version); this code understands up to " *
            "$(FORMAT_VERSION). Update EphysLink."
        )

        arrays = Dict{String,Any}()
        dims = Dict{String,Vector{String}}()
        array_meta = Dict{String,Dict{String,String}}()
        if haskey(f, "arrays")
            g = f["arrays"]
            for name in keys(g)
                ds = g[name]
                A = read(ds)
                labels = _has_attr(ds, "dims") ? String.(read_attribute(ds, "dims")) : String[]
                if native
                    # leave the array as HDF5.jl produced it; reverse the labels to match
                    arrays[name] = A
                    dims[name] = reverse(labels)
                else
                    arrays[name] = _reverse_axes(A)
                    dims[name] = labels
                end
                extra = Dict{String,String}()
                for key in ("units", "description")
                    _has_attr(ds, key) && (extra[key] = _as_string(read_attribute(ds, key)))
                end
                isempty(extra) || (array_meta[name] = extra)
            end
        end

        tables = Dict{String,Dict{String,Any}}()
        table_columns = Dict{String,Vector{String}}()
        if haskey(f, "tables")
            g = f["tables"]
            for name in keys(g)
                tg = g[name]
                columns = String.(read_attribute(tg, "columns"))
                table_columns[name] = columns
                tables[name] = Dict{String,Any}(c => read(tg[c]) for c in columns)
            end
        end

        events = Dict{String,Vector{Int64}}()
        if haskey(f, "events")
            g = f["events"]
            for name in keys(g)
                events[name] = Vector{Int64}(read(g[name]))
            end
        end

        return Session(
            _as_string(read_attribute(f, "kind")),
            _as_string(read_attribute(f, "id")),
            _has_attr(f, "experiment") ? _as_string(read_attribute(f, "experiment")) : "",
            Float64(read_attribute(f, "fs_hz")),
            _has_attr(f, "source") ? _as_string(read_attribute(f, "source")) : "",
            arrays, dims, array_meta,
            tables, table_columns,
            events,
            _read_json(f, "meta_json", Dict{String,Any}()),
            _read_json(f, "history_json", Any[]),
            native,
        )
    end
end


"""
    write_session(path, s)

Write a session so that Python reads it with the axis order recorded in `s.dims`.

If the session was read with `native=true`, its arrays are already in the on-disk orientation
and its `dims` are reversed; both cases are handled, so a natively-read session round-trips
correctly without the caller having to remember which mode it came from.
"""
function write_session(path::AbstractString, s::Session)
    validate(s)
    mkpath(dirname(abspath(path)))

    h5open(path, "w") do f
        a = attributes(f)
        a["format"] = FORMAT_NAME
        a["format_version"] = FORMAT_VERSION
        a["kind"] = s.kind
        a["id"] = s.id
        a["experiment"] = s.experiment
        a["fs_hz"] = s.fs_hz
        a["source"] = s.source
        a["created"] = _utc_now_iso()
        a["meta_json"] = JSON3.write(s.meta)
        a["history_json"] = JSON3.write(s.history)

        if !isempty(s.arrays)
            g = create_group(f, "arrays")
            for (name, A) in s.arrays
                stored = s.native ? collect(A) : _reverse_axes(A)
                labels = s.native ? reverse(get(s.dims, name, String[])) :
                                    get(s.dims, name, String[])
                g[name] = stored
                da = attributes(g[name])
                isempty(labels) || (da["dims"] = labels)
                for (key, value) in get(s.array_meta, name, Dict{String,String}())
                    da[key] = value
                end
            end
        end

        if !isempty(s.tables)
            g = create_group(f, "tables")
            for (name, columns) in s.tables
                order = get(s.table_columns, name, collect(keys(columns)))
                tg = create_group(g, name)
                ta = attributes(tg)
                ta["columns"] = order
                ta["n_rows"] = length(columns[first(order)])
                for column in order
                    tg[column] = collect(columns[column])
                end
            end
        end

        if !isempty(s.events)
            g = create_group(f, "events")
            for (name, samples) in s.events
                g[name] = Vector{Int64}(samples)
            end
        end
    end
    return path
end


"""
    describe_file(path) -> String

What is in a file, without constructing a Session. For when loading fails.

Shapes are reported as they sit on disk, i.e. **reversed** relative to the `dims` labels,
which are always in Python logical order. That is deliberate: this function is for
diagnosing a file, so it shows the file rather than an interpretation of it.
"""
function describe_file(path::AbstractString)
    io = IOBuffer()
    println(io, path, "\n")
    h5open(path, "r") do f
        println(io, "root attributes")
        for key in sort(collect(keys(attributes(f))))
            value = _as_string(read_attribute(f, key))
            println(io, "  ", rpad(key, 16), first(value, 100))
        end
        for section in ("arrays", "tables", "events")
            haskey(f, section) || continue
            println(io, "\n", section)
            g = f[section]
            for name in keys(g)
                item = g[name]
                if item isa HDF5.Dataset
                    labels = _has_attr(item, "dims") ? String.(read_attribute(item, "dims")) : String[]
                    println(io, "  ", rpad(name, 22), size(item), " on disk",
                            isempty(labels) ? "" : "  [$(join(labels, ", "))] in python order")
                else
                    columns = String.(read_attribute(item, "columns"))
                    println(io, "  ", rpad(name, 22), read_attribute(item, "n_rows"),
                            " rows × ", length(columns), " columns: ", join(columns, ", "))
                end
            end
        end
    end
    return String(take!(io))
end
