# The Julia half of `check.py`. Not meant to be run by hand.
#
#   julia --project=modules/EphysLink.jl modules/EphysLink.jl/test/bridge.jl IN.h5 OUTDIR
#
# It does three things and makes no judgements — every assertion happens on the Python side,
# so there is exactly one place where "correct" is defined:
#
#   1. reads IN in both layout modes and writes everything it saw to OUTDIR/report.json,
#      including each array's elements **in Python's C order**, so Python can compare
#      element by element rather than trusting a shape;
#   2. writes the session back out from each mode;
#   3. builds a fresh array here in Julia, inside a natively-read session, and writes that too
#      — the path most likely to be wrong and least likely to be noticed.

using JSON3

include(joinpath(@__DIR__, "..", "src", "EphysLink.jl"))
using .EphysLink

length(ARGS) == 2 || error("usage: bridge.jl IN.h5 OUTDIR")
infile, outdir = ARGS[1], ARGS[2]
mkpath(outdir)

matched = read_session(infile; native = false)   # Python's axis order
native  = read_session(infile; native = true)    # Julia's own, the default

"""
Elements in Python's C order, whatever the array's rank.

`vec` flattens column-major, so reversing the axes first makes it walk the array in exactly
the order `numpy.ravel(order="C")` does. This is what lets Python check the *values*, not just
the shape — a shape can match while the contents are scrambled.
"""
c_order(A) = ndims(A) <= 1 ? collect(A) : vec(permutedims(A, ndims(A):-1:1))

"""
JSON3 writes bare `NaN`, which is not valid JSON and which Python's strict parser rejects.
Non-finite values become `nothing` (JSON null) — the Python side compares with `equal_nan`,
so a null and a NaN both count as "not a number here".
"""
json_safe(v::AbstractFloat) = isfinite(v) ? Float64(v) : nothing
json_safe(v) = v
json_safe(v::AbstractVector) = [json_safe(x) for x in v]

report = Dict{String,Any}(
    "julia_version" => string(VERSION),
    "kind" => matched.kind,
    "id" => matched.id,
    "experiment" => matched.experiment,
    "fs_hz" => matched.fs_hz,
    "meta" => matched.meta,
    "history_steps" => [get(h, "step", "?") for h in matched.history],
    "arrays" => Dict{String,Any}(),
    "tables" => Dict{String,Any}(),
    "events" => Dict{String,Any}(),
)

for name in keys(matched.arrays)
    A, N = matched.arrays[name], native.arrays[name]
    report["arrays"][name] = Dict{String,Any}(
        "matched_shape" => collect(size(A)),
        "matched_dims"  => matched.dims[name],
        "native_shape"  => collect(size(N)),
        "native_dims"   => native.dims[name],
        "eltype"        => string(eltype(A)),
        "c_order"       => json_safe(collect(c_order(A))),
        # the accessor result, so Python can check it agrees with indexing by hand
        "n_along"       => Dict(d => n_along(matched, name, d) for d in matched.dims[name]),
    )
end

for name in keys(matched.tables)
    columns = matched.table_columns[name]
    report["tables"][name] = Dict{String,Any}(
        "columns" => columns,
        "values"  => Dict{String,Any}(c => json_safe(collect(matched.tables[name][c]))
                                      for c in columns),
    )
end

for (name, samples) in matched.events
    report["events"][name] = collect(samples)
end

open(joinpath(outdir, "report.json"), "w") do io
    JSON3.write(io, report)
end

write_session(joinpath(outdir, "from_matched.h5"), matched)
write_session(joinpath(outdir, "from_native.h5"), native)

# an array built here, in a session flagged native — the subtle path
added = read_session(infile)                      # native, the default
add_array!(added, "julia_built", Float32.(reshape(1:60, (3, 4, 5))), ["a", "b", "c"])
write_session(joinpath(outdir, "with_added.h5"), added)

println("bridge ok")
