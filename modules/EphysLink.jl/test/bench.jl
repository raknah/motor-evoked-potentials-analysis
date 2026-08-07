# Does the layout choice actually buy Julia anything, and does it cost seamlessness?
# Driven by check.py; not run by hand.
#
#   julia --project=modules/EphysLink.jl test/bench.jl SESSION.h5 OUT.json
#
# Two questions, measured rather than argued:
#
#   1. OPTIMISATION — is `native` genuinely faster for the work Julia is here to do?
#      Per-channel reductions are the honest test: they are memory-bound, which is where
#      layout decides everything. Under `native` one channel is a contiguous column; under
#      `matched` it is a strided row, stepping n_channels elements at a time.
#
#   2. SEAMLESSNESS — do both modes give the *same answers*, through the same code?
#      The loops below are written once, with `n_channels` and `channel`, and run against
#      both.
#
# METHODOLOGY, because the first version of this file got it wrong and reported the default
# read as 1.7x SLOWER — which is impossible, since it does strictly less work:
#
#   - every timing is the MINIMUM of several runs, not a single shot. The minimum is the
#     standard estimator here: it is the run least disturbed by the scheduler and by GC.
#   - `GC.gc()` before each run, so a collection triggered by the *previous* measurement is
#     not charged to this one. With 100 MB arrays, allocation dominates and an uncontrolled
#     GC pause is larger than the effect being measured.
#   - everything is run once before timing starts, so compilation is not in the number.

using JSON3

include(joinpath(@__DIR__, "..", "src", "EphysLink.jl"))
using .EphysLink

length(ARGS) == 2 || error("usage: bench.jl SESSION.h5 OUT.json")
path, outfile = ARGS[1], ARGS[2]

const REPEATS = 5

"""Minimum of `n` runs, each preceded by a collection. Returns (seconds, last result)."""
function best_of(f; n::Int = REPEATS)
    f()                                    # compile
    best, value = Inf, nothing
    for _ in 1:n
        GC.gc()
        t = @elapsed value = f()
        best = min(best, t)
    end
    return best, value
end

"""The identical code both modes are measured with — the seamlessness claim in one function."""
function per_channel_sum(s)
    total = 0.0
    for i in 1:n_channels(s)
        total += sum(channel(s, i))
    end
    return total
end

"""Per-channel sums, kept separate.

The grand total is invariant to *how* the samples are grouped into channels, so summing
everything cannot detect a layout error at all — only a value error. These per-channel
numbers can: get the grouping wrong and every entry changes.
"""
channel_sums(s) = [Float64(sum(channel(s, i))) for i in 1:n_channels(s)]

"""A second pattern, to show the effect is not specific to one reduction."""
function per_channel_extrema(s)
    total = 0.0
    for i in 1:n_channels(s)
        c = channel(s, i)
        total += maximum(c) - minimum(c)
    end
    return total
end

t_read_native,  _ = best_of(() -> read_session(path; native = true))
t_read_matched, _ = best_of(() -> read_session(path; native = false))

native  = read_session(path; native = true)
matched = read_session(path; native = false)

t_sum_native,  v_sum_native  = best_of(() -> per_channel_sum(native))
t_sum_matched, v_sum_matched = best_of(() -> per_channel_sum(matched))
t_ext_native,  v_ext_native  = best_of(() -> per_channel_extrema(native))
t_ext_matched, v_ext_matched = best_of(() -> per_channel_extrema(matched))

A = native.arrays["continuous"]
t_permutedims, _ = best_of(() -> collect(permutedims(A, ndims(A):-1:1)))

result = Dict{String,Any}(
    "julia_version" => string(VERSION),
    "repeats" => REPEATS,
    "threads" => Threads.nthreads(),
    "n_channels" => Dict("native" => n_channels(native), "matched" => n_channels(matched)),
    "n_samples"  => Dict("native" => n_samples(native),  "matched" => n_samples(matched)),
    "axis_of_channel" => Dict("native" => axis(native, "continuous", "channel"),
                              "matched" => axis(matched, "continuous", "channel")),
    "shape" => Dict("native" => collect(size(native.arrays["continuous"])),
                    "matched" => collect(size(matched.arrays["continuous"]))),
    "bytes" => sizeof(A),
    # seamlessness: the same code, the same answers
    "value_sum" => Dict("native" => v_sum_native, "matched" => v_sum_matched),
    "value_extrema" => Dict("native" => v_ext_native, "matched" => v_ext_matched),
    # grouping-sensitive: one number per channel, so a layout error cannot cancel out
    "channel_sums" => Dict("native" => channel_sums(native),
                           "matched" => channel_sums(matched)),
    # a channel must be contiguous under native — the mechanism, not the symptom
    "channel_stride" => Dict("native" => only(strides(channel(native, 1))),
                             "matched" => only(strides(channel(matched, 1)))),
    "seconds" => Dict(
        "read_native" => t_read_native,
        "read_matched" => t_read_matched,
        "per_channel_sum_native" => t_sum_native,
        "per_channel_sum_matched" => t_sum_matched,
        "per_channel_extrema_native" => t_ext_native,
        "per_channel_extrema_matched" => t_ext_matched,
        "permutedims_copy" => t_permutedims,
    ),
)

open(outfile, "w") do io
    JSON3.write(io, result)
end

println("bench ok")
