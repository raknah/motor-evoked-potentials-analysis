# One round of the Python↔Julia chain test. Driven by check.py; not run by hand.
#
#   julia --project=modules/EphysLink.jl test/chain.jl INDIR OUTDIR
#
# For every .h5 in INDIR it produces two files in OUTDIR:
#
#   <name>__N.h5   read with native=true  (the default layout), written back
#   <name>__M.h5   read with native=false (Python's index order), written back
#
# check.py runs this repeatedly, writing a Python copy of everything between rounds. After
# three rounds that is every sequence of layout modes up to depth three, with a Python hop
# between each one — which is what "every variant of the back and forth" means. Every file at
# every depth is then compared against the *original*, so an error that cancels itself over a
# single round trip still shows up.
#
# This file makes no judgements. It reports what it saw; check.py decides.

using JSON3

include(joinpath(@__DIR__, "..", "src", "EphysLink.jl"))
using .EphysLink

length(ARGS) == 2 || error("usage: chain.jl INDIR OUTDIR")
indir, outdir = ARGS[1], ARGS[2]
mkpath(outdir)

"""Order-independent fingerprint, enough to flag a value change at a glance.
The authoritative value check is done by check.py on the written file."""
function checksum(A)
    isempty(A) && return 0.0
    return sum(x -> Float64(x), A)
end

report = Dict{String,Any}()

for file in sort(readdir(indir))
    endswith(file, ".h5") || continue
    base = replace(file, ".h5" => "")

    for (tag, native) in (("N", true), ("M", false))
        s = read_session(joinpath(indir, file); native = native)
        out = joinpath(outdir, "$(base)__$(tag).h5")
        write_session(out, s)

        report[basename(out)] = Dict{String,Any}(
            "from" => file,
            "mode" => tag,
            "native_flag" => s.native,
            "arrays" => Dict{String,Any}(
                name => Dict{String,Any}(
                    "shape" => collect(size(A)),
                    "dims" => s.dims[name],
                    "checksum" => checksum(A),
                    # the pairing that must hold in every mode at every depth
                    "n_along" => Dict(d => n_along(s, name, d) for d in s.dims[name]),
                ) for (name, A) in s.arrays
            ),
        )
    end
end

open(joinpath(outdir, "chain_report.json"), "w") do io
    JSON3.write(io, report)
end

println("chain ok: $(length(report)) files written")
