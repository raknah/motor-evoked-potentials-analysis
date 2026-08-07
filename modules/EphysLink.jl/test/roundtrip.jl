# Cross-language round-trip test.
#
#   julia --project=modules/EphysLink.jl modules/EphysLink.jl/test/roundtrip.jl [fixture.h5]
#
# With no argument it tests Julia → file → Julia only. Given the fixture written by
# `python modules/ephyslink/selftest.py --write-fixture …`, it additionally:
#
#   1. reads the Python file and checks every shape, value and axis label;
#   2. writes it back out beside the input as `*_jl.h5`, for Python to verify with
#      `python modules/ephyslink/selftest.py --check-fixture …_jl.h5`.
#
# The fixture arrays are deliberately non-square on every axis (4×7, 3×5×11, 2×3×5×7), so an
# axis reversal applied an odd number of times changes the shape and cannot pass unnoticed. A
# square test array is worse than no test here — it would have let the old shape-guessing
# implementation through.
#
# THE INVARIANT everything below checks, in one line:
#
#     dims[i] describes size(A, i), in whichever language and whichever layout mode.
#
# `read_session` defaults to native=true (zero copy, Julia's own memory order). Pass
# native=false for Python's literal index order.

using Test

include(joinpath(@__DIR__, "..", "src", "EphysLink.jl"))
using .EphysLink

const SHAPE_2D = (4, 7)
const SHAPE_3D = (3, 5, 11)
const SHAPE_4D = (2, 3, 5, 7)

"""Every axis name must report the length of the axis it sits on."""
function check_invariant(s, label)
    for (name, labels) in s.dims
        A = s.arrays[name]
        @test length(labels) == ndims(A)
        for (i, axis_name) in enumerate(labels)
            @test n_along(s, name, axis_name) == size(A, i)
        end
    end
end

@testset "EphysLink" begin

    @testset "julia → file → julia" begin
        s = Session("KS", "julia_fixture", 30_000.0; experiment = "roundtrip")
        add_array!(s, "spike_times", Int64.(sort(rand(1:10^7, 500))), ["spike"])
        add_array!(s, "matrix2d", Float32.(reshape(1:prod(SHAPE_2D), SHAPE_2D)),
                   ["channel", "sample"]; units = "µV")
        add_array!(s, "matrix3d", Float32.(reshape(1:prod(SHAPE_3D), SHAPE_3D)),
                   ["channel", "sample", "epoch"])
        add_table!(s, "clusters",
                   Dict("cluster_id" => Int32.(0:8),
                        "KSLabel" => ["good", "mua", "good", "mua", "good",
                                      "mua", "good", "mua", "good"]);
                   order = ["cluster_id", "KSLabel"])
        add_events!(s, "flicker_40Hz", Int64.(0:750:9999))
        s.meta = Dict("animal" => "XU20", "genotype" => "5xFAD")
        log!(s, "julia roundtrip"; note = "synthetic")

        path = joinpath(mktempdir(), "julia.h5")
        write_session(path, s)

        # matched read must return exactly what was built — a session built here declares its
        # dims in the order its arrays are actually in, so that is the mode that matches
        matched = read_session(path; native = false)
        @test matched.kind == s.kind && matched.id == s.id && matched.fs_hz == s.fs_hz
        for name in keys(s.arrays)
            @test size(matched.arrays[name]) == size(s.arrays[name])
            @test matched.arrays[name] == s.arrays[name]
            @test matched.dims[name] == s.dims[name]
        end
        @test matched.table_columns["clusters"] == ["cluster_id", "KSLabel"]
        @test matched.tables["clusters"]["KSLabel"] == s.tables["clusters"]["KSLabel"]
        @test matched.events["flicker_40Hz"] == s.events["flicker_40Hz"]
        @test matched.meta["animal"] == "XU20"
        check_invariant(matched, "matched")

        # the default native read reports reversed axes with reversed labels — different
        # index order, same data, and every axis name still reports the right length
        native = read_session(path)
        @test native.native
        for name in keys(s.arrays)
            @test size(native.arrays[name]) == reverse(size(s.arrays[name]))
            @test native.dims[name] == reverse(s.dims[name])
            for axis_name in s.dims[name]
                @test n_along(native, name, axis_name) == n_along(matched, name, axis_name)
            end
        end
        check_invariant(native, "native")
    end

    @testset "an array BUILT in julia, in native mode, reaches python correctly" begin
        # The subtle path: read native (so the session is flagged native), build a *new* array
        # here, write, read back. If the invariant survives this it survives everything.
        source = Session("KS", "native_add", 30_000.0)
        add_array!(source, "seed", Float32.(reshape(1:prod(SHAPE_2D), SHAPE_2D)),
                   ["channel", "sample"])
        dir = mktempdir()
        write_session(joinpath(dir, "seed.h5"), source)

        s = read_session(joinpath(dir, "seed.h5"))          # native by default
        @test s.native
        B = Float32.(reshape(1:prod(SHAPE_3D), SHAPE_3D))   # 3 × 5 × 11, built here
        add_array!(s, "built_in_julia", B, ["p", "q", "r"])
        write_session(joinpath(dir, "out.h5"), s)

        back = read_session(joinpath(dir, "out.h5"); native = false)   # python's order
        @test size(back.arrays["built_in_julia"]) == reverse(SHAPE_3D)
        @test back.dims["built_in_julia"] == ["r", "q", "p"]
        check_invariant(back, "built-in-julia, matched")

        again = read_session(joinpath(dir, "out.h5"))                  # native again
        @test again.arrays["built_in_julia"] == B
        @test again.dims["built_in_julia"] == ["p", "q", "r"]
    end

    @testset "layout-independent accessors" begin
        # The same code, the same answers, in both modes. This is what makes the layout
        # question stop mattering at the call site.
        s = Session("OE", "accessors", 1000.0)
        data = Float32.(reshape(1:prod(SHAPE_2D), SHAPE_2D))    # (channel, sample) = 4 × 7
        add_array!(s, "continuous", data, ["channel", "sample"])
        path = joinpath(mktempdir(), "oe.h5")
        write_session(path, s)

        for mode in (true, false)
            t = read_session(path; native = mode)
            @test n_channels(t) == 4
            @test n_samples(t) == 7
            @test length(channel(t, 2)) == 7
            @test collect(channel(t, 2)) == data[2, :]
            @test length(collect(each_channel(t))) == 4
            # a view, not a copy — writing through it reaches the session
            channel(t, 1)[1] = -99.0f0
            @test slice_along(t, "continuous", "channel", 1)[1] == -99.0f0
        end

        # under the default layout a channel is a contiguous column; that is the point
        native = read_session(path)
        @test axis(native, "continuous", "sample") == 1
        @test strides(channel(native, 1)) == (1,)
    end

    @testset "guards" begin
        s = Session("OE", "guard", 1000.0)
        @test_throws ErrorException add_array!(s, "bad", zeros(3, 4), ["only_one"])

        s.arrays["sneaky"] = zeros(3, 4)          # bypass add_array!
        @test_throws ErrorException validate(s)
        delete!(s.arrays, "sneaky")

        add_array!(s, "matrix", zeros(3, 4), ["channel", "sample"])
        @test axis(s, "matrix", "sample") == 2    # 1-based, as Julia expects
        @test_throws ErrorException axis(s, "matrix", "frequency")
    end

    if length(ARGS) >= 1 && isfile(ARGS[1])
        fixture = ARGS[1]
        @testset "a file written by python" begin
            # Deliberately GENERIC: this suite is handed whatever fixture the caller has, so
            # it must not hard-code array names. (It used to expect matrix2d/3d/4d and errored
            # on nine tests when check.py passed its own, richer fixture.) The value checking
            # lives in bridge.jl + check.py, where Python owns the expected answers; what is
            # checked here is the invariant, in both modes, for whatever is in the file.
            matched = read_session(fixture; native = false)
            native  = read_session(fixture)

            @test matched.kind == native.kind
            @test matched.id == native.id
            @test matched.fs_hz == native.fs_hz
            @test !isempty(matched.arrays)

            check_invariant(matched, "python file, matched")
            check_invariant(native, "python file, native")

            @test sort(collect(keys(matched.arrays))) == sort(collect(keys(native.arrays)))

            for (name, A) in matched.arrays
                N = native.arrays[name]
                # native is the reverse of matched, in shape and in labels
                @test size(N) == reverse(size(A))
                @test native.dims[name] == reverse(matched.dims[name])
                # and every axis name reports the same length in both
                for d in matched.dims[name]
                    @test n_along(native, name, d) == n_along(matched, name, d)
                end
                # reversing the native array must recover the matched one exactly
                @test (ndims(N) <= 1 ? collect(N) : permutedims(N, ndims(N):-1:1)) == A
            end

            for (name, columns) in matched.tables
                @test matched.table_columns[name] == native.table_columns[name]
                for c in matched.table_columns[name]
                    @test columns[c] == native.tables[name][c]
                end
            end
            for (name, e) in matched.events
                @test e == native.events[name]
            end

            out = replace(fixture, r"\.h5$" => "_jl.h5")
            write_session(out, matched)
            println("\nwrote $(out)")
        end
    else
        @info "No fixture given; skipping the python → julia checks. Generate one with:\n" *
              "  python modules/ephyslink/selftest.py --write-fixture /tmp/fixture_py.h5"
    end
end
