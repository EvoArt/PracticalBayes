# Benchmarks EVERY model in bench/corpus/{posteriordb,tutorials}/ across
# every AD backend actually loaded in this session, both Float32 and
# Float64, on three layers: logdensity-only, logdensity_and_gradient, and a
# short end-to-end NUTS run. Results are appended to
# bench/results/history_corpus.jsonl (same JSON-Lines format/mechanism as
# bench/suite.jl's history.jsonl, just a separate file since this covers a
# very different — much larger, structurally varied — set of models).
#
# Deliberately reuses each bench/corpus/*/run_smoke_tests*.jl file's own
# `MODELS` list (already correctness-verified there — logdensity finite,
# gradient matches finite differences) rather than redefining model/data
# pairs here: this file is ONLY responsible for timing, not correctness.
#
# Run with every optional AD backend available by using the SAME
# environment `Pkg.test()` builds from (this package's `[extras]`), e.g.:
#   julia --project=. -e 'import Pkg; Pkg.instantiate(); include("bench/corpus_bench.jl")'
# (see bench/suite.jl's header for why hand-rolled timing, not
# BenchmarkTools/Chairmarks, is used throughout this package's benchmarks.)

using PracticalBayes
using ADTypes
using Random
using Dates: now
using Statistics: mean, median, std
import LogDensityProblems
import AbstractMCMC
import AdvancedHMC

# Byte-index slicing (`s[1:150]`) on an error message can land mid-UTF-8-
# character (many AD backends' error messages contain box-drawing
# characters like `│`, each 3 bytes) and throw `StringIndexError` — INSIDE
# a `catch` block's own error-reporting code, where it isn't caught by
# anything, silently escaping and killing the whole benchmark run. Confirmed
# the hard way: this crashed a multi-hour corpus-wide run partway through on
# a Mooncake error message. `first(s, n)` counts CHARACTERS, not bytes, and
# is always safe.
_trunc_err(e; n=200) = first(sprint(showerror, e), n)

# ===========================================================================
# Timing harness — identical methodology to bench/suite.jl (kept as a
# separate copy, not a shared include, since bench/suite.jl is itself meant
# to be run standalone and this file's model set is entirely different).
# ===========================================================================

struct TimingResult
    label::String
    first_call_s::Float64
    min_s::Float64
    median_s::Float64
    mean_s::Float64
    std_s::Float64
    reps::Int
end

function time_reps(f, label; reps=10)
    first_call_s = @elapsed f()
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        times[i] = @elapsed f()
    end
    return TimingResult(label, first_call_s, minimum(times), median(times), mean(times), std(times), reps)
end

# ===========================================================================
# Results recording — same JSONL mechanism as bench/suite.jl, written to a
# separate history file since this is a structurally distinct benchmark
# (many small varied models vs a few hand-picked shapes).
# ===========================================================================

const _RESULTS = NamedTuple[]

function record!(; package="PracticalBayes", corpus, model, layer, precision, backend, r::TimingResult)
    push!(
        _RESULTS,
        (;
            package, corpus, model, layer, precision, backend,
            first_call_s=r.first_call_s, min_s=r.min_s, median_s=r.median_s, mean_s=r.mean_s, std_s=r.std_s, reps=r.reps,
        ),
    )
    return r
end

function _git_commit()
    try
        return strip(read(`git -C $(dirname(@__DIR__)) rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

_json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"")
_json_value(x::AbstractString) = "\"" * _json_escape(x) * "\""
_json_value(x::Real) = isfinite(x) ? string(x) : "null"
_json_value(x::Integer) = string(x)
_to_json_line(nt::NamedTuple) = "{" * join(("\"$(k)\":$(_json_value(v))" for (k, v) in pairs(nt)), ",") * "}"

function write_history!(path=joinpath(@__DIR__, "results", "history_corpus.jsonl"))
    mkpath(dirname(path))
    commit = _git_commit()
    timestamp = string(now())
    open(path, "a") do io
        for r in _RESULTS
            println(io, _to_json_line(merge((; timestamp, commit), r)))
        end
    end
    println("Wrote ", length(_RESULTS), " results to ", path, " (commit ", commit, ")")
    return path
end

# ===========================================================================
# Normalizing each corpus file's own MODELS list into `(name, () -> Model)`
# pairs — batch1 files use 4-tuples `(name, modelfn, datafn, init)`, later
# batches use 3-tuples `(name, model_closure, init)` (data baked into the
# closure). Both normalize to the same `(name, buildfn, init)` shape.
# ===========================================================================

function _normalize_models(raw_models)
    out = Tuple{String,Function,NamedTuple}[]
    for entry in raw_models
        if length(entry) == 4
            name, modelfn, datafn, init = entry
            push!(out, (name, () -> modelfn(datafn(1)...), init))
        elseif length(entry) == 3
            name, buildfn, init = entry
            push!(out, (name, buildfn, init))
        else
            error("unrecognized MODELS entry shape: $entry")
        end
    end
    return out
end

# Each corpus file is `include`d in its OWN Module to avoid name collisions
# between files that reuse model names (e.g. multiple `make_*_data` helpers)
# — every batch file was written as a standalone script assuming it owns
# the global namespace, so this file must not `include` them all into one
# shared scope directly.
function _load_corpus_file(path)
    mod = Module(Symbol("CorpusMod_", replace(basename(path), r"[^A-Za-z0-9]" => "_")))
    # A bare `Module()` has no `include`/`eval` of its own bound as callable
    # from WITHIN itself — each corpus file's own body calls a plain,
    # relative `include("models.jl")`, which needs to resolve to THIS
    # module's `include`, not `Main`'s. `Core.eval`ing these two one-liners
    # into `mod` gives it exactly that (the same pattern `Base.include_string`
    # / `Base.eval(:(module M ... end))` uses internally for "load this file
    # into its own fresh namespace").
    Core.eval(mod, :(eval(x) = Core.eval($mod, x)))
    Core.eval(mod, :(include(x) = Base.include($mod, x)))
    Base.include(mod, path)
    return _normalize_models(Base.eval(mod, :MODELS))
end

const CORPUS_FILES = [
    ("posteriordb", joinpath(@__DIR__, "corpus", "posteriordb", "run_smoke_tests.jl")),
    ("posteriordb", joinpath(@__DIR__, "corpus", "posteriordb", "run_smoke_tests_batch2.jl")),
    ("posteriordb", joinpath(@__DIR__, "corpus", "posteriordb", "run_smoke_tests_batch3.jl")),
    ("posteriordb", joinpath(@__DIR__, "corpus", "posteriordb", "run_smoke_tests_batch4.jl")),
    ("posteriordb", joinpath(@__DIR__, "corpus", "posteriordb", "run_smoke_tests_batch5.jl")),
    ("tutorials", joinpath(@__DIR__, "corpus", "tutorials", "run_smoke_tests.jl")),
    ("tutorials", joinpath(@__DIR__, "corpus", "tutorials", "run_smoke_tests_batch2.jl")),
]

# ===========================================================================
# AD backend registration — same pattern as bench/suite.jl: only registered
# if actually installed AND `import`ed in this session.
# ===========================================================================

const _AD_BACKENDS = Pair{String,Any}["ForwardDiff" => AutoForwardDiff()]
if !isnothing(Base.find_package("Mooncake"))
    @eval import Mooncake
    push!(_AD_BACKENDS, "Mooncake" => AutoMooncake(; config=nothing))
end
if !isnothing(Base.find_package("ReverseDiff"))
    @eval import ReverseDiff
    push!(_AD_BACKENDS, "ReverseDiff" => AutoReverseDiff())
end
# Enzyme deliberately EXCLUDED from this corpus-wide sweep (unlike
# bench/suite.jl's small hand-picked model set, where it's known to work):
# across this many structurally varied models it hit a native crash that
# killed the whole benchmark process outright (not a catchable Julia
# exception — confirmed directly, the run silently died mid-model with no
# stack trace, just the OS process disappearing), on top of the
# `EnzymeRuntimeActivityError`/`IllegalTypeAnalysisException`/
# `EnzymeNoTypeError` failures already seen (and safely caught) on other
# models. Enzyme's stability is genuinely worse across this much model
# diversity than the other three backends; re-enable only if/when that's
# independently resolved.
# if !isnothing(Base.find_package("Enzyme"))
#     @eval import Enzyme
#     push!(_AD_BACKENDS, "Enzyme" => AutoEnzyme())
# end

# ===========================================================================
# Per-model benchmarking. `T` (Float32/Float64) only affects `θ0`'s element
# type — the model's OWN literals/data stay whatever the corpus file wrote
# them as (see bench/suite.jl's tiny/large models for the "genuinely
# Float32 end to end" version; the corpus models were written for
# correctness/coverage, not precision sweeps, so this measures "Float32
# parameter vector through an otherwise-Float64 model," a real and
# meaningful configuration in its own right, just not a full-Float32 one).
# ===========================================================================

# `buildfn` (and everything built from its result — `m.f`, called
# internally by `evaluate`/`build_layout`/`LogDensityFunction`) closes over
# methods defined by a DYNAMICALLY `include`d corpus file
# (`_load_corpus_file`, above). Calling any of that from a function that was
# ALREADY COMPILING before those methods existed hits Julia's world-age
# barrier (`MethodError: ... method too new to be called from this world
# context`) — not just at the outermost `buildfn()` call, but at every
# nested dynamic dispatch through `m.f` too (`build_layout`, `evaluate`,
# `LogDensityFunction`'s constructor, `LogDensityProblems.logdensity`, ...).
# The general fix is `Base.invokelatest` around the WHOLE per-model
# benchmarking body, not just the model-construction call — hence this is
# split into its own function invoked via `invokelatest` from
# `bench_one_model`, rather than peppering `invokelatest` through every
# individual call site below.
function _bench_one_model_at_latest_world(corpus, name, buildfn, init, T; reps, nuts_samples, nuts_reps)
    local m, layout, θ0, store0
    try
        m = buildfn()
        layout, θ0, store0 = build_layout(m; init=init, T=T)
    catch e
        println(rpad("$corpus/$name", 40), "  [$T] BUILD FAILED — ", _trunc_err(e))
        return nothing
    end

    # Layer 1: logdensity only (backend-independent).
    try
        ldf = LogDensityFunction(m, layout, store0)
        r = time_reps(() -> LogDensityProblems.logdensity(ldf, θ0), "$corpus/$name logdensity ($T)"; reps=reps)
        record!(; corpus, model=name, layer="logdensity", precision=string(T), backend="none", r)
    catch e
        println(rpad("$corpus/$name", 40), "  [$T] logdensity FAILED — ", _trunc_err(e))
    end

    # Layer 2: gradient, per AD backend.
    for (bname, adtype) in _AD_BACKENDS
        try
            ldf = LogDensityFunction(m, layout, store0, adtype; θ0=θ0)
            r = time_reps(() -> LogDensityProblems.logdensity_and_gradient(ldf, θ0), "$corpus/$name grad ($bname,$T)"; reps=reps)
            record!(; corpus, model=name, layer="gradient", precision=string(T), backend=bname, r)
        catch e
            println(rpad("$corpus/$name", 40), "  [$T/$bname] gradient FAILED — ", _trunc_err(e))
        end
    end

    # Layer 3: short end-to-end NUTS (ForwardDiff only — matches
    # bench/suite.jl's own choice; NUTS timing is dominated by
    # AdvancedHMC's own per-iteration overhead more than by AD backend
    # choice for these small/medium models).
    try
        δ = T(0.8)
        ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
        ldm = AbstractMCMC.LogDensityModel(ldf)
        run() = AbstractMCMC.sample(
            Random.Xoshiro(1), ldm, AdvancedHMC.NUTS(δ), nuts_samples; initial_params=θ0, progress=false,
        )
        r = time_reps(run, "$corpus/$name NUTS ($T)"; reps=nuts_reps)
        record!(; corpus, model=name, layer="nuts", precision=string(T), backend="ForwardDiff", r)
    catch e
        println(rpad("$corpus/$name", 40), "  [$T] NUTS FAILED — ", _trunc_err(e))
    end
    return nothing
end

function bench_one_model(corpus, name, buildfn, init; reps=8, nuts_samples=50, nuts_reps=3)
    for T in (Float64, Float32)
        Base.invokelatest(
            _bench_one_model_at_latest_world, corpus, name, buildfn, init, T; reps=reps, nuts_samples=nuts_samples,
            nuts_reps=nuts_reps,
        )
    end
    return nothing
end

function run_corpus_bench()
    empty!(_RESULTS)
    println("AD backends available: ", join(first.(_AD_BACKENDS), ", "))
    for (corpus, path) in CORPUS_FILES
        println("\n", "="^100, "\nLoading: ", path, "\n", "="^100)
        models = try
            _load_corpus_file(path)
        catch e
            println("FAILED TO LOAD ", path, ": ", _trunc_err(e; n=300))
            continue
        end
        for (name, buildfn, init) in models
            println("  benchmarking ", corpus, "/", name, " ...")
            bench_one_model(corpus, name, buildfn, init)
        end
    end
    write_history!()
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_corpus_bench()
