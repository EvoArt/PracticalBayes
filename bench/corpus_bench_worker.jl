# Worker: benchmarks exactly ONE model (identified by corpus-file path +
# model name) and appends its results directly to
# bench/results/history_corpus.jsonl. Invoked as a fresh `julia` SUBPROCESS
# per model by corpus_bench.jl's driver loop — this is what makes the
# corpus-wide sweep resilient to a single model crashing the whole process
# (confirmed necessary: Enzyme has produced an OS-level process death, not a
# catchable Julia exception, on at least one model in this corpus; running
# every model in one shared process meant that ONE crash silently ended the
# entire multi-hour sweep with no diagnostic).
#
# Usage: julia --project=<env> corpus_bench_worker.jl <corpus_file_path> <model_name> <corpus_label>
#
# Not meant to be run directly by a person — corpus_bench.jl's driver spawns
# this once per (corpus file, model) pair via `Base.run`.

using PracticalBayes
using ADTypes
using Random
using Dates: now
using Statistics: mean, median, std
import LogDensityProblems
import AbstractMCMC
import AdvancedHMC

_trunc_err(e; n=200) = first(sprint(showerror, e), n)  # see corpus_bench.jl's matching comment: byte-slicing a UTF-8 error message can throw INSIDE a catch block

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
    println("Wrote ", length(_RESULTS), " results (commit ", commit, ")")
    return path
end

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

function _load_corpus_file(path)
    mod = Module(Symbol("CorpusMod_", replace(basename(path), r"[^A-Za-z0-9]" => "_")))
    Core.eval(mod, :(eval(x) = Core.eval($mod, x)))
    Core.eval(mod, :(include(x) = Base.include($mod, x)))
    Base.include(mod, path)
    return _normalize_models(Base.eval(mod, :MODELS))
end

# ===========================================================================
# AD backends. ReverseDiff dropped entirely (redundant with Mooncake — both
# reverse-mode, Mooncake is the actively-developed one this package's test
# suite already leans on). Enzyme RE-ENABLED (see corpus_bench.jl's
# driver-level comment): subprocess isolation means a per-model Enzyme crash
# now only loses that one model's results, not the whole sweep. Used for
# BOTH the gradient layer and (see bench_nuts below) the NUTS layer.
# ===========================================================================

const _AD_BACKENDS = Pair{String,Any}["ForwardDiff" => AutoForwardDiff()]
if !isnothing(Base.find_package("Mooncake"))
    @eval import Mooncake
    push!(_AD_BACKENDS, "Mooncake" => AutoMooncake(; config=nothing))
end
if !isnothing(Base.find_package("Enzyme"))
    @eval import Enzyme
    # Bare `AutoEnzyme()` defaults to a mode with runtime activity analysis
    # OFF — several of the earlier `EnzymeRuntimeActivityError: Detected
    # potential need for runtime activity` failures on this corpus (models
    # where a constant/store value flows into a differentiable computation,
    # e.g. any model with a ValueSlot-backed latent, or a `:=`-computed
    # quantity feeding back into an observe) are directly caused by this
    # default, not a fundamental incompatibility. `Enzyme.set_runtime_activity`
    # enables the (slightly slower, but far more permissive) analysis mode
    # that handles this correctly. Does NOT fix the separate
    # `IllegalTypeAnalysisException`/`EnzymeNoTypeError` failures seen on a
    # few other models — those stem from Enzyme's static type-analysis
    # hitting a genuine Union type in the generated code, a different root
    # cause runtime-activity mode doesn't address.
    push!(_AD_BACKENDS, "Enzyme" => AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)))
end

# ===========================================================================
# NUTS timing: adaptation schedule EXPLICITLY MATCHED to Turing's own
# convention, not AdvancedHMC's bare default. `Turing.sample(model,
# NUTS(0.8), N)` (nadapts unspecified) resolves internally to
# `n_adapts = min(1000, N÷2)` and discards all `n_adapts` warmup draws
# (`packages/Turing/*/src/mcmc/hmc.jl`, the `nadapts == -1` branch) — i.e.
# it actually runs `n_adapts + N` total transitions and reports N. AdvancedHMC's
# OWN default (`n_adapts = min(N÷10, 1000)`, N total transitions INCLUDING
# adaptation) is a materially different, much shorter adaptation budget —
# confirmed via direct investigation to be the dominant cause of PB
# appearing far slower than Turing on hierarchical models: with only 1/10th
# of Turing's adaptation budget, PB's step size/mass matrix are undertuned,
# NUTS trees run deeper, and each kept sample costs many more gradient
# evaluations, even though the RAW gradient-eval speed is at parity or
# faster. Fixed by passing `n_adapts=N÷2` and `discard_initial=n_adapts`
# explicitly here, mirroring Turing's resolved values exactly so both sides
# run the same total number of transitions (n_adapts adaptation + nuts_samples
# kept) and report timing over the same definition of "kept sample".
#
# ONE short chain first (untimed) purely to force JIT compilation of the
# whole call path, THEN one timed chain at the real sample count. Multiple
# full-length timed reps (the earlier design) made a 55-model sweep
# impractically slow — 500-sample NUTS chains are themselves long enough
# that a single post-warmup timing is already fairly stable, so the
# statistical value of repeating the whole chain 3-5× isn't worth 3-5× the
# wall-clock cost here (contrast with the logdensity/gradient layers, which
# are cheap enough that many `reps` cost almost nothing and materially
# reduce noise).
#
# Swept across every available AD backend (ForwardDiff/Mooncake/Enzyme),
# matching the gradient layer — previously NUTS only ever used ForwardDiff.
# ===========================================================================

function bench_nuts(m, layout, θ0, store0, T, adtype; nuts_samples=500, compile_samples=20)
    δ = T(0.8)
    n_adapts = nuts_samples ÷ 2
    ldf = LogDensityFunction(m, layout, store0, adtype; θ0=θ0)
    ldm = AbstractMCMC.LogDensityModel(ldf)
    # Compile warmup: short chain, discarded, not timed as `first_call_s` either
    # (a genuinely separate call shape/sample count from the timed run below,
    # so it wouldn't be a meaningful "first call" number anyway).
    AbstractMCMC.sample(Random.Xoshiro(1), ldm, AdvancedHMC.NUTS(δ), compile_samples; n_adapts=compile_samples ÷ 2, discard_initial=compile_samples ÷ 2, initial_params=θ0, progress=false)
    first_call_s = @elapsed AbstractMCMC.sample(Random.Xoshiro(1), ldm, AdvancedHMC.NUTS(δ), nuts_samples; n_adapts=n_adapts, discard_initial=n_adapts, initial_params=θ0, progress=false)
    t = @elapsed AbstractMCMC.sample(Random.Xoshiro(2), ldm, AdvancedHMC.NUTS(δ), nuts_samples; n_adapts=n_adapts, discard_initial=n_adapts, initial_params=θ0, progress=false)
    return TimingResult("NUTS ($T)", first_call_s, t, t, t, 0.0, 1)
end

# ===========================================================================
# Benchmark exactly one model, both precisions.
# ===========================================================================

function _bench_one_model_at_latest_world(corpus, name, buildfn, init, T; reps, nuts_samples)
    local m, layout, θ0, store0
    try
        m = buildfn()
        layout, θ0, store0 = build_layout(m; init=init, T=T)
    catch e
        println(rpad("$corpus/$name", 40), "  [$T] BUILD FAILED — ", _trunc_err(e))
        return nothing
    end

    try
        ldf = LogDensityFunction(m, layout, store0)
        r = time_reps(() -> LogDensityProblems.logdensity(ldf, θ0), "$corpus/$name logdensity ($T)"; reps=reps)
        record!(; corpus, model=name, layer="logdensity", precision=string(T), backend="none", r)
    catch e
        println(rpad("$corpus/$name", 40), "  [$T] logdensity FAILED — ", _trunc_err(e))
    end

    for (bname, adtype) in _AD_BACKENDS
        try
            ldf = LogDensityFunction(m, layout, store0, adtype; θ0=θ0)
            r = time_reps(() -> LogDensityProblems.logdensity_and_gradient(ldf, θ0), "$corpus/$name grad ($bname,$T)"; reps=reps)
            record!(; corpus, model=name, layer="gradient", precision=string(T), backend=bname, r)
        catch e
            println(rpad("$corpus/$name", 40), "  [$T/$bname] gradient FAILED — ", _trunc_err(e))
        end
    end

    for (bname, adtype) in _AD_BACKENDS
        try
            r = bench_nuts(m, layout, θ0, store0, T, adtype; nuts_samples=nuts_samples)
            record!(; corpus, model=name, layer="nuts", precision=string(T), backend=bname, r)
        catch e
            println(rpad("$corpus/$name", 40), "  [$T/$bname] NUTS FAILED — ", _trunc_err(e))
        end
    end
    return nothing
end

function bench_one_model(corpus, name, buildfn, init; reps=8, nuts_samples=500)
    for T in (Float64, Float32)
        Base.invokelatest(_bench_one_model_at_latest_world, corpus, name, buildfn, init, T; reps=reps, nuts_samples=nuts_samples)
    end
    return nothing
end

function main()
    length(ARGS) == 3 || error("usage: julia corpus_bench_worker.jl <corpus_file_path> <model_name> <corpus_label>")
    path, model_name, corpus_label = ARGS
    println("AD backends available: ", join(first.(_AD_BACKENDS), ", "))
    models = _load_corpus_file(path)
    idx = findfirst(m -> m[1] == model_name, models)
    idx === nothing && error("model `$model_name` not found in $path")
    name, buildfn, init = models[idx]
    bench_one_model(corpus_label, name, buildfn, init)
    write_history!()
end

main()
