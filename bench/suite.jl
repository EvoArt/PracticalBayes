# Hand-rolled benchmark harness — NOT BenchmarkTools/Chairmarks.
#
# Why hand-rolled: BenchmarkTools/Chairmarks both use an adaptive sampling
# budget (run until some total time/sample-count target is hit), and for
# anything in the realistic multi-millisecond-per-call range (a real NUTS
# step on a non-trivial model), that budget can be exhausted after a single
# sample — there is no portable "minimum N samples regardless of cost" knob
# in either package. Since we specifically want to compare stable
# medians/spreads across model shapes, AD backends, and precisions without
# silently degenerating to N=1 on the expensive configurations, this file
# implements the simplest thing that reliably gives N explicit repetitions:
# run once (discarded, to trigger JIT compilation), then time `reps` more
# calls individually and report min/median/mean/std of that vector.
#
# What this suite measures, deliberately split into three layers so a slow
# number can be attributed to the right cause instead of blamed on "the
# framework":
#   1. logdensity only (no gradient) — isolates model-evaluation overhead
#      from any AD cost at all.
#   2. logdensity_and_gradient, per AD backend — isolates AD-backate cost;
#      this is what NUTS calls at every leapfrog step, so it's the number
#      that actually matters for real sampling.
#   3. a short end-to-end NUTS run — the number a user actually experiences,
#      which includes tree-building, step-size adaptation, and everything
#      else AdvancedHMC does per iteration on top of the gradient call. For
#      simple, cheap models this will be dominated by AdvancedHMC/adaptation
#      overhead, NOT by anything this package controls — reported for
#      honesty, not held to the same "should match a raw loop" bar as (1).
#
# Model shapes: "tiny" (few params, few observations — where any FIXED
# per-call framework overhead would show up most clearly as a fraction of
# total time) and "large" (many observations — where raw floating-point
# work should dominate and framework overhead should be a vanishing
# fraction, if the design is doing its job). Comparing the SAME
# relative-overhead metric across both shapes is what actually tells you
# whether overhead is a fixed cost or scales with the problem.

using PracticalBayes
using Distributions: Normal, Exponential, Poisson, logpdf
using Random
using Statistics: mean, median, std
using LinearAlgebra: dot
using ADTypes
using Dates: now
import LogDensityProblems
import AbstractMCMC
import AdvancedHMC

# ===========================================================================
# Timing harness
# ===========================================================================

struct TimingResult
    label::String
    first_call_s::Float64  # includes JIT compilation — what a user sees the very first time
    min_s::Float64
    median_s::Float64
    mean_s::Float64
    std_s::Float64
    reps::Int
end

function Base.show(io::IO, r::TimingResult)
    print(
        io,
        rpad(r.label, 46),
        "  first_call=", _fmt(r.first_call_s),
        "  min=", _fmt(r.min_s), "  median=", _fmt(r.median_s), "  mean=", _fmt(r.mean_s), "  std=", _fmt(r.std_s),
        "  (n=", r.reps, ")",
    )
end
_fmt(t) = t < 1e-3 ? string(round(t * 1e6; digits=2), "µs") : string(round(t * 1e3; digits=3), "ms")

# ===========================================================================
# Results recording — tagged by date + git commit so `bench/results/history.jsonl`
# becomes a regression-spotting history across runs, not just a single
# snapshot. Deliberately hand-rolled JSON (one flat record per line, only
# strings/numbers, no nesting) rather than adding a JSON dependency just for
# this — see `_json_escape`/`to_jsonl` below.
# ===========================================================================

const _RESULTS = NamedTuple[]  # populated by `record!` during a run_suite() call

"""
    record!(; package, layer, model, shape, precision, backend, r::TimingResult)

Appends one benchmark result to the in-memory `_RESULTS` collector (flushed
to `bench/results/history.jsonl` by `write_history!` at the end of
`run_suite()`). `package` is `"PracticalBayes"` here (this file only ever
benchmarks our own side — see `test/comparison_env/generate_turing_reference.jl`
for the equivalent on the Turing side, written to a parallel file).
"""
function record!(; package, layer, model, shape, precision, backend, r::TimingResult)
    push!(
        _RESULTS,
        (;
            package, layer, model, shape, precision, backend,
            first_call_s=r.first_call_s, min_s=r.min_s, median_s=r.median_s, mean_s=r.mean_s, std_s=r.std_s, reps=r.reps,
        ),
    )
    return r
end

# Returns `nothing` (via git) as the commit hash if this isn't run inside a
# git checkout, or if git itself isn't on PATH — history is still useful
# without it, just without the "which commit was this" regression anchor.
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

function _to_json_line(nt::NamedTuple)
    pairs_str = join(("\"$(k)\":$(_json_value(v))" for (k, v) in pairs(nt)), ",")
    return "{" * pairs_str * "}"
end

"""
    write_history!(path = joinpath(@__DIR__, "results", "history.jsonl"))

Appends every result recorded via `record!` since the last call, one JSON
object per line, each stamped with `timestamp` (ISO-8601-ish,
`Dates.now()`) and `commit` (this repo's current `HEAD`, so a regression
can be traced to the exact commit that caused it). Appending (not
overwriting) is deliberate: `history.jsonl` is meant to accumulate across
many runs over time, committed to git alongside the code it measured —
`git log -p bench/results/history.jsonl` doubles as a benchmark changelog.
"""
function write_history!(path=joinpath(@__DIR__, "results", "history.jsonl"))
    mkpath(dirname(path))
    commit = _git_commit()
    timestamp = string(now())
    open(path, "a") do io
        for r in _RESULTS
            full = merge((; timestamp, commit), r)
            println(io, _to_json_line(full))
        end
    end
    println("Wrote ", length(_RESULTS), " results to ", path, " (commit ", commit, ")")
    return path
end

"""
    time_reps(f, label; reps=30) -> TimingResult

Calls `f()` once and records that call's time SEPARATELY as `first_call_s`
(this includes JIT compilation — the latency a user actually experiences
the very first time they run a given model/backend/precision combination,
which matters disproportionately for small/quick models where steady-state
cost is tiny by comparison), then calls it `reps` more times to build the
steady-state min/median/mean/std distribution. Explicit fixed repetition
count, unlike BenchmarkTools' adaptive budget, specifically so expensive
configurations (a real NUTS run, Enzyme/Mooncake tape building, etc.) still
get a meaningful steady-state sample size instead of silently collapsing to
N=1.
"""
function time_reps(f, label; reps=30)
    first_call_s = @elapsed f()
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        times[i] = @elapsed f()
    end
    return TimingResult(label, first_call_s, minimum(times), median(times), mean(times), std(times), reps)
end

# ===========================================================================
# Model shapes
# ===========================================================================

# "tiny": 2 params, 20 observations — any fixed per-call overhead should be
# most visible here as a fraction of total time.
@model function tiny_model(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end

# "large": 2 params, 50_000 observations — raw floating-point work over `y`
# should dominate; framework overhead should be a vanishing fraction if the
# hot path is truly allocation-free per requirement 1.
@model function large_model(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end

function make_data(::Type{T}, n, seed) where {T<:Real}
    rng = Random.Xoshiro(seed)
    return T.(randn(rng, n))
end

# "manyparam": K=200 regression coefficients, N=2000 observations — the
# shape where reverse-mode AD (Mooncake/ReverseDiff/Enzyme) SHOULD start
# beating forward-mode (ForwardDiff), since forward-mode's cost scales
# roughly linearly with the number of DIFFERENTIATED parameters (K+1 here)
# while reverse-mode's is closer to O(1) in parameter count (a small
# constant times the cost of one forward pass, regardless of K). The tiny/
# large shapes above both have exactly 2 parameters and can never surface
# this crossover — that was a real gap in the original benchmark design.
@model function manyparam_model(X, y)
    beta = Vector{paramtype(__mode__)}(undef, size(X, 2))
    for k in 1:size(X, 2)
        beta[k] ~ Normal(0, 1)
    end
    sigma ~ Exponential(1)
    y .~ Normal.(X * beta, sigma)
end

function make_manyparam_data(::Type{T}, k, n, seed) where {T<:Real}
    rng = Random.Xoshiro(seed)
    X = T.(randn(rng, n, k))
    true_beta = randn(rng, k)
    y = T.(X * true_beta .+ randn(rng, n) .* 0.5)
    return X, y
end

# "poisson": discrete-LIKELIHOOD regression (observations are draws from a
# discrete distribution — Poisson counts — not a discrete LATENT variable;
# fully supported by ordinary `~`/`.~`, no marginalization needed). Added
# because every shape so far used a continuous Normal likelihood; this
# checks the hot path is equally allocation-free/fast when `logpdf` is a
# discrete pmf instead of a continuous density.
@model function poisson_model(X, y)
    beta = Vector{paramtype(__mode__)}(undef, size(X, 2))
    for k in 1:size(X, 2)
        beta[k] ~ Normal(0, 1)
    end
    y .~ Poisson.(exp.(X * beta))
end

function make_poisson_data(k, n, seed)
    rng = Random.Xoshiro(seed)
    X = randn(rng, n, k)
    true_beta = randn(rng, k) .* 0.3
    y = [rand(rng, Poisson(exp(clamp(dot(X[i, :], true_beta), -10, 10)))) for i in 1:n]
    return X, Float64.(y)
end

# ===========================================================================
# Layer 1: logdensity only (no AD) — pure model-evaluation overhead vs a
# hand-written loop computing the identical quantity with no framework at all.
# ===========================================================================

function bench_logdensity_only(::Type{T}, n; reps=50) where {T<:Real}
    y = make_data(T, n, 1)
    m = n <= 100 ? tiny_model(y) : large_model(y)
    layout, θ0, store0 = build_layout(m; T=T)
    ldf = LogDensityFunction(m, layout, store0)

    raw(mu, sigma) = sum(logpdf.(Normal(mu, sigma), y))
    r_raw = time_reps(() -> raw(zero(T), one(T)), "raw loop (T=$T, n=$n)"; reps=reps)
    r_pb = time_reps(() -> LogDensityProblems.logdensity(ldf, θ0), "PracticalBayes logdensity (T=$T, n=$n)"; reps=reps)
    return r_raw, r_pb
end

# ===========================================================================
# Layer 2: logdensity_and_gradient, swept over every AD backend that's
# actually loaded in this session (see the bottom of this file for how
# backends get registered — each one is optional and skipped if not loaded).
# ===========================================================================

const _AD_BACKENDS = Pair{String,Any}[]  # populated below, once per available backend

function bench_gradient(::Type{T}, n; reps=50) where {T<:Real}
    y = make_data(T, n, 2)
    m = n <= 100 ? tiny_model(y) : large_model(y)
    layout, θ0, store0 = build_layout(m; T=T)

    results = TimingResult[]
    for (name, adtype) in _AD_BACKENDS
        ldf = LogDensityFunction(m, layout, store0, adtype; θ0=θ0)
        r = time_reps(
            () -> LogDensityProblems.logdensity_and_gradient(ldf, θ0), "$name gradient (T=$T, n=$n)"; reps=reps
        )
        push!(results, r)
    end
    return results
end

# Same idea as `bench_gradient` above, but on the K=200-parameter regression
# — this is the shape that should actually distinguish forward- from
# reverse-mode AD, unlike the 2-parameter tiny/large models.
#
# Each backend is wrapped in a try/catch: a single backend failing on a
# given (model, precision) combination (e.g. Enzyme's Union-type analysis
# tripping on Float32 for this model — a real limitation observed in
# practice, not something worth chasing down before reporting the other
# backends) must not abort the whole benchmark run. Failures are reported
# as `nothing` in the results and printed, not silently dropped.
function bench_manyparam_gradient(::Type{T}, k, n; reps=20) where {T<:Real}
    X, y = make_manyparam_data(T, k, n, 5)
    m = manyparam_model(X, y)
    layout, θ0, store0 = build_layout(m; T=T)

    results = Pair{String,Union{TimingResult,Nothing}}[]
    for (name, adtype) in _AD_BACKENDS
        label = "$name gradient (manyparam K=$k, T=$T, n=$n)"
        try
            ldf = LogDensityFunction(m, layout, store0, adtype; θ0=θ0)
            r = time_reps(() -> LogDensityProblems.logdensity_and_gradient(ldf, θ0), label; reps=reps)
            push!(results, name => r)
        catch e
            println(label, ": FAILED — ", sprint(showerror, e)[1:min(end, 200)])
            push!(results, name => nothing)
        end
    end
    return results
end

# Discrete-likelihood (Poisson) density-only comparison against a raw loop,
# same shape as `bench_logdensity_only` but with a non-Gaussian observation
# distribution.
function bench_poisson_density(k, n; reps=50)
    X, y = make_poisson_data(k, n, 6)
    m = poisson_model(X, y)
    layout, θ0, store0 = build_layout(m)
    ldf = LogDensityFunction(m, layout, store0)

    raw(beta) = sum(logpdf.(Poisson.(exp.(X * beta)), y))
    beta0 = zeros(k)
    r_raw = time_reps(() -> raw(beta0), "raw Poisson loop (k=$k, n=$n)"; reps=reps)
    r_pb = time_reps(() -> LogDensityProblems.logdensity(ldf, θ0), "PracticalBayes Poisson density (k=$k, n=$n)"; reps=reps)
    return r_raw, r_pb
end

# ===========================================================================
# Layer 3: short end-to-end NUTS run. Reported for honesty — this is what a
# user actually experiences, but for a 2-parameter model it will likely be
# dominated by AdvancedHMC's own per-iteration bookkeeping (tree building,
# step-size/mass-matrix adaptation) rather than by this package's model
# evaluation, so it should NOT be held to the same "must match a raw loop"
# bar as layer 1. `δ` must be typed to match `T` (AdvancedHMC's NUTS(δ) fixes
# its internal step-size type parameter to `typeof(δ)` — passing a Float64
# `δ` with a Float32 position vector fails inside step-size adaptation; this
# is an AdvancedHMC requirement, not something this package controls).
#
# n_adapts/discard_initial passed EXPLICITLY, matching Turing's own default
# resolution (`n_adapts = min(1000, N÷2)`, discarding all of it) rather than
# AdvancedHMC's much shorter bare default (`n_adapts = min(N÷10, 1000)`) —
# see bench/corpus_bench_worker.jl's matching comment for why this
# alignment is what "PB and Turing doing the same thing" actually requires.
# ===========================================================================

function bench_nuts(::Type{T}, n; n_samples=1000, reps=5) where {T<:Real}
    y = make_data(T, n, 3)
    m = n <= 100 ? tiny_model(y) : large_model(y)
    layout, θ0, store0 = build_layout(m; T=T)
    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    ldm = AbstractMCMC.LogDensityModel(ldf)
    δ = T(0.8)
    n_adapts = n_samples ÷ 2
    run() = AbstractMCMC.sample(Random.Xoshiro(1), ldm, AdvancedHMC.NUTS(δ), n_samples; n_adapts=n_adapts, discard_initial=n_adapts, initial_params=θ0, progress=false)
    return time_reps(run, "NUTS $n_samples samples (T=$T, n=$n)"; reps=reps)
end

# ===========================================================================
# Backend registration — each guarded so the suite runs (skipping what's not
# installed) rather than failing outright; run bench/suite.jl in an
# environment with the AD backends you want to compare actually added.
# ===========================================================================

push!(_AD_BACKENDS, "ForwardDiff" => AutoForwardDiff())
if !isnothing(Base.find_package("Mooncake"))
    @eval import Mooncake
    push!(_AD_BACKENDS, "Mooncake" => AutoMooncake(; config=nothing))
end
# ReverseDiff dropped entirely — redundant with Mooncake (both reverse-mode;
# Mooncake is the actively-developed one this package's own test suite
# already leans on), matching the corpus-wide benchmark scripts.
if !isnothing(Base.find_package("Enzyme"))
    @eval import Enzyme
    # set_runtime_activity — see bench/corpus_bench_worker.jl's matching
    # comment: bare AutoEnzyme() defaults to runtime-activity analysis OFF.
    push!(_AD_BACKENDS, "Enzyme" => AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)))
end

# ===========================================================================
# Run everything and print a report.
# ===========================================================================

function run_suite()
    empty!(_RESULTS)

    println("="^100)
    println("Layer 1: logdensity only, Float64")
    println("="^100)
    for n in (20, 50_000)
        shape = n <= 100 ? "tiny" : "large"
        r_raw, r_pb = bench_logdensity_only(Float64, n)
        println(r_raw)
        println(r_pb)
        println("  -> PracticalBayes / raw ratio (median): ", round(r_pb.median_s / r_raw.median_s; digits=3))
        record!(; package="PracticalBayes", layer="logdensity", model="normal", shape, precision="Float64", backend="none", r=r_pb)
        record!(; package="PracticalBayes", layer="logdensity_raw", model="normal", shape, precision="Float64", backend="none", r=r_raw)
    end

    println()
    println("="^100)
    println("Layer 1: logdensity only, Float32")
    println("="^100)
    for n in (20, 50_000)
        shape = n <= 100 ? "tiny" : "large"
        r_raw, r_pb = bench_logdensity_only(Float32, n)
        println(r_raw)
        println(r_pb)
        println("  -> PracticalBayes / raw ratio (median): ", round(r_pb.median_s / r_raw.median_s; digits=3))
        record!(; package="PracticalBayes", layer="logdensity", model="normal", shape, precision="Float32", backend="none", r=r_pb)
        record!(; package="PracticalBayes", layer="logdensity_raw", model="normal", shape, precision="Float32", backend="none", r=r_raw)
    end

    println()
    println("="^100)
    println("Layer 2: logdensity_and_gradient, per AD backend (Float64)")
    println("="^100)
    for n in (20, 50_000)
        shape = n <= 100 ? "tiny" : "large"
        for (name, r) in zip(first.(_AD_BACKENDS), bench_gradient(Float64, n))
            println(r)
            record!(; package="PracticalBayes", layer="gradient", model="normal", shape, precision="Float64", backend=name, r)
        end
    end

    println()
    println("="^100)
    println("Layer 2: logdensity_and_gradient, per AD backend (Float32)")
    println("="^100)
    for n in (20, 50_000)
        shape = n <= 100 ? "tiny" : "large"
        for (name, r) in zip(first.(_AD_BACKENDS), bench_gradient(Float32, n))
            println(r)
            record!(; package="PracticalBayes", layer="gradient", model="normal", shape, precision="Float32", backend=name, r)
        end
    end

    println()
    println("="^100)
    println("Layer 3: end-to-end NUTS (ForwardDiff only, both precisions)")
    println("  NOTE: for a model this simple, expect this to be dominated by")
    println("  AdvancedHMC's own per-iteration overhead, not model evaluation.")
    println("="^100)
    for n in (20, 50_000)
        shape = n <= 100 ? "tiny" : "large"
        r64 = bench_nuts(Float64, n)
        r32 = bench_nuts(Float32, n)
        println(r64)
        println(r32)
        record!(; package="PracticalBayes", layer="nuts", model="normal", shape, precision="Float64", backend="ForwardDiff", r=r64)
        record!(; package="PracticalBayes", layer="nuts", model="normal", shape, precision="Float32", backend="ForwardDiff", r=r32)
    end

    println()
    println("="^100)
    println("Layer 2b: MANY-PARAMETER model (K=200, N=2000) — where reverse-mode")
    println("  AD should start beating forward-mode, unlike the 2-parameter shapes above.")
    println("="^100)
    for T in (Float64, Float32)
        for (name, r) in bench_manyparam_gradient(T, 200, 2000)
            if r !== nothing
                println(r)
                record!(; package="PracticalBayes", layer="gradient", model="manyparam", shape="K200_N2000", precision=string(T), backend=name, r)
            end
        end
    end

    println()
    println("="^100)
    println("Layer 1b: DISCRETE-LIKELIHOOD model (Poisson regression, k=5, n=2000/50000)")
    println("="^100)
    for n in (2000, 50_000)
        r_raw, r_pb = bench_poisson_density(5, n)
        println(r_raw)
        println(r_pb)
        println("  -> PracticalBayes / raw ratio (median): ", round(r_pb.median_s / r_raw.median_s; digits=3))
        record!(; package="PracticalBayes", layer="logdensity", model="poisson", shape="k5_n$n", precision="Float64", backend="none", r=r_pb)
        record!(; package="PracticalBayes", layer="logdensity_raw", model="poisson", shape="k5_n$n", precision="Float64", backend="none", r=r_raw)
    end

    write_history!()
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_suite()
