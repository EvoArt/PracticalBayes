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
using Distributions: Normal, Exponential, logpdf
using Random
using Statistics: mean, median, std
using ADTypes
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
# ===========================================================================

function bench_nuts(::Type{T}, n; n_samples=200, reps=5) where {T<:Real}
    y = make_data(T, n, 3)
    m = n <= 100 ? tiny_model(y) : large_model(y)
    layout, θ0, store0 = build_layout(m; T=T)
    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    ldm = AbstractMCMC.LogDensityModel(ldf)
    δ = T(0.8)
    run() = AbstractMCMC.sample(Random.Xoshiro(1), ldm, AdvancedHMC.NUTS(δ), n_samples; initial_params=θ0, progress=false)
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
if !isnothing(Base.find_package("ReverseDiff"))
    @eval import ReverseDiff
    push!(_AD_BACKENDS, "ReverseDiff" => AutoReverseDiff())
end
if !isnothing(Base.find_package("Enzyme"))
    @eval import Enzyme
    push!(_AD_BACKENDS, "Enzyme" => AutoEnzyme())
end

# ===========================================================================
# Run everything and print a report.
# ===========================================================================

function run_suite()
    println("="^100)
    println("Layer 1: logdensity only, Float64")
    println("="^100)
    for n in (20, 50_000)
        r_raw, r_pb = bench_logdensity_only(Float64, n)
        println(r_raw)
        println(r_pb)
        println("  -> PracticalBayes / raw ratio (median): ", round(r_pb.median_s / r_raw.median_s; digits=3))
    end

    println()
    println("="^100)
    println("Layer 1: logdensity only, Float32")
    println("="^100)
    for n in (20, 50_000)
        r_raw, r_pb = bench_logdensity_only(Float32, n)
        println(r_raw)
        println(r_pb)
        println("  -> PracticalBayes / raw ratio (median): ", round(r_pb.median_s / r_raw.median_s; digits=3))
    end

    println()
    println("="^100)
    println("Layer 2: logdensity_and_gradient, per AD backend (Float64)")
    println("="^100)
    for n in (20, 50_000)
        for r in bench_gradient(Float64, n)
            println(r)
        end
    end

    println()
    println("="^100)
    println("Layer 2: logdensity_and_gradient, per AD backend (Float32)")
    println("="^100)
    for n in (20, 50_000)
        for r in bench_gradient(Float32, n)
            println(r)
        end
    end

    println()
    println("="^100)
    println("Layer 3: end-to-end NUTS (ForwardDiff only, both precisions)")
    println("  NOTE: for a model this simple, expect this to be dominated by")
    println("  AdvancedHMC's own per-iteration overhead, not model evaluation.")
    println("="^100)
    for n in (20, 50_000)
        println(bench_nuts(Float64, n))
        println(bench_nuts(Float32, n))
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_suite()
