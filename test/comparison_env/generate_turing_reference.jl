# Run ONCE (manually, in this directory's own environment: `julia --project=.
# generate_turing_reference.jl`) to produce `turing_reference.jl` — a plain
# Julia source file of literal numbers that `test/turing_comparison.jl` reads
# back and compares against, WITHOUT ever loading Turing itself.
#
# Why this exists at all: Turing 0.45 (newest released) depends on
# AbstractPPL 0.14.2, but PracticalBayes depends on AbstractPPL 0.15.x (the
# VectorBijectors-based Bijectors API the whole layout system is built on) —
# these two genuinely cannot be loaded in the same Julia process, in any
# environment, with any compat tuning. So instead of a live side-by-side
# comparison, Turing's numbers are captured once here and frozen into a file
# PracticalBayes' own test suite checks against later, in a separate process
# that never touches Turing at all.
#
# Regenerate this file (by rerunning this script in test/comparison_env/)
# only if the reference MODELS themselves change; ordinary PracticalBayes
# code changes should never need this rerun.
#
# FULL PARITY with bench/suite.jl: same tiny(n=20)/large(n=50_000) x
# Float64/Float32 matrix, AND the same three layers — (1) density only,
# (2) density+gradient per available AD backend (via
# `DynamicPPL.LogDensityFunction(model; adtype=...)`, Turing's own
# LogDensityProblems-based gradient path — mirrors our own
# `LogDensityFunction` almost exactly), (3) end-to-end NUTS timing. Whatever
# this package benchmarks about itself, Turing gets benchmarked on too.

using Pkg
Pkg.activate(@__DIR__)

using Turing
using DynamicPPL
using LogDensityProblems
using ADTypes
using StableRNGs: StableRNG
using Statistics: mean, median, std
using Distributions: Normal, Exponential, Poisson, MvNormal, logpdf
using Dates: now
using Random
using LinearAlgebra: I, dot

println("Turing version: ", pkgversion(Turing))

# ===========================================================================
# Same hand-rolled timing harness as bench/suite.jl (first-call time
# reported separately from steady-state min/median/mean/std) — kept as a
# verbatim copy here rather than a shared dependency, since this script runs
# in an entirely separate environment from the main package (see above).
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

function time_reps(f, label; reps=30)
    first_call_s = @elapsed f()
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        times[i] = @elapsed f()
    end
    return TimingResult(label, first_call_s, minimum(times), median(times), mean(times), std(times), reps)
end

# ===========================================================================
# Results recording — same JSONL history mechanism as bench/suite.jl, written
# to a parallel file (`bench/results/history_turing.jsonl`) so a report can
# join both sides by (layer, model, shape, precision, backend) without ever
# loading Turing and PracticalBayes in the same process. `package` is always
# "Turing" here. Deliberately a verbatim copy of bench/suite.jl's recorder
# rather than a shared dependency — this script runs in its own environment.
# ===========================================================================

const _RESULTS = NamedTuple[]

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

function _git_commit()
    try
        return strip(read(`git -C $(joinpath(@__DIR__, "..", "..")) rev-parse HEAD`, String))
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

function write_history!(path=joinpath(@__DIR__, "..", "..", "bench", "results", "history_turing.jsonl"))
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

# ===========================================================================
# Accuracy reference: conjugate Normal-Normal model, y_i ~ Normal(mu, 1),
# mu ~ Normal(0, 1). Same data-generation seed/procedure as
# test/turing_comparison.jl's PracticalBayes side.
# ===========================================================================
rng = StableRNG(42)
y_accuracy = randn(rng, 30) .+ 2.0

Turing.@model function turing_conjugate(y)
    mu ~ Normal(0, 1)
    for i in eachindex(y)
        y[i] ~ Normal(mu, 1)
    end
end

turing_chain = Turing.sample(StableRNG(1), turing_conjugate(y_accuracy), Turing.NUTS(0.8), 2000; progress=false)
# Turing 0.45 returns a FlexiChains-backed `VNChain`, not the older
# `MCMCChains.Chains` — `chain[:mu]` gives a DimMatrix (iter x chain) that
# plain `vec`/`mean`/`std` handle like any other array once sliced.
turing_mu_draws = vec(Array(turing_chain[:mu])[501:end, :])
turing_mu_mean = mean(turing_mu_draws)
turing_mu_std = std(turing_mu_draws)

println("Turing conjugate posterior: mean=", turing_mu_mean, " std=", turing_mu_std)

# ===========================================================================
# Model definitions — unconditional top-level (NOT inside a function with an
# if/else picking between two `@model function m(...)` definitions: that
# pattern redefines the same top-level name `m` conditionally, which
# produces "Method definition overwritten" warnings and an UndefVarError
# inside the function's local scope, since `@model`'s generated
# `function m(...)` def always binds at module/global scope regardless of
# which branch of the `if` textually contains it — discovered the hard way
# earlier in this file's history).
#
# IMPORTANT finding, captured explicitly rather than silently worked around:
# DynamicPPL's `getlogjoint` PROMOTES BACK TO Float64 even when a model is
# written entirely with Float32 literals (`Normal(0f0, 1f0)` etc) — confirmed
# by checking `typeof(DynamicPPL.getlogjoint(vi))` on a Float32-literal model.
# This is exactly the "Turing/DynamicPPL forces everything to Float64"
# limitation motivating this package's numeric-type genericity requirement
# in the first place. So the "Float32" rows below reflect Float32
# data/model literals but a Float64 computation internally in most cases —
# NOT an apples-to-apples Float32-compute comparison on Turing's side; that
# gap is itself part of what's measured.
# ===========================================================================

Turing.@model function turing_density_f64(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    for i in eachindex(y)
        y[i] ~ Normal(mu, sigma)
    end
end

Turing.@model function turing_density_f32(y)
    mu ~ Normal(0.0f0, 1.0f0)
    sigma ~ Exponential(1.0f0)
    for i in eachindex(y)
        y[i] ~ Normal(mu, sigma)
    end
end

Turing.@model function turing_addlogprob_f64(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    Turing.@addlogprob! sum(logpdf.(Normal(mu, sigma), y))
end

Turing.@model function turing_addlogprob_f32(y)
    mu ~ Normal(0.0f0, 1.0f0)
    sigma ~ Exponential(1.0f0)
    Turing.@addlogprob! sum(logpdf.(Normal(mu, sigma), y))
end

density_model(T, y) = T == Float64 ? turing_density_f64(y) : turing_density_f32(y)
addlogprob_model(T, y) = T == Float64 ? turing_addlogprob_f64(y) : turing_addlogprob_f32(y)

# ===========================================================================
# Layer 1: density only (model(vi) call, matching bench/suite.jl's
# raw-loop-vs-framework comparison).
# ===========================================================================

function bench_turing_density(T, n; reps=30)
    rng_local = StableRNG(n + (T == Float32 ? 1 : 0))
    y = T.(randn(rng_local, n))
    model = density_model(T, y)
    vi = DynamicPPL.VarInfo(model)
    return time_reps(() -> model(vi), "Turing density (T=$T, n=$n)"; reps=reps)
end

function bench_turing_addlogprob(T, n; reps=30)
    rng_local = StableRNG(n + (T == Float32 ? 1 : 0) + 100)
    y = T.(randn(rng_local, n))
    model = addlogprob_model(T, y)
    vi = DynamicPPL.VarInfo(model)
    return time_reps(() -> model(vi), "Turing @addlogprob! (T=$T, n=$n)"; reps=reps)
end

# ===========================================================================
# Layer 2: density + gradient, per AD backend, via
# `DynamicPPL.LogDensityFunction(model; adtype=...)` — Turing's own
# LogDensityProblems-based gradient path, the same interface
# `LogDensityProblems.logdensity_and_gradient` our own `LogDensityFunction`
# implements. Backends registered the same way as bench/suite.jl: only added
# if actually installed AND actually `import`ed in this session (DI's
# per-backend extensions only activate once the backend package is loaded,
# not merely installed).
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
if !isnothing(Base.find_package("Enzyme"))
    @eval import Enzyme
    push!(_AD_BACKENDS, "Enzyme" => AutoEnzyme())
end

function bench_turing_gradient(T, n, backend_name, adtype; reps=30)
    rng_local = StableRNG(n + (T == Float32 ? 1 : 0) + 200)
    y = T.(randn(rng_local, n))
    model = density_model(T, y)
    ldf = DynamicPPL.LogDensityFunction(model; adtype=adtype)
    params = zeros(T, LogDensityProblems.dimension(ldf))
    return time_reps(
        () -> LogDensityProblems.logdensity_and_gradient(ldf, params), "Turing $backend_name gradient (T=$T, n=$n)"; reps=reps
    )
end

# ===========================================================================
# Layer 3: end-to-end NUTS, same n_samples/reps as bench/suite.jl's
# bench_nuts, using Turing's own `Turing.sample`. `δ` fixed at Float64 0.8
# regardless of `T` — unlike AdvancedHMC used directly, Turing's NUTS
# constructor doesn't require a type-matched acceptance target (Turing
# handles the promotion internally, consistent with the Float64-promotion
# finding above).
# ===========================================================================

function bench_turing_nuts(T, n; n_samples=200, reps=5)
    rng_local = StableRNG(n + (T == Float32 ? 1 : 0) + 300)
    y = T.(randn(rng_local, n))
    model = density_model(T, y)
    run() = Turing.sample(StableRNG(1), model, Turing.NUTS(0.8), n_samples; progress=false)
    return time_reps(run, "Turing NUTS $n_samples samples (T=$T, n=$n)"; reps=reps)
end

# ===========================================================================
# Layer 2b: MANY-PARAMETER model (K=200, N=2000) — matches
# bench/suite.jl's `manyparam_model` exactly (same coefficient-vector
# regression), so forward-vs-reverse-mode AD crossover behavior can be
# compared on both sides, not just PracticalBayes in isolation.
# ===========================================================================

Turing.@model function turing_manyparam(X, y)
    K = size(X, 2)
    beta ~ MvNormal(zeros(K), I)
    sigma ~ Exponential(1)
    for i in eachindex(y)
        y[i] ~ Normal(dot(view(X, i, :), beta), sigma)
    end
end

function make_manyparam_data(k, n, seed)
    rng_local = StableRNG(seed)
    X = randn(rng_local, n, k)
    true_beta = randn(rng_local, k)
    y = X * true_beta .+ randn(rng_local, n) .* 0.5
    return X, y
end

function bench_turing_manyparam_gradient(k, n, backend_name, adtype; reps=20)
    X, y = make_manyparam_data(k, n, 5)
    model = turing_manyparam(X, y)
    ldf = DynamicPPL.LogDensityFunction(model; adtype=adtype)
    params = zeros(LogDensityProblems.dimension(ldf))
    return time_reps(
        () -> LogDensityProblems.logdensity_and_gradient(ldf, params),
        "Turing $backend_name gradient (manyparam K=$k, n=$n)";
        reps=reps,
    )
end

# ===========================================================================
# Layer 1b: DISCRETE-LIKELIHOOD (Poisson regression) — matches
# bench/suite.jl's `poisson_model` exactly.
# ===========================================================================

Turing.@model function turing_poisson(X, y)
    K = size(X, 2)
    beta ~ MvNormal(zeros(K), I)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(dot(view(X, i, :), beta)))
    end
end

function make_poisson_data(k, n, seed)
    rng_local = StableRNG(seed)
    X = randn(rng_local, n, k)
    true_beta = randn(rng_local, k) .* 0.3
    y = [rand(rng_local, Poisson(exp(clamp(dot(view(X, i, :), true_beta), -10, 10)))) for i in 1:n]
    return X, Float64.(y)
end

function bench_turing_poisson_density(k, n; reps=50)
    X, y = make_poisson_data(k, n, 6)
    model = turing_poisson(X, y)
    vi = DynamicPPL.VarInfo(model)
    return time_reps(() -> model(vi), "Turing Poisson density (k=$k, n=$n)"; reps=reps)
end

# ===========================================================================
# Run everything.
# ===========================================================================

shapes = ((20, "tiny"), (50_000, "large"))
precisions = (Float64, Float32)

speed_results = Dict{Tuple{String,String},TimingResult}()
for (n, shapename) in shapes, T in precisions
    r_tilde = bench_turing_density(T, n)
    r_add = bench_turing_addlogprob(T, n)
    println(r_tilde.label, ": median=", r_tilde.median_s, "s  first_call=", r_tilde.first_call_s, "s")
    println(r_add.label, ": median=", r_add.median_s, "s  first_call=", r_add.first_call_s, "s")
    speed_results[(shapename, "$(T)_tilde")] = r_tilde
    speed_results[(shapename, "$(T)_addlogprob")] = r_add
    record!(; package="Turing", layer="logdensity", model="normal", shape=shapename, precision=string(T), backend="none", r=r_tilde)
    record!(; package="Turing", layer="logdensity_addlogprob", model="normal", shape=shapename, precision=string(T), backend="none", r=r_add)
end

gradient_results = Dict{Tuple{String,String,String},TimingResult}()
for (n, shapename) in shapes, T in precisions, (backend_name, adtype) in _AD_BACKENDS
    r = bench_turing_gradient(T, n, backend_name, adtype)
    println(r.label, ": median=", r.median_s, "s  first_call=", r.first_call_s, "s")
    gradient_results[(shapename, "$T", backend_name)] = r
    record!(; package="Turing", layer="gradient", model="normal", shape=shapename, precision=string(T), backend=backend_name, r)
end

nuts_results = Dict{Tuple{String,String},TimingResult}()
for (n, shapename) in shapes, T in precisions
    r = bench_turing_nuts(T, n)
    println(r.label, ": median=", r.median_s, "s  first_call=", r.first_call_s, "s")
    nuts_results[(shapename, "$T")] = r
    record!(; package="Turing", layer="nuts", model="normal", shape=shapename, precision=string(T), backend="ForwardDiff", r)
end

manyparam_results = Dict{Tuple{String,String},TimingResult}()
for T in (Float64, Float32), (backend_name, adtype) in _AD_BACKENDS
    T == Float32 && continue  # MvNormal(zeros(K), I) is Float64-only in this Turing model; skip rather than force a mismatch
    r = bench_turing_manyparam_gradient(200, 2000, backend_name, adtype)
    println(r.label, ": median=", r.median_s, "s  first_call=", r.first_call_s, "s")
    manyparam_results[("$T", backend_name)] = r
    record!(; package="Turing", layer="gradient", model="manyparam", shape="K200_N2000", precision=string(T), backend=backend_name, r)
end

poisson_results = Dict{Int,TimingResult}()
for n in (2000, 50_000)
    r = bench_turing_poisson_density(5, n)
    println(r.label, ": median=", r.median_s, "s  first_call=", r.first_call_s, "s")
    poisson_results[n] = r
    record!(; package="Turing", layer="logdensity", model="poisson", shape="k5_n$n", precision="Float64", backend="none", r)
end

write_history!()

# ===========================================================================
# Write the reference file. Plain Julia literals only — no Turing types
# leak into it, so it's safe to `include` from an environment that has never
# heard of Turing.
# ===========================================================================
out_path = joinpath(@__DIR__, "..", "turing_reference.jl")
_fmt_result(r) = "(first_call_s=$(r.first_call_s), min_s=$(r.min_s), median_s=$(r.median_s), mean_s=$(r.mean_s), std_s=$(r.std_s))"
open(out_path, "w") do io
    println(io, "# AUTO-GENERATED by test/comparison_env/generate_turing_reference.jl")
    println(io, "# Regenerate only if the reference models themselves change (see that")
    println(io, "# script for why this can't just be a live comparison at test time).")
    println(io, "# Captured against Turing v$(pkgversion(Turing)) on $(Sys.MACHINE), $(now())")
    println(io, "#")
    println(io, "# NOTE: the Float32 rows use Float32 data/model literals, but Turing/")
    println(io, "# DynamicPPL's getlogjoint promotes internally back to Float64 regardless")
    println(io, "# (confirmed directly) — this is NOT an apples-to-apples Float32-compute")
    println(io, "# comparison on Turing's side; that gap is itself part of what's measured.")
    println(io, "#")
    println(io, "# AD backends benchmarked here: ", join(first.(_AD_BACKENDS), ", "), " (whichever were")
    println(io, "# installed + `import`ed when this was generated — see the script for how")
    println(io, "# backends are registered).")
    println(io)
    println(io, "const TURING_REFERENCE = (")
    println(io, "    accuracy = (mu_mean = ", turing_mu_mean, ", mu_std = ", turing_mu_std, "),")
    println(io, "    speed = Dict(")
    for ((shapename, key), r) in speed_results
        println(io, "        (\"$shapename\", \"$key\") => ", _fmt_result(r), ",")
    end
    println(io, "    ),")
    println(io, "    gradient = Dict(")
    for ((shapename, Tname, backend_name), r) in gradient_results
        println(io, "        (\"$shapename\", \"$Tname\", \"$backend_name\") => ", _fmt_result(r), ",")
    end
    println(io, "    ),")
    println(io, "    nuts = Dict(")
    for ((shapename, Tname), r) in nuts_results
        println(io, "        (\"$shapename\", \"$Tname\") => ", _fmt_result(r), ",")
    end
    println(io, "    ),")
    println(io, "    manyparam = Dict(")
    for ((Tname, backend_name), r) in manyparam_results
        println(io, "        (\"$Tname\", \"$backend_name\") => ", _fmt_result(r), ",")
    end
    println(io, "    ),")
    println(io, "    poisson = Dict(")
    for (n, r) in poisson_results
        println(io, "        $n => ", _fmt_result(r), ",")
    end
    println(io, "    ),")
    println(io, ")")
end
println("Wrote reference to: ", out_path)
