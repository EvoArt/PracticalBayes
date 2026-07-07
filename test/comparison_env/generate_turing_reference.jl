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
# Matches the shape/precision matrix in bench/suite.jl (tiny=20 obs,
# large=50_000 obs; Float64 and Float32) so the two sides are actually
# comparable, not just a single arbitrarily-chosen model size.

using Pkg
Pkg.activate(@__DIR__)

using Turing
using DynamicPPL
using StableRNGs: StableRNG
using Statistics: mean, median, std
using Distributions: Normal, Exponential, logpdf
using Dates: now

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
# Speed reference, density-only, over the SAME tiny/large x Float64/Float32
# matrix as bench/suite.jl. `Turing.VarInfo` isn't directly exported in this
# version; the underlying type lives in DynamicPPL (added here directly).
#
# IMPORTANT finding, captured explicitly rather than silently worked around:
# DynamicPPL's `getlogjoint` PROMOTES BACK TO Float64` even when a model is
# written entirely with Float32 literals (`Normal(0f0, 1f0)` etc) — confirmed
# by checking `typeof(DynamicPPL.getlogjoint(vi))` on a Float32-literal model
# before writing this script. This is exactly the "Turing/DynamicPPL forces
# everything to Float64" limitation motivating this package's numeric-type
# genericity requirement in the first place (see devlog). So the "Float32"
# row for Turing below reflects Float32 DATA/model-literals but a Float64
# computation internally — it is not an apples-to-apples Float32 compute
# comparison, and that gap IS the point being measured.
# ===========================================================================

# Defined unconditionally at top level (NOT inside a function with an
# if/else picking between two `@model function m(...)` definitions — that
# pattern redefines the same top-level name `m` conditionally, which
# produces "Method definition overwritten" warnings and, worse, an
# UndefVarError inside the function's local scope, since `@model`'s
# generated `function m(...)` def always binds at module/global scope
# regardless of which branch of the `if` textually contains it).
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

function bench_turing_density(T, n; reps=30)
    rng_local = StableRNG(n + (T == Float32 ? 1 : 0))
    y = T.(randn(rng_local, n))
    model = T == Float64 ? turing_density_f64(y) : turing_density_f32(y)
    vi = DynamicPPL.VarInfo(model)
    return time_reps(() -> model(vi), "Turing density (T=$T, n=$n)"; reps=reps)
end

function bench_turing_addlogprob(T, n; reps=30)
    rng_local = StableRNG(n + (T == Float32 ? 1 : 0) + 100)
    y = T.(randn(rng_local, n))
    model = T == Float64 ? turing_addlogprob_f64(y) : turing_addlogprob_f32(y)
    vi = DynamicPPL.VarInfo(model)
    return time_reps(() -> model(vi), "Turing @addlogprob! (T=$T, n=$n)"; reps=reps)
end

shapes = ((20, "tiny"), (50_000, "large"))
precisions = (Float64, Float32)

results = Dict{Tuple{String,String},TimingResult}()
for (n, shapename) in shapes, T in precisions
    r_tilde = bench_turing_density(T, n)
    r_add = bench_turing_addlogprob(T, n)
    println(r_tilde.label, ": median=", r_tilde.median_s, "s  first_call=", r_tilde.first_call_s, "s")
    println(r_add.label, ": median=", r_add.median_s, "s  first_call=", r_add.first_call_s, "s")
    results[(shapename, "$(T)_tilde")] = r_tilde
    results[(shapename, "$(T)_addlogprob")] = r_add
end

# ===========================================================================
# Write the reference file. Plain Julia literals only — no Turing types
# leak into it, so it's safe to `include` from an environment that has never
# heard of Turing.
# ===========================================================================
out_path = joinpath(@__DIR__, "..", "turing_reference.jl")
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
    println(io)
    println(io, "const TURING_REFERENCE = (")
    println(io, "    accuracy = (mu_mean = ", turing_mu_mean, ", mu_std = ", turing_mu_std, "),")
    println(io, "    speed = Dict(")
    for ((shapename, key), r) in results
        println(
            io,
            "        (\"$shapename\", \"$key\") => (first_call_s=", r.first_call_s,
            ", min_s=", r.min_s, ", median_s=", r.median_s, ", mean_s=", r.mean_s, ", std_s=", r.std_s, "),",
        )
    end
    println(io, "    ),")
    println(io, ")")
end
println("Wrote reference to: ", out_path)
