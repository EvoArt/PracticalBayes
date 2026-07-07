# Requirement-1 microbenchmark: `y ~ Normal(mu, sigma)` on data must cost the
# same as `Turing.@addlogprob! logpdf(Normal(mu, sigma), y)`. This is the
# concrete acceptance gate for "no VarName, no context walk, no accumulator-
# tuple rebuild" — if this benchmark shows PracticalBayes noticeably slower
# than a raw `sum(logpdf.(...))` loop, something in the tilde/EvalMode hot
# path has regressed.
#
# This is a QUICK single-shot check (one BenchmarkTools @btime per side).
# For the fuller picture — multiple model sizes, multiple AD backends,
# Float32, and an honest breakdown of density-only vs gradient vs full-NUTS
# cost (with explicit fixed-repetition timing, not BenchmarkTools' adaptive
# budget, which can degenerate to a single sample on expensive
# configurations) — see `bench/suite.jl` instead.
#
# Run manually with: julia --project=. bench/observe_overhead.jl
# (Requires BenchmarkTools; not a package dependency — install into your own
# global/shared environment if you don't already have it.)

using PracticalBayes
using Distributions: Normal, Exponential, logpdf
using BenchmarkTools: @btime
import LogDensityProblems

const N = 10_000
const Y = randn(N)

# Baseline: the theoretical floor. No framework at all, just a loop/broadcast.
raw_logpdf(mu, sigma, y) = sum(logpdf.(Normal(mu, sigma), y))

# PracticalBayes: the model whose `.~` observe site should hit exactly the
# same cost as `raw_logpdf` above (see tilde.jl `tilde(::EvalMode, ...)` for
# the scalar path and `tilde_dot` for the vectorized path exercised here).
@model function observe_bench(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end

model = observe_bench(Y)
layout, θ0, store0 = build_layout(model)
ldf = LogDensityFunction(model, layout, store0)

println("Baseline: sum(logpdf.(Normal(mu, sigma), y)) over N=$N")
@btime raw_logpdf(0.0, 1.0, $Y)

println("\nPracticalBayes: LogDensityProblems.logdensity(ldf, θ0) over N=$N")
@btime LogDensityProblems.logdensity($ldf, $θ0)

println()
println("If the PracticalBayes number above is within ~1.1x of the baseline,")
println("the observe-overhead requirement is met (M1 acceptance gate).")
