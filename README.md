# PracticalBayes

[![CI](https://github.com/EvoArt/PracticalBayes/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoArt/PracticalBayes/actions/workflows/ci.yml)
[![Benchmark Sweep](https://github.com/EvoArt/PracticalBayes/actions/workflows/benchmark.yml/badge.svg)](https://github.com/EvoArt/PracticalBayes/actions/workflows/benchmark.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://EvoArt.github.io/PracticalBayes)

PracticalBayes is a performance-oriented probabilistic programming package built to keep clean Turing-style modeling syntax while targeting fast inference paths from day one.

## Why this package exists

- Target **GPU-oriented workflows** and **latent-state samplers** from the start, not as an afterthought.
- Keep model code clean and familiar (`@model`, `~`, `arraydist`, etc.) while still getting top performance.
- Make high performance the default so users do not need to memorize optimizer/compiler tricks to get good results.
- Stay compatible with modern AD tooling via **DifferentiationInterface** (ForwardDiff, Mooncake, Enzyme).
- Include practical features for large models, including **untracked nuisance parameters** and **Float32-first paths**.

## Relationship to Turing

Turing is probably my favourite Julia package of all time.

PracticalBayes is built almost entirely on the Turing ecosystem (AbstractPPL, DynamicPPL-adjacent patterns, AdvancedHMC, Bijectors, Distributions, LogDensityProblems, and related tooling). The goal is not to replace Turing's scope.

PracticalBayes will likely never be as general as Turing, by design. It intentionally keeps a narrower focus so the core goals above remain first-class.

## Example

```julia
using PracticalBayes
using ADTypes
using Random

@model function demo_model(y)
    μ ~ Normal(0, 1)
    σ ~ Exponential(1)
    y ~ arraydist(Normal.(μ, σ))
end

m = demo_model(randn(100))
chain = sample(Random.default_rng(), m, NUTS(0.8), 1000; adtype=AutoForwardDiff())
```

## Features

- Static layout + efficient constrained/unconstrained parameter transforms.
- `LogDensityFunction` interface for direct integration with samplers/optimizers.
- Nuisance-parameter support (`untracked`) for large latent models.
- MAP/MLE/Laplace tools with optional Optimization.jl extension.
- Float32 performance path for parameter-heavy models.
- AD backend flexibility via DifferentiationInterface.
- **Trimmable models**: compile a model to a standalone native binary with
  `juliac --trim=safe` — a few MB, no Julia install needed at the target, and
  milliseconds to start instead of JIT latency. This needs an AD backend whose
  derivative code exists as ordinary compiled Julia, which rules out every
  established package (they each build it at runtime, via a config, a thunk, an
  interpreter, or `eval`). [Piste.jl](https://github.com/EvoArt/Piste.jl)
  provides one: `import Piste` and pass `AutoPBForwardDiff()`. Forward mode, so
  reverse mode still wins on parameter-heavy models — reach for it when you
  want a binary. Requires Julia 1.12+; Piste is a weak dependency, so nothing
  changes for anyone who is not trimming.
- **`depends=` annotations for Gibbs**: `@addlogprob! expr depends=(:a, :b)`
  tells the evaluator which variables a term actually involves, so a Gibbs block
  owning none of them skips it entirely. Exact, not approximate — but only if
  the annotation is right, and under-declaring fails SILENTLY (the block’s
  gradient quietly loses a contribution). `check_depends` is the guard: it
  compares each block’s gradient as written against the same gradient with
  nothing skipped, so the hint is verified rather than trusted.

## Benchmark summary (auto-updated)

Ratios below are median gradient time `PracticalBayes / Turing` (lower is better for PracticalBayes).

<!-- BENCH:START -->
Ratios are median gradient time `PracticalBayes / Turing` (`< 1` means PracticalBayes faster). Corner cells come from the 5x5 sweep grid: N in [50, 200, 1000, 5000, 20000], NPARAMS in [2, 10, 50, 100, 200].

### Forwarddiff

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 40.111 | 0.073 | 0.535 | 0.011 |
| `normal` | `Float32` | 39.35 | 0.159 | 0.854 | 0.009 |
| `poisson` | `Float64` | 2.641 | 3.926 | 0.261 | 3.204 |
| `poisson` | `Float32` | 5.976 | 2.586 | 0.235 | 3.881 |
| `bernoulli_logit` | `Float64` | 18.076 | 0.065 | 0.578 | 0.015 |
| `bernoulli_logit` | `Float32` | 9.944 | 0.079 | 0.548 | 0.013 |

### Mooncake

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 9.984 | 1.075 | 0.132 | 0.318 |
| `normal` | `Float32` | 4.797 | 1.128 | 0.154 | 0.183 |
| `poisson` | `Float64` | 1.841 | 55.336 | 0.127 | 54.004 |
| `poisson` | `Float32` | 3.48 | 49.193 | 0.126 | 64.537 |
| `bernoulli_logit` | `Float64` | 2.238 | 0.895 | 0.139 | 0.251 |
| `bernoulli_logit` | `Float32` | 1.867 | 0.572 | 0.237 | 0.212 |

### Enzyme

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 17.392 | 1.754 | 0.277 | 0.229 |
| `normal` | `Float32` | 9.122 | 2.646 | 0.232 | 0.195 |
| `poisson` | `Float64` | n/a | n/a | n/a | n/a |
| `poisson` | `Float32` | n/a | n/a | n/a | n/a |
| `bernoulli_logit` | `Float64` | n/a | n/a | n/a | n/a |
| `bernoulli_logit` | `Float32` | n/a | n/a | n/a | n/a |


### Piste (PracticalBayes' own AD)

[Piste](https://github.com/EvoArt/Piste.jl) is the only backend here that survives `juliac --trim`, i.e. the only one that lets a model compile to a standalone binary. Turing cannot use it, so there is no PB/Turing ratio to give; these ratios are instead `Piste / fastest of ForwardDiff, Mooncake, Enzyme` on the same cell, with `< 1` meaning Piste is faster.

Read the two tables together. **Forward mode** wins at small problems (roughly 20x faster at N=50, P=2, where the other backends' per-call setup dominates) and loses badly at large P, because its cost is O(P) by construction -- the 100x+ figures in the bottom-right are that, not a defect. **Reverse mode** is the one to use on parameter-heavy models; it costs one sweep regardless of P.

#### Forward mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 0.047 | 17.99 | 3.21 | 192.932 |
| `normal` | `Float32` | 0.037 | 15.082 | 2.026 | 101.51 |
| `poisson` | `Float64` | 0.098 | 0.264 | 2.125 | 0.583 |
| `poisson` | `Float32` | 0.086 | 0.188 | 2.013 | 0.215 |
| `bernoulli_logit` | `Float64` | 0.088 | 22.927 | 1.584 | 123.901 |
| `bernoulli_logit` | `Float32` | 0.081 | 11.693 | 1.662 | 68.591 |


#### Reverse mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 0.136 | 0.985 | 11.861 | 5.074 |
| `normal` | `Float32` | 0.136 | 2.382 | 13.803 | 27.952 |
| `poisson` | `Float64` | 0.223 | 0.018 | 5.082 | 0.019 |
| `poisson` | `Float32` | 0.199 | 0.028 | 7.322 | 0.041 |
| `bernoulli_logit` | `Float64` | 0.26 | 1.237 | 3.82 | 4.582 |
| `bernoulli_logit` | `Float32` | 0.229 | 1.634 | 4.377 | 10.727 |

<!-- BENCH:END -->

Heatmaps generated by CI (5x5 N x NPARAMS grid):

- `benchmarks/figures/fig1_forwarddiff.png`
- `benchmarks/figures/fig2_mooncake.png`
- `benchmarks/figures/fig3_enzyme.png`
- `benchmarks/figures/fig4_fastest_per_ppl.png`

## Development

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

For benchmark sweeps:

```julia
julia --project=benchmarks benchmarks/sweep.jl
julia --project=benchmarks benchmarks/make_readme_table.jl
```
