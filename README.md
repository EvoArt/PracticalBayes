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
| `normal` | `Float64` | 1.0 | 1.079 | 1.206 | 0.986 |
| `normal` | `Float32` | 0.9 | 1.009 | 1.052 | 0.992 |
| `poisson` | `Float64` | 0.46 | 1.146 | 0.574 | 1.094 |
| `poisson` | `Float32` | 0.483 | 0.914 | 0.504 | 0.893 |
| `bernoulli_logit` | `Float64` | 1.0 | 1.09 | 1.013 | 0.981 |
| `bernoulli_logit` | `Float32` | 0.971 | 1.011 | 0.959 | 1.005 |

### Mooncake

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.042 | 1.143 | 1.009 | 0.765 |
| `normal` | `Float32` | 1.196 | 1.107 | 1.608 | 1.164 |
| `poisson` | `Float64` | 1.512 | 1.418 | 0.677 | 1.014 |
| `poisson` | `Float32` | 0.687 | 0.942 | 0.737 | 0.874 |
| `bernoulli_logit` | `Float64` | 0.976 | 1.052 | 0.889 | 1.012 |
| `bernoulli_logit` | `Float32` | 0.959 | 0.596 | 0.998 | 1.107 |

### Enzyme

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 0.949 | 1.126 | 0.95 | 1.107 |
| `normal` | `Float32` | 0.658 | 0.89 | 0.909 | 1.008 |
| `poisson` | `Float64` | n/a | n/a | n/a | n/a |
| `poisson` | `Float32` | n/a | n/a | n/a | n/a |
| `bernoulli_logit` | `Float64` | n/a | n/a | n/a | n/a |
| `bernoulli_logit` | `Float32` | n/a | n/a | n/a | n/a |


### Piste (PracticalBayes' own AD)

[Piste](https://github.com/EvoArt/Piste.jl) is the only backend here that survives `juliac --trim`, i.e. the only one that lets a model compile to a standalone binary. Turing cannot use it, so there is no PB/Turing ratio to give; these ratios are instead `Piste / fastest of ForwardDiff, Mooncake, Enzyme` on the same cell, with `< 1` meaning Piste is faster.

Read the tables together. **Forward mode** wins at small problems (the other backends' per-call setup dominates a tiny gradient) and loses badly at large P, because its cost is O(P) by construction -- the large bottom-right figures are that, not a defect. **Reverse mode** is the one to use on parameter-heavy models; it costs one sweep regardless of P.

**GradMode** is not AD at all: for a GLM-shaped model PracticalBayes recognises the shape and evaluates the analytic gradient directly (`X' * resid`, one BLAS call, no tape), which is why it beats every backend here. All three sweep models are GLM-shaped, so it applies to every cell; a model outside the recognised grammar falls back to ordinary AD and gets none of this.

#### Forward mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.667 | 24.368 | 1.024 | 39.034 |
| `normal` | `Float32` | 1.667 | 33.113 | 1.415 | 19.13 |
| `poisson` | `Float64` | 1.828 | 12.19 | 0.997 | 30.242 |
| `poisson` | `Float32` | 1.034 | 9.757 | 1.03 | 19.317 |
| `bernoulli_logit` | `Float64` | 1.394 | 16.121 | 0.996 | 32.784 |
| `bernoulli_logit` | `Float32` | 0.794 | 23.365 | 0.906 | 12.446 |


#### Reverse mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 5.333 | 1.393 | 4.565 | 1.183 |
| `normal` | `Float32` | 5.667 | 3.558 | 9.625 | 2.822 |
| `poisson` | `Float64` | 2.5 | 0.661 | 2.101 | 1.041 |
| `poisson` | `Float32` | 2.69 | 1.428 | 2.59 | 2.261 |
| `bernoulli_logit` | `Float64` | 2.379 | 1.075 | 2.18 | 1.439 |
| `bernoulli_logit` | `Float32` | 2.529 | 3.345 | 3.005 | 2.354 |


#### GradMode (analytic, GLM-shaped models only)

Note: these figures predate a 3.8x speedup to GradMode's fixed per-call cost (`_gm_positive_support` was calling `minimum`/`maximum` on an `MvNormal` prior on every gradient, ~10us and allocating). Regenerate the sweep to pick it up; the small-N columns here are the ones most understated.

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 36.056 | 1.186 | 0.28 | 0.284 |
| `normal` | `Float32` | 41.389 | 2.093 | 0.72 | 0.204 |
| `poisson` | `Float64` | 10.862 | 39.628 | 0.442 | 53.892 |
| `poisson` | `Float32` | 10.328 | 59.007 | 0.471 | 75.243 |
| `bernoulli_logit` | `Float64` | 11.121 | 1.249 | 0.537 | 0.281 |
| `bernoulli_logit` | `Float32` | 9.559 | 1.808 | 0.597 | 0.186 |

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
