# PracticalBayes.jl

PracticalBayes is a performance-oriented probabilistic programming package with Turing-style model syntax and a strong focus on fast inference paths for large models.

## Core ideas

- Keep model authoring ergonomic (`@model`, `~`, vectorized distributions).
- Keep AD and sampler internals explicit and composable.
- Support large-model workloads with features like untracked nuisance parameters and Float32 parameter paths.
- Work with DifferentiationInterface-compatible backends, including ForwardDiff, Mooncake, and Enzyme.

## Getting started

```@example
using PracticalBayes
using ADTypes
using LogDensityProblems
using Distributions

PracticalBayes.@model function demo(y)
    μ ~ Normal(0, 1)
    σ ~ Exponential(1)
    y .~ Normal.(μ, σ)
end

m = demo(randn(20))
layout, θ0, store0 = PracticalBayes.build_layout(m)
ldf = PracticalBayes.LogDensityFunction(m, layout, store0, ADTypes.AutoForwardDiff(); θ0=θ0)
LogDensityProblems.logdensity_and_gradient(ldf, θ0)
```

See [Sampling](sampling.md) for the sampling and chain APIs (`sample`, `Gibbs`, `AbstractLatentKernel`, `SymChain`).
