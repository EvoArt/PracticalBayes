# Sampling and chains

This page documents the sampling and chain-building machinery introduced around `src/sample.jl`, `src/gibbs.jl`, and `src/latent.jl`.

```@setup sampling
using PracticalBayes
using Distributions
using AdvancedHMC
using Random
import AbstractMCMC
```

## One-shot HMC/NUTS sampling

`PracticalBayes` extends `AbstractMCMC.sample` for `Model` + AdvancedHMC samplers.

```@docs
AbstractMCMC.sample
```

### Behavior

- Builds `Layout`, `θ0`, and `LogDensityFunction` from the model.
- Runs AdvancedHMC through `AbstractMCMC.LogDensityModel`.
- By default, returns a `FlexiChains.SymChain` with constrained parameter fields and sampler diagnostics.
- Use `chain_type=nothing` to receive raw `Vector{AdvancedHMC.Transition}`.

## Gibbs blocks for mixed latent/continuous models

```@docs
PracticalBayes.Gibbs
PracticalBayes.GibbsState
PracticalBayes.AbstractLatentKernel
PracticalBayes.ModelConditional
PracticalBayes.latent_step
```

### Block design

`Gibbs` composes blocks of the form:

- `:name => NUTS(0.8)` for continuous/HMC-updated variables
- `:name => MyLatentKernel()` for discrete or custom latent updates

Every assumed variable must belong to exactly one block.

### Latent kernel contract

To implement a custom latent update:

1. Define a subtype of `AbstractLatentKernel`.
2. Implement:

```julia
PracticalBayes.latent_step(rng, kernel::MyKernel, block_names, c::ModelConditional)
```

3. Return a `NamedTuple` with exactly the keys in `block_names`.

The latent step runs once per Gibbs sweep and is never called from inside gradient/leapfrog internals.

## Example: Gibbs with a custom latent kernel

```@example
using PracticalBayes
using Distributions
using AdvancedHMC
using Random
import AbstractMCMC

PracticalBayes.@model function toy(y)
    μ ~ Normal(0, 1)
    z ~ Normal(μ, 1)
    y ~ Normal(z, 0.5)
end

struct SimpleLatent <: PracticalBayes.AbstractLatentKernel end

function PracticalBayes.latent_step(rng, ::SimpleLatent, block_names, c::PracticalBayes.ModelConditional)
    block_names == (:z,) || error("SimpleLatent handles only :z")
    return (; z = rand(rng, Normal(c.values.μ, 1.0)))
end

m = toy(2.0)
spl = PracticalBayes.Gibbs(:μ => NUTS(0.8), :z => SimpleLatent())
rng = Random.Xoshiro(1)
transition, state = AbstractMCMC.step(rng, m, spl)
transition
```

## Notes for large models

- Use `build_layout(...; T=Float32)` and Float32 distribution literals (`1f0`, `0f0`) to keep parameter/AD paths in Float32.
- Use untracked parameters where appropriate to avoid unnecessary optimization/sampling dimensions.
- For long Gibbs runs, keep `n_adapts` fixed from the first sweep onward for HMC blocks.
