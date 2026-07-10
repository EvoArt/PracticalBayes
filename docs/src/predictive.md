# Predictive utilities and log-density accessors

This page documents `src/predict.jl` — prior-predictive sampling, posterior
predictive sampling, and point-wise log-density evaluation, all built on top
of `PracticalBayes.PriorMode` and `PracticalBayes.FixedMode` (see their own docstrings for
the exact per-site semantics; this file is the convenience layer on top).

## Prior and prior-predictive sampling

```@docs
Random.rand(::Random.AbstractRNG, ::PracticalBayes.Model)
```

Every assumed site is drawn fresh from its prior; every observe site uses
real data if the model has any bound (via arguments or conditioning),
otherwise is drawn from the likelihood too. The return value is a
`NamedTuple` covering every site — assumed *and* observed — since a common
use is generating a synthetic `(params..., data...)` dataset for the model
itself, not just the parameters.

```@example
using PracticalBayes
using Distributions
using Random

PracticalBayes.@model function toy(n)
    μ ~ Normal(0, 1)
    σ ~ Exponential(1)
    y ~ PracticalBayes.arraydist(Normal.(fill(μ, n), σ))
end

m = toy(5)
rand(Random.Xoshiro(1), m)
```

## Log-density at a fixed point

```@docs
PracticalBayes.logjoint(::PracticalBayes.Model, ::NamedTuple; rng)
```

`logjoint`/`logprior`/`loglikelihood_at` all evaluate the model at a fixed
`NamedTuple` of parameter values (e.g. one posterior draw). Unlike
`predict` below, every observe site must resolve to real data — there is no
"sample instead" fallback, since you can't score a log-density against a
missing observation.

## `returned`: read back the model's own return value

```@docs
PracticalBayes.returned
```

Useful for models whose body computes and returns a derived quantity (a
transformed parameter, a summary statistic) that you want evaluated at a
specific posterior draw rather than re-derived by hand outside the model.

## Posterior predictive sampling

```@docs
PracticalBayes.predict
PracticalBayes.chain_draws
```

`predict` is the one place un-conditioned observe sites *are* sampled
(unlike `logjoint`/`returned`) — that's the whole point of a posterior
predictive check. `chain_draws` turns a `SymChain` (as returned by
`AbstractMCMC.sample`, see [Sampling](sampling.md)) back into the `Vector{NamedTuple}` form `predict` expects,
so the usual flow is:

```@example
using PracticalBayes
using Distributions
using AdvancedHMC
using Random
import AbstractMCMC

PracticalBayes.@model function reg(x, y)
    β ~ Normal(0, 1)
    σ ~ Exponential(1)
    y ~ PracticalBayes.arraydist(Normal.(β .* x, σ))
end

x = randn(Random.Xoshiro(1), 20)
y = 2.0 .* x .+ randn(Random.Xoshiro(2), 20) .* 0.3
m = reg(x, y)

chn = AbstractMCMC.sample(Random.Xoshiro(3), m, NUTS(0.8), 200;
    n_adapts=100, discard_initial=100, progress=false)
draws = PracticalBayes.chain_draws(chn)
preds = PracticalBayes.predict(Random.Xoshiro(4), m, draws[1:10])
length(preds)
```

## Pointwise log-likelihoods (LOO-CV/WAIC)

```@docs
PracticalBayes.pointwise_loglikelihoods
```

`loglikelihood_at` above gives one summed scalar; LOO-CV/WAIC-style model
comparison (e.g. via ParetoSmooth.jl or exporting to ArviZ) instead needs
one log-likelihood value *per observation*. This is a separate, opt-in
re-evaluation of the model — never run during sampling — because computing
per-observation values requires `logpdf.(dist, y)` at every `.~` site, which
is deliberately NOT what the hot HMC path computes (see
`PracticalBayes.PointwiseMode`'s own docstring for why the fast summed
`.~` path can't just be reused here).

```@example
using PracticalBayes
using Distributions
using Random

PracticalBayes.@model function reg2(x, y)
    β ~ Normal(0, 1)
    σ ~ Exponential(1)
    y .~ Normal.(β .* x, σ)
end

x = randn(Random.Xoshiro(1), 20)
y = 2.0 .* x .+ randn(Random.Xoshiro(2), 20) .* 0.3
m = reg2(x, y)

pw = PracticalBayes.pointwise_loglikelihoods(m, (β=1.9, σ=0.35))
sum(pw.y) ≈ PracticalBayes.loglikelihood_at(m, (β=1.9, σ=0.35))
```
