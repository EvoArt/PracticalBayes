# Latent-kernel contract for Gibbs (gibbs.jl): a user plugs in a custom
# sampler for a block of variables that can't (or shouldn't) go through
# HMC/NUTS — a discrete latent state, a whole trajectory sampled via exact
# forward-filtering/backward-sampling (FFBS), etc. Deliberately minimal: no
# VarName/context machinery, just plain NamedTuple field access, so writing
# a kernel is "one struct, one method" (see `latent_step`'s docstring for a
# worked example).

"""
    AbstractLatentKernel

Supertype for user-defined Gibbs latent-block samplers. A subtype `K` must
implement `latent_step(rng, kernel::K, block_names, c::ModelConditional)`
(see its docstring). `Gibbs` (gibbs.jl) never differentiates through a
latent kernel and calls it exactly once per outer sweep — never from inside
a NUTS leapfrog/tree-doubling step — so a kernel is free to do arbitrary,
non-differentiable work (discrete sampling, matrix factorizations, calling
back into `rand`, etc).

If a kernel's own struct carries mutable internal state (e.g. an adaptive
proposal's running acceptance count), ensure that state is safe to `deepcopy`:
`MCMCThreads` isolates chains by deep-copying the sampler.
"""
abstract type AbstractLatentKernel end

"""
    ModelConditional(model, values)

Passed to `latent_step`. `values` is a NamedTuple holding the current
(constrained-space) value of every assumed model variable — i.e. everything
needed to evaluate this block's conditional distribution given the rest of the
model. This INCLUDES this block's own current value (it's built from the full
merged state before the current block is resampled), so a
Metropolis-within-Gibbs kernel that needs its own previous value (e.g. a
move-events / occult-infection proposal) can read it from `c.values` directly;
an exact-conditional kernel (FFBS, conjugate Gibbs) simply ignores its own
entry and reads only the OTHER blocks' values it conditions on. Data isn't
duplicated into `values`: read it directly off `c.model.args` (the model's own
call arguments) or `c.model.conditioned`, exactly as you would from inside the
`@model` body itself.
"""
struct ModelConditional{M<:Model,V<:NamedTuple}
    model::M
    values::V
end

"""
    logjoint(c::ModelConditional; overrides...) -> Real

Convenience oracle: evaluates the model's full log-joint density at
`merge(c.values, NamedTuple(overrides))` (via `FixedMode` — the same
machinery `logprior`/`loglikelihood`/`returned` use to evaluate a model at a
fixed point). `overrides` lets a kernel probe the joint density at trial
values without mutating `c.values` itself, e.g. for a generic
Metropolis-within-Gibbs kernel that isn't an exact conjugate/FFBS update.

Most exact-conditional kernels (FFBS, conjugate Gibbs updates) will NOT call
this — they compute their closed-form conditional directly from `c.values`
and the model's data, which is both simpler and avoids re-evaluating the
whole model just to get one block's conditional. `logjoint` exists for the
generic case where no closed form is available.
"""
function logjoint(c::ModelConditional; overrides...)
    nt = merge(c.values, NamedTuple(overrides))
    mode = FixedMode(Random.default_rng(), nt, c.model.conditioned; predict=false)
    _, acc = evaluate(c.model, mode, Accum(0.0))
    return logjoint(acc)
end

"""
    latent_step(rng, kernel::AbstractLatentKernel, block_names::Tuple{Vararg{Symbol}},
                c::ModelConditional) -> NamedTuple

User-implemented. Must return a `NamedTuple` with EXACTLY the keys in
`block_names`, holding the new constrained-space draw for each. Called
ONCE per Gibbs sweep by `Gibbs`'s sampling loop (gibbs.jl) — never inside a
gradient call, never mid-leapfrog-step — so `latent_step` is free to use
plain `Float64`/`Int` arithmetic throughout; nothing here is ever
differentiated or promoted to a `Dual`.

# Example: a hand-written forward-filter/backward-sample (FFBS) kernel for a
# 2-state Gaussian HMM's discrete state sequence `z`

```julia
using Distributions, StatsFuns, PracticalBayes

@model function hmm(y)
    p_stay ~ Beta(8, 2)
    sigma ~ Exponential(1)
    N = length(y)
    z = Vector{Int}(undef, N)   # discrete latent container — plain Int is fine,
                                # NEVER touched by AD (see `paramtype` docstring:
                                # only CONTINUOUS assumed containers need it)
    z[1] ~ Categorical([0.5, 0.5])
    for t in 2:N
        z[t] ~ Categorical(z[t - 1] == 1 ? [p_stay, 1 - p_stay] : [1 - p_stay, p_stay])
    end
    for t in 1:N
        y[t] ~ Normal(z[t] == 1 ? 0.0 : 5.0, sigma)
    end
end

struct FFBS <: AbstractLatentKernel end

function PracticalBayes.latent_step(rng, ::FFBS, block_names, c::ModelConditional)
    block_names == (:z,) || error("this FFBS kernel only handles the `:z` block")
    y = c.model.args.y
    p_stay, sigma = c.values.p_stay, c.values.sigma
    N = length(y)
    P = [p_stay 1-p_stay; 1-p_stay p_stay]
    means = (0.0, 5.0)

    # forward filter
    logα = Matrix{Float64}(undef, 2, N)
    logα[:, 1] .= log(0.5) .+ logpdf.(Normal.(means, sigma), y[1])
    for t in 2:N, j in 1:2
        logα[j, t] = logsumexp(logα[:, t - 1] .+ log.(P[:, j])) + logpdf(Normal(means[j], sigma), y[t])
    end

    # backward sample
    z = Vector{Int}(undef, N)
    z[N] = rand(rng, Categorical(softmax(logα[:, N])))
    for t in (N - 1):-1:1
        w = softmax(logα[:, t] .+ log.(P[:, z[t + 1]]))
        z[t] = rand(rng, Categorical(w))
    end
    return (; z=z)
end

chain = ...  # Gibbs(:p_stay => NUTS(0.8), :sigma => NUTS(0.8), :z => FFBS())
```
"""
function latent_step end
