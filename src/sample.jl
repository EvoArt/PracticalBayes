using AbstractMCMC: AbstractMCMC
using AdvancedHMC: AdvancedHMC
using FlexiChains: FlexiChains, SymChain

"""
    SymChain

Re-exported from FlexiChains.jl — the default `chain_type` returned by
[`sample`](@ref). A `Symbol`-keyed `FlexiChains.FlexiChain`, indexable by
`chn[:paramname]`; see FlexiChains.jl's own documentation for the full chain
API (`FlexiChains.rhat`, `chainscat`/`chainsstack` for multi-chain combination,
etc.). PracticalBayes only re-exports the name for convenience — it defines no
behavior of its own on top of it.
"""
SymChain

"""
    sample(rng, model::Model, spl::AdvancedHMC.AbstractHMCSampler, N;
           adtype=AutoForwardDiff(), init=NamedTuple(), chain_type=SymChain,
           kwargs...)

Run `spl` (e.g. `AdvancedHMC.NUTS(0.8)`) on `model` for `N` iterations and
return the posterior draws.

By default the result is a `FlexiChains.SymChain`: every constrained parameter
name in `model` is a `Parameter(:name)` key (mapped back to constrained space
from the sampler's unconstrained draws), and every AdvancedHMC per-iteration
statistic (`log_density`, `acceptance_rate`, `n_steps`, ...) is an
`Extra(:stat_name)` key. Index and summarize the result as usual: `chn[:mu]`,
`chn[Extra(:n_steps)]`, `summarystats(chn)`.

`init` supplies starting values for named parameters (defaults to a draw from
the prior). Pass `chain_type=nothing` to get AdvancedHMC's raw
`Vector{AdvancedHMC.Transition}` instead — the unconstrained θ vectors, without
the constrained-space mapping.

Multiple chains use the standard AbstractMCMC interface with no extra setup:

    sample(rng, model, spl, MCMCThreads(), N, nchains; kwargs...)

runs `nchains` independent chains and combines them into a single `SymChain`
with a chain dimension.
"""
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::AdvancedHMC.AbstractHMCSampler,
    N::Integer;
    adtype=ADTypes.AutoForwardDiff(),
    init=NamedTuple(),
    chain_type=SymChain,
    kwargs...,
)
    layout, θ0, store0 = build_layout(model; init=init)
    ldf = LogDensityFunction(model, layout, store0, adtype; θ0=θ0)
    ldm = AbstractMCMC.LogDensityModel(ldf)

    if chain_type === nothing
        return AbstractMCMC.sample(rng, ldm, spl, N; initial_params=θ0, kwargs...)
    end

    raw_transitions, state = _pb_sample_transitions(rng, ldm, spl, N; initial_params=θ0, kwargs...)
    nt_transitions = [_pb_transition_to_nt(layout, t) for t in raw_transitions]
    return AbstractMCMC.bundle_samples(nt_transitions, ldm, spl, state, chain_type)
end

# `bundle_samples` receives whatever `AbstractMCMC.step` returns as a
# "transition" — for AdvancedHMC that's `AdvancedHMC.Transition` (a `PhasePoint`
# plus a stats `NamedTuple`), not a plain `NamedTuple`. We collect the raw
# transitions ourselves and convert each to `(constrained_params, stats)` before
# handing off to `bundle_samples`, so FlexiChains' generic `NamedTuple` path does
# the chain-building rather than a bespoke `FlexiChain{Symbol}` method here.
function _pb_sample_transitions(rng, ldm, spl, N; initial_params, kwargs...)
    transitions = Vector{Any}(undef, N)
    sample1, state = AbstractMCMC.step(rng, ldm, spl; initial_params=initial_params, kwargs...)
    transitions[1] = sample1
    for i in 2:N
        transitions[i], state = AbstractMCMC.step(rng, ldm, spl, state; kwargs...)
    end
    return transitions, state
end

# Keeps the constrained-parameter NamedTuple and AdvancedHMC's per-transition
# stat NamedTuple as two separate fields. `FlexiChains.to_nt_and_stats` routes
# the first into the chain's parameters and the second into its extras; a merged
# NamedTuple would land entirely in the parameters half, mislabeling every
# sampler diagnostic as a parameter. `to_nt_and_stats` is FlexiChains' extension
# point for a custom sampler's transition type.
struct PBTransitionNT{P<:NamedTuple,S<:NamedTuple}
    params::P
    stats::S
end
FlexiChains.to_nt_and_stats(t::PBTransitionNT) = (t.params, t.stats)

function _pb_transition_to_nt(layout::Layout, t::AdvancedHMC.Transition)
    params = invlink(layout, t.z.θ)
    stats = AdvancedHMC.stat(t)
    return PBTransitionNT(params, stats)
end
