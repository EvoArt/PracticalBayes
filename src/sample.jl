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

Build a `Layout` + `LogDensityFunction` for `model` and run `spl` (e.g.
`AdvancedHMC.NUTS(0.8)`) for `N` iterations via `AbstractMCMC.sample` on the
resulting `LogDensityModel`. Returns a `FlexiChains.SymChain`
(`FlexiChain{Symbol}`) by default — every constrained parameter name in
`model` becomes a `Parameter(:name)` key (`invlink`ed back from the sampler's
raw unconstrained draws via `Layout`), and every AdvancedHMC per-iteration
statistic (`log_density`, `acceptance_rate`, `n_steps`, ...) becomes an
`Extra(:stat_name)` key alongside it — so `chn[:mu]`, `chn[Extra(:n_steps)]`,
`summarystats(chn)`, etc. all work exactly as they would on a chain from a
plain `NamedTuple`-transition AbstractMCMC sampler.

NOTE on how this actually works: `FlexiChains.to_nt_and_stats(nt::NamedTuple)`
puts the ENTIRE NamedTuple into the params half and returns EMPTY stats —
confirmed directly (a first attempt that `merge`d params and stats into one
flat NamedTuple before calling `bundle_samples` silently produced a SymChain
where every AdvancedHMC diagnostic showed up as a `Parameter`, not an
`Extra`). The params/stats split has to survive as two SEPARATE fields for
FlexiChains to route them correctly, so `_pb_transition_to_nt` returns a
small `PBTransitionNT` wrapper (below) with its own `to_nt_and_stats` method,
rather than a bare `NamedTuple`.

`chain_type=nothing` returns AdvancedHMC's raw `Vector{AdvancedHMC.Transition}`
instead (skips `invlink`/FlexiChains entirely) for callers who want the
unconstrained θ vectors directly (e.g. diagnostics, `predict`).

## Multi-chain: `AbstractMCMC.MCMCThreads`/`MCMCDistributed` work with NO
## PracticalBayes-side code beyond this single-chain method

`AbstractMCMC.sample(rng, model, spl, MCMCThreads(), N, nchains; kwargs...)`
just works — confirmed directly, no extra method needed. `AbstractMCMC`'s own
`mcmcsample` implementation for `MCMCThreads` calls this exact single-chain
method once per thread (via `StatsBase.sample`, `deepcopy`d model/sampler per
thread) and combines the resulting `SymChain`s with `chainsstack`/`chainscat`
(`reduce(chainscat, chains)`, i.e. `cat(chains...; dims=3)`) — `FlexiChain`
(backed by `DimensionalData`) supports this `cat` generically, with no
FlexiChains-side or PracticalBayes-side multi-chain glue required. Verified:
4 threaded NUTS chains on a conjugate model combine into one correctly-shaped
`(niters, nchains)` `SymChain`, and `FlexiChains.rhat` on the result reports
< 1.01 for both parameters (the M2 milestone's own multi-chain gate).
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

# `AbstractMCMC.sample`'s own `chain_type` dispatch expects `bundle_samples`
# to receive whatever `AbstractMCMC.step` returns as a "transition" — for
# AdvancedHMC that's `AdvancedHMC.Transition` (holding a `PhasePoint` and a
# stats `NamedTuple`), never a plain `NamedTuple`. Rather than write our own
# `FlexiChain{Symbol}`-specific `bundle_samples` method (duplicating logic
# FlexiChains already provides generically for `NamedTuple` transitions —
# see `FlexiChains.to_nt_and_stats(nt::NamedTuple) = (nt, (;))`), we collect
# the raw transitions ourselves (mirroring `AbstractMCMC.mcmcsample`'s own
# loop) and convert each one to `(constrained_params_nt, stats_nt)` BEFORE
# handing off to `AbstractMCMC.bundle_samples` — so FlexiChains' existing
# generic method does all the actual chain-building work.
function _pb_sample_transitions(rng, ldm, spl, N; initial_params, kwargs...)
    transitions = Vector{Any}(undef, N)
    sample1, state = AbstractMCMC.step(rng, ldm, spl; initial_params=initial_params, kwargs...)
    transitions[1] = sample1
    for i in 2:N
        transitions[i], state = AbstractMCMC.step(rng, ldm, spl, state; kwargs...)
    end
    return transitions, state
end

# Keeps the invlink'd constrained-parameter NamedTuple and AdvancedHMC's own
# per-transition stat NamedTuple as two SEPARATE fields (see the docstring
# above for why merging them into one flat NamedTuple doesn't work —
# `FlexiChains.to_nt_and_stats(::NamedTuple)` can't recover the split
# afterward). `FlexiChains.to_nt_and_stats` is the documented extension
# point for a custom AbstractMCMC sampler's transition type (see
# FlexiChains' conversions.jl docstring).
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
