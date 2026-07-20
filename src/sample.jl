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

"""
    sample(rng, model::Model, spl::Gibbs, N;
           init=NamedTuple(), adtype=AutoForwardDiff(), n_adapts=0,
           discard_initial=0, save_states=nothing, chain_type=SymChain, kwargs...)

Run a [`Gibbs`](@ref) sampler on `model` for `N` sweeps and return the draws.

Each sweep updates every block once (NUTS for HMC blocks, `latent_step` for
latent-kernel blocks). By default the result is a `FlexiChains.SymChain` with
one key per model variable. `discard_initial` drops that many leading sweeps
(burn-in); `n_adapts` is passed to every HMC block on every sweep and, as with
AdvancedHMC generally, should be its final target value from the first sweep on
(see [`Gibbs`](@ref)'s step docstring). `init` supplies starting values for any
named variables.

Pass `chain_type=nothing` to get the raw `Vector` of per-sweep `NamedTuple`
transitions instead of a chain.

By default a variable whose value is an array (e.g. a latent state matrix) is
retained per draw as that array. For a large latent that is fine to condition on
but too big to keep every draw of — a discrete trajectory `X` can be megabytes
per sweep, so thousands of sweeps materialise gigabytes in RAM — use
`save_states` to say, per variable, where its per-sweep value should go. The
sampler ALWAYS keeps the current value live for conditioning; `save_states` only
changes the OUTPUT:

    save_states = (X = :buffer,)          # keep live, DROP from the chain
    save_states = (X = ("X.jld2", 100),)  # keep live, append to disk every 100 sweeps
    save_states = (X = :chain,)           # keep in the chain (the default)

Values may be `:chain`/`true` (default; retain in the chain), `:buffer`/`false`
(omit from the chain, write nothing), or `(path, every)` (stream to disk every
`every` sweeps, plus a final flush; omitted from the chain). Any variable not
named keeps the default `:chain` behaviour, so scalar parameters are unaffected.
The disk form needs a backend loaded (`using JLD2` activates the bundled sink);
see [`write_state_chunk!`](@ref) to plug in another format.
"""
# Project a full sweep NamedTuple onto a statically-known subset of its keys.
# `Val{names}` makes `names` a compile-time constant, so the returned NamedTuple's
# type is inferred and no runtime key search happens — the per-sweep cost is a
# handful of field loads, and (crucially) it drops references to the omitted
# large arrays so they can be freed instead of accumulating in the chain.
@inline function _select_names(nt::NamedTuple, ::Val{names}) where {names}
    NamedTuple{names}(nt)
end

# The disk-streaming bookkeeping: one growable buffer per streamed variable,
# flushed to its OWN file every `every` sweeps. Kept out of the sweep loop's type
# domain (a plain Vector of little structs) — the loop just pushes and, on the
# flush boundary, hands whole chunks off. `Any`-typed buffers are fine here: this
# is once-per-sweep bookkeeping on already-materialised values, not the hot path,
# and the values are heterogeneous arrays we only ever move, never compute on.
# `n_written` counts sweeps already flushed to disk, so the next flush knows its
# `iters_x_to_y` range.
mutable struct _DiskStream
    name::Symbol
    disp::SaveToDisk
    buffer::Vector{Any}
    n_written::Int
end

function _init_disk_streams(save_states::NamedTuple)
    streams = _DiskStream[]
    for name in keys(save_states)
        disp = save_states[name]
        disp isa SaveToDisk && push!(streams, _DiskStream(name, disp, Vector{Any}(), 0))
    end
    streams
end

# Append one sweep's value for each streamed variable; flush any buffer that has
# reached its interval. `_flush_disk!` forces a final flush of trailing partials.
function _record_disk!(streams::Vector{_DiskStream}, t::NamedTuple)
    @inbounds for s in streams
        push!(s.buffer, getfield(t, s.name))
        length(s.buffer) >= s.disp.every && _flush_stream!(s)
    end
end
function _flush_disk!(streams::Vector{_DiskStream})
    for s in streams
        isempty(s.buffer) || _flush_stream!(s)
    end
end
function _flush_stream!(s::_DiskStream)
    first_iter = s.n_written + 1
    last_iter = s.n_written + length(s.buffer)
    write_state_chunk!(s.disp, s.name, s.buffer, first_iter, last_iter)
    s.n_written = last_iter
    empty!(s.buffer)
    nothing
end

function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Gibbs,
    N::Integer;
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    n_adapts::Int=0,
    discard_initial::Int=0,
    save_states=nothing,
    chain_type=SymChain,
    kwargs...,
)
    save = _normalize_save_states(save_states)
    streams = _init_disk_streams(save)

    # First sweep builds the per-block layouts/preps and takes one step; every
    # later sweep reuses them. We keep `N` sweeps after discarding the first
    # `discard_initial` as burn-in.
    t, state = AbstractMCMC.step(rng, model, spl; init=init, adtype=adtype, n_adapts=n_adapts, kwargs...)
    # A `save_states` name that isn't a real model variable is almost certainly a
    # typo — catch it here (once, cheaply) rather than silently doing nothing.
    for name in keys(save)
        haskey(t, name) || throw(ArgumentError(
            "`save_states` names variable `$name`, which is not a variable of this model " *
            "(its variables are: $(join(keys(t), ", ")))"))
    end
    for _ in 1:discard_initial
        t, state = AbstractMCMC.step(rng, model, spl, state; n_adapts=n_adapts, kwargs...)
    end

    # Which of the sweep's variables survive into the chain, as a compile-time
    # constant tuple (the sweep always returns the same key set, so this is fixed
    # for the whole run). Everything else is either dropped (`:buffer`) or
    # streamed to disk — in both cases it must NOT be retained here, which is the
    # whole point: retaining a 3 MB `X` for 5000 sweeps is what exhausts memory.
    retained = Val(_retained_keys(t, save))

    kept = _select_names(t, retained)
    transitions = Vector{typeof(kept)}(undef, N)
    # When there is no burn-in, the very first sweep (`t` above) is the first
    # kept draw; otherwise `t` currently holds the last discarded sweep and the
    # first kept draw comes from the next step.
    if discard_initial == 0
        transitions[1] = kept
        _record_disk!(streams, t)
        for i in 2:N
            t, state = AbstractMCMC.step(rng, model, spl, state; n_adapts=n_adapts, kwargs...)
            transitions[i] = _select_names(t, retained)
            _record_disk!(streams, t)
        end
    else
        for i in 1:N
            t, state = AbstractMCMC.step(rng, model, spl, state; n_adapts=n_adapts, kwargs...)
            transitions[i] = _select_names(t, retained)
            _record_disk!(streams, t)
        end
    end
    _flush_disk!(streams)  # trailing partial buffers

    chain_type === nothing && return transitions
    # A Gibbs transition is already a constrained-parameter NamedTuple, so
    # FlexiChains' generic `to_nt_and_stats(::NamedTuple)` bundles it directly
    # (no params/stats split to preserve, unlike the AdvancedHMC path below).
    return AbstractMCMC.bundle_samples(transitions, model, spl, state, chain_type)
end

# The chain-retained key subset of a full sweep NamedTuple, preserving the
# sweep's own key order. Runs ONCE per `sample` call (not per sweep), so the
# `keys`/`haskey` work here is immaterial.
function _retained_keys(t::NamedTuple, save::NamedTuple)
    Tuple(k for k in keys(t) if _retained_in_chain(save, k))
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
