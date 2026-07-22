using AdvancedHMC:
    AdvancedHMC, AbstractHMCSampler, HMCKernel, Trajectory, EndPointTS, FixedNSteps, HMCState

"""
    NUTSthenHMC(δ=0.8; n_leapfrog=nothing, kwargs...)

An HMC sampler that runs **NUTS during warm-up and plain HMC afterwards**,
inheriting everything NUTS adapted: the mass matrix (metric), the step size, and
the integrator.

The motivation is that NUTS' tree-doubling is what makes warm-up robust — it
finds a sensible trajectory length without being told one — but after adaptation
that same doubling costs an unpredictable, sometimes large number of gradient
evaluations per draw. Once the metric and step size have settled, a fixed number
of leapfrog steps gives a constant, predictable cost per iteration, which is
often faster in wall-clock terms for the same effective sample size (and is what
makes the per-draw cost usable in a `Gibbs` sweep, where a runaway NUTS tree in
one block stalls the whole sweep).

`n_leapfrog` is the fixed step count used after warm-up. Leave it `nothing`
(the default) to pick it from what NUTS actually did: the median tree depth over
the adaptation phase, `n = 2^depth - 1`, clamped to `[1, 2^max_depth - 1]`. Pass
an integer to fix it yourself.

All other keyword arguments (`metric`, `integrator`, `max_depth`, `Δ_max`,
`init_ϵ`) are forwarded to the underlying `AdvancedHMC.NUTS` and so mean exactly
what they do there.

Use it like any other sampler; the switch happens at iteration `n_adapts`, so
that argument is not optional here:

    chn = sample(model, NUTSthenHMC(0.8), 2000; n_adapts=1000)

Note that with `discard_initial` unset, AbstractMCMC keeps the warm-up draws in
the chain as usual — pass `discard_initial=n_adapts` to drop them.
"""
struct NUTSthenHMC{T<:Real,I,M} <: AbstractHMCSampler
    "The wrapped NUTS sampler used for the whole warm-up phase."
    nuts::AdvancedHMC.NUTS{T,I,M}
    "Fixed leapfrog step count for the post-warm-up HMC phase, or `nothing` to infer it."
    n_leapfrog::Union{Int,Nothing}
end

function NUTSthenHMC(δ=0.8; n_leapfrog=nothing, kwargs...)
    return NUTSthenHMC(AdvancedHMC.NUTS(δ; kwargs...), n_leapfrog)
end

# `sampler_eltype` is the one piece of AdvancedHMC's sampler interface that is
# reached with the wrapper itself rather than the inner NUTS — callers use it to
# decide the element type of `initial_params` (and PracticalBayes' Float32 path
# depends on getting the right answer here). The rest of the `make_*` family
# (metric, step size, integrator, kernel, adaptor) is never called on the
# wrapper: the `step` methods below hand the whole initialization to `spl.nuts`,
# so AdvancedHMC's generic initializer sees a plain `NUTS` and builds exactly the
# state NUTS would have — same metric, same step-size search, same
# `StanHMCAdaptor` (mass matrix + dual averaging). We only intervene later, at
# the moment adaptation ends.
AdvancedHMC.sampler_eltype(spl::NUTSthenHMC) = AdvancedHMC.sampler_eltype(spl.nuts)

"""
    NUTSthenHMCState

Wraps AdvancedHMC's own `HMCState` and adds the two pieces of bookkeeping the
phase switch needs: whether it has already happened (`switched`), and the tree
depths seen so far during warm-up (`depths`), which are what a `nothing`
`n_leapfrog` is inferred from.

`depths` is emptied at the switch — it is warm-up-only bookkeeping and there is
no reason to keep growing it (or holding it) for the rest of the run.

`inner` is deliberately abstractly typed (`HMCState`, not a concrete type
parameter): swapping the trajectory sampler at the switch changes `HMCState`'s
own `TKernel` parameter (`MultinomialTS`/`GeneralisedNoUTurn` becomes
`EndPointTS`/`FixedNSteps`), so a concretely-parameterised field could not hold
both phases' states. The cost is one dynamic dispatch per `step` call, which is
noise against a trajectory of leapfrog steps and their gradients.
"""
mutable struct NUTSthenHMCState
    inner::HMCState
    switched::Bool
    depths::Vector{Int}
end

# Forward the state accessors AbstractMCMC/AdvancedHMC (and anything downstream,
# e.g. resuming or `Gibbs`' block handling) expects to find on an HMC state.
AdvancedHMC.getadaptor(s::NUTSthenHMCState) = AdvancedHMC.getadaptor(s.inner)
AdvancedHMC.getmetric(s::NUTSthenHMCState) = AdvancedHMC.getmetric(s.inner)
AdvancedHMC.getintegrator(s::NUTSthenHMCState) = AdvancedHMC.getintegrator(s.inner)
AbstractMCMC.getparams(s::NUTSthenHMCState) = AbstractMCMC.getparams(s.inner)
AbstractMCMC.getstats(s::NUTSthenHMCState) = AbstractMCMC.getstats(s.inner)
function AbstractMCMC.setparams!!(model, s::NUTSthenHMCState, params)
    s.inner = AbstractMCMC.setparams!!(model, s.inner, params)
    return s
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::NUTSthenHMC;
    kwargs...,
)
    # Delegate initialization to NUTS itself. Note AdvancedHMC's initializer
    # already takes the first real step (it tail-calls the `state`-taking method),
    # so this returns a genuine first transition, not just a state.
    t, inner = AbstractMCMC.step(rng, model, spl.nuts; kwargs...)
    state = NUTSthenHMCState(inner, false, Int[])
    _record_depth!(state, t)
    return t, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::NUTSthenHMC,
    state::NUTSthenHMCState;
    n_adapts::Int=0,
    kwargs...,
)
    # Warm-up: behave exactly as NUTS, and remember the tree depth so a
    # `nothing` `n_leapfrog` has something to be inferred from.
    if !state.switched && state.inner.i < n_adapts
        t, inner = AbstractMCMC.step(rng, model, spl.nuts, state.inner; n_adapts=n_adapts, kwargs...)
        state.inner = inner
        _record_depth!(state, t)
        return t, state
    end

    # First iteration at or past `n_adapts`: adaptation is finished, so the
    # metric and the integrator's ϵ in `state.inner` are the *adapted* ones. We
    # keep both and swap only the trajectory sampler — multinomial sampling over
    # a doubled tree becomes an end-point sample after a fixed number of
    # leapfrog steps.
    if !state.switched
        integrator = AdvancedHMC.getintegrator(state.inner)
        n = _post_warmup_n_leapfrog(spl, state)
        κ = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n)))
        state.inner = HMCState(
            state.inner.i, state.inner.transition, state.inner.metric, κ, state.inner.adaptor
        )
        state.switched = true
        empty!(state.depths)
    end

    # Post-warm-up: a plain HMC step against the adapted metric. `n_adapts=0`
    # here is what makes `adapt!` a no-op — the adaptor is still carried in the
    # state (so `getadaptor` keeps working and the run stays resumable), it just
    # never fires again.
    t, inner = AbstractMCMC.step(rng, model, spl.nuts, state.inner; n_adapts=0, kwargs...)
    state.inner = inner
    return t, state
end

# NUTS reports the realised tree depth per transition; HMC transitions don't have
# the field, hence the `haskey` rather than a bare access.
function _record_depth!(state::NUTSthenHMCState, t)
    stats = AdvancedHMC.stat(t)
    haskey(stats, :tree_depth) && push!(state.depths, Int(stats.tree_depth))
    return nothing
end

# The fixed step count for the HMC phase. An explicit `n_leapfrog` wins; otherwise
# take the median warm-up tree depth `d` and use `2^d - 1`, the number of leapfrog
# steps a NUTS tree of that depth actually performs. Clamped below at 1 (a depth-0
# tree still means one step) and above at NUTS' own `max_depth` budget, so a
# pathological warm-up cannot hand the HMC phase an enormous trajectory.
function _post_warmup_n_leapfrog(spl::NUTSthenHMC, state::NUTSthenHMCState)
    spl.n_leapfrog === nothing || return spl.n_leapfrog
    isempty(state.depths) && return 2^spl.nuts.max_depth - 1  # no warm-up ran at all
    # Median by hand rather than pulling in Statistics as a dependency for one
    # call. `depths` is warm-up-length and this runs once, so sorting a copy is
    # not worth optimizing.
    d = sort(state.depths)[cld(length(state.depths), 2)]
    return clamp(2^d - 1, 1, 2^spl.nuts.max_depth - 1)
end
