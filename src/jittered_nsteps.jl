using AdvancedHMC:
    AdvancedHMC,
    Trajectory,
    FixedNSteps,
    StaticTerminationCriterion,
    PhasePoint,
    Hamiltonian
using Random: Random, AbstractRNG

"""
    JitteredNSteps(L)

Termination criterion for a static HMC trajectory whose length is redrawn
uniformly from `1:L` at every iteration, instead of being pinned at `L`.

This is the drop-in sibling of AdvancedHMC's own `FixedNSteps`: it plugs into
exactly the same `Trajectory{EndPointTS}` kernel, so a sampler opts in by
swapping which of the two it constructs and changes nothing else.

# Why jitter at all

A trajectory of *fixed* length can resonate with a periodic posterior — after
`L` steps the sampler keeps landing near where it started, and the chain crawls.
Redrawing the length each iteration breaks that periodicity. Neal (2011,
sec. 4.2) recommends it as standard practice for static HMC. It is also free on
average: a uniform draw from `1:L` costs `(L+1)/2` gradient evaluations per
iteration rather than `L`, so a jittered run is roughly *half* the work of a
fixed-`L` one (which is why you may want a larger `L` when you turn it on, not
the same one).

# Why the drawn value is stored in the struct

AdvancedHMC's static-trajectory `transition` calls `nsteps(τ)` **twice** per
iteration: once inside `sample_phasepoint` to actually integrate, and once more
when it assembles the `n_steps` entry of the transition's stats NamedTuple.
Neither call site is handed an RNG. So `nsteps` cannot itself be the random
draw — it would (a) have no RNG to draw from, and (b) return two *different*
values within one iteration, integrating one trajectory length while reporting
another in the diagnostics.

Instead the draw happens once, in the one place that does have the RNG (the
`transition` method below), and is cached in this mutable field. Both `nsteps`
calls then read that same value, so the reported `n_steps` is always the
trajectory that was really simulated.

Mutating a field of the sampler's own criterion object is safe here because a
`Trajectory` is rebuilt per chain, so two chains sampling in parallel (threaded
or otherwise) each hold their own `JitteredNSteps` and never share this slot.
This "nominal value + current draw" layout is also exactly how AdvancedHMC's own
`JitteredLeapfrog` carries its jittered step size, so it is the library's
existing idiom rather than a workaround.

Note that `JitteredLeapfrog` jitters the step size `\u03f5` while holding the step
*count* fixed; this type does the opposite. They are independent and can be
combined.

# Correctness

Detailed balance is preserved because the step count is drawn **independently of
the current state** — it depends only on the RNG, never on `z`. That is the
condition Neal's argument needs: the trajectory length is part of the proposal
mechanism, not a function of where the chain currently is, so the usual
Metropolis correction on the end point remains valid unchanged.
"""
mutable struct JitteredNSteps <: StaticTerminationCriterion
    "Maximum number of leapfrog steps; each iteration draws uniformly from `1:L`."
    L::Int
    "The step count drawn for the current iteration; read by `nsteps`."
    current::Int
end

function JitteredNSteps(L::Int)
    L >= 1 || throw(ArgumentError("L must be at least 1, got $L"))
    # Seed `current` with `L` so that a criterion which somehow reaches `nsteps`
    # before any `transition` has drawn for it still yields a valid trajectory
    # (the fixed-`L` one) rather than a zero-step no-op.
    return JitteredNSteps(L, L)
end

# Both of AdvancedHMC's `nsteps(τ)` call sites land here and read the value the
# `transition` below drew for this iteration.
AdvancedHMC.nsteps(τ::Trajectory{TS,I,TC}) where {TS,I,TC<:JitteredNSteps} =
    τ.termination_criterion.current

# The single point where the per-iteration draw happens. We draw, then swap in a
# plain `FixedNSteps` holding that draw and hand straight over to AdvancedHMC's
# own static-trajectory `transition`: accept/reject, momentum reversal, energy
# bookkeeping and stats assembly are all unchanged and stay in AdvancedHMC where
# they belong.
#
# Rebuilding the `Trajectory` around a `FixedNSteps` (rather than `invoke`-ing the
# generic method on our own type) keeps the dispatch honest -- the delegated call
# is an ordinary call to an existing public method, with no dependence on the
# internal signature of the method we would otherwise be bypassing.
#
# This method is strictly more specific than AdvancedHMC's
# (`TC<:StaticTerminationCriterion`), so it takes precedence for `JitteredNSteps`
# without touching the `FixedNSteps` path at all.
function AdvancedHMC.transition(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    tau::Trajectory{TS,I,TC},
    z::PhasePoint,
) where {TS<:AdvancedHMC.AbstractTrajectorySampler,I,TC<:JitteredNSteps}
    tc = tau.termination_criterion
    # `rand(rng, 1:L)` on a vector of RNGs (AdvancedHMC's matrix-parallel mode)
    # would be ambiguous about which RNG to use, so take the first one there;
    # all chains in that mode already share a coupled trajectory length.
    r = rng isa AbstractVector ? first(rng) : rng
    L = rand(r, 1:tc.L)
    # Keep `current` in step with the draw so `nsteps(tau)` on THIS object still
    # reports the trajectory that was simulated, for anything that inspects the
    # criterion directly rather than going through the delegated trajectory.
    tc.current = L
    fixed = Trajectory{TS}(tau.integrator, FixedNSteps(L))
    return AdvancedHMC.transition(rng, h, fixed, z)
end

"""
    _n_steps_criterion(L, jitter)

Pick the termination criterion for a static HMC trajectory of nominal length `L`:
`JitteredNSteps` when `jitter` is set, AdvancedHMC's plain `FixedNSteps`
otherwise. Shared by `AdaptiveHMC` and `NUTSthenHMC` so the two samplers cannot
drift apart in what `jitter=true` means.
"""
_n_steps_criterion(L::Int, jitter::Bool) =
    jitter ? JitteredNSteps(L) : FixedNSteps(L)
