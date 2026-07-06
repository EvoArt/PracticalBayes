"""
    AbstractEvalMode

Every `@model`-generated evaluator takes an `AbstractEvalMode` as its first
argument after `self`. All role decisions (assume vs observe, where a value
comes from) dispatch on the *type* of the mode plus `Val{name}`, so that for a
concrete mode type the tilde branch compiles away entirely — there is no
runtime context-stack walk like DynamicPPL's `AbstractContext` chain.
"""
abstract type AbstractEvalMode end

"""
    getcond(mode, ::Val{s}) -> value or nothing

Returns the conditioned value for symbol `s`, or `nothing` if `s` is not
conditioned. For concrete `TCond` NamedTuple types this is resolved by the
compiler via `hasfield`/`getfield` and folds away at compile time.
"""
@inline function getcond(mode::AbstractEvalMode, ::Val{s}) where {s}
    _getcond(mode.conditioned, Val(s))
end
# Split out from `getcond` so it can be called directly with a bare NamedTuple
# (TraceMode/PriorMode/FixedMode's tilde methods do this) without needing a
# full mode object. `hasfield`/`getfield` on a *concrete* NamedTuple type are
# both compile-time constant-foldable — the compiler knows at inference time
# whether `s` is one of `cond`'s field names, so this whole function reduces
# to either `nothing` or a direct field load, not a runtime lookup.
@inline function _getcond(cond::NamedTuple, ::Val{s}) where {s}
    hasfield(typeof(cond), s) ? getfield(cond, s) : nothing
end

# ---------------------------------------------------------------------------
# TraceMode: one-shot discovery run. Not on the hot path — allocations here
# are fine (a Vector{SiteRecord} grows as the model runs). Values are drawn
# from `init` if the caller supplied a starting value, else `rand(rng, dist)`.
# ---------------------------------------------------------------------------

"""
    TraceMode(rng, conditioned, init)

Discovery-phase evaluation mode. Running a model once under `TraceMode`
produces the ordered list of `SiteRecord`s consumed by `build_layout`.
"""
mutable struct TraceMode{R<:AbstractRNG,TCond<:NamedTuple,TInit<:NamedTuple} <: AbstractEvalMode
    rng::R
    conditioned::TCond
    init::TInit
    sites::Vector{Any}  # Vector{SiteRecord}; typed `Any` here to avoid a layout.jl->modes.jl cycle
end

# `sites` always starts empty and is appended to by every `tilde`/`tilde_index`/
# `tilde_dot` method specialized on `TraceMode` (see tilde.jl) as the model
# body runs top to bottom — so after one `evaluate(model, tmode, ...)` call,
# `tmode.sites` is the ordered list `build_layout` (layout.jl) partitions into
# flat-vector vs. value-store slots.
TraceMode(rng, conditioned, init) = TraceMode(rng, conditioned, init, Any[])

# ---------------------------------------------------------------------------
# EvalMode: the hot path. `θ` is the flat unconstrained parameter vector for
# this evaluation; `store` holds constant (non-differentiated) values for
# latents / other Gibbs blocks, keyed by variable name.
# ---------------------------------------------------------------------------

"""
    EvalMode(layout, θ, store, conditioned)

Type-stable hot-path evaluation mode. `layout.slots` is a NamedTuple of
isbits slot descriptors so `getproperty(layout.slots, s)` dispatches on a
concrete type per variable and the assume/observe/latent branch compiles away.
"""
struct EvalMode{L,V<:AbstractVector,S<:NamedTuple,TCond<:NamedTuple} <: AbstractEvalMode
    layout::L    # a Layout (layout.jl) — tells each tilde site where to find its value
    θ::V         # flat unconstrained parameter vector; V is left abstract so Vector{Float32},
                 # Vector{Float64}, or a CuVector all work without any code change here
    store::S     # constant values (latents, other Gibbs blocks) keyed by variable name;
                 # passed to the AD backend as `DI.Constant` in logdensity.jl, so gradients
                 # never flow through it and updating it never triggers an HMC leapfrog step
    conditioned::TCond
end
# `EvalMode` is deliberately immutable (unlike the other three modes) — it is
# constructed fresh for every logdensity/gradient evaluation, so there is
# nothing to mutate in place; this also makes it trivially safe to share
# across threads (one `EvalMode` per chain/thread, never written to).

# ---------------------------------------------------------------------------
# PriorMode: draw every assumed site from its prior; record all values
# (assumed and observed) into a NamedTuple for `rand(model)`.
# ---------------------------------------------------------------------------

mutable struct PriorMode{R<:AbstractRNG,TCond<:NamedTuple} <: AbstractEvalMode
    rng::R
    conditioned::TCond
    values::Ref{NamedTuple}  # mutated (not rebuilt functionally) as sites are visited — fine here
                              # since PriorMode is never differentiated and only ever run once per `rand` call
end

# `values` starts as an empty NamedTuple and each tilde site does
# `p.values[] = merge(p.values[], NamedTuple{(s,)}((x,)))` — i.e. we grow the
# NamedTuple one field at a time via `merge`. This re-specializes the type of
# `values[]` on every site (a new NamedTuple type each time a field is added),
# so PriorMode is deliberately NOT type-stable — that's an accepted tradeoff
# since `rand(model)` runs the model exactly once per call, not per HMC step.
PriorMode(rng, conditioned) = PriorMode(rng, conditioned, Ref{NamedTuple}(NamedTuple()))

# ---------------------------------------------------------------------------
# FixedMode: assumed sites read from a fixed NamedTuple of values (e.g. one
# posterior draw); observe sites that are NOT conditioned are *sampled* from
# the likelihood (this is what `predict` uses). Used also for `returned` and
# for `logprior`/`loglikelihood`/`logjoint` evaluation at a point.
# ---------------------------------------------------------------------------

mutable struct FixedMode{R<:AbstractRNG,TVals<:NamedTuple,TCond<:NamedTuple} <: AbstractEvalMode
    rng::R
    fixed::TVals  # the point (e.g. one posterior draw) to evaluate at; assume sites read from here
    conditioned::TCond
    predict::Bool  # if true, un-conditioned observe sites are sampled instead of erroring
    values::Ref{NamedTuple}  # accumulates every site's value (both assumed and observed) as
                              # the model runs, so callers can read back a full draw afterwards
end

# Three uses share this one mode, distinguished only by the `predict` flag and
# by which keys are present in `fixed`/`conditioned`:
#   * `logjoint`/`logprior`/`loglikelihood` at a point: `predict=false`, every
#     observe site must resolve to real data (via args or `conditioned`) or
#     this errors — you can't evaluate a density without the data term.
#   * `returned(model, chain)`: same as above, just reading the return value.
#   * `predict(rng, model, chain)`: `predict=true`, so observe sites with no
#     data attached are drawn from the likelihood instead of erroring — this
#     is exactly how posterior-predictive sampling works.
FixedMode(rng, fixed, conditioned; predict=false) =
    FixedMode(rng, fixed, conditioned, predict, Ref{NamedTuple}(NamedTuple()))
