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

"""
    paramtype(mode) -> Type{<:Real}

The numeric type this mode is currently evaluating in — `eltype(mode.θ)` for
`EvalMode` (so it tracks whatever `θ`'s element type is: `Float64`,
`Float32`, or a `ForwardDiff.Dual`/other AD-backend number type mid-gradient),
`Float64` for the other three modes (never differentiated).

Model authors need this whenever a model pre-declares a container for an
indexed family (`x[i] ~ dist`): `x = Vector{Float64}(undef, n)` looks
harmless but is a real bug under any AD backend, since the container then
can't hold the `Dual` (or other AD-backend) numbers a gradient call needs to
write into it — this crashes with a `MethodError`/`InexactError` inside
`_assume_index`, not silently. Write `x = Vector{paramtype(__mode__)}(undef, n)`
instead (the same fix DynamicPPL's `@model` uses internally, via a
`::Type{T}=Float64` model argument it rewrites per call — `paramtype` is the
simpler equivalent this package's simpler compiler can offer directly,
reading the type straight off the mode rather than rewriting model arguments
at construction time).
"""
@inline paramtype(::AbstractEvalMode) = Float64  # overridden for EvalMode below, once it's defined

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

# Overrides the `AbstractEvalMode` fallback above: `EvalMode` DOES vary its
# working number type (Float64/Float32 density calls, `Dual` under
# ForwardDiff, backend-specific number types under Mooncake/Enzyme/etc), so
# `paramtype` must track `θ`'s actual element type rather than hardcoding
# `Float64` — this is precisely what lets a model-declared container
# (`Vector{paramtype(__mode__)}(undef, n)`) come out correctly typed for
# whichever AD backend (or none) is currently differentiating through it.
@inline paramtype(mode::EvalMode) = eltype(mode.θ)

# ---------------------------------------------------------------------------
# PriorMode: draw every assumed site from its prior; record all values
# (assumed and observed) into a NamedTuple for `rand(model)`.
# ---------------------------------------------------------------------------

"""
    PriorMode{R<:AbstractRNG,TCond<:NamedTuple} <: AbstractEvalMode

Evaluation mode backing `rand(model)`/`rand(model, n)`: every assumed site is
drawn from its prior distribution (ignoring any conditioned value), and every
site's value (assumed and observed) is recorded into `values[]` as the model
runs, giving a full prior-predictive draw once evaluation completes.

Deliberately not type-stable (`values[]`'s NamedTuple type grows one field per
site) — an accepted tradeoff since `rand(model)` runs the model exactly once
per call, unlike [`EvalMode`](@ref)'s hot HMC path.
"""
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

"""
    FixedMode{R<:AbstractRNG,TVals<:NamedTuple,TCond<:NamedTuple} <: AbstractEvalMode

Evaluation mode that reads every assumed site from a fixed `NamedTuple` of
values (`fixed`, e.g. one posterior draw) rather than sampling from a prior or
reading from a flat `θ` vector. Backs [`returned`](@ref), the `Model`-level
[`logjoint`](@ref)/[`logprior`](@ref)/[`loglikelihood_at`](@ref), and
[`predict`](@ref) (via the `predict` flag below) — see each function's own
docstring for the user-facing entry point; this struct is the shared mode they
all evaluate the model under.

Fields:
- `fixed`: the point to evaluate assumed sites at.
- `conditioned`: the model's own conditioned data (as for any other mode).
- `predict`: if `true`, observe sites with no data attached (neither in
  `conditioned` nor as a model argument) are *sampled* from the likelihood
  instead of erroring — this is what makes `predict` distinct from
  `logjoint`/`returned`, which require every observe site to resolve to real
  data or fail loudly.
- `values`: accumulates every site's value (assumed and observed) as the model
  runs, so callers can read back a full draw afterwards.
"""
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

"""
    PointwiseMode{TVals<:NamedTuple,TCond<:NamedTuple} <: AbstractEvalMode

Evaluation mode backing [`pointwise_loglikelihoods`](@ref) (LOO-CV/WAIC-style
model comparison, e.g. via ParetoSmooth.jl/ArviZ). Reads every assumed site
from a fixed `NamedTuple` of values exactly like [`FixedMode`](@ref) with
`predict=false` — same "every observe site must resolve to real data or
error" contract — but instead of only summing observe-site log-densities into
`Accum`, it ALSO records each site's PER-OBSERVATION `logpdf` values into
`pointwise[]`, keyed by site name.

This is a genuinely separate mode (not a `FixedMode` flag) because computing
per-observation values requires `logpdf.(dist, y)` at every `.~` site — the
opposite of the hot `EvalMode` path's whole optimization, which uses the
allocation-free summed `Distributions.loglikelihood(dist, y)` specifically to
avoid ever materializing that per-observation array (see tilde.jl's own
`_dot_loglik` docstring). Keeping this as its own mode means the hot HMC path
is completely untouched — this is a strictly opt-in, post-hoc re-evaluation,
never run during sampling.

Fields:
- `fixed`: the point to evaluate assumed sites at (e.g. one posterior draw).
- `conditioned`: the model's own conditioned data (as for any other mode).
- `pointwise`: accumulates each observe site's `Vector` of per-observation
  `logpdf` values, keyed by site name.
"""
mutable struct PointwiseMode{TVals<:NamedTuple,TCond<:NamedTuple} <: AbstractEvalMode
    fixed::TVals
    conditioned::TCond
    pointwise::Ref{NamedTuple}
end
PointwiseMode(fixed, conditioned) = PointwiseMode(fixed, conditioned, Ref{NamedTuple}(NamedTuple()))
