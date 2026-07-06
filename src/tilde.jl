using Distributions: Distributions, logpdf, Discrete, ValueSupport

# Every method in this file is named `tilde`, `tilde_index`, or `tilde_dot`
# and is called by compiler-generated code (see compiler.jl's
# `_tilde_expansion`/`_dot_tilde_expansion`) with a first argument that is one
# of the four AbstractEvalMode subtypes from modes.jl. There is NO if/else
# branching on "what kind of evaluation is this" anywhere — Julia's multiple
# dispatch picks the right method purely from the runtime types of the
# arguments, and because the compiler always generates the exact same call
# shape (`tilde(__mode__, Val(:x), dist, value_or_nothing, __acc__)`), the
# *type* of `__mode__` alone determines which of the ~13 methods below runs.
# For `EvalMode` specifically, every argument's type is known at compile time
# for a given model + Layout, so the whole call inlines away to straight-line
# code — this is what replaces DynamicPPL's runtime context-stack walk.

_is_discrete(d) = Distributions.value_support(typeof(d)) <: Discrete

# ===========================================================================
# `.~` (dot-tilde) — MVP supports OBSERVE only: `y .~ Normal.(μ, σ)` where `y`
# is already-bound data (a model argument or conditioned value). This is
# sugar for `sum(logpdf.(dist, y))` — a single vectorized accumulation, never
# a per-element loop, so it stays GPU-friendly. Assuming (unknown values) via
# `.~` is not supported: use `x[i] ~ dist` for indexed families of unknowns,
# or an array distribution (`product_distribution`, `MvNormal`) via plain `~`.
# ===========================================================================

"""
    tilde_dot(mode::EvalMode, ::Val{s}, y, dist_bcast, acc) where {s}

`dist_bcast` is the broadcast distribution expression's result (e.g.
`Normal.(μ, σ)`, itself an array/broadcast of `Distribution`s); `y` must be
already-bound data. Accumulates `sum(logpdf.(dist_bcast, y))` in one shot.
"""
@inline function tilde_dot(::EvalMode, ::Val, y, dist_bcast, acc::Accum)
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    return acc_lik(acc, sum(logpdf.(dist_bcast, y)))
end

function tilde_dot(t::TraceMode, ::Val{s}, y, dist_bcast, acc::Accum) where {s}
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    push!(t.sites, SiteRecord(s, dist_bcast, 0, :observed, y))
    return acc_lik(acc, sum(logpdf.(dist_bcast, y)))
end

function tilde_dot(::PriorMode, ::Val, y, dist_bcast, acc::Accum)
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    return acc_lik(acc, sum(logpdf.(dist_bcast, y)))
end

function tilde_dot(::FixedMode, ::Val, y, dist_bcast, acc::Accum)
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    return acc_lik(acc, sum(logpdf.(dist_bcast, y)))
end

# ===========================================================================
# EvalMode — the hot path. This is the code that must cost the same as
# `Turing.@addlogprob!` on the observe side, and a single bijector transform
# + logpdf on the assume side. No VarName is constructed, no context stack is
# walked, and the accumulator is a plain immutable pair (see accumulator.jl).
# ===========================================================================

"""
    tilde(mode::EvalMode, ::Val{s}, dist, value, acc) where {s}

OBSERVE path: `value` is a concrete (non-`nothing`, non-`missing`) data value
supplied either as a model argument or via `getcond`. This is exactly
`logpdf(dist, value)` accumulated into `acc` — no lookup, no allocation.
"""
@inline function tilde(::EvalMode, ::Val, dist, value, acc::Accum)
    # This is the entire observe path. Compare to Turing's `@addlogprob!`,
    # which does `acc = accloglikelihood!!(acc, val)` after the caller has
    # already computed `logpdf` themselves — here `logpdf(dist, value)` is
    # computed inline and folded straight into the accumulator. No VarName,
    # no NamedTuple rebuild, no dispatch beyond this one method: this is the
    # requirement-1 benchmark target (bench/observe_overhead.jl).
    return value, acc_lik(acc, logpdf(dist, value))
end

"""
    tilde(mode::EvalMode, ::Val{s}, dist, ::Nothing, acc) where {s}

ASSUME path: no conditioned/argument value for `s`, so its value comes from
`mode.layout.slots[s]`. Dispatch on the slot's concrete type selects between
reading a sub-vector of `θ` (bijector transform + Jacobian) or reading a
constant from `mode.store` (latent / other Gibbs block) — this branch is
resolved entirely at compile time because `layout.slots` is an isbits
NamedTuple.
"""
@inline function tilde(m::EvalMode, ::Val{s}, dist, ::Nothing, acc::Accum) where {s}
    # `getproperty(m.layout.slots, s)` on an isbits NamedTuple with a
    # compile-time-known field name `s` is resolved to a concrete-typed field
    # load during type inference — so `_assume`'s dispatch on `slot`'s
    # concrete type (FlatSlot vs ValueSlot) is decided at compile time, not
    # runtime, for any given (model, Layout) pair. `Val(s)` is passed through
    # so the ValueSlot method (below) can look `s` up in `m.store` the same way.
    return _assume(getproperty(m.layout.slots, s), m, Val(s), dist, acc)
end

# FlatSlot: this variable's unconstrained representation lives in
# `m.θ[slot.range]`. `from_linked_vec(dist)` builds (from the *runtime* `dist`
# argument, so this is correct even when `dist`'s support depends on other
# parameters, e.g. `Uniform(0, τ)`) the bijector mapping that sub-vector to
# `dist`'s constrained value space; `with_logabsdet_jacobian` gives us both
# the value AND the log|det Jacobian| of that transform in one call, which we
# must add to the accumulated log-density for a correct change-of-variables
# (this is the "includes bijector log-|det J|" part of Accum's docstring).
@inline function _assume(slot::FlatSlot, m::EvalMode, ::Val, dist, acc::Accum)
    x, logjac = with_logabsdet_jacobian(from_linked_vec(dist), view(m.θ, slot.range))
    return x, acc_prior(acc, logpdf(dist, x) + logjac)
end

# ValueSlot: this variable's value is a plain constant read out of `m.store`
# — never touches θ, never touches a bijector (it's already in constrained
# space, e.g. a latent state or another Gibbs block's current value). No
# Jacobian term is added because there's no transform happening here.
@inline function _assume(::ValueSlot, m::EvalMode, ::Val{s}, dist, acc::Accum) where {s}
    z = getfield(m.store, s)
    return z, acc_prior(acc, logpdf(dist, z))
end

"""
    tilde_index(mode::EvalMode, ::Val{s}, container, (i,), dist, acc) where {s}

Handles `x[i] ~ dist` sites. `container` is the pre-declared array bound to
`x`; the element at linear index `i` is written (assume) or read (observe)
and the corresponding sub-range of `θ` is selected via `elem_range`.
"""
@inline function tilde_index(m::EvalMode, ::Val{s}, container, idx::Tuple, dist, acc::Accum) where {s}
    # `idx` is whatever the user wrote inside `x[...]`, e.g. `(i,)` for a
    # vector or `(i, j)` for a matrix — `LinearIndices(size(container))[idx...]`
    # converts that to the single linear index `k` that `FlatArraySlot`'s
    # `elem_range` (layout.jl) expects, regardless of how many dimensions
    # `container` has.
    k = LinearIndices(size(container))[idx...]
    return _assume_index(getproperty(m.layout.slots, s), m, Val(s), container, k, dist, acc)
end

# Same idea as `_assume(::FlatSlot, ...)` above, but for one element of an
# indexed family: `elem_range(slot, k)` picks out that element's own
# sub-range of θ, and we additionally write the drawn/transformed value back
# into `container[k]` — this is what makes `x` (the user's pre-declared
# array) hold real values after the model runs, e.g. for use in a `return x`.
@inline function _assume_index(slot::FlatArraySlot, m::EvalMode, ::Val, container, k::Int, dist, acc::Accum)
    x, logjac = with_logabsdet_jacobian(from_linked_vec(dist), view(m.θ, elem_range(slot, k)))
    container[k] = x
    return x, acc_prior(acc, logpdf(dist, x) + logjac)
end

# ValueSlot version: the whole indexed family lives as one array in
# `m.store` (keyed by the family's name `s`); we just index into it and
# mirror the value into `container` for consistency with the FlatArraySlot
# case above.
@inline function _assume_index(::ValueSlot, m::EvalMode, ::Val{s}, container, k::Int, dist, acc::Accum) where {s}
    z = getfield(m.store, s)
    container[k] = z[k]
    return z[k], acc_prior(acc, logpdf(dist, z[k]))
end

# ===========================================================================
# TraceMode — discovery only. Allocation and dynamic dispatch here are fine.
# ===========================================================================

function tilde(t::TraceMode, ::Val{s}, dist, value, acc::Accum) where {s}
    # Three cases, checked in order, each recorded as a SiteRecord so
    # `build_layout` can see exactly what role every name played:
    if value !== nothing && value !== missing
        # 1) data supplied directly as a model argument — definitely observed.
        # `linked_len=0` since observed sites never occupy space in θ.
        push!(t.sites, SiteRecord(s, dist, 0, :observed, value))
        return value, acc_lik(acc, logpdf(dist, value))
    end
    cond = _getcond(t.conditioned, Val(s))
    if cond !== nothing
        # 2) not passed as an argument, but bound via `model | (; s=...)` —
        # also observed.
        push!(t.sites, SiteRecord(s, dist, 0, :observed, cond))
        return cond, acc_lik(acc, logpdf(dist, cond))
    end
    # 3) genuinely unknown: this is a parameter (or latent, if `dist` is
    # discrete — checked via `_is_discrete`). Its trace-time value comes from
    # a caller-supplied starting point (`t.init`) if given, else a prior draw
    # — either way we need SOME concrete value to continue evaluating the
    # rest of the model body (e.g. if a later line uses this variable).
    x0 = haskey(t.init, s) ? getfield(t.init, s) : rand(t.rng, dist)
    role = _is_discrete(dist) ? :latent : :param
    push!(t.sites, SiteRecord(s, dist, linked_vec_length(dist), role, x0))
    return x0, acc_prior(acc, logpdf(dist, x0))
end

function tilde_index(t::TraceMode, ::Val{s}, container, idx::Tuple, dist, acc::Accum) where {s}
    # Indexed families are always treated as unknowns during tracing (there's
    # no argument/conditioning mechanism for individual array elements in the
    # MVP — see the "Conditioning granularity" restriction in the plan), so
    # this is simpler than the scalar `tilde` above: always draw and record.
    k = LinearIndices(size(container))[idx...]
    x0 = rand(t.rng, dist)
    container[k] = x0
    role = _is_discrete(dist) ? :latent : :param
    push!(t.sites, SiteRecord(s, dist, linked_vec_length(dist), role, x0))
    return x0, acc_prior(acc, logpdf(dist, x0))
end

# ===========================================================================
# PriorMode — draw everything (assumed sites from prior; observed sites from
# the likelihood, unless conditioned, in which case use the conditioned data).
# ===========================================================================

function tilde(p::PriorMode, ::Val{s}, dist, value, acc::Accum) where {s}
    # An observe site under PriorMode still uses the REAL data if it's
    # available (e.g. `model | (; y=obs))`) — this lets `rand(model)` on a
    # partially-conditioned model reproduce the conditioned values exactly
    # while still drawing everything else fresh from the prior. Only when
    # there's truly no data does this fall back to drawing from `dist`.
    cond = value !== nothing && value !== missing ? value : _getcond(p.conditioned, Val(s))
    x = cond === nothing ? rand(p.rng, dist) : cond
    p.values[] = merge(p.values[], NamedTuple{(s,)}((x,)))
    return x, acc_lik(acc, logpdf(dist, x))
end

function tilde_index(p::PriorMode, ::Val{s}, container, idx::Tuple, dist, acc::Accum) where {s}
    k = LinearIndices(size(container))[idx...]
    x = rand(p.rng, dist)
    container[k] = x
    return x, acc_prior(acc, logpdf(dist, x))
end

# ===========================================================================
# FixedMode — assumed sites come from `mode.fixed`; observe sites use the
# conditioned/argument value if present, else (only when `predict=true`) are
# sampled, else it's an error (can't evaluate a joint density without data).
# ===========================================================================

function tilde(f::FixedMode, ::Val{s}, dist, value, acc::Accum) where {s}
    # Observe site — three cases in priority order:
    if value !== nothing && value !== missing
        # 1) real data passed as a model argument.
        f.values[] = merge(f.values[], NamedTuple{(s,)}((value,)))
        return value, acc_lik(acc, logpdf(dist, value))
    end
    cond = _getcond(f.conditioned, Val(s))
    if cond !== nothing
        # 2) real data bound via conditioning.
        f.values[] = merge(f.values[], NamedTuple{(s,)}((cond,)))
        return cond, acc_lik(acc, logpdf(dist, cond))
    end
    if f.predict
        # 3) no data anywhere, but we're in `predict` mode — sample this
        # observation from the likelihood instead of treating it as an error.
        # This is the entire mechanism behind posterior-predictive sampling.
        x = rand(f.rng, dist)
        f.values[] = merge(f.values[], NamedTuple{(s,)}((x,)))
        return x, acc_lik(acc, logpdf(dist, x))
    end
    # Not predicting and no data available: there is no sensible logdensity
    # to compute (a likelihood term with no observation), so fail loudly
    # rather than silently skipping the term or making up a value.
    throw(ArgumentError("observe site `$s` has no data and `predict=false`; cannot evaluate logdensity"))
end

function tilde(f::FixedMode, ::Val{s}, dist, ::Nothing, acc::Accum) where {s}
    # Assume site: read from the fixed point if it's there (the normal case —
    # evaluating at one posterior draw), otherwise draw fresh (lets `predict`
    # work even for parameters a particular chain/draw didn't include).
    x = hasfield(typeof(f.fixed), s) ? getfield(f.fixed, s) : rand(f.rng, dist)
    f.values[] = merge(f.values[], NamedTuple{(s,)}((x,)))
    return x, acc_prior(acc, logpdf(dist, x))
end

function tilde_index(f::FixedMode, ::Val{s}, container, idx::Tuple, dist, acc::Accum) where {s}
    k = LinearIndices(size(container))[idx...]
    x = if hasfield(typeof(f.fixed), s)
        getfield(f.fixed, s)[k]
    else
        rand(f.rng, dist)
    end
    container[k] = x
    return x, acc_prior(acc, logpdf(dist, x))
end
