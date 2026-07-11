using Distributions: Distributions, Distribution, logpdf, loglikelihood, Discrete, ValueSupport

# `y .~ Normal.(mu, sigma)` with SCALAR `mu`/`sigma` broadcasts to a single
# `Normal` (Julia's broadcast collapses to a scalar when every argument is a
# scalar), not an array of distributions — this is the common, GPU-friendly
# idiom this package's docs recommend. For that case,
# `Distributions.loglikelihood(dist, y)` is the purpose-built, allocation-free
# total-log-likelihood function (it does NOT materialize an intermediate
# per-observation array the way `logpdf.(dist, y)` does before summing).
# Measured on a 10_000-observation model: `sum(logpdf.(dist, y))` allocates
# ~80KB per call and is ~4x slower than a hand-written loop; `loglikelihood`
# is ~16 bytes and matches the hand-written loop's speed. Only fall back to
# the sum-of-logpdf form when `dist_bcast` is genuinely an array of DIFFERENT
# distributions (e.g. `Normal.(mus, sigma)` with a vector `mus`), which
# `loglikelihood` doesn't support.
@inline _dot_loglik(dist_bcast::Distribution, y) = loglikelihood(dist_bcast, y)
@inline _dot_loglik(dist_bcast, y) = sum(logpdf.(dist_bcast, y))

# Posterior-predictive sampling for `.~` sites: `predict(rng, model, chain)`
# calls the model with the observed argument replaced by an
# `AbstractArray{Missing}` of the desired output SHAPE (the standard PPL
# convention — DynamicPPL's own `predict` docs use exactly this
# `fill(missing, length(...))` pattern) rather than a bare `missing` scalar,
# because a scalar-broadcast `dist_bcast` (`Normal.(mu,sigma)` with scalar
# `mu`/`sigma`) carries NO shape information on its own — the number of
# observations to draw has to come from somewhere, and `y`'s shape is the
# only place that's available once `y` is no longer real data.
# `dist_bcast::Distribution` (the scalar-broadcast case) draws directly at
# `y`'s size; the array-of-distinct-distributions case broadcasts `rand`
# elementwise (each element may have different parameters).
@inline _dot_rand(rng, dist_bcast::Distribution, y) = rand(rng, dist_bcast, size(y))
@inline _dot_rand(rng, dist_bcast, y) = rand.(rng, dist_bcast)

# `y` signals "sample me" for `.~` the same way a bare `missing`/`nothing`
# does for scalar `~` — but shape must survive, so it's an ARRAY of missing,
# not a bare `missing`. `y === nothing` (no argument at all, e.g. `rand(model)`
# calling a model that was never given `y`) is treated the same way as long
# as `dist_bcast` itself carries a shape (the array-of-distributions case);
# for the scalar-broadcast case with `y === nothing` there's no shape
# anywhere and this still needs a real `y` argument (whole-array-of-`missing`
# or real data) — see the `y === nothing` guard left in each mode's own
# method below, unchanged from before this predictive-sampling addition.
_dot_all_missing(y) = y isa AbstractArray && all(ismissing, y)

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

# Coerce a bijector's constrained-space output back to the working precision
# `T = eltype(mode.θ)`. This is the fix for a silent Float64 promotion in the
# assume path: `Bijectors.VectorBijectors`'s constrained transforms hardcode
# Float64 bounds (e.g. `from_linked_vec(Exponential(1f0))` builds
# `Exp{Float64}(0.0, 1)`, whose `sign*exp(y) + 0.0` promotes a Float32 `y` to
# Float64 regardless of the distribution's own eltype, since
# `minimum(Exponential(1f0)) === 0.0`). Left uncoerced, that Float64 value
# flows into any downstream likelihood distribution built from it (e.g.
# `Normal.(mu, sigma)`), forcing the dominant large-`n` likelihood loop into
# Float64 and erasing the point of sampling in Float32.
#
# `T(x)` / `T.(x)` is a no-op when `x` is already `T` (the Float64 path is
# unchanged). Under AD, `T = eltype(mode.θ)` is the backend's Dual/tracked
# number type, so converting to it preserves derivatives — it just keeps the
# value and partials at the intended precision rather than the widened one.
@inline _to_paramtype(::Type{T}, x::Real) where {T<:Real} = convert(T, x)
@inline _to_paramtype(::Type{T}, x::AbstractArray) where {T<:Real} = convert(AbstractArray{T}, x)
# Non-Real / non-numeric values (e.g. a discrete draw, or a value read from an
# untransformed ValueSlot) are passed through untouched — there's no bijector
# promotion to undo there.
@inline _to_paramtype(::Type, x) = x

# ===========================================================================
# `.~` (dot-tilde) — MVP supports OBSERVE only: `y .~ Normal.(μ, σ)` where `y`
# is already-bound data (a model argument or conditioned value). This
# compiles to a single vectorized `_dot_loglik` call — allocation-free via
# `Distributions.loglikelihood` for the common scalar-broadcast case — never
# a per-element loop, so it stays GPU-friendly. Assuming (unknown values) via
# `.~` is not supported: use `x[i] ~ dist` for indexed families of unknowns,
# or an array distribution (`product_distribution`, `MvNormal`) via plain `~`.
# ===========================================================================

"""
    tilde_dot(mode::EvalMode, ::Val{s}, y, dist_bcast, acc) where {s}

`dist_bcast` is the broadcast distribution expression's result — a single
`Distribution` for the common scalar-broadcast case (e.g. `Normal.(μ, σ)`
with scalar `μ`/`σ`), or an array of distributions for `Normal.(μs, σ)`; `y`
must be already-bound data. Accumulates the total log-likelihood in one shot
via `_dot_loglik` (above), which picks the allocation-free
`Distributions.loglikelihood` path when possible.
"""
@inline function tilde_dot(::EvalMode, ::Val, y, dist_bcast, acc::Accum)
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    return acc_lik(acc, _dot_loglik(dist_bcast, y))
end

function tilde_dot(t::TraceMode, ::Val{s}, y, dist_bcast, acc::Accum) where {s}
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    push!(t.sites, SiteRecord(s, dist_bcast, 0, :observed, y))
    return acc_lik(acc, _dot_loglik(dist_bcast, y))
end

function tilde_dot(p::PriorMode, ::Val{s}, y, dist_bcast, acc::Accum) where {s}
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    if _dot_all_missing(y)
        # rand(model) on a model whose observed argument is an array of
        # `missing` (e.g. `predict`'s own machinery, or a user directly
        # generating prior-predictive data) — sample fresh values instead of
        # computing a likelihood against placeholders.
        x = _dot_rand(p.rng, dist_bcast, y)
        p.values[] = merge(p.values[], NamedTuple{(s,)}((x,)))
        return acc_lik(acc, _dot_loglik(dist_bcast, x))
    end
    return acc_lik(acc, _dot_loglik(dist_bcast, y))
end

function tilde_dot(f::FixedMode, ::Val{s}, y, dist_bcast, acc::Accum) where {s}
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    if _dot_all_missing(y)
        if !f.predict
            throw(ArgumentError("observe site `$s` has no data and `predict=false`; cannot evaluate logdensity"))
        end
        # See PriorMode's method above and `_dot_rand`'s docstring comment:
        # this is the `.~` half of posterior-predictive sampling — the
        # scalar-`~` equivalent of this branch already exists in `tilde`'s
        # own FixedMode method (case 3 there).
        x = _dot_rand(f.rng, dist_bcast, y)
        f.values[] = merge(f.values[], NamedTuple{(s,)}((x,)))
        return acc_lik(acc, _dot_loglik(dist_bcast, x))
    end
    return acc_lik(acc, _dot_loglik(dist_bcast, y))
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
    x_raw, logjac = with_logabsdet_jacobian(from_linked_vec(dist), view(m.θ, slot.range))
    # Undo any Float64 promotion introduced by the constrained bijector's
    # hardcoded Float64 bounds (see `_to_paramtype` above), so `x` stays in
    # the working precision `eltype(m.θ)` and any downstream distribution
    # built from it does too.
    x = _to_paramtype(eltype(m.θ), x_raw)
    return x, acc_prior(acc, logpdf(dist, x) + _to_paramtype(eltype(m.θ), logjac))
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

REQUIRED: `container` (`x`) MUST be declared as `Vector{paramtype(__mode__)}(undef, n)`
(see `paramtype`, modes.jl), NOT `Vector{Float64}(undef, n)`. `paramtype`
tracks whatever number type the CURRENT evaluation actually needs — plain
`Float64`/`Float32` for a density-only call, but a `ForwardDiff.Dual` (or
other AD-backend-specific number type) mid-gradient. Declaring the container
as hardcoded `Float64` crashes under every AD backend (`MethodError`/
`InexactError` trying to store a `Dual` into a `Float64` array) — this is
exactly the fix DynamicPPL's own `@model` applies internally, via a
`::Type{T}=Float64` model argument it rewrites per call; `paramtype` is the
simpler equivalent our compiler can offer directly, reading the type off the
mode rather than rewriting model arguments at construction time.

KNOWN LIMITATION (allocation, not correctness): `container`'s declaration
line reruns on every single `EvalMode` evaluation (every NUTS leapfrog
step), so it reallocates fresh each time — for `n` on the order of
thousands, expect on the order of `n * sizeof(eltype)` bytes/call (~8KB/call
measured for a 1000-element `Float64` family). Caching/reusing this buffer
was considered and deliberately NOT implemented: it would only help the
no-gradient (order-0) path, since AD backends need a fresh
Dual-typed/backend-specific buffer per call regardless, and NUTS always
requests gradients — a cached buffer wouldn't touch the actual sampling hot
path, while adding real complexity (thread-safety for `MCMCThreads`, cache
invalidation). If this allocation matters for your model, prefer an array
distribution via plain `~` (`product_distribution`, `MvNormal`) over an
indexed `x[i] ~ dist` loop — those don't need a pre-declared container at all.
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
    x_raw, logjac = with_logabsdet_jacobian(from_linked_vec(dist), view(m.θ, elem_range(slot, k)))
    # Same Float64-promotion fix as `_assume(::FlatSlot,...)`. `container`'s
    # element type is already `paramtype(__mode__) == eltype(m.θ)` (see the
    # `tilde_index` docstring), so the coerced `x` stores without a further
    # conversion/InexactError.
    x = _to_paramtype(eltype(m.θ), x_raw)
    container[k] = x
    return x, acc_prior(acc, logpdf(dist, x) + _to_paramtype(eltype(m.θ), logjac))
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
    # `linked_vec_length` is the size this site would occupy in the flat
    # unconstrained vector θ — but a `:latent`-role site NEVER lives in θ (it
    # goes to the value-store as a `ValueSlot`, see build_layout), so its
    # linked length is meaningless AND `linked_vec_length` isn't even defined
    # for arbitrary discrete distributions (e.g. a custom
    # `DiscreteMatrixDistribution` whole-trajectory latent — the entire point
    # of the epi/iFFBS use case). Only ask for a linked length when the site
    # is actually a continuous parameter; record 0 for latents.
    linked_len = role == :latent ? 0 : linked_vec_length(dist)
    push!(t.sites, SiteRecord(s, dist, linked_len, role, x0))
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
    # See the scalar `tilde` above: a discrete/latent indexed family never
    # occupies θ, so skip `linked_vec_length` (undefined for custom discrete
    # element distributions) and record 0.
    linked_len = role == :latent ? 0 : linked_vec_length(dist)
    push!(t.sites, SiteRecord(s, dist, linked_len, role, x0))
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

# ===========================================================================
# PointwiseMode — post-hoc re-evaluation for LOO-CV/WAIC-style pointwise
# log-likelihoods (see modes.jl's PointwiseMode docstring for why this is a
# separate mode rather than a FixedMode flag). Assume-site handling is
# identical in spirit to FixedMode's (read from the fixed point, no
# recording needed — pointwise values are an OBSERVE-side concept). Observe
# sites additionally push each observation's own `logpdf` into
# `p.pointwise[]`, keyed by site name, alongside the usual summed `Accum`
# accumulation (so `logjoint`-style totals computed via this mode still
# agree with the `FixedMode` path).
# ===========================================================================

function tilde(p::PointwiseMode, ::Val{s}, dist, value, acc::Accum) where {s}
    # Observe site — same three-case priority as FixedMode's own method,
    # minus the `predict` branch: pointwise evaluation always requires real
    # data (same rationale as `logjoint`'s own `predict=false`-only contract
    # — a per-observation likelihood term is meaningless without an
    # observation to score it against).
    x = if value !== nothing && value !== missing
        value
    else
        cond = _getcond(p.conditioned, Val(s))
        cond === nothing && throw(ArgumentError("observe site `$s` has no data; cannot evaluate pointwise logdensity"))
        cond
    end
    lp = logpdf(dist, x)
    p.pointwise[] = merge(p.pointwise[], NamedTuple{(s,)}(([lp],)))
    return x, acc_lik(acc, lp)
end

function tilde(p::PointwiseMode, ::Val{s}, dist, ::Nothing, acc::Accum) where {s}
    # Assume site: read from the fixed point (normal case), or draw fresh —
    # matches FixedMode's own assume method, since a value has to come from
    # somewhere even for a parameter the fixed point didn't include.
    x = hasfield(typeof(p.fixed), s) ? getfield(p.fixed, s) : rand(Random.default_rng(), dist)
    return x, acc_prior(acc, logpdf(dist, x))
end

function tilde_index(p::PointwiseMode, ::Val{s}, container, idx::Tuple, dist, acc::Accum) where {s}
    k = LinearIndices(size(container))[idx...]
    x = hasfield(typeof(p.fixed), s) ? getfield(p.fixed, s)[k] : rand(Random.default_rng(), dist)
    container[k] = x
    return x, acc_prior(acc, logpdf(dist, x))
end

function tilde_dot(p::PointwiseMode, ::Val{s}, y, dist_bcast, acc::Accum) where {s}
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    # Same "no data, no predict escape hatch" contract as the scalar `tilde`
    # method above — an all-`missing` `y` (the shape-carrying placeholder
    # `predict`'s machinery uses, see `_dot_all_missing`'s own docstring)
    # means there is genuinely no observation to score a pointwise
    # log-likelihood against, so this must error, not call `logpdf` on
    # `missing` (which would otherwise hit a bare `MethodError` instead of a
    # clear, intentional one).
    _dot_all_missing(y) && throw(ArgumentError("observe site `$s` has no data; cannot evaluate pointwise logdensity"))
    # This is the one place pointwise evaluation genuinely differs in cost
    # from FixedMode: `.~`'s hot-path optimization (`_dot_loglik`, see this
    # file's top-of-file comment) deliberately avoids materializing
    # per-observation values via the summed `Distributions.loglikelihood`
    # fast path — but pointwise values are the entire point here, so this
    # method always takes the `logpdf.(dist_bcast, y)` form instead.
    lps = logpdf.(dist_bcast, y)
    p.pointwise[] = merge(p.pointwise[], NamedTuple{(s,)}((vec(lps),)))
    return acc_lik(acc, sum(lps))
end
