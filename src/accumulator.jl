"""
    Accum{T<:Real}

Immutable log-density accumulator threaded through model evaluation.
Split into prior (includes bijector log-|det J|) and likelihood parts so that
`logprior`, `loglikelihood`, and `logjoint` can all be read off cheaply, matching
DynamicPPL's accumulator split without the per-statement NamedTuple rebuild.

Being immutable (rather than mutable-struct or Ref) keeps every tilde site a pure
function `Accum -> Accum`, which is what makes the hot path AD- and GPU-friendly:
no aliasing, no heap mutation, trivially handled by ForwardDiff/Mooncake/Enzyme.
"""
struct Accum{T<:Real}
    logprior::T
    loglik::T
end

Accum(z) = Accum(z, z)
Accum{T}() where {T<:Real} = Accum{T}(zero(T), zero(T))
# Explicit promoting constructor: `struct Accum{T}`'s implicit default
# constructor requires BOTH fields to already be the same concrete `T`, but
# `acc_prior`/`acc_lik` below routinely combine values of different types
# (e.g. accumulating a Float64 `logpdf` result into a Float32-seeded `Accum`,
# which happens whenever a model mixes distributions/constants of different
# precisions during TraceMode, or under mixed-precision Dual arithmetic).
# Promoting here means the accumulator's type naturally widens instead of
# throwing a MethodError on the first mismatched site.
Accum(logprior, loglik) = Accum(promote(logprior, loglik)...)

@inline acc_prior(a::Accum, x) = Accum(a.logprior + x, a.loglik)
@inline acc_lik(a::Accum, x) = Accum(a.logprior, a.loglik + x)

"""
    logprior(a::Accum)

The accumulated log-prior density (sum of `logpdf` + bijector log-|det J| over
every assume site seen so far). See also [`logprior(model, nt)`](@ref) for the
whole-model, named-tuple-argument version built on top of this.
"""
@inline logprior(a::Accum) = a.logprior

# Named with a trailing underscore (not `loglikelihood`) to avoid colliding
# with the `loglikelihood` generic function widely exported by
# StatsAPI/Distributions/etc — a user doing `using PracticalBayes, Distributions`
# would otherwise hit an ambiguous-name error merely from loading both.
"""
    loglikelihood_(a::Accum)

The accumulated log-likelihood (sum of `logpdf` over every observe site seen so
far). Trailing underscore avoids colliding with `Base`/`StatsAPI`'s exported
`loglikelihood`. See also [`loglikelihood_at`](@ref) for the whole-model version.
"""
@inline loglikelihood_(a::Accum) = a.loglik

"""
    logjoint(a::Accum)

`logprior(a) + loglikelihood_(a)` — the full log-joint density accumulated so
far. See also [`logjoint(model, nt)`](@ref) for the whole-model version.
"""
@inline logjoint(a::Accum) = a.logprior + a.loglik

Base.eltype(::Accum{T}) where {T} = T
Base.zero(::Type{Accum{T}}) where {T} = Accum{T}()
