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

@inline acc_prior(a::Accum, x) = Accum(a.logprior + x, a.loglik)
@inline acc_lik(a::Accum, x) = Accum(a.logprior, a.loglik + x)

@inline logprior(a::Accum) = a.logprior
# Named with a trailing underscore (not `loglikelihood`) to avoid colliding
# with the `loglikelihood` generic function widely exported by
# StatsAPI/Distributions/etc — a user doing `using PracticalBayes, Distributions`
# would otherwise hit an ambiguous-name error merely from loading both.
@inline loglikelihood_(a::Accum) = a.loglik
@inline logjoint(a::Accum) = a.logprior + a.loglik

Base.eltype(::Accum{T}) where {T} = T
Base.zero(::Type{Accum{T}}) where {T} = Accum{T}()
