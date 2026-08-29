using Distributions: Distributions

# Improper priors, matching Turing's `stdlib/distributions.jl` (not part of
# Distributions.jl itself). Defined here as ordinary `Distribution` subtypes
# — no special-casing needed anywhere else in the package: `Bijectors`'
# `from_linked_vec`/`to_linked_vec` build unconstrained-space transforms
# generically from `minimum`/`maximum`, so these plug straight into the
# existing `FlatSlot`/`build_layout`/`tilde` machinery exactly like `Normal`
# or `Exponential`.

"""
    Flat()

Improper flat prior over all reals: `logpdf(Flat(), x) == 0` everywhere.
Common in PosteriorDB models expressing "no informative prior."
"""
struct Flat <: Distributions.ContinuousUnivariateDistribution end

Base.minimum(::Flat) = -Inf
Base.maximum(::Flat) = Inf
Base.rand(rng::AbstractRNG, ::Flat) = rand(rng)
Distributions.logpdf(::Flat, x::Real) = zero(x)

"""
    FlatPos(l)

Improper flat prior over `(l, Inf)`: `logpdf` is `0` above `l`, `-Inf` at or
below it.
"""
struct FlatPos{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    l::T
end

Base.minimum(d::FlatPos) = d.l
Base.maximum(::FlatPos) = Inf
Base.rand(rng::AbstractRNG, d::FlatPos) = rand(rng) + d.l
function Distributions.logpdf(d::FlatPos, x::Real)
    z = float(zero(x))
    return x <= d.l ? oftype(z, -Inf) : z
end

# `filldist`/`arraydist`: thin wrappers around `product_distribution`, ported
# verbatim (same semantics) from DynamicPPL's `distribution_wrappers.jl` so
# PosteriorDB/tutorial model source can be copied over with minimal editing.
# Neither needs any PracticalBayes-side layout/tilde support beyond what
# `product_distribution` already gets for free (see src/distributions.jl's
# module docstring above) — these exist purely for source compatibility with
# Turing model code, not because the underlying mechanism needs them.

"""
    filldist(dist::Distribution, dim::Int, dims::Int...)

`product_distribution(fill(dist, dim, dims...))` — an array of `dim x dims...`
i.i.d. copies of `dist`, as one array-valued distribution suitable for a
single `~` site.
"""
filldist(dist::Distributions.Distribution, dim::Int, dims::Int...) = Distributions.product_distribution(fill(dist, dim, dims...))

"""
    arraydist(dists::AbstractArray{<:Distribution})

`product_distribution(dists)` — an array of independent (not necessarily
identically distributed) sub-distributions as one array-valued distribution.
"""
arraydist(dists::AbstractArray{<:Distributions.Distribution}) = Distributions.product_distribution(dists)

# Discrete-likelihood reparameterizations, again matching Turing's
# `stdlib/distributions.jl` verbatim (logit/log-linked parameterizations of
# Binomial/Poisson are common in PosteriorDB's GLM-family models). Discrete,
# so these only ever appear as observe-site likelihoods (never behind a
# `~` prior needing a bijector) — no layout-side work needed at all, only
# the `logpdf` definition itself.

"""
    LogPoisson(logλ)

`Poisson` reparameterized by log-rate: `logpdf(LogPoisson(logλ), k)` for
`k = 0, 1, 2, ...`.
"""
struct LogPoisson{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    logλ::T
end

Base.minimum(::LogPoisson) = 0
Base.maximum(::LogPoisson) = Inf
Base.rand(rng::AbstractRNG, d::LogPoisson) = rand(rng, Distributions.Poisson(exp(d.logλ)))
function Distributions.logpdf(d::LogPoisson, k::Real)
    insupp = Distributions.insupport(d, k)
    kk = insupp ? round(Int, k) : 0
    logp = kk * d.logλ - exp(d.logλ) - Distributions.logfactorial(kk)
    return insupp ? logp : oftype(logp, -Inf)
end

"""
    BernoulliCLogLog(η)

`Bernoulli` under the complementary log-log link: `p = 1 - exp(-exp(η))`.

The cloglog link is the standard choice for discrete-time survival / "hazard"
data, where `η` is a log-hazard and the observation is whether an event
occurred in the interval — it arises exactly from `P(event) = 1 - exp(-hazard)`
for a rate `exp(η)`. That makes it the natural link for epidemic
force-of-infection models (see e.g. the SEIC cloglog models in the badgers /
jolly island work), which is why it is here alongside `LogPoisson` and
`BinomialLogit` rather than left to a hand-written `@addlogprob!`.

Taking `η` directly (rather than `Bernoulli(1 - exp(-exp(η)))`) keeps the
whole computation on the log-hazard scale, which is both better conditioned
for large `|η|` and what `GradMode`'s closed-form gradient recognizes.
"""
struct BernoulliCLogLog{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    η::T
end

Base.minimum(::BernoulliCLogLog) = 0
Base.maximum(::BernoulliCLogLog) = 1
# `-expm1(-m)` is `1 - exp(-m)` computed accurately when `m` is small, which is
# the regime where the naive form loses all its significant digits.
_cloglog_p(d::BernoulliCLogLog) = -expm1(-exp(d.η))
Base.rand(rng::AbstractRNG, d::BernoulliCLogLog) = rand(rng) < _cloglog_p(d)
function Distributions.logpdf(d::BernoulliCLogLog, x::Real)
    Distributions.insupport(d, x) || return oftype(float(d.η), -Inf)
    m = exp(d.η)
    # log(1-p) = -m exactly, so the failure branch needs no logarithm at all.
    return x > 0 ? log(-expm1(-m)) : -m
end

"""
    BinomialLogit(n, logitp)

`Binomial` reparameterized by the logit of the success probability.
"""
struct BinomialLogit{T<:Real} <: Distributions.DiscreteUnivariateDistribution
    n::Int
    logitp::T
end

Base.minimum(::BinomialLogit) = 0
Base.maximum(d::BinomialLogit) = d.n
Base.rand(rng::AbstractRNG, d::BinomialLogit) = rand(rng, Distributions.Binomial(d.n, Distributions.logistic(d.logitp)))
function Distributions.logpdf(d::BinomialLogit, k::Real)
    insupp = Distributions.insupport(d, k)
    kk = insupp ? round(Int, k) : 0
    logp = -(log1p(d.n) + d.n * Distributions.StatsFuns.log1pexp(d.logitp)) + kk * d.logitp -
           Distributions.StatsFuns.logbeta(d.n - kk + 1, kk + 1)
    return insupp ? logp : oftype(logp, -Inf)
end
