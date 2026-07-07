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
