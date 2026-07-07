# A representative slice of TuringLang/docs tutorials
# (https://github.com/TuringLang/docs/tree/main/tutorials), ported to
# PracticalBayes' `@model` syntax. Chosen to cover: the introductory
# conjugate example with `filldist` over a scalar-parameter likelihood
# (coin-flipping), plain multivariate regression (linear/logistic/poisson
# regression, one scalar-loop and one MvNormal-array-observe variant each),
# and two "harder" structural patterns worth confirming explicitly:
# `Categorical`/`softmax` multi-class likelihoods (multinomial-logistic-
# regression) and a model that RETURNS a derived NamedTuple plus takes a
# plain Julia function as a model argument (bayesian-time-series-analysis's
# `op` argument, e.g. `+` vs `*` for additive/multiplicative decomposition).
#
# Same synthetic-data approach as bench/corpus/posteriordb/models.jl: no
# original tutorial datasets fetched, just shape/type-matched synthetic
# data sufficient to prove the model structure runs correctly end to end.

using PracticalBayes
using Distributions
using Distributions: logistic
using LinearAlgebra: I
using Random
using StatsFuns: softmax

# ===========================================================================
# coin-flipping — the introductory tutorial: conjugate Beta-Bernoulli via
# filldist over a scalar-parameter likelihood, plus keyword-argument model
# constructor (`coinflip(; N)`) conditioned via `|`.
# ===========================================================================
@model function coinflip(; N::Int)
    p ~ Beta(1, 1)
    y ~ filldist(Bernoulli(p), N)
    return y
end
coinflip(y::AbstractVector{<:Real}) = coinflip(; N=length(y)) | (; y)
function make_coinflip_data(seed)
    rng = Random.Xoshiro(seed)
    return (y=Float64.(rand(rng, Bernoulli(0.5), 100)),)
end

# ===========================================================================
# bayesian-linear-regression — scalar priors + MvNormal-array-observe with
# an explicit `return y ~ ...` (the tilde expression itself as the return
# value, exercising that `~` is a normal expression, not just a statement).
# ===========================================================================
@model function linear_regression(x, y)
    σ² ~ truncated(Normal(0, 100); lower=0)
    intercept ~ Normal(0, sqrt(3))
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(zeros(nfeatures), 10.0 * I)
    mu = intercept .+ x * coefficients
    return y ~ MvNormal(mu, σ² * I)
end
function make_linear_regression_data(seed)
    rng = Random.Xoshiro(seed)
    N, D = 32, 3
    x = randn(rng, N, D)
    true_coef = randn(rng, D)
    y = 1.0 .+ x * true_coef .+ randn(rng, N) .* 0.5
    return (x=x, y=y)
end

# ===========================================================================
# bayesian-logistic-regression — scalar-loop `~` with a per-row indexed
# design matrix (`x[1,i]`/`x[2,i]`/`x[3,i]`), Bernoulli-via-logistic-link
# likelihood.
# ===========================================================================
#
# NOTE ON PORTING: the original tutorial uses `y[i] ~ Bernoulli(v)` inside a
# loop over already-observed data `y`. PracticalBayes' `x[i] ~ dist` is
# assume-only by design (it always draws into a pre-declared container — see
# compiler.jl's docstring); observing element-by-element against a
# pre-bound data vector is exactly what `.~` is for, so this ports to a
# vectorized `.~` over the whole `v` vector instead of a per-element loop.
@model function logistic_regression(x, y, σ)
    intercept ~ Normal(0, σ)
    student ~ Normal(0, σ)
    balance ~ Normal(0, σ)
    income ~ Normal(0, σ)
    v = logistic.(intercept .+ student .* x[1, :] .+ balance .* x[2, :] .+ income .* x[3, :])
    y .~ Bernoulli.(v)
end
function make_logistic_regression_data(seed)
    rng = Random.Xoshiro(seed)
    N = 40
    x = randn(rng, 3, N)
    true_coef = [0.5, -0.3, 0.8, 0.1]  # intercept, student, balance, income
    p = logistic.(true_coef[1] .+ true_coef[2] .* x[1, :] .+ true_coef[3] .* x[2, :] .+ true_coef[4] .* x[3, :])
    y = Float64.(rand(rng, N) .< p)
    return (x=x, y=y, σ=3.0)
end

# ===========================================================================
# bayesian-poisson-regression — discrete (Poisson) log-link likelihood.
# Ported to a vectorized `.~` for the same reason as logistic_regression
# above (`y[i] ~ dist` inside a loop over already-observed `y` isn't the
# assume-into-preallocated-container pattern `x[i] ~ dist` supports).
# ===========================================================================
@model function poisson_regression(x, y, n, σ²)
    b0 ~ Normal(0, σ²)
    b1 ~ Normal(0, σ²)
    b2 ~ Normal(0, σ²)
    b3 ~ Normal(0, σ²)
    theta = b0 .+ b1 .* x[:, 1] .+ b2 .* x[:, 2] .+ b3 .* x[:, 3]
    y .~ Poisson.(exp.(theta))
end
function make_poisson_regression_data(seed)
    rng = Random.Xoshiro(seed)
    n = 30
    x = randn(rng, n, 3) .* 0.3
    true_coef = [0.5, 0.2, -0.1, 0.3]
    y = [rand(rng, Poisson(exp(true_coef[1] + true_coef[2] * x[i, 1] + true_coef[3] * x[i, 2] + true_coef[4] * x[i, 3]))) for i in 1:n]
    return (x=x, y=Float64.(y), n=n, σ²=10.0)
end

# ===========================================================================
# multinomial-logistic-regression — multi-class likelihood via `Categorical`
# over a `softmax`-normalized score vector, one of the few genuinely
# "different shape" patterns in the corpus (discrete K>2-category outcome).
#
# NOTE ON PORTING: same `y[i] ~ dist`-in-a-loop issue as logistic/poisson
# above; ported to `.~` over an ARRAY of distinct per-row `Categorical`s
# (`Categorical.(vs)`, a `Vector{Categorical}` since each row has its own
# softmax-derived probability vector) — `tilde_dot`'s fallback path
# (`sum(logpdf.(dist_bcast, y))`) handles an array of distributions exactly
# like this without any special-casing.
# ===========================================================================
@model function multinomial_logistic_regression(x, y, σ)
    n = size(x, 2)
    intercept_versicolor ~ Normal(0, σ)
    intercept_virginica ~ Normal(0, σ)
    coefficients_versicolor ~ MvNormal(zeros(4), σ^2 * I)
    coefficients_virginica ~ MvNormal(zeros(4), σ^2 * I)
    values_versicolor = intercept_versicolor .+ (coefficients_versicolor' * x)
    values_virginica = intercept_virginica .+ (coefficients_virginica' * x)
    vs = [softmax([0.0, values_versicolor[i], values_virginica[i]]) for i in 1:n]
    y .~ Categorical.(vs)
end
function make_multinomial_data(seed)
    rng = Random.Xoshiro(seed)
    n = 30
    x = randn(rng, 4, n) .* 0.5
    y = rand(rng, 1:3, n)
    return (x=x, y=y, σ=3.0)
end

# ===========================================================================
# bayesian-time-series-analysis — model RETURNS a derived NamedTuple
# (`(; trend, cyclic)`) rather than just the observed variable, and takes a
# plain Julia function (`op`, e.g. `+`) as a model argument controlling how
# trend/cyclic components combine — neither needs any special support, this
# just confirms both patterns (arbitrary return value, function-typed
# argument) work through our compiler as ordinary Julia semantics.
# ===========================================================================
@model function decomp_model(t, c, op)
    α ~ Normal(0, 10)
    βt ~ Normal(0, 2)
    βc ~ MvNormal(zeros(size(c, 2)), I)
    σ ~ truncated(Normal(0, 0.1); lower=0)
    cyclic = c * βc
    trend = α .+ βt .* t
    μ = op(trend, cyclic)
    y ~ MvNormal(μ, σ^2 * I)
    return (; trend, cyclic)
end
function make_decomp_data(seed)
    rng = Random.Xoshiro(seed)
    n, k = 40, 2
    t = collect(1.0:n)
    c = randn(rng, n, k) .* 0.5
    return (t=t, c=c, op=+)
end
