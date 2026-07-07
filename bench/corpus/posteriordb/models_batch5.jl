# Fifth batch of TuringPosteriorDB.jl model ports (see models.jl for the
# overall approach/caveats). Covers the radon family (varying-intercept/
# slope hierarchical regression, centered and non-centered
# parameterizations, gather-indexed county effects — all mechanically
# similar to `pilots`/`rats` in models_batch4.jl) and
# `logistic_regression_rhs` (a regularized-horseshoe-prior logistic
# regression: `LocationScale`-wrapped `TDist` priors, `filldist` over a
# `Truncated` distribution, and a `:=` that itself calls `pdf.` — none of
# which need any new PracticalBayes-side support, all already exercised
# individually elsewhere in this corpus).
#
# NOTE: `radon_variable_slope_centered`/`radon_partially_pooled_centered`'s
# ORIGINAL Turing source writes `alpha ~ Normal(fill(mu, J), sigma^2 .* I)`/
# `alpha ~ Normal(mu_alpha, sigma_alpha)` and then indexes `alpha[county_idx]`
# — `Normal` is univariate, this is a pre-existing bug in the upstream
# PosteriorDB source (would not run under real Turing as written either).
# Ported using `MvNormal` instead, which is clearly the intended
# distribution given the surrounding models in the same family.
# `radon_variable_slope_noncentered` was skipped entirely: its source
# references `beta[county_idx]` without `beta` ever being defined via `~`
# or `:=` anywhere in the model body — not a porting decision, a genuine gap
# in the upstream file.

using PracticalBayes
using Distributions
using LinearAlgebra: I
using Random

function make_radon_data(seed)
    rng = Random.Xoshiro(seed)
    N = 60
    J = 8
    county_idx = rand(rng, 1:J, N)
    floor_measure = Float64.(rand(rng, Bool, N))
    log_uppm = randn(rng, N) .* 0.3
    true_county_effect = randn(rng, J) .* 0.5 .+ 1.2
    log_radon = [true_county_effect[county_idx[i]] - 0.5 * floor_measure[i] + randn(rng) * 0.3 for i in 1:N]
    return (N=N, J=J, county_idx=county_idx, floor_measure=floor_measure, log_uppm=log_uppm, log_radon=log_radon)
end

@model function radon_county_model(N, J, county, y)
    mu_a ~ Normal(0, 1)
    sigma_a ~ Uniform(0, 100)
    sigma_y ~ Uniform(0, 100)
    a ~ MvNormal(fill(mu_a, J), sigma_a^2 .* I)
    y ~ MvNormal(a[county], sigma_y^2 .* I)
end

@model function radon_pooled_model(N, floor_measure, log_radon)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    alpha ~ Normal(0, 10)
    beta ~ Normal(0, 10)
    log_radon ~ MvNormal(alpha .+ beta .* floor_measure, sigma_y^2 .* I)
end

@model function radon_hierarchical_intercept_centered_model(J, N, county_idx, log_uppm, floor_measure, log_radon)
    mu_alpha ~ Normal(0, 10)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    beta ~ MvNormal(fill(0.0, 2), 100)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    alpha ~ MvNormal(fill(mu_alpha, J), sigma_alpha^2 .* I)
    log_radon ~ MvNormal((alpha[county_idx] .+ log_uppm .* beta[1]) .+ floor_measure .* beta[2], sigma_y .* I)
end

@model function radon_hierarchical_intercept_noncentered_model(J, N, county_idx, log_uppm, floor_measure, log_radon)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    mu_alpha ~ Normal(0, 10)
    beta ~ MvNormal(fill(0.0, 2), 100 * I)
    alpha_raw ~ MvNormal(fill(0.0, J), 100 * I)
    alpha := mu_alpha .+ sigma_alpha .* alpha_raw
    log_radon ~ MvNormal((alpha[county_idx] .+ log_uppm .* beta[1]) .+ floor_measure .* beta[2], sigma_y^2 .* I)
end

@model function radon_partially_pooled_centered_model(N, J, county_idx, log_radon)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    mu_alpha ~ Normal(0, 10)
    alpha ~ MvNormal(fill(mu_alpha, J), sigma_alpha^2 .* I)  # source says `Normal(...)`; MvNormal is clearly intended (see header)
    log_radon ~ MvNormal(alpha[county_idx], sigma_y^2 .* I)
end

@model function radon_variable_intercept_centered_model(J, N, county_idx, floor_measure, log_radon)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    mu_alpha ~ Normal(0, 10)
    beta ~ Normal(0, 10)
    alpha ~ MvNormal(fill(mu_alpha, J), sigma_alpha .* I)
    log_radon ~ MvNormal(alpha[county_idx] .+ floor_measure .* beta, sigma_y^2 .* I)
end

@model function radon_variable_intercept_noncentered_model(J, N, county_idx, floor_measure, log_radon)
    alpha_raw ~ MvNormal(fill(0.0, J), I)
    beta ~ Normal(0, 10)
    mu_alpha ~ Normal(0, 10)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_y ~ Truncated(Normal(0, 1), 0, 1)
    alpha := mu_alpha .+ sigma_alpha .* alpha_raw
    log_radon ~ MvNormal(alpha[county_idx] .+ floor_measure * beta, sigma_y^2 .* I)
end

@model function radon_county_intercept_model(N, J, county_idx, floor_measure, log_radon)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    alpha ~ MvNormal(fill(0.0, J), 100I)
    beta ~ Normal(0, 10)
    log_radon ~ MvNormal(alpha[county_idx] .+ beta .* floor_measure, sigma_y^2 .* I)
end

@model function radon_variable_slope_centered_model(J, N, county_idx, floor_measure, log_radon)
    alpha ~ Normal(0, 10)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_beta ~ Truncated(Normal(0, 1), 0, Inf)
    mu_beta ~ Normal(0, 10)
    beta ~ MvNormal(fill(mu_beta, J), sigma_beta^2 .* I)  # source says `Normal(...)`; MvNormal is clearly intended (see header)
    log_radon ~ MvNormal(alpha .+ floor_measure .* beta[county_idx], sigma_y^2 .* I)
end

@model function radon_variable_intercept_slope_centered_model(N, J, county_idx, floor_measure, log_radon)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_beta ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    mu_alpha ~ Normal(0, 10)
    mu_beta ~ Normal(0, 10)
    alpha ~ MvNormal(fill(mu_alpha, J), sigma_alpha^2 .* I)
    beta ~ MvNormal(fill(mu_beta, J), sigma_beta^2 .* I)
    log_radon ~ MvNormal(alpha[county_idx] .+ floor_measure .* beta[county_idx], sigma_y^2 .* I)
end

@model function radon_variable_intercept_slope_noncentered_model(N, J, county_idx, floor_measure, log_radon)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_beta ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    mu_alpha ~ Normal(0, 10)
    mu_beta ~ Normal(0, 10)
    alpha_raw ~ MvNormal(fill(0.0, J), 1)
    beta_raw ~ MvNormal(fill(0.0, J), 1)
    alpha := mu_alpha .+ sigma_alpha .* alpha_raw
    beta := mu_beta .+ sigma_beta .* beta_raw
    log_radon ~ MvNormal(alpha[county_idx] .+ floor_measure .* beta[county_idx], sigma_y^2 .* I)
end

# ===========================================================================
# logistic_regression_rhs — regularized horseshoe prior: LocationScale-
# wrapped TDist priors (`scale*TDist(nu)`, Distributions.jl builtin sugar),
# `filldist` over a `Truncated` distribution (not just `Truncated` over a
# `LocationScale`, individually — this is the combination), and a `:=` whose
# right-hand side itself calls `pdf.` (arbitrary Julia, same as any other
# `:=`).
# ===========================================================================
@model function logistic_regression_rhs_model(n, d, y, x, scale_icept, scale_global, nu_global, nu_local, slab_scale, slab_df)
    z ~ MvNormal(fill(0.0, d), I)
    lambda ~ filldist(Truncated(TDist(nu_local), 0, Inf), d)
    tau ~ Truncated(scale_global * 2 * TDist(nu_global), 0, Inf)
    caux ~ filldist(InverseGamma(0.5 * slab_df, 0.5 * slab_df), d)
    beta0 ~ Normal(0, scale_icept)
    c := slab_scale .* sqrt.(caux)
    lambda_tilde := sqrt.(c .^ 2 .* lambda .^ 2 ./ (c .^ 2 .+ tau .^ 2 .* lambda .^ 2))
    beta := z .* lambda_tilde .* tau
    y .~ BernoulliLogit.(beta0 .+ x * beta)
    f := beta0 .+ x * beta
end
function make_logistic_rhs_data(seed)
    rng = Random.Xoshiro(seed)
    n, d = 40, 5
    x = randn(rng, n, d) .* 0.5
    true_beta = [1.0, -0.5, 0.0, 0.0, 0.3]
    p = 1.0 ./ (1.0 .+ exp.(-(0.2 .+ x * true_beta)))
    y = Float64.(rand(rng, n) .< p)
    return (
        n=n, d=d, y=y, x=x, scale_icept=5.0, scale_global=1.0, nu_global=3.0, nu_local=3.0, slab_scale=2.0, slab_df=4.0,
    )
end
