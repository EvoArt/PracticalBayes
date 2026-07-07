# Fourth batch of TuringPosteriorDB.jl model ports (see models.jl for the
# overall approach/caveats). Covers: boolean-mask-derived covariates (nes),
# gather-indexed random-effects regression (pilots, rats — the latter also
# using `filldist(Normal(mu,sigma), N)`, a non-Flat filldist prior, unlike
# every earlier filldist use which was `filldist(Flat(), K)`), the seeds
# family (`UniformScaling` as an MvNormal covariance, `BinomialLogit.` GLM
# with a random effect), a trivial 2-parameter regression
# (sesame_one_pred_a), and a model with three chained `:=` quantities plus
# an InverseGamma prior (surgical_model).

using PracticalBayes
using Distributions
using Distributions: logistic
using LinearAlgebra: I, UniformScaling
using Random
using Statistics: mean

# ===========================================================================
# nes — regression with boolean-mask-derived dummy covariates
# (`age_discrete .== 2`, etc.) built from an integer-coded categorical
# argument.
# ===========================================================================
@model function nes_model(N, partyid7, real_ideo, race_adj, educ1, gender, income, age_discrete)
    sigma ~ FlatPos(0.0)
    beta ~ filldist(Flat(), 9)
    age30_44 = age_discrete .== 2
    age45_64 = age_discrete .== 3
    age65up = age_discrete .== 4
    partyid7 ~ MvNormal(
        beta[1] .+ beta[2] .* real_ideo .+ beta[3] .* race_adj .+ beta[4] .* age30_44 .+
        beta[5] .* age45_64 .+ beta[6] .* age65up .+ beta[7] .* educ1 .+ beta[8] .* gender .+
        beta[9] .* income,
        sigma^2 .* I,
    )
end
function make_nes_data(seed)
    rng = Random.Xoshiro(seed)
    N = 40
    real_ideo = randn(rng, N)
    race_adj = randn(rng, N) .* 0.5
    educ1 = Float64.(rand(rng, 1:4, N))
    gender = Float64.(rand(rng, Bool, N))
    income = randn(rng, N)
    age_discrete = rand(rng, 1:4, N)
    partyid7 = 4.0 .+ 0.5 .* real_ideo .+ randn(rng, N) .* 1.5
    return (N=N, partyid7=partyid7, real_ideo=real_ideo, race_adj=race_adj, educ1=educ1, gender=gender, income=income, age_discrete=age_discrete)
end

# ===========================================================================
# pilots — two independent GATHER-indexed random effects (group, scenario)
# combined additively into one linear predictor, `:=` computing the reported
# combined mean before the final observe.
# ===========================================================================
@model function pilots_model(N, n_groups, n_scenarios, group_id, scenario_id, y)
    sigma_y ~ Uniform(0, 100)
    mu_a ~ Normal(0, 1)
    sigma_a ~ Uniform(0, 100)
    a ~ MvNormal(fill(10 * mu_a, n_groups), sigma_a^2 .* I)
    mu_b ~ Normal(0, 1)
    sigma_b ~ Uniform(0, 100)
    b ~ MvNormal(fill(10 * mu_b, n_scenarios), sigma_b^2 .* I)
    y_hat := a[group_id] .+ b[scenario_id]
    y ~ MvNormal(y_hat, sigma_y^2 .* I)
end
function make_pilots_data(seed)
    rng = Random.Xoshiro(seed)
    N = 40
    n_groups, n_scenarios = 5, 8
    group_id = rand(rng, 1:n_groups, N)
    scenario_id = rand(rng, 1:n_scenarios, N)
    true_a = randn(rng, n_groups) .* 2
    true_b = randn(rng, n_scenarios) .* 2
    y = [true_a[group_id[i]] + true_b[scenario_id[i]] + randn(rng) for i in 1:N]
    return (N=N, n_groups=n_groups, n_scenarios=n_scenarios, group_id=group_id, scenario_id=scenario_id, y=y)
end

# ===========================================================================
# rats — GATHER-indexed varying-intercept/varying-slope regression;
# `filldist(Normal(mu_alpha, sigma_alpha), N)` is a non-Flat filldist prior
# (every earlier filldist use was `filldist(Flat(), K)`), confirming
# filldist works identically regardless of the underlying scalar
# distribution.
# ===========================================================================
@model function rats_model(N, Npts, rat, x, y, xbar)
    mu_alpha ~ Normal(0, 100)
    mu_beta ~ Normal(0, 100)
    sigma_y ~ FlatPos(0.0)
    sigma_alpha ~ FlatPos(0.0)
    sigma_beta ~ FlatPos(0.0)
    alpha ~ filldist(Normal(mu_alpha, sigma_alpha), N)
    beta ~ filldist(Normal(mu_beta, sigma_beta), N)
    y ~ MvNormal(alpha[rat] .+ beta[rat] .* (x .- xbar), sigma_y^2 .* I)
    alpha0 := mu_alpha - xbar * mu_beta
end
function make_rats_data(seed)
    rng = Random.Xoshiro(seed)
    N = 6     # rats
    Nt = 5    # timepoints per rat
    Npts = N * Nt
    rat = repeat(1:N; inner=Nt)
    xvals = [8.0, 15.0, 22.0, 29.0, 36.0]
    x = repeat(xvals, N)
    xbar = mean(xvals)
    true_alpha = randn(rng, N) .* 10 .+ 100
    true_beta = randn(rng, N) .* 2 .+ 6
    y = [true_alpha[rat[i]] + true_beta[rat[i]] * (x[i] - xbar) + randn(rng) * 3 for i in 1:Npts]
    return (N=N, Npts=Npts, rat=rat, x=x, y=y, xbar=xbar)
end

# ===========================================================================
# seeds / seeds_centered / seeds_stanified — `UniformScaling` as an MvNormal
# covariance (`sigma^2 .* I` and `UniformScaling(sigma^2)` are equivalent —
# these three variants use the latter form verbatim, both already exercised
# elsewhere), `.~ BinomialLogit.(...)` GLM with a random effect added
# directly into the logit-linear predictor.
# ===========================================================================
@model function seeds_model(I_, n, N, x1, x2; x1x2=x1 .* x2)
    alpha0 ~ Normal(0.0, 1.0e3)
    alpha1 ~ Normal(0.0, 1.0e3)
    alpha2 ~ Normal(0.0, 1.0e3)
    alpha12 ~ Normal(0.0, 1.0e3)
    tau ~ Gamma(1.0e3, 1.0e-3)
    sigma := 1.0 / sqrt(tau)
    b ~ MvNormal(fill(0.0, I_), UniformScaling(sigma^2))
    n .~ BinomialLogit.(N, alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+ alpha12 .* x1x2 .+ b)
end

@model function seeds_centered_model(I_, n, N, x1, x2; x1x2=x1 .* x2)
    alpha0 ~ Normal(0, 1)
    alpha1 ~ Normal(0, 1)
    alpha2 ~ Normal(0, 1)
    alpha12 ~ Normal(0, 1)
    sigma ~ Truncated(Cauchy(0, 1), 0, Inf)
    c ~ MvNormal(fill(0.0, I_), UniformScaling(sigma^2))
    b := c .- mean(c)
    n .~ BinomialLogit.(N, alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+ alpha12 .* x1x2 .+ b)
end

@model function seeds_stanified_model(I_, n, N, x1, x2; x1x2=x1 .* x2)
    alpha0 ~ Normal(0, 1)
    alpha1 ~ Normal(0, 1)
    alpha2 ~ Normal(0, 1)
    alpha12 ~ Normal(0, 1)
    sigma ~ Truncated(Cauchy(0, 1), 0, Inf)
    b ~ MvNormal(fill(0.0, I_), UniformScaling(sigma^2))
    n .~ BinomialLogit.(N, alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+ alpha12 .* x1x2 .+ b)
end
function make_seeds_data(seed)
    rng = Random.Xoshiro(seed)
    I_ = 21
    N = fill(20, I_)
    x1 = randn(rng, I_) .* 0.3
    x2 = randn(rng, I_) .* 0.3
    true_p = logistic.(0.1 .+ 0.3 .* x1 .- 0.2 .* x2)
    n = [rand(rng, Binomial(N[i], true_p[i])) for i in 1:I_]
    return (I_=I_, n=n, N=N, x1=x1, x2=x2)
end

# ===========================================================================
# sesame_one_pred_a — trivial 2-parameter regression, same shape as
# earn_height (models.jl).
# ===========================================================================
@model function sesame_one_pred_a_model(N, encouraged, watched)
    beta ~ filldist(Flat(), 2)
    sigma ~ FlatPos(0.0)
    watched ~ MvNormal(beta[1] .+ beta[2] .* encouraged, sigma^2 * I)
end
function make_sesame_data(seed)
    rng = Random.Xoshiro(seed)
    N = 40
    encouraged = Float64.(rand(rng, Bool, N))
    watched = 20.0 .+ 10.0 .* encouraged .+ randn(rng, N) .* 5
    return (N=N, encouraged=encouraged, watched=watched)
end

# ===========================================================================
# surgical_model — InverseGamma prior, three chained `:=` quantities
# (`sigma` feeding `b`'s prior; `p`/`pop_mean` purely reported), MvNormal
# random-effect prior + `.~ BinomialLogit.(...)` observe.
# ===========================================================================
@model function surgical_model(N, r, n)
    mu ~ Normal(0, 1000)
    sigmasq ~ InverseGamma(0.001, 0.001)
    sigma := sqrt.(sigmasq)
    b ~ MvNormal(fill(mu, N), sigma^2 .* I)
    p := logistic.(b)
    r .~ BinomialLogit.(n, b)
    pop_mean := logistic.(mu)
end
function make_surgical_data(seed)
    rng = Random.Xoshiro(seed)
    N = 12
    n = fill(50, N)
    true_b = randn(rng, N) .* 0.5 .- 2.0
    r = [rand(rng, Binomial(n[i], logistic(true_b[i]))) for i in 1:N]
    return (N=N, r=r, n=n)
end
