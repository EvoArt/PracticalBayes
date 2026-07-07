# Second batch of TuringPosteriorDB.jl model ports (see models.jl for the
# overall approach/caveats — synthetic shape-matched data, structural smoke
# test only, no original datasets fetched). This batch covers: the rest of
# the Rate_* family (posterior-predictive `:=` quantities involving `rand`
# inside the model body), a Binomial-logit-link GLM, a random-effects GLMM
# with a gather-indexed discrete likelihood, a large multi-group varying-
# intercept model (election88_full), a 2-parameter-per-item IRT model
# (per-row distinct `arraydist`), and the remaining kidscore/kilpisjarvi
# variants (centering/standardizing covariates before the same regression
# shape already covered in models.jl).

using PracticalBayes
using Distributions
using Distributions: logistic
using LinearAlgebra: I
using Random
using Statistics: mean, std

# ===========================================================================
# Rate_2/3/4/5 — variations on the Beta-Binomial theme: two independent
# rates with a `:=` difference (Rate_2); a single shared rate across two
# Binomials (Rate_3); posterior/prior PREDICTIVE quantities computed via
# `rand(...)` inside a `:=` (Rate_4, Rate_5) — this exercises that `:=` is
# just an ordinary runtime assignment, so arbitrary Julia (including further
# random draws) is allowed on its right-hand side.
# ===========================================================================
@model function Rate_2_model(n1, n2, k1, k2)
    theta1 ~ Beta(1, 1)
    theta2 ~ Beta(1, 1)
    k1 ~ Binomial(n1, theta1)
    k2 ~ Binomial(n2, theta2)
    delta := theta1 - theta2
end
make_Rate_2_data(seed) = (n1=20, n2=25, k1=7, k2=9)

@model function Rate_3_model(n1, n2, k1, k2)
    theta ~ Beta(1, 1)
    k1 ~ Binomial(n1, theta)
    k2 ~ Binomial(n2, theta)
end
make_Rate_3_data(seed) = (n1=20, n2=25, k1=7, k2=9)

@model function Rate_4_model(n, k)
    theta ~ Beta(1, 1)
    thetaprior ~ Beta(1, 1)
    k ~ Binomial(n, theta)
    postpredk := rand(Binomial(n, theta))
    priorpredk := rand(Binomial(n, thetaprior))
end
make_Rate_4_data(seed) = (n=20, k=7)

@model function Rate_5_model(n1, n2, k1, k2)
    theta ~ Beta(1, 1)
    k1 ~ Binomial(n1, theta)
    k2 ~ Binomial(n2, theta)
    postpredk1 := rand(Binomial(n1, theta))
    postpredk2 := rand(Binomial(n2, theta))
end
make_Rate_5_data(seed) = (n1=20, n2=25, k1=7, k2=9)

# ===========================================================================
# GLM_Binomial — Binomial-logit-link GLM (quadratic-in-year mean), `:=`
# computing a REPORTED (non-conditioned) transformed-probability quantity.
# ===========================================================================
@model function GLM_Binomial_model(nyears, C, N, year; year_squared=year .* year)
    alpha ~ Normal(0, 100)
    beta1 ~ Normal(0, 100)
    beta2 ~ Normal(0, 100)
    logit_p := alpha .+ beta1 .* year .+ beta2 .* year_squared
    C .~ BinomialLogit.(N, logit_p)
    p := logistic.(logit_p)
end
function make_GLM_Binomial_data(seed)
    rng = Random.Xoshiro(seed)
    nyears = 15
    year = collect(range(-1.0, 1.0; length=nyears))
    N = fill(20, nyears)
    true_alpha, true_beta1 = 0.2, 0.5
    C = [rand(rng, Binomial(N[i], logistic(true_alpha + true_beta1 * year[i]))) for i in 1:nyears]
    return (nyears=nyears, C=C, N=N, year=year)
end

# ===========================================================================
# GLMM_Poisson — Poisson GLM with an observation-level random effect `eps`
# (an MvNormal-distributed noise vector added to the linear predictor before
# the log-link), same log-Poisson `.~` pattern as the earlier GLM_Poisson
# port, plus a `:=` computing a reported vector.
# ===========================================================================
@model function GLMM_Poisson_model(n, C, year; year_squared=year .^ 2, year_cubed=year .^ 3)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 10)
    beta3 ~ Uniform(-10, 10)
    sigma ~ Uniform(0, 5)
    eps ~ MvNormal(fill(0.0, n), sigma^2 .* I)
    log_lambda := alpha .* beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed .+ eps
    C .~ LogPoisson.(log_lambda)
    lambda := exp.(log_lambda)
end
function make_GLMM_Poisson_data(seed)
    rng = Random.Xoshiro(seed)
    n = 15
    year = collect(range(-1.0, 1.0; length=n))
    true_alpha, true_beta1 = 1.0, 0.3
    C = Float64.(rand.(rng, Poisson.(exp.(true_alpha .* true_beta1 .* year))))
    return (n=n, C=C, year=year)
end

# ===========================================================================
# GLMM1 — random-effects Poisson GLMM with a GATHER-indexed discrete
# likelihood: each observation's rate is read from a [year, site]-indexed
# matrix built via `repeat`, using per-observation index vectors
# (`obsyear[i]`, `obssite[i]`) — a genuinely different indexing shape from
# every earlier port. `y[i] ~ dist_i` in a loop over data doesn't fit
# `x[i] ~ dist`'s assume-only container pattern (same reasoning as the
# tutorial ports), so this becomes a vectorized gather (`log_lambda[CartesianIndex.(obsyear,obssite)]`)
# followed by one `.~`.
# ===========================================================================
@model function GLMM1_model(nobs, nyear, nsite, obs, obsyear, obssite)
    mu_alpha ~ Normal(0, 10)
    sd_alpha ~ Uniform(0, 5)
    alpha ~ MvNormal(fill(mu_alpha, nsite), sd_alpha^2 .* I)
    log_lambda := repeat(alpha, 1, nyear)'
    rates = [log_lambda[obsyear[i], obssite[i]] for i in 1:nobs]
    obs .~ LogPoisson.(rates)
end
function make_GLMM1_data(seed)
    rng = Random.Xoshiro(seed)
    nyear, nsite = 4, 5
    nobs = 15
    obsyear = rand(rng, 1:nyear, nobs)
    obssite = rand(rng, 1:nsite, nobs)
    true_alpha = randn(rng, nsite) .* 0.5
    obs = Float64.([rand(rng, Poisson(exp(true_alpha[obssite[i]]))) for i in 1:nobs])
    return (nobs=nobs, nyear=nyear, nsite=nsite, obs=obs, obsyear=obsyear, obssite=obssite)
end

# ===========================================================================
# election88_full — large multi-group varying-intercept model: several
# independent MvNormal random-effect vectors, gathered by INTEGER GROUP
# INDEX (`a[age]`, `b[edu]`, ...) into one linear predictor, observed via a
# BernoulliLogit `.~`. A good stress test for multiple simultaneous
# gather-indexed random effects in one model.
# ===========================================================================
@model function election88_full_model(
    n_age, n_age_edu, n_edu, n_region_full, n_state, age, age_edu, black, edu, female, region_full, state,
    v_prev_full, y,
)
    sigma_a ~ Uniform(0, 100)
    sigma_b ~ Uniform(0, 100)
    sigma_c ~ Uniform(0, 100)
    sigma_d ~ Uniform(0, 100)
    sigma_e ~ Uniform(0, 100)
    a ~ MvNormal(fill(0.0, n_age), sigma_a^2 .* I)
    b ~ MvNormal(fill(0.0, n_edu), sigma_b^2 .* I)
    c ~ MvNormal(fill(0.0, n_age_edu), sigma_c^2 .* I)
    d ~ MvNormal(fill(0.0, n_state), sigma_d^2 .* I)
    e ~ MvNormal(fill(0.0, n_region_full), sigma_e^2 .* I)
    beta ~ MvNormal(fill(0.0, 5), 100^2 .* I)
    y_hat := beta[1] .+ beta[2] .* black .+ beta[3] .* female .+
             beta[5] .* female .* black .+ beta[4] .* v_prev_full .+
             a[age] .+ b[edu] .+ c[age_edu] .+ d[state] .+ e[region_full]
    y .~ BernoulliLogit.(y_hat)
end
function make_election88_data(seed)
    rng = Random.Xoshiro(seed)
    N = 60
    n_age, n_edu, n_age_edu, n_state, n_region_full = 4, 4, 16, 10, 5
    age = rand(rng, 1:n_age, N)
    edu = rand(rng, 1:n_edu, N)
    age_edu = rand(rng, 1:n_age_edu, N)
    state = rand(rng, 1:n_state, N)
    region_full = rand(rng, 1:n_region_full, N)
    black = Float64.(rand(rng, Bool, N))
    female = Float64.(rand(rng, Bool, N))
    v_prev_full = rand(rng, N) .* 0.4 .+ 0.3
    y = Float64.(rand(rng, Bool, N))
    return (
        N=N, n_age=n_age, n_age_edu=n_age_edu, n_edu=n_edu, n_region_full=n_region_full, n_state=n_state,
        age=age, age_edu=age_edu, black=black, edu=edu, female=female, region_full=region_full, state=state,
        v_prev_full=v_prev_full, y=y,
    )
end
election88_full_model(d::NamedTuple) = election88_full_model(
    d.n_age, d.n_age_edu, d.n_edu, d.n_region_full, d.n_state, d.age, d.age_edu, d.black, d.edu, d.female,
    d.region_full, d.state, d.v_prev_full, d.y,
)

# ===========================================================================
# irt_2pl — 2-parameter logistic item-response model.
#
# NOTE ON PORTING: the original tutorial uses `y[i, :] ~ arraydist(...)`
# inside a loop over already-observed data `y` — same class of issue as the
# tutorial ports in models.jl (`x[i] ~ dist` is assume-only by design, it
# always draws INTO a pre-declared container; it has no path for "observe
# against already-bound data" the way plain `~`/`.~` do). Ported to a single
# `.~` over the WHOLE I×J matrix, using a matrix-shaped array of distinct
# per-cell `BernoulliLogit`s (broadcasting naturally handles the 2D shape;
# `tilde_dot`'s `sum(logpdf.(dist_bcast, y))` fallback doesn't care whether
# `dist_bcast`/`y` are vectors or matrices).
# ===========================================================================
@model function irt_2pl_model(I, J, y)
    sigma_theta ~ Truncated(Cauchy(0, 2), 0, Inf)
    theta ~ MvNormal(fill(0.0, J), sigma_theta)
    sigma_a ~ Truncated(Cauchy(0, 2), 0, Inf)
    a ~ MvLogNormal(fill(0.0, I), sigma_a)
    mu_b ~ Normal(0, 5)
    sigma_b ~ Truncated(Cauchy(0, 2), 0, Inf)
    b ~ MvNormal(fill(mu_b, I), sigma_b)
    dists = [BernoulliLogit(a[i] * (theta[j] - b[i])) for i in 1:I, j in 1:J]
    y .~ dists
end
function make_irt_2pl_data(seed)
    rng = Random.Xoshiro(seed)
    I, J = 6, 10
    true_theta = randn(rng, J)
    true_a = abs.(randn(rng, I)) .+ 0.5
    true_b = randn(rng, I) .* 0.5
    y = [Float64(rand(rng, Bernoulli(logistic(true_a[i] * (true_theta[j] - true_b[i]))))) for i in 1:I, j in 1:J]
    return (I=I, J=J, y=y)
end

# ===========================================================================
# kidscore_* remaining variants — same regression shape as
# kidscore_interaction (models.jl), just with centered/standardized
# covariates computed before the model (or, here, inside it via `mean`/`std`
# on the data arguments) and fewer/different predictor sets.
# ===========================================================================
@model function kidscore_interaction_c_model(N, kid_score, mom_hs, mom_iq)
    c_mom_hs = mom_hs .- mean(mom_hs)
    c_mom_iq = mom_iq .- mean(mom_iq)
    inter = c_mom_hs .* c_mom_iq
    sigma ~ FlatPos(0.0)
    beta ~ filldist(Flat(), 4)
    kid_score ~ MvNormal(beta[1] .+ beta[2] .* c_mom_hs .+ beta[3] .* c_mom_iq .+ beta[4] .* inter, sigma^2 * I)
end

@model function kidscore_interaction_z_model(N, kid_score, mom_hs, mom_iq)
    z_mom_hs = (mom_hs .- mean(mom_hs)) ./ (2 .* std(mom_hs))
    z_mom_iq = (mom_iq .- mean(mom_iq)) ./ (2 .* std(mom_iq))
    inter = z_mom_iq .* z_mom_hs
    beta ~ filldist(Flat(), 4)
    sigma ~ FlatPos(0.0)
    kid_score ~ MvNormal(beta[1] .+ beta[2] .* z_mom_hs .+ beta[3] .* z_mom_iq .+ beta[4] .* inter, sigma^2 * I)
end

@model function kidscore_momhs_model(N, kid_score, mom_hs)
    sigma ~ Truncated(Cauchy(0, 2.5), 0, Inf)
    beta ~ filldist(Flat(), 2)
    kid_score ~ MvNormal(beta[1] .+ mom_hs .* beta[2], sigma^2 .* I)
end

@model function kidscore_momiq_model(N, kid_score, mom_iq)
    sigma ~ Truncated(Cauchy(0, 2.5), 0, Inf)
    beta ~ filldist(Flat(), 2)
    kid_score ~ MvNormal(beta[1] .+ beta[2] .* mom_iq, sigma^2 * I)
end
function make_kidscore2_data(seed)
    rng = Random.Xoshiro(seed)
    N = 40
    mom_hs = Float64.(rand(rng, Bool, N))
    mom_iq = randn(rng, N) .* 15 .+ 100
    kid_score = 20.0 .+ 5.0 .* mom_hs .+ 0.3 .* mom_iq .+ randn(rng, N) .* 10
    return (N=N, kid_score=kid_score, mom_hs=mom_hs, mom_iq=mom_iq)
end

# ===========================================================================
# kilpisjarvi — simple linear regression with informative Normal priors on
# alpha/beta (parameterized by ARGUMENTS, not literals — `pmualpha` etc are
# themselves model arguments, a common PosteriorDB pattern of passing prior
# hyperparameters in as data) and a Flat(Pos) prior on sigma.
# ===========================================================================
@model function kilpisjarvi_model(N, x, y, pmualpha, psalpha, pmubeta, psbeta)
    alpha ~ Normal(pmualpha, psalpha)
    beta ~ Normal(pmubeta, psbeta)
    sigma ~ FlatPos(0.0)
    y ~ MvNormal(alpha .+ beta .* x, sigma^2 .* I)
end
function make_kilpisjarvi_data(seed)
    rng = Random.Xoshiro(seed)
    N = 25
    x = collect(1.0:N)
    true_alpha, true_beta = 2.0, 0.05
    y = true_alpha .+ true_beta .* x .+ randn(rng, N) .* 0.3
    return (N=N, x=x, y=y, pmualpha=0.0, psalpha=10.0, pmubeta=0.0, psbeta=10.0)
end
