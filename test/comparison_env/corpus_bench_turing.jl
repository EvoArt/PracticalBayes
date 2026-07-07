# Turing-side counterpart to bench/corpus_bench.jl: benchmarks a
# REPRESENTATIVE SUBSET (~20 of the 53) of the model corpus on Turing, one
# per structural family, across every available AD backend and both
# Float64/Float32, on the same three layers (logdensity, gradient, NUTS).
# Run from THIS directory's own environment (Turing/PracticalBayes can
# never coexist — see this file's siblings for the full story):
#   julia --project=. corpus_bench_turing.jl
#
# Models are near-verbatim ports of the SAME PracticalBayes corpus models
# (bench/corpus/{posteriordb,tutorials}/models*.jl) back to plain Turing
# syntax — in most cases this is close to the ORIGINAL PosteriorDB/tutorial
# source, since the only PracticalBayes-side changes were the
# `x[i]~dist`-is-assume-only workarounds (`.~` instead of a per-element
# loop) and `Turing.Flat()`/`filldist(Turing.Flat(),...)` -> `Flat()`/
# `filldist(Flat(),...)` renames — both trivially reversible here.

using Turing
using DynamicPPL
using LinearAlgebra
using StatsFuns: logsumexp, softmax, logistic
using Random
using Dates: now
using ADTypes
import LogDensityProblems
import AbstractMCMC

println("Turing version: ", pkgversion(Turing))

# Byte-index slicing on an error message can land mid-UTF-8-character and
# throw `StringIndexError` INSIDE a `catch` block's own reporting code,
# uncaught — see bench/corpus_bench.jl's matching helper for the full story
# (confirmed directly: this killed a real corpus-wide run on the
# PracticalBayes side via a Mooncake error message containing `│`).
_trunc_err(e; n=200) = first(sprint(showerror, e), n)

# ===========================================================================
# Timing harness — identical methodology to bench/suite.jl /
# bench/corpus_bench.jl (kept as its own copy since this runs in a
# permanently separate environment).
# ===========================================================================

struct TimingResult
    label::String
    first_call_s::Float64
    min_s::Float64
    median_s::Float64
    mean_s::Float64
    std_s::Float64
    reps::Int
end

using Statistics: mean, median, std

function time_reps(f, label; reps=8)
    first_call_s = @elapsed f()
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        times[i] = @elapsed f()
    end
    return TimingResult(label, first_call_s, minimum(times), median(times), mean(times), std(times), reps)
end

const _RESULTS = NamedTuple[]
function record!(; package="Turing", corpus, model, layer, precision, backend, r::TimingResult)
    push!(
        _RESULTS,
        (;
            package, corpus, model, layer, precision, backend,
            first_call_s=r.first_call_s, min_s=r.min_s, median_s=r.median_s, mean_s=r.mean_s, std_s=r.std_s, reps=r.reps,
        ),
    )
    return r
end

function _git_commit()
    try
        return strip(read(`git -C $(joinpath(@__DIR__, "..", "..")) rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

_json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"")
_json_value(x::AbstractString) = "\"" * _json_escape(x) * "\""
_json_value(x::Real) = isfinite(x) ? string(x) : "null"
_json_value(x::Integer) = string(x)
_to_json_line(nt::NamedTuple) = "{" * join(("\"$(k)\":$(_json_value(v))" for (k, v) in pairs(nt)), ",") * "}"

function write_history!(path=joinpath(@__DIR__, "..", "..", "bench", "results", "history_corpus_turing.jsonl"))
    mkpath(dirname(path))
    commit = _git_commit()
    timestamp = string(now())
    open(path, "a") do io
        for r in _RESULTS
            println(io, _to_json_line(merge((; timestamp, commit), r)))
        end
    end
    println("Wrote ", length(_RESULTS), " results to ", path, " (commit ", commit, ")")
    return path
end

# ===========================================================================
# AD backends — same registration pattern as generate_turing_reference.jl.
# ===========================================================================

const _AD_BACKENDS = Pair{String,Any}["ForwardDiff" => AutoForwardDiff()]
if !isnothing(Base.find_package("Mooncake"))
    @eval import Mooncake
    push!(_AD_BACKENDS, "Mooncake" => AutoMooncake(; config=nothing))
end
if !isnothing(Base.find_package("ReverseDiff"))
    @eval import ReverseDiff
    push!(_AD_BACKENDS, "ReverseDiff" => AutoReverseDiff())
end
# Enzyme deliberately excluded — see bench/corpus_bench.jl's matching
# comment: across this many structurally varied models it produced a
# native-crash-level failure that killed the whole benchmark process, on
# top of several safely-caught EnzymeRuntimeActivityError/
# IllegalTypeAnalysisException failures on other models.
# if !isnothing(Base.find_package("Enzyme"))
#     @eval import Enzyme
#     push!(_AD_BACKENDS, "Enzyme" => AutoEnzyme())
# end

# ===========================================================================
# Model definitions — 20 representative models, one (or two) per structural
# family covered by the PracticalBayes corpus.
# ===========================================================================

# --- Rate_1: simplest conjugate scalar model ---
Turing.@model function Rate_1_model(n, k)
    theta ~ Beta(1, 1)
    k ~ Binomial(n, theta)
end
make_Rate_1_data() = (n=20, k=7)

# --- blr: filldist + MvNormal regression ---
Turing.@model function blr_model(N, D, X, y)
    beta ~ filldist(Normal(0, 10), D)
    sigma ~ Truncated(Normal(0, 10), 0, Inf)
    y ~ MvNormal(X * beta, sigma^2 * I)
end
function make_blr_data()
    rng = Random.Xoshiro(1)
    N, D = 30, 4
    X = randn(rng, N, D)
    true_beta = randn(rng, D)
    y = X * true_beta .+ randn(rng, N) .* 0.5
    return (N=N, D=D, X=X, y=y)
end

# --- eight_schools_centered: hierarchical MvNormal ---
Turing.@model function eight_schools_centered_model(J, y, sigma)
    tau ~ Truncated(Cauchy(0, 5), 0, Inf)
    mu ~ Normal(0, 5)
    theta ~ MvNormal(fill(mu, J), sqrt(tau) * I)
    y ~ MvNormal(theta, Diagonal(sqrt.(sigma)))
end
function make_eight_schools_data()
    rng = Random.Xoshiro(1)
    J = 8
    sigma = abs.(randn(rng, J)) .* 5 .+ 5
    true_theta = randn(rng, J) .* 5
    y = true_theta .+ randn(rng, J) .* sqrt.(sigma)
    return (J=J, y=y, sigma=sigma)
end

# --- dugongs: nonlinear regression, := reporting ---
Turing.@model function dugongs_model(N, x, Y)
    alpha ~ Normal(0, 1000)
    beta ~ Normal(0, 1000)
    lambda ~ Uniform(0.5, 1.0)
    tau ~ Gamma(1.0e4, 1.0e-4)
    sigma = 1 / sqrt(tau)
    Y ~ MvNormal(alpha .- beta .* lambda .^ x, sigma^2 .* I)
end
function make_dugongs_data()
    rng = Random.Xoshiro(1)
    N = 27
    x = collect(1.0:N)
    Y = 2.5 .- 1.5 .* 0.9 .^ x .+ randn(rng, N) .* 0.1
    return (N=N, x=x, Y=Y)
end

# --- GLM_Poisson: discrete-likelihood GLM ---
Turing.@model function GLM_Poisson_model(n, C, year; year_squared=year .^ 2, year_cubed=year .^ 3)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 10)
    beta3 ~ Uniform(-10, 10)
    log_lambda = alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed
    C ~ product_distribution(Poisson.(exp.(log_lambda)))
end
function make_GLM_Poisson_data()
    rng = Random.Xoshiro(1)
    n = 20
    year = collect(range(-1.0, 1.0; length=n))
    C = Float64.(rand.(rng, Poisson.(exp.(1.0 .+ 0.5 .* year))))
    return (n=n, C=C, year=year)
end

# --- kidscore_interaction: Flat priors + interaction regression ---
Turing.@model function kidscore_interaction_model(N, kid_score, mom_iq, mom_hs)
    inter = mom_hs .* mom_iq
    beta ~ filldist(Flat(), 4)
    sigma ~ Truncated(Cauchy(0, 2.5), 0, Inf)
    kid_score ~ MvNormal(beta[1] .+ beta[2] .* mom_hs .+ beta[3] .* mom_iq .+ beta[4] .* inter, sigma^2 * I)
end
function make_kidscore_data()
    rng = Random.Xoshiro(1)
    N = 40
    mom_hs = Float64.(rand(rng, Bool, N))
    mom_iq = randn(rng, N) .* 15 .+ 100
    inter = mom_hs .* mom_iq
    kid_score = 20.0 .+ 5.0 .* mom_hs .+ 0.5 .* mom_iq .- 0.05 .* inter .+ randn(rng, N) .* 10
    return (N=N, kid_score=kid_score, mom_iq=mom_iq, mom_hs=mom_hs)
end

# --- wells_dae: Flat-prior logistic GLM ---
Turing.@model function wells_dae_model(N, switched, dist, arsenic, educ)
    dist100 = dist ./ 100
    educ4 = educ ./ 4.0
    x = hcat(dist100, arsenic, educ4)
    alpha ~ Flat()
    beta ~ filldist(Flat(), 3)
    switched ~ product_distribution(BernoulliLogit.(alpha .+ x * beta))
end
function make_wells_data()
    rng = Random.Xoshiro(1)
    N = 60
    dist = abs.(randn(rng, N)) .* 50 .+ 20
    arsenic = abs.(randn(rng, N)) .* 2 .+ 1
    educ = Float64.(rand(rng, 0:12, N))
    true_p = logistic.(0.2 .- 0.01 .* dist .- 0.1 .* arsenic)
    switched = Float64.(rand(rng, N) .< true_p)
    return (N=N, switched=switched, dist=dist, arsenic=arsenic, educ=educ)
end

# --- radon_variable_intercept_slope_centered: gather-indexed hierarchical ---
Turing.@model function radon_vis_model(N, J, county_idx, floor_measure, log_radon)
    sigma_y ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_beta ~ Truncated(Normal(0, 1), 0, Inf)
    sigma_alpha ~ Truncated(Normal(0, 1), 0, Inf)
    mu_alpha ~ Normal(0, 10)
    mu_beta ~ Normal(0, 10)
    alpha ~ MvNormal(fill(mu_alpha, J), sigma_alpha^2 .* I)
    beta ~ MvNormal(fill(mu_beta, J), sigma_beta^2 .* I)
    log_radon ~ MvNormal(alpha[county_idx] .+ floor_measure .* beta[county_idx], sigma_y^2 .* I)
end
function make_radon_data()
    rng = Random.Xoshiro(1)
    N, J = 60, 8
    county_idx = rand(rng, 1:J, N)
    floor_measure = Float64.(rand(rng, Bool, N))
    true_county_effect = randn(rng, J) .* 0.5 .+ 1.2
    log_radon = [true_county_effect[county_idx[i]] - 0.5 * floor_measure[i] + randn(rng) * 0.3 for i in 1:N]
    return (N=N, J=J, county_idx=county_idx, floor_measure=floor_measure, log_radon=log_radon)
end

# --- seeds_model: UniformScaling + BinomialLogit GLM ---
Turing.@model function seeds_model(I_, n, N, x1, x2; x1x2=x1 .* x2)
    alpha0 ~ Normal(0.0, 1.0e3)
    alpha1 ~ Normal(0.0, 1.0e3)
    alpha2 ~ Normal(0.0, 1.0e3)
    alpha12 ~ Normal(0.0, 1.0e3)
    tau ~ Gamma(1.0e3, 1.0e-3)
    sigma = 1.0 / sqrt(tau)
    b ~ MvNormal(fill(0.0, I_), UniformScaling(sigma^2))
    n ~ product_distribution(BinomialLogit.(N, alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+ alpha12 .* x1x2 .+ b))
end
function make_seeds_data()
    rng = Random.Xoshiro(1)
    I_ = 21
    N = fill(20, I_)
    x1 = randn(rng, I_) .* 0.3
    x2 = randn(rng, I_) .* 0.3
    true_p = logistic.(0.1 .+ 0.3 .* x1 .- 0.2 .* x2)
    n = [rand(rng, Binomial(N[i], true_p[i])) for i in 1:I_]
    return (I_=I_, n=n, N=N, x1=x1, x2=x2)
end

# --- rats: gather-indexed varying-intercept/slope, non-Flat filldist ---
Turing.@model function rats_model(N, Npts, rat, x, y, xbar)
    mu_alpha ~ Normal(0, 100)
    mu_beta ~ Normal(0, 100)
    sigma_y ~ Flat()
    sigma_alpha ~ Flat()
    sigma_beta ~ Flat()
    alpha ~ filldist(Normal(mu_alpha, sigma_alpha), N)
    beta ~ filldist(Normal(mu_beta, sigma_beta), N)
    y ~ MvNormal(alpha[rat] .+ beta[rat] .* (x .- xbar), sigma_y^2 .* I)
end
function make_rats_data()
    rng = Random.Xoshiro(1)
    N, Nt = 6, 5
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

# --- logistic_regression_rhs: regularized horseshoe ---
Turing.@model function logistic_regression_rhs_model(n, d, y, x, scale_icept, scale_global, nu_global, nu_local, slab_scale, slab_df)
    z ~ MvNormal(fill(0.0, d), I)
    lambda ~ filldist(Truncated(TDist(nu_local), 0, Inf), d)
    tau ~ Truncated(scale_global * 2 * TDist(nu_global), 0, Inf)
    caux ~ filldist(InverseGamma(0.5 * slab_df, 0.5 * slab_df), d)
    beta0 ~ Normal(0, scale_icept)
    c = slab_scale .* sqrt.(caux)
    lambda_tilde = sqrt.(c .^ 2 .* lambda .^ 2 ./ (c .^ 2 .+ tau .^ 2 .* lambda .^ 2))
    beta = z .* lambda_tilde .* tau
    y ~ product_distribution(BernoulliLogit.(beta0 .+ x * beta))
end
function make_logistic_rhs_data()
    rng = Random.Xoshiro(1)
    n, d = 40, 5
    x = randn(rng, n, d) .* 0.5
    true_beta = [1.0, -0.5, 0.0, 0.0, 0.3]
    p = 1.0 ./ (1.0 .+ exp.(-(0.2 .+ x * true_beta)))
    y = Float64.(rand(rng, n) .< p)
    return (n=n, d=d, y=y, x=x, scale_icept=5.0, scale_global=1.0, nu_global=3.0, nu_local=3.0, slab_scale=2.0, slab_df=4.0)
end

# --- pilots: two simultaneous gather-indexed random effects ---
Turing.@model function pilots_model(N, n_groups, n_scenarios, group_id, scenario_id, y)
    sigma_y ~ Uniform(0, 100)
    mu_a ~ Normal(0, 1)
    sigma_a ~ Uniform(0, 100)
    a ~ MvNormal(fill(10 * mu_a, n_groups), sigma_a^2 .* I)
    mu_b ~ Normal(0, 1)
    sigma_b ~ Uniform(0, 100)
    b ~ MvNormal(fill(10 * mu_b, n_scenarios), sigma_b^2 .* I)
    y_hat = a[group_id] .+ b[scenario_id]
    y ~ MvNormal(y_hat, sigma_y^2 .* I)
end
function make_pilots_data()
    rng = Random.Xoshiro(1)
    N = 40
    n_groups, n_scenarios = 5, 8
    group_id = rand(rng, 1:n_groups, N)
    scenario_id = rand(rng, 1:n_scenarios, N)
    true_a = randn(rng, n_groups) .* 2
    true_b = randn(rng, n_scenarios) .* 2
    y = [true_a[group_id[i]] + true_b[scenario_id[i]] + randn(rng) for i in 1:N]
    return (N=N, n_groups=n_groups, n_scenarios=n_scenarios, group_id=group_id, scenario_id=scenario_id, y=y)
end

# --- election88_full: five simultaneous random effects ---
Turing.@model function election88_full_model(
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
    y_hat = beta[1] .+ beta[2] .* black .+ beta[3] .* female .+
            beta[5] .* female .* black .+ beta[4] .* v_prev_full .+
            a[age] .+ b[edu] .+ c[age_edu] .+ d[state] .+ e[region_full]
    y ~ product_distribution(BernoulliLogit.(y_hat))
end
function make_election88_data()
    rng = Random.Xoshiro(1)
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
        n_age=n_age, n_age_edu=n_age_edu, n_edu=n_edu, n_region_full=n_region_full, n_state=n_state,
        age=age, age_edu=age_edu, black=black, edu=edu, female=female, region_full=region_full, state=state,
        v_prev_full=v_prev_full, y=y,
    )
end

# --- irt_2pl: 2D matrix-shaped observe ---
Turing.@model function irt_2pl_model(I, J, y)
    sigma_theta ~ Truncated(Cauchy(0, 2), 0, Inf)
    theta ~ MvNormal(fill(0.0, J), sigma_theta)
    sigma_a ~ Truncated(Cauchy(0, 2), 0, Inf)
    a ~ MvLogNormal(fill(0.0, I), sigma_a)
    mu_b ~ Normal(0, 5)
    sigma_b ~ Truncated(Cauchy(0, 2), 0, Inf)
    b ~ MvNormal(fill(mu_b, I), sigma_b)
    for i in 1:I
        y[i, :] ~ arraydist([BernoulliLogit(a[i] * (theta[j] - b[i])) for j in 1:J])
    end
end
function make_irt_2pl_data()
    rng = Random.Xoshiro(1)
    I, J = 6, 10
    true_theta = randn(rng, J)
    true_a = abs.(randn(rng, I)) .+ 0.5
    true_b = randn(rng, I) .* 0.5
    y = [Float64(rand(rng, Bernoulli(logistic(true_a[i] * (true_theta[j] - true_b[i]))))) for i in 1:I, j in 1:J]
    return (I=I, J=J, y=y)
end

# --- coin-flipping ---
Turing.@model function coinflip(y)
    p ~ Beta(1, 1)
    y .~ Bernoulli.(p)
end
function make_coinflip_data()
    rng = Random.Xoshiro(1)
    return (y=Float64.(rand(rng, Bernoulli(0.5), 100)),)
end

# --- bayesian-linear-regression ---
Turing.@model function linear_regression(x, y)
    σ² ~ Truncated(Normal(0, 100), 0, Inf)
    intercept ~ Normal(0, sqrt(3))
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(zeros(nfeatures), 10.0 * I)
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, σ² * I)
end
function make_linear_regression_data()
    rng = Random.Xoshiro(1)
    N, D = 32, 3
    x = randn(rng, N, D)
    true_coef = randn(rng, D)
    y = 1.0 .+ x * true_coef .+ randn(rng, N) .* 0.5
    return (x=x, y=y)
end

# --- bayesian-logistic-regression ---
Turing.@model function logistic_regression(x, y, σ)
    intercept ~ Normal(0, σ)
    student ~ Normal(0, σ)
    balance ~ Normal(0, σ)
    income ~ Normal(0, σ)
    v = logistic.(intercept .+ student .* x[1, :] .+ balance .* x[2, :] .+ income .* x[3, :])
    y ~ product_distribution(Bernoulli.(v))
end
function make_logistic_regression_data()
    rng = Random.Xoshiro(1)
    N = 40
    x = randn(rng, 3, N)
    true_coef = [0.5, -0.3, 0.8, 0.1]
    p = logistic.(true_coef[1] .+ true_coef[2] .* x[1, :] .+ true_coef[3] .* x[2, :] .+ true_coef[4] .* x[3, :])
    y = Float64.(rand(rng, N) .< p)
    return (x=x, y=y, σ=3.0)
end

# --- bayesian-poisson-regression ---
Turing.@model function poisson_regression(x, y, n, σ²)
    b0 ~ Normal(0, σ²)
    b1 ~ Normal(0, σ²)
    b2 ~ Normal(0, σ²)
    b3 ~ Normal(0, σ²)
    theta = b0 .+ b1 .* x[:, 1] .+ b2 .* x[:, 2] .+ b3 .* x[:, 3]
    y ~ product_distribution(Poisson.(exp.(theta)))
end
function make_poisson_regression_data()
    rng = Random.Xoshiro(1)
    n = 30
    x = randn(rng, n, 3) .* 0.3
    true_coef = [0.5, 0.2, -0.1, 0.3]
    y = [rand(rng, Poisson(exp(true_coef[1] + true_coef[2] * x[i, 1] + true_coef[3] * x[i, 2] + true_coef[4] * x[i, 3]))) for i in 1:n]
    return (x=x, y=Float64.(y), n=n, σ²=10.0)
end

# --- multinomial-logistic-regression ---
Turing.@model function multinomial_logistic_regression(x, y, σ)
    n = size(x, 2)
    intercept_versicolor ~ Normal(0, σ)
    intercept_virginica ~ Normal(0, σ)
    coefficients_versicolor ~ MvNormal(zeros(4), σ^2 * I)
    coefficients_virginica ~ MvNormal(zeros(4), σ^2 * I)
    values_versicolor = intercept_versicolor .+ (coefficients_versicolor' * x)
    values_virginica = intercept_virginica .+ (coefficients_virginica' * x)
    vs = [softmax([0.0, values_versicolor[i], values_virginica[i]]) for i in 1:n]
    y ~ product_distribution(Categorical.(vs))
end
function make_multinomial_data()
    rng = Random.Xoshiro(1)
    n = 30
    x = randn(rng, 4, n) .* 0.5
    y = rand(rng, 1:3, n)
    return (x=x, y=y, σ=3.0)
end

# --- probabilistic-pca ---
Turing.@model function pPCA(X, k::Int)
    N, D = size(X)
    W ~ filldist(Normal(), D, k)
    Z ~ filldist(Normal(), k, N)
    mu ~ MvNormal(fill(0.0, D), I)
    genes_mean = W * Z .+ mu
    X ~ arraydist([MvNormal(m, I) for m in eachcol(genes_mean')])
end
function make_pPCA_data()
    rng = Random.Xoshiro(1)
    N, D, k = 20, 4, 2
    true_W = randn(rng, D, k)
    true_Z = randn(rng, k, N)
    X = (true_W * true_Z)' .+ randn(rng, N, D) .* 0.3
    return (X=X, k=k)
end

# ===========================================================================
# Benchmark driver
# ===========================================================================

const MODELS = [
    ("posteriordb", "Rate_1", () -> Rate_1_model(make_Rate_1_data()...), Dict(:theta => 0.3)),
    ("posteriordb", "blr", () -> blr_model(make_blr_data()...), Dict(:sigma => 1.0)),
    ("posteriordb", "eight_schools_centered", () -> eight_schools_centered_model(make_eight_schools_data()...), Dict(:tau => 1.0, :mu => 0.0)),
    ("posteriordb", "dugongs", () -> dugongs_model(make_dugongs_data()...), Dict(:alpha => 2.5, :beta => 1.5, :lambda => 0.9, :tau => 1.0)),
    ("posteriordb", "GLM_Poisson", () -> GLM_Poisson_model(make_GLM_Poisson_data()...), Dict(:alpha => 0.0, :beta1 => 0.0, :beta2 => 0.0, :beta3 => 0.0)),
    ("posteriordb", "kidscore_interaction", () -> kidscore_interaction_model(make_kidscore_data()...), Dict(:sigma => 10.0)),
    ("posteriordb", "wells_dae", () -> wells_dae_model(make_wells_data()...), Dict{Symbol,Any}()),
    ("posteriordb", "radon_variable_intercept_slope_centered", () -> radon_vis_model(make_radon_data()...), Dict(:sigma_y => 1.0, :sigma_beta => 1.0, :sigma_alpha => 1.0, :mu_alpha => 0.0, :mu_beta => 0.0)),
    ("posteriordb", "seeds", () -> seeds_model(make_seeds_data()...), Dict(:tau => 1.0)),
    ("posteriordb", "rats", () -> rats_model(make_rats_data()...), Dict(:sigma_y => 3.0, :sigma_alpha => 10.0, :sigma_beta => 2.0)),
    ("posteriordb", "logistic_regression_rhs", () -> logistic_regression_rhs_model(make_logistic_rhs_data()...), Dict(:beta0 => 0.0)),
    ("posteriordb", "pilots", () -> pilots_model(make_pilots_data()...), Dict(:mu_a => 0.0, :sigma_a => 1.0, :mu_b => 0.0, :sigma_b => 1.0)),
    ("posteriordb", "election88_full", () -> election88_full_model(make_election88_data()...), Dict{Symbol,Any}()),
    ("posteriordb", "irt_2pl", () -> irt_2pl_model(make_irt_2pl_data()...), Dict(:sigma_theta => 1.0, :sigma_a => 1.0, :mu_b => 0.0, :sigma_b => 1.0)),
    ("tutorials", "coin-flipping", () -> coinflip(make_coinflip_data()...), Dict(:p => 0.5)),
    ("tutorials", "bayesian-linear-regression", () -> linear_regression(make_linear_regression_data()...), Dict(Symbol("σ²") => 1.0)),
    ("tutorials", "bayesian-logistic-regression", () -> logistic_regression(make_logistic_regression_data()...), Dict{Symbol,Any}()),
    ("tutorials", "bayesian-poisson-regression", () -> poisson_regression(make_poisson_regression_data()...), Dict{Symbol,Any}()),
    ("tutorials", "multinomial-logistic-regression", () -> multinomial_logistic_regression(make_multinomial_data()...), Dict{Symbol,Any}()),
    ("tutorials", "probabilistic-pca", () -> pPCA(make_pPCA_data()...), Dict{Symbol,Any}()),
]

function bench_one_model(corpus, name, modelfn; reps=8, nuts_samples=50, nuts_reps=3)
    for T in (Float64,)  # Turing/DynamicPPL promotes to Float64 internally regardless (documented elsewhere in this repo) — Float32 sweep is not meaningful here
        local model, vi
        try
            model = modelfn()
            vi = DynamicPPL.VarInfo(model)
        catch e
            println(rpad("$corpus/$name", 40), "  [$T] BUILD FAILED — ", _trunc_err(e))
            continue
        end

        try
            r = time_reps(() -> model(vi), "$corpus/$name logdensity ($T)"; reps=reps)
            record!(; corpus, model=name, layer="logdensity", precision=string(T), backend="none", r)
        catch e
            println(rpad("$corpus/$name", 40), "  [$T] logdensity FAILED — ", _trunc_err(e))
        end

        for (bname, adtype) in _AD_BACKENDS
            try
                ldf = DynamicPPL.LogDensityFunction(model; adtype=adtype)
                params = zeros(T, LogDensityProblems.dimension(ldf))
                r = time_reps(() -> LogDensityProblems.logdensity_and_gradient(ldf, params), "$corpus/$name grad ($bname,$T)"; reps=reps)
                record!(; corpus, model=name, layer="gradient", precision=string(T), backend=bname, r)
            catch e
                println(rpad("$corpus/$name", 40), "  [$T/$bname] gradient FAILED — ", _trunc_err(e))
            end
        end

        try
            run() = Turing.sample(Random.Xoshiro(1), model, Turing.NUTS(0.8), nuts_samples; progress=false)
            r = time_reps(run, "$corpus/$name NUTS ($T)"; reps=nuts_reps)
            record!(; corpus, model=name, layer="nuts", precision=string(T), backend="ForwardDiff", r)
        catch e
            println(rpad("$corpus/$name", 40), "  [$T] NUTS FAILED — ", _trunc_err(e))
        end
    end
end

function run_corpus_bench_turing()
    empty!(_RESULTS)
    println("AD backends available: ", join(first.(_AD_BACKENDS), ", "))
    for (corpus, name, modelfn, _init) in MODELS
        println("  benchmarking ", corpus, "/", name, " ...")
        bench_one_model(corpus, name, modelfn)
    end
    write_history!()
end

run_corpus_bench_turing()
