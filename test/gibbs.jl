# Tests for src/gibbs.jl + src/latent.jl — the three M3 milestone gates
# (see the design plan's "M3 implementation plan" section):
#   (a) an exact-conditional latent kernel matches the analytically-computed
#       joint posterior of a small conjugate model;
#   (b) a 2-state Gaussian HMM sampled via Gibbs(NUTS, NUTS, FFBS-kernel)
#       matches NUTS run directly on the same model with the discrete state
#       marginalized out by hand (forward algorithm, via the new
#       `@addlogprob!`);
#   (c) a kernel that asserts its inputs are never `ForwardDiff.Dual` proves
#       latents genuinely never reach a gradient call.

using Distributions: Distributions, Normal, Beta, Categorical, logpdf
using StatsFuns: logsumexp, softmax
using AdvancedHMC: NUTS
using StableRNGs: StableRNG
import AbstractMCMC
using Statistics: mean, std
using ForwardDiff: ForwardDiff
using LinearAlgebra: diag
using ADTypes: ADTypes

# ===========================================================================
# Gate (a): exact-conditional kernel matches analytic posterior.
#
# Model: mu ~ Normal(0,1); z ~ Normal(mu,1); y ~ Normal(z, 0.5), y fixed.
# (mu, z) jointly Gaussian a priori with covariance [[1,1],[1,2]]; observing
# y (linear-Gaussian) gives a closed-form joint posterior via standard
# Gaussian conditioning — computed independently below (small 2x2 linear
# algebra), NOT via any PracticalBayes machinery, so this is a genuine
# external reference.
# ===========================================================================

@testset "gibbs.jl: exact-conditional kernel matches analytic posterior" begin
    @model function conj_model(y)
        mu ~ Normal(0, 1)
        z ~ Normal(mu, 1)
        y ~ Normal(z, 0.5)
    end

    struct ExactNormalKernel <: AbstractLatentKernel end
    function PracticalBayes.latent_step(rng, ::ExactNormalKernel, block_names, c::ModelConditional)
        block_names == (:z,) || error("this test kernel only handles the `:z` block")
        y = c.model.args.y
        mu = c.values.mu
        prior_prec = 1.0             # z ~ Normal(mu, 1)
        lik_prec = 1 / 0.5^2         # y ~ Normal(z, 0.5)
        post_prec = prior_prec + lik_prec
        post_mean = mu + lik_prec * (y - mu) / post_prec
        return (; z=rand(rng, Normal(post_mean, sqrt(1 / post_prec))))
    end

    y_obs = 2.0
    m = conj_model(y_obs)
    spl = Gibbs(:mu => NUTS(0.8), :z => ExactNormalKernel())

    # Independent analytic reference: (mu,z) ~ N(0, [[1,1],[1,2]]) a priori;
    # observation y ~ N(z, 0.25) with H = [0 1]. Standard Gaussian
    # conditioning: post_prec = prior_prec + H'*inv(R)*H, post_mean =
    # post_cov * H' * inv(R) * y.
    Sigma_prior = [1.0 1.0; 1.0 2.0]
    H = [0.0 1.0]
    R = 0.25
    post_prec = inv(Sigma_prior) + H' * (1 / R) * H
    post_cov = inv(post_prec)
    post_mean = post_cov * (H' * (1 / R) * [y_obs])
    analytic_mean = post_mean            # [mu_mean, z_mean]
    analytic_sd = sqrt.(diag(post_cov))

    rng = StableRNG(1)
    n_sweeps = 4000
    n_burn = 1000
    n_adapts = 500
    mu_draws = Vector{Float64}(undef, n_sweeps)
    z_draws = Vector{Float64}(undef, n_sweeps)
    transition, state = AbstractMCMC.step(rng, m, spl)
    for i in 1:n_sweeps
        # `n_adapts` must be passed as its FINAL target value from the very
        # first sweep onward, NOT ramped up (e.g. `min(i, 500)`) — AdvancedHMC's
        # windowed StanHMCAdaptor calls `initialize!(adaptor, n_adapts)`
        # (which locks in the Stan-style adaptation WINDOW SCHEDULE) only on
        # this block's first-ever step (`i_internal == 1`); passing a small
        # `n_adapts` on that first call permanently fixes a tiny window for
        # the rest of the run, no matter what's passed on later sweeps.
        # Confirmed directly: `min(i, 500)` starting from `n_adapts=1` on
        # sweep 1 left the step size frozen at its poor initial guess for
        # 1500+ sweeps (mass-matrix/step-size adaptation effectively
        # disabled), producing a barely-moving, highly autocorrelated chain
        # that looked like a Gibbs correctness bug but was purely this
        # AdvancedHMC calling-convention gotcha.
        transition, state = AbstractMCMC.step(rng, m, spl, state; n_adapts=n_adapts)
        mu_draws[i] = transition.mu
        z_draws[i] = transition.z
    end
    mu_post = mu_draws[(n_burn + 1):end]
    z_post = z_draws[(n_burn + 1):end]

    mu_se = std(mu_post) / sqrt(length(mu_post))
    z_se = std(z_post) / sqrt(length(z_post))
    @test abs(mean(mu_post) - analytic_mean[1]) < 3 * mu_se
    @test abs(mean(z_post) - analytic_mean[2]) < 3 * z_se
    @test isapprox(std(mu_post), analytic_sd[1]; rtol=0.15)
    @test isapprox(std(z_post), analytic_sd[2]; rtol=0.15)
end

# ===========================================================================
# Gate (b): 2-state Gaussian HMM — Gibbs(NUTS, NUTS, FFBS) matches NUTS on
# the hand-marginalized model.
# ===========================================================================

@testset "gibbs.jl: HMM Gibbs+FFBS matches NUTS on the marginalized model" begin
    means = (0.0, 5.0)

    # `y[t] ~ dist` in a loop over already-observed data doesn't fit
    # `x[i] ~ dist`'s assume-only container pattern (same porting idiom
    # established throughout bench/corpus/ this session) — vectorized `.~`
    # over the whole per-timestep mean vector instead.
    @model function hmm_latent(y)
        p_stay ~ Beta(8, 2)
        sigma ~ Distributions.Exponential(1)
        N = length(y)
        z = Vector{Int}(undef, N)
        z[1] ~ Categorical([0.5, 0.5])
        for t in 2:N
            z[t] ~ Categorical(z[t - 1] == 1 ? [p_stay, 1 - p_stay] : [1 - p_stay, p_stay])
        end
        mu = [zi == 1 ? 0.0 : 5.0 for zi in z]
        y .~ Normal.(mu, sigma)
    end

    struct FFBS <: AbstractLatentKernel end
    function PracticalBayes.latent_step(rng, ::FFBS, block_names, c::ModelConditional)
        block_names == (:z,) || error("this FFBS kernel only handles the `:z` block")
        y = c.model.args.y
        p_stay, sigma = c.values.p_stay, c.values.sigma
        N = length(y)
        P = [p_stay 1-p_stay; 1-p_stay p_stay]

        logα = Matrix{Float64}(undef, 2, N)
        logα[:, 1] .= log(0.5) .+ logpdf.(Normal.(means, sigma), y[1])
        for t in 2:N, j in 1:2
            logα[j, t] = logsumexp(logα[:, t - 1] .+ log.(P[:, j])) + logpdf(Normal(means[j], sigma), y[t])
        end

        z = Vector{Int}(undef, N)
        z[N] = rand(rng, Categorical(softmax(logα[:, N])))
        for t in (N - 1):-1:1
            w = softmax(logα[:, t] .+ log.(P[:, z[t + 1]]))
            z[t] = rand(rng, Categorical(w))
        end
        return (; z=z)
    end

    @model function hmm_marginal(y)
        p_stay ~ Beta(8, 2)
        sigma ~ Distributions.Exponential(1)
        N = length(y)
        P = [p_stay 1-p_stay; 1-p_stay p_stay]
        logα = [log(0.5) + logpdf(Normal(means[1], sigma), y[1]), log(0.5) + logpdf(Normal(means[2], sigma), y[1])]
        for t in 2:N
            logα = [
                logsumexp(logα .+ log.(P[:, 1])) + logpdf(Normal(means[1], sigma), y[t]),
                logsumexp(logα .+ log.(P[:, 2])) + logpdf(Normal(means[2], sigma), y[t]),
            ]
        end
        @addlogprob! logsumexp(logα)
    end

    # Synthetic data from a known true path so both chains have real signal
    # to identify p_stay/sigma from.
    rng_data = StableRNG(7)
    N = 40
    true_p_stay, true_sigma = 0.85, 0.7
    z_true = Vector{Int}(undef, N)
    z_true[1] = 1
    for t in 2:N
        z_true[t] = rand(rng_data) < (z_true[t - 1] == 1 ? true_p_stay : 1 - true_p_stay) ? z_true[t - 1] : 3 - z_true[t - 1]
    end
    y = [rand(rng_data, Normal(means[z_true[t]], true_sigma)) for t in 1:N]

    # Chain 1: Gibbs + FFBS.
    m_latent = hmm_latent(y)
    spl = Gibbs(:p_stay => NUTS(0.8), :sigma => NUTS(0.8), :z => FFBS())
    rng1 = StableRNG(11)
    n_sweeps, n_burn = 1500, 500
    n_adapts = 300  # passed as a FIXED value from sweep 1 onward — see gate (a)'s comment on why
    p_stay_draws1 = Vector{Float64}(undef, n_sweeps)
    sigma_draws1 = Vector{Float64}(undef, n_sweeps)
    transition, state = AbstractMCMC.step(rng1, m_latent, spl)
    for i in 1:n_sweeps
        transition, state = AbstractMCMC.step(rng1, m_latent, spl, state; n_adapts=n_adapts)
        p_stay_draws1[i] = transition.p_stay
        sigma_draws1[i] = transition.sigma
    end
    p1 = p_stay_draws1[(n_burn + 1):end]
    s1 = sigma_draws1[(n_burn + 1):end]

    # Chain 2: NUTS directly on the marginalized model.
    m_marginal = hmm_marginal(y)
    layout2, θ0_2, store0_2 = build_layout(m_marginal)
    ldf2 = LogDensityFunction(m_marginal, layout2, store0_2, ADTypes.AutoForwardDiff(); θ0=θ0_2)
    ldm2 = AbstractMCMC.LogDensityModel(ldf2)
    rng2 = StableRNG(12)
    _, hstate = AbstractMCMC.step(rng2, ldm2, NUTS(0.8); initial_params=θ0_2)
    n2 = n_sweeps
    p_stay_draws2 = Vector{Float64}(undef, n2)
    sigma_draws2 = Vector{Float64}(undef, n2)
    for i in 1:n2
        _, hstate = AbstractMCMC.step(rng2, ldm2, NUTS(0.8), hstate; n_adapts=n_adapts)
        θ = AbstractMCMC.getparams(hstate)
        nt = invlink(layout2, θ)
        p_stay_draws2[i] = nt.p_stay
        sigma_draws2[i] = nt.sigma
    end
    p2 = p_stay_draws2[(n_burn + 1):end]
    s2 = sigma_draws2[(n_burn + 1):end]

    combined_se_p = sqrt(std(p1)^2 / length(p1) + std(p2)^2 / length(p2))
    combined_se_s = sqrt(std(s1)^2 / length(s1) + std(s2)^2 / length(s2))
    @test abs(mean(p1) - mean(p2)) < 3 * combined_se_p
    @test abs(mean(s1) - mean(s2)) < 3 * combined_se_s
end

# ===========================================================================
# Gate (c): a kernel that asserts no Duals reach the store.
# ===========================================================================

@testset "gibbs.jl: latent values never reach a gradient call as Duals" begin
    @model function dualcheck_model(y)
        mu ~ Normal(0, 1)
        z ~ Normal(0, 1)
        y ~ Normal(mu + z, 0.5)
    end

    struct DualCheckKernel <: AbstractLatentKernel end
    function PracticalBayes.latent_step(rng, ::DualCheckKernel, block_names, c::ModelConditional)
        block_names == (:z,) || error("bad block")
        @assert eltype([c.values.mu]) <: Union{Float32,Float64} "latent kernel saw a non-plain-float value: $(typeof(c.values.mu))"
        @assert !(c.values.mu isa ForwardDiff.Dual) "latent kernel saw a Dual!"
        return (; z=rand(rng, Normal(c.values.mu, 1.0)))
    end

    m = dualcheck_model(2.0)
    spl = Gibbs(:mu => NUTS(0.8), :z => DualCheckKernel())
    rng = StableRNG(3)
    transition, state = AbstractMCMC.step(rng, m, spl)
    for i in 1:300
        transition, state = AbstractMCMC.step(rng, m, spl, state; n_adapts=100)
    end
    @test true  # reaching here without the kernel's internal @assert firing IS the test
end
