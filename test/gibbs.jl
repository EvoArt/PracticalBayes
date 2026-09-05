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
using Random: Random
using ForwardDiff: ForwardDiff
using LinearAlgebra: diag
using ADTypes: ADTypes
using LogDensityProblems: LogDensityProblems
using PracticalBayes: AdaptiveHMC

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

# ===========================================================================
# Whole-array (matrix) latent block round-trips through a Gibbs sweep.
#
# The individual-level iFFBS / epidemic use case stores an entire hidden-state
# trajectory (an individual x time `Matrix{Int}`) as ONE latent variable in
# the value-store, sampled by a custom kernel and read back by an
# `@addlogprob!` likelihood term as an AD-constant. This gate confirms a
# whole `DiscreteMatrixDistribution` latent site: (1) traces without needing a
# `linked_vec_length` (custom discrete matrix dists don't define one, and a
# latent site never occupies θ anyway); (2) round-trips as a `Matrix{Int}`
# through `build_layout`'s value-store; and (3) preserves its matrix shape
# across a full Gibbs sweep while the continuous NUTS block still identifies
# its parameter. This is the PracticalBayes-side prerequisite for the
# EpidemicTrajectories package's iFFBS kernel.
# ===========================================================================

@testset "gibbs.jl: whole-matrix latent block round-trips" begin
    # A minimal discrete matrix "latent trajectory" distribution: Discrete
    # value support so `build_layout` tags the site `:latent` and routes it to
    # the value-store. Only needs `size`/`rand`/`logpdf` to trace.
    struct GridLatent <: Distributions.DiscreteMatrixDistribution
        nrows::Int
        ncols::Int
    end
    Base.size(d::GridLatent) = (d.nrows, d.ncols)
    Distributions.rand(rng::Random.AbstractRNG, d::GridLatent) = rand(rng, 0:1, d.nrows, d.ncols)
    Distributions.logpdf(::GridLatent, X::AbstractMatrix) = 0.0  # improper; real coupling via @addlogprob!

    @model function grid_model(y, nrows, ncols)
        mu ~ Normal(0, 1)
        X ~ GridLatent(nrows, ncols)
        # `X` enters the likelihood as an AD-constant (ValueSlot): the summed
        # state is pulled toward `mu`, giving NUTS real signal on `mu` while
        # the discrete `X` is never differentiated.
        @addlogprob! -0.5 * (sum(X) - mu)^2
        y ~ Normal(mu, 1.0)
    end

    # Kernel: resample every cell as Bernoulli(logistic(mu)). Returns a Matrix{Int}.
    struct GridKernel <: AbstractLatentKernel
        nrows::Int
        ncols::Int
    end
    function PracticalBayes.latent_step(rng, k::GridKernel, block_names, c::ModelConditional)
        block_names == (:X,) || error("only :X")
        p = 1 / (1 + exp(-c.values.mu))
        return (; X=Int.(rand(rng, k.nrows, k.ncols) .< p))
    end

    nrows, ncols = 4, 5
    m = grid_model(3.0, nrows, ncols)

    # (1)+(2): traces, and X round-trips as a Matrix{Int} in the store.
    layout, θ0, store0 = build_layout(m; values=(:X,))
    @test store0.X isa AbstractMatrix{<:Integer}
    @test size(store0.X) == (nrows, ncols)
    @test length(θ0) == 1  # only `mu` occupies θ; `X` does not

    # (3): shape survives a full Gibbs sweep loop; NUTS block keeps stepping.
    spl = Gibbs(:mu => NUTS(0.8), :X => GridKernel(nrows, ncols))
    rng = StableRNG(1)
    transition, state = AbstractMCMC.step(rng, m, spl)
    @test transition.X isa AbstractMatrix{<:Integer}
    @test size(transition.X) == (nrows, ncols)
    for i in 1:50
        transition, state = AbstractMCMC.step(rng, m, spl, state; n_adapts=50)
        @test size(transition.X) == (nrows, ncols)
    end
    @test transition.mu isa Float64
end

# ===========================================================================
# `@addlogprob! ... depends=(...)` — skipping terms a Gibbs block cannot move.
#
# Without `depends`, every HMC block evaluates and differentiates EVERY
# `@addlogprob!` term, including ones that are constant w.r.t. that block's
# parameters. `depends` lets the evaluator skip those.
#
# The skip is EXACT, and these tests pin down exactly why: for a block that
# skips a term, the term is an additive CONSTANT, so
#   (1) the gradient is bit-identical,
#   (2) log-density DIFFERENCES (all Metropolis ever uses) are identical,
#   (3) only the ABSOLUTE log density shifts, by exactly the term's value.
#
# NOTE these do NOT assert that two sampled CHAINS match draw-for-draw. They
# don't, and that is correct: AdvancedHMC's initial step-size search reads the
# absolute density, so the constant offset makes it pick a slightly different
# step size, after which the chains diverge like any two runs with different
# step sizes — while targeting the same posterior.
@testset "@addlogprob! depends: skipped terms are exact constants" begin
    n = 200
    rng = StableRNG(20260726)
    y = randn(rng, n) .+ 2.0
    z = randn(rng, n) .- 1.0

    term_a(a, y) = -0.5 * sum(abs2, y .- a)
    term_b(b, z) = -0.5 * sum(abs2, z .- b)

    @model function m_plain(y, z)
        a ~ Normal(0, 10)
        b ~ Normal(0, 10)
        @addlogprob! term_a(a, y)
        @addlogprob! term_b(b, z)
    end
    @model function m_dep(y, z)
        a ~ Normal(0, 10)
        b ~ Normal(0, 10)
        @addlogprob! term_a(a, y) depends=(:a,)
        @addlogprob! term_b(b, z) depends=(:b,)
    end

    # Build a block-`b` layout (b flat, a in the constant store) for each model.
    function ldf_b(model, aval)
        layout, θ0, _ = PracticalBayes.build_layout(model; flat=(:b,), values=(:a,),
                                                    init=(; a=aval, b=0.0))
        PracticalBayes.LogDensityFunction(model, layout, (; a=aval),
                                          ADTypes.AutoForwardDiff(); θ0=θ0)
    end

    aval = 2.0
    f_p, f_d = ldf_b(m_plain(y, z), aval), ldf_b(m_dep(y, z), aval)
    bs = -2.0:0.5:2.0

    # (1) gradients identical — the skipped term has zero derivative here.
    for b in bs
        _, g_p = LogDensityProblems.logdensity_and_gradient(f_p, [b])
        _, g_d = LogDensityProblems.logdensity_and_gradient(f_d, [b])
        @test g_p ≈ g_d atol=0 rtol=0
    end

    # (3) the offset is constant in b and equals the skipped term's value.
    offs = [LogDensityProblems.logdensity(f_p, [b]) -
            LogDensityProblems.logdensity(f_d, [b]) for b in bs]
    @test maximum(offs) - minimum(offs) < 1e-9
    @test offs[1] ≈ term_a(aval, y) rtol=1e-10

    # (2) Metropolis-relevant differences are unchanged.
    ref = -1.0
    d_p = [LogDensityProblems.logdensity(f_p, [b]) -
           LogDensityProblems.logdensity(f_p, [ref]) for b in bs]
    d_d = [LogDensityProblems.logdensity(f_d, [b]) -
           LogDensityProblems.logdensity(f_d, [ref]) for b in bs]
    @test d_p ≈ d_d rtol=1e-9

    # A block owning BOTH names skips nothing: identical absolute density too.
    function ldf_all(model)
        layout, θ0, _ = PracticalBayes.build_layout(model; flat=(:a, :b), values=(),
                                                    init=(; a=0.0, b=0.0))
        PracticalBayes.LogDensityFunction(model, layout, NamedTuple(),
                                          ADTypes.AutoForwardDiff(); θ0=θ0)
    end
    g_p, g_d = ldf_all(m_plain(y, z)), ldf_all(m_dep(y, z))
    for θ in ([0.0, 0.0], [1.5, -0.5], [2.2, -1.3])
        @test LogDensityProblems.logdensity(g_p, θ) ==
              LogDensityProblems.logdensity(g_d, θ)
    end

    # Sampling with depends recovers the right posterior (both blocks).
    spl = Gibbs(:a => AdaptiveHMC(0.8; n_leapfrog=8),
                :b => AdaptiveHMC(0.8; n_leapfrog=8))
    chn = AbstractMCMC.sample(StableRNG(3), m_dep(y, z), spl, 400;
                              init=(; a=0.0, b=0.0), n_adapts=150, progress=false)
    @test mean(vec(chn[:a])) ≈ 2.0 atol=0.2
    @test mean(vec(chn[:b])) ≈ -1.0 atol=0.2
end

# `depends` must reject anything it cannot resolve at macro-expansion time —
# the whole point is a compile-time constant, and a runtime value would
# silently degrade into a dynamic branch that defeats the optimization.
@testset "@addlogprob! depends: rejects non-literal name lists" begin
    @test_throws LoadError @eval @model function _bad_dep1(y)
        a ~ Normal()
        @addlogprob! 0.0 depends=(a,)          # not a QuoteNode symbol
    end
    @test_throws LoadError @eval @model function _bad_dep2(y)
        a ~ Normal()
        @addlogprob! 0.0 depends=somevar       # not a literal tuple
    end
    @test_throws LoadError @eval @model function _bad_dep3(y)
        a ~ Normal()
        @addlogprob! 0.0 wrongkw=(:a,)         # wrong keyword name
    end
end

@testset "gibbs.jl: multi-block Gibbs with a VECTOR parameter (regression)" begin
    # Any multi-block Gibbs with a vector-valued parameter used to die on its
    # SECOND sweep with `DifferentiationInterface.PreparationMismatchError`.
    #
    # The mechanism sat between two deliberate behaviours. `invlink` builds its
    # values through `_linked_view`, which must return a `Base.SubArray` (see
    # its docstring — PolyesterForwardDiff), so a vector site came back as a
    # VIEW. That NamedTuple becomes the other block's `store`, which is handed
    # to `DI.prepare_gradient` as a `DI.Constant` — and that prep is cached and
    # deliberately never rebuilt. So sweep 1 prepared against `Vector{Float64}`
    # initial values, sweep 2 supplied a `SubArray`, and DI's strict type check
    # threw.
    #
    # Neither behaviour was wrong, which is why the fix is at the boundary:
    # `invlink` materialises. This test is the cheapest possible statement of
    # the bug — two blocks, one scalar, one `filldist` vector, five sweeps.
    @model function _vec_block_model(y)
        mu ~ Normal(0, 1)
        z ~ filldist(Normal(0, 1), 3)
        y ~ Normal(mu + sum(z), 1.0)
    end

    m = _vec_block_model(1.5)
    spl = Gibbs(:mu => NUTS(0.8), :z => NUTS(0.8))
    chn = AbstractMCMC.sample(StableRNG(7), m, spl, 5; progress=false)
    @test size(chn, 1) == 5

    # And the reported draws must be plain arrays, not views — that IS the fix.
    layout, θ0, _ = build_layout(m)
    nt = invlink(layout, θ0)
    @test nt.z isa Vector{Float64}
    @test !(nt.z isa SubArray)
end
