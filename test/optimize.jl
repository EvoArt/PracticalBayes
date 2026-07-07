# Tests for maximum_a_posteriori/maximum_likelihood/laplace_approximation
# (optimize.jl) against closed-form conjugate-model answers. Point estimation
# is a secondary feature (full Bayesian NUTS sampling is the main target of
# this package), so there is no built-in optimizer — everything here requires
# `using Optimization` + a solver package, and the whole file is skipped (not
# failed) if those aren't installed, matching the pattern used for
# Turing/Mooncake/Enzyme elsewhere in this test suite.

using Distributions: Normal, Exponential
using LinearAlgebra: diag
using StableRNGs: StableRNG

const _HAVE_OPTIMIZATION = !isnothing(Base.find_package("Optimization")) && !isnothing(Base.find_package("OptimizationOptimJL"))

if _HAVE_OPTIMIZATION
    import Optimization
    import OptimizationOptimJL

    @testset "optimize.jl: maximum_a_posteriori matches conjugate Normal-Normal posterior mode" begin
        # y_i ~ Normal(mu, 1), mu ~ Normal(0, 1): posterior is Normal(post_mean,
        # post_var), so the MAP (= posterior mode = posterior mean for a
        # Gaussian) has a closed form to check against.
        rng = StableRNG(1)
        y = randn(rng, 20) .+ 3.0

        @model function conjugate(y)
            mu ~ Normal(0, 1)
            y .~ Normal.(mu, 1.0)
        end

        n = length(y)
        post_var = 1 / (1 + n)
        post_mean = post_var * sum(y)

        m = conjugate(y)
        est = maximum_a_posteriori(m, OptimizationOptimJL.BFGS(); rng=StableRNG(2))
        @test est.constrained.mu ≈ post_mean atol = 1e-4
    end

    @testset "optimize.jl: maximum_likelihood ignores the prior" begin
        # Same model as above, but MLE should recover the SAMPLE MEAN (no prior
        # pull toward 0), which for n=20 and a Normal(0,1) prior is visibly
        # different from the MAP computed in the previous testset.
        rng = StableRNG(1)
        y = randn(rng, 20) .+ 3.0

        @model function conjugate(y)
            mu ~ Normal(0, 1)
            y .~ Normal.(mu, 1.0)
        end

        m = conjugate(y)
        est = maximum_likelihood(m, OptimizationOptimJL.BFGS(); rng=StableRNG(2))
        @test est.constrained.mu ≈ sum(y) / length(y) atol = 1e-4
    end

    @testset "optimize.jl: laplace_approximation matches analytic posterior variance" begin
        rng = StableRNG(1)
        y = randn(rng, 20) .+ 3.0

        @model function conjugate(y)
            mu ~ Normal(0, 1)
            y .~ Normal.(mu, 1.0)
        end

        n = length(y)
        post_var = 1 / (1 + n)
        post_mean = post_var * sum(y)

        m = conjugate(y)
        la = laplace_approximation(m, OptimizationOptimJL.BFGS(); rng=StableRNG(2))
        @test la.map.constrained.mu ≈ post_mean atol = 1e-4
        # `mu` is identity-linked (Normal has full-real support), so unconstrained
        # and constrained space coincide here — the Laplace covariance should
        # match the analytic posterior variance essentially exactly for an
        # actually-Gaussian posterior.
        @test only(la.covariance) ≈ post_var atol = 1e-4
        @test only(diag(la.covariance)) > 0

        d = laplace_mvnormal(la)
        @test d.μ[1] ≈ post_mean atol = 1e-4
    end

    @testset "optimize.jl: untracked params still optimized, included in the point estimate" begin
        # Unlike a full chain (where `invlink` hides untracked/nuisance sites by
        # default to save memory across thousands of draws), a single
        # PointEstimate always includes them — see `PointEstimate`'s docstring:
        # there's no memory-saving reason to hide one point. `untracked` still
        # fully participates in the optimization (it occupies real space in
        # `theta` and the objective/gradient), it's just about what gets
        # materialized for reporting.
        @model function many_params(y)
            mu ~ Normal(0, 1)
            nuisance ~ Normal(0, 1)
            y ~ Normal(mu, 1)
        end
        m = many_params(3.0)
        est = maximum_a_posteriori(m, OptimizationOptimJL.BFGS(); untracked=(:nuisance,), rng=StableRNG(3))
        @test haskey(est.constrained, :nuisance)
        @test length(est.theta) == 2  # nuisance still occupies flat space
    end

    @testset "optimize.jl: store keeps a name fixed during optimization" begin
        @model function with_fixed(y)
            mu ~ Normal(0, 1)
            z ~ Normal(0, 1)
            y ~ Normal(mu + z, 1)
        end
        m = with_fixed(5.0)
        est = maximum_a_posteriori(m, OptimizationOptimJL.BFGS(); store=(; z=10.0), rng=StableRNG(4))
        @test est.store.z == 10.0
        @test length(est.theta) == 1  # only `mu` is optimized over
    end
else
    @testset "optimize.jl" begin
        @test_skip "Optimization.jl/OptimizationOptimJL not available in this environment"
    end
end
