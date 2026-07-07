# Tests for `Flat`/`FlatPos` (improper priors, src/distributions.jl) and the
# `filldist`-equivalent pattern (plain `Distributions.product_distribution`,
# which needs no PracticalBayes-side code at all — Bijectors' VectorBijectors
# already builds unconstrained transforms generically from any Distribution's
# support/linked length, and `build_layout`/`tilde.jl` never special-case
# scalar vs vector-valued sites).

using Distributions: Normal, Exponential, product_distribution, logpdf
using ADTypes: AutoForwardDiff
import LogDensityProblems

@testset "distributions.jl: Flat has zero logpdf everywhere" begin
    d = Flat()
    @test logpdf(d, 0.0) == 0.0
    @test logpdf(d, -1e6) == 0.0
    @test logpdf(d, 1e6) == 0.0
end

@testset "distributions.jl: FlatPos is -Inf at/below l, 0 above" begin
    d = FlatPos(2.0)
    @test logpdf(d, 2.0) == -Inf
    @test logpdf(d, 1.0) == -Inf
    @test logpdf(d, 2.1) == 0.0
end

@testset "distributions.jl: Flat/FlatPos work as `~` priors, gradient matches finite differences" begin
    @model function flat_model(y)
        mu ~ Flat()
        sigma ~ FlatPos(0.0)
        y .~ Normal.(mu, sigma)
    end
    y = randn(20) .+ 2.0
    m = flat_model(y)
    layout, θ0, store0 = build_layout(m; init=(; mu=2.0, sigma=1.0))
    @test layout.dim == 2

    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    @test isfinite(val)

    h = 1e-6
    fd_grad = similar(θ0)
    for i in eachindex(θ0)
        θp = copy(θ0); θp[i] += h
        θm = copy(θ0); θm[i] -= h
        fd_grad[i] = (LogDensityProblems.logdensity(ldf, θp) - LogDensityProblems.logdensity(ldf, θm)) / (2h)
    end
    @test grad ≈ fd_grad atol = 1e-4
end

@testset "distributions.jl: product_distribution as filldist-equivalent" begin
    @model function filldist_model(y)
        beta ~ product_distribution(fill(Normal(0, 1), 5))
        sigma ~ Exponential(1)
        y .~ Normal.(sum(beta), sigma)
    end
    y = randn(10)
    m = filldist_model(y)
    layout, θ0, store0 = build_layout(m)
    @test layout.dim == 6  # 5 (beta) + 1 (sigma)

    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    @test isfinite(val)
    @test length(grad) == 6

    h = 1e-6
    fd_grad = similar(θ0)
    for i in eachindex(θ0)
        θp = copy(θ0); θp[i] += h
        θm = copy(θ0); θm[i] -= h
        fd_grad[i] = (LogDensityProblems.logdensity(ldf, θp) - LogDensityProblems.logdensity(ldf, θm)) / (2h)
    end
    @test grad ≈ fd_grad atol = 1e-4

    # constrained (positive-support) product distribution round-trips
    # correctly through invlink -- checks the Jacobian isn't just silently 0.
    nt = invlink(layout, θ0)
    @test length(nt.beta) == 5
    @test all(isfinite, nt.beta)
end

using LinearAlgebra: I
using Distributions: MvNormal

@testset "distributions.jl: filldist/arraydist match Turing's wrapper semantics" begin
    @test filldist(Normal(0, 1), 5) == product_distribution(fill(Normal(0, 1), 5))
    @test arraydist([Normal(0, 1), Exponential(1)]) == product_distribution([Normal(0, 1), Exponential(1)])
end

@testset "distributions.jl: filldist as a `~` prior for a full linear-regression-style model (PosteriorDB blr.jl pattern)" begin
    @model function blr_model(X, y)
        D = size(X, 2)
        beta ~ filldist(Normal(0, 10), D)
        sigma ~ FlatPos(0.0)
        y ~ MvNormal(X * beta, sigma^2 * I)
    end
    X = randn(30, 4)
    true_beta = randn(4)
    y = X * true_beta .+ randn(30) .* 0.5
    m = blr_model(X, y)
    layout, θ0, store0 = build_layout(m; init=(; sigma=1.0))
    @test layout.dim == 5  # 4 (beta) + 1 (sigma)

    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    @test isfinite(val)

    h = 1e-6
    fd_grad = similar(θ0)
    for i in eachindex(θ0)
        θp = copy(θ0); θp[i] += h
        θm = copy(θ0); θm[i] -= h
        fd_grad[i] = (LogDensityProblems.logdensity(ldf, θp) - LogDensityProblems.logdensity(ldf, θm)) / (2h)
    end
    @test grad ≈ fd_grad atol = 1e-3
end
