# Tests for the LogDensityProblems interface + gradient correctness with the
# default AD backend (ForwardDiff). Cross-backend agreement is checked
# separately in ad_backends.jl; this file focuses on: interface compliance,
# gradient correctness vs finite differences, and that latents in `store`
# are truly invisible to the gradient (requirement 2's core guarantee).

using Distributions: Normal, Exponential, logpdf
using ADTypes: AutoForwardDiff
import LogDensityProblems

@testset "logdensity.jl: LogDensityProblems interface, order 0" begin
    @model function two_params(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    m = two_params(2.0)
    layout, θ0, store0 = build_layout(m)
    ldf = LogDensityFunction(m, layout, store0)  # no adtype -> order 0
    @test LogDensityProblems.dimension(ldf) == 2
    @test LogDensityProblems.capabilities(typeof(ldf)) == LogDensityProblems.LogDensityOrder{0}()
    @test LogDensityProblems.logdensity(ldf, θ0) isa Real
end

@testset "logdensity.jl: order 1 (gradient) matches finite differences" begin
    @model function two_params(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    m = two_params(2.0)
    layout, θ0, store0 = build_layout(m)
    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    @test LogDensityProblems.capabilities(typeof(ldf)) == LogDensityProblems.LogDensityOrder{1}()

    val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    @test val ≈ LogDensityProblems.logdensity(ldf, θ0)

    # central finite-difference check
    h = 1e-6
    fd_grad = similar(θ0)
    for i in eachindex(θ0)
        θp = copy(θ0); θp[i] += h
        θm = copy(θ0); θm[i] -= h
        fd_grad[i] = (LogDensityProblems.logdensity(ldf, θp) - LogDensityProblems.logdensity(ldf, θm)) / (2h)
    end
    @test grad ≈ fd_grad atol = 1e-4
end

@testset "logdensity.jl: latents in `store` are invisible to the gradient" begin
    # `z` sits behind a ValueSlot (constant, in `store`), not in θ at all —
    # so the gradient dimension must equal only the FlatSlot count, and
    # perturbing θ must never touch z's contribution beyond what's already
    # baked into the (constant) logdensity offset.
    @model function with_latent(y)
        mu ~ Normal(0, 1)
        z ~ Normal(0, 1)  # pretend latent (continuous here only to keep the test simple)
        y ~ Normal(mu + z, 1)
    end
    m = with_latent(2.0)
    layout, θ0, store0 = build_layout(m; values=(:z,), init=(; z=0.5))
    @test layout.dim == 1  # only mu
    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    val1, grad1 = LogDensityProblems.logdensity_and_gradient(ldf, θ0)

    # Changing z in the store changes the VALUE (z enters the likelihood
    # mean) but must not change the gradient's SHAPE/eltype, and the gradient
    # w.r.t. mu should be unaffected in structure (still a length-1 Float64
    # vector, no promotion to Dual leaking through).
    ldf2 = LogDensityFunction(m, layout, (; z=5.0), AutoForwardDiff(); θ0=θ0)
    val2, grad2 = LogDensityProblems.logdensity_and_gradient(ldf2, θ0)
    @test val1 != val2  # z does affect the density value
    @test eltype(grad1) == eltype(grad2) == Float64  # but never promotes/leaks a Dual
    @test length(grad1) == length(grad2) == 1
end
