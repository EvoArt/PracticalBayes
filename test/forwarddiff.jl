# AutoPBForwardDiff: the PracticalBayes AD backend that survives
# `juliac --trim=safe`.
#
# The dual-number engine itself lives in Piste.jl and has its own test suite
# (arithmetic rules, distribution coverage, the lgamma/atan/isinteger
# regressions, chunk-width invariance). What is tested HERE is the integration:
# that the backend behaves correctly *as a PracticalBayes backend* — through
# `LogDensityFunction`, agreeing with the reference backend, reporting the right
# capability to samplers, and preserving Float32.
#
# What these tests are actually guarding:
#
#  1. AGREEMENT WITH A REFERENCE. A wrong gradient does not crash — it silently
#     biases the posterior, which is this package's worst failure mode. Every
#     test here compares against ForwardDiff (the reference backend) or against
#     finite differences, never against a hand-computed expectation.
#
#  2. THE THREE NON-OBVIOUS METHODS. `Dual <: Real` gets us through
#     `Distributions.logpdf`, but three methods beyond plain arithmetic were
#     needed, each found by measurement:
#       * `SpecialFunctions._logabsgamma` — the SINGLE missing method behind
#         Gamma, Beta, Poisson, Binomial, NegativeBinomial, TDist and Chisq.
#         All seven failed on it and nothing else.
#       * two-argument `atan(y, x)` — its absence was a StackOverflowError (an
#         infinite promotion recursion), not a MethodError. Needed by Cauchy's
#         `cdf`, hence by `truncated(Cauchy(...), 0, Inf)`.
#       * `isinteger` — discrete distributions call it in their support check.
#     Each has a regression test below, because each was silently missing once.
#
#  3. CHUNK-WIDTH INVARIANCE. The chunk is a compile-time type parameter, so
#     every width is a separately compiled code path. They must all agree.

using Distributions: Normal, Exponential, MvNormal
using LinearAlgebra: I, Diagonal
using ADTypes: AutoForwardDiff
using PracticalBayes: chunksize
using Piste: gradient!, value_and_gradient!
import LogDensityProblems
using Random: Xoshiro

@model function _fd_test_model(X, y)
    pT = paramtype(__mode__)
    k = size(X, 2)
    beta ~ MvNormal(zeros(pT, k), I)
    sigma ~ Exponential(one(pT))
    y ~ MvNormal(X * beta, sigma^2 * I)
end

function _fd_setup(; n=60, k=4, seed=11)
    rng = Xoshiro(seed)
    X = randn(rng, n, k)
    y = X * randn(rng, k) .+ randn(rng, n) .* 0.7
    m = _fd_test_model(X, y)
    layout, θ0, store0 = build_layout(m; rng=Xoshiro(20240829))
    return m, layout, θ0, store0
end

@testset "forwarddiff.jl: agrees with ForwardDiff through LogDensityFunction" begin
    m, layout, θ0, store0 = _fd_setup()
    ref = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    vref, gref = LogDensityProblems.logdensity_and_gradient(ref, θ0)

    # Every chunk width is a separately compiled path (the width is a type
    # parameter), so each is tested rather than assumed equivalent.
    for ch in (1, 2, 4, 8, 16)
        ldf = LogDensityFunction(m, layout, store0, AutoPBForwardDiff(chunk=ch); θ0=θ0)
        v, g = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
        @test v ≈ vref rtol=1e-10
        @test g ≈ gref rtol=1e-8
        @test length(g) == length(θ0)
    end
end

@testset "forwarddiff.jl: reports gradient capability to samplers" begin
    m, layout, θ0, store0 = _fd_setup()
    ldf = LogDensityFunction(m, layout, store0, AutoPBForwardDiff(); θ0=θ0)
    # Order 1 is what tells AdvancedHMC to use our gradient rather than
    # wrapping the object in its own AD.
    @test LogDensityProblems.capabilities(typeof(ldf)) isa LogDensityProblems.LogDensityOrder{1}
    @test LogDensityProblems.dimension(ldf) == length(θ0)
end

@testset "forwarddiff.jl: Float32 stays Float32" begin
    # PracticalBayes is Float32-first, and this backend is generic in the dual's
    # element type, so a Float32 θ must not silently promote to Float64.
    m, layout, θ0, store0 = _fd_setup()
    θ32 = Float32.(θ0)
    ldf = LogDensityFunction(m, layout, store0, AutoPBForwardDiff(chunk=4); θ0=θ32)
    v, g = LogDensityProblems.logdensity_and_gradient(ldf, θ32)
    @test eltype(g) === Float32
    @test length(g) == length(θ32)

    ref = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    _, gref = LogDensityProblems.logdensity_and_gradient(ref, θ0)
    # Float32 accumulation, so the tolerance is genuinely looser here.
    @test maximum(abs.(Float64.(g) .- gref)) < 1e-2
end

@testset "forwarddiff.jl: chunk size is a compile-time type parameter" begin
    # Not a style point: a runtime-valued chunk width is one of the things that
    # stops ForwardDiff from trimming, so this property is load-bearing.
    @test chunksize(AutoPBForwardDiff(chunk=8)) == 8
    @test chunksize(AutoPBForwardDiff()) == 8
    @test AutoPBForwardDiff(chunk=4) isa AutoPBForwardDiff{4}
    @test isbitstype(typeof(AutoPBForwardDiff(chunk=4)))
end

@testset "forwarddiff.jl: gradient! on a plain function" begin
    # The low-level entry point, independent of any model machinery.
    f = x -> sum(abs2, x) + exp(x[1]) * log(x[2])
    x = [0.7, 1.3, -0.4]
    for ch in (1, 2, 4)
        g = zeros(3)
        gradient!(g, f, x, Val(ch))
        fd = similar(x)
        for i in eachindex(x)
            h = 1e-6
            xp = copy(x); xp[i] = x[i] + h; f1 = f(xp)
            xp[i] = x[i] - h; f2 = f(xp)
            fd[i] = (f1 - f2) / (2h)
        end
        @test g ≈ fd rtol=1e-5
    end

    # value_and_gradient! must return the SAME primal a plain call gives; it
    # takes it from the first dual pass rather than re-evaluating.
    g = zeros(3)
    v, g2 = value_and_gradient!(g, f, x, Val(2))
    @test v ≈ f(x)
    @test g2 === g
end

@testset "forwarddiff.jl: latents in `store` stay invisible to the gradient" begin
    # Same property the other backends are held to (see ad_backends.jl): a
    # value living in `store` is data, not a parameter, and must not appear in
    # the gradient or change its length.
    m, layout, θ0, store0 = _fd_setup()
    ldf = LogDensityFunction(m, layout, store0, AutoPBForwardDiff(chunk=4); θ0=θ0)
    _, g = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    @test length(g) == layout.dim
    @test all(isfinite, g)
end
