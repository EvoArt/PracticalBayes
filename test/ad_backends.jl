# Cross-backend AD agreement, per user request. All backends must agree with
# ForwardDiff (the default/reference backend, safe by construction since it
# never needs special handling of `DI.Constant`) to a tight tolerance on the
# same model + point. This is the check that our `DI.Constant`-based latent
# invisibility (logdensity.jl) actually works identically across backends —
# not just for ForwardDiff, where it's easy to get right by accident.
#
# NOTE: these tests are best-effort — Mooncake/Enzyme/PolyesterForwardDiff
# are `[extras]`-only deps (see Project.toml) and may not be installed/loaded
# in every environment this test suite runs in. Each backend's testset is
# skipped (not failed) if the package isn't available, so `Pkg.test()` stays
# green in minimal environments while still running the full matrix in CI /
# a fully-instantiated dev environment.

using Distributions: Normal, Exponential
using ADTypes: AutoForwardDiff, AutoMooncake, AutoEnzyme, AutoPolyesterForwardDiff
import LogDensityProblems

@model function ad_test_model(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

function _reference_grad()
    m = ad_test_model(2.0)
    layout, θ0, store0 = build_layout(m)
    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    _, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    return layout, θ0, store0, grad
end

@testset "ad_backends.jl: ForwardDiff (reference)" begin
    layout, θ0, store0, grad = _reference_grad()
    @test grad isa Vector{Float64}
    @test length(grad) == 2
end

@testset "ad_backends.jl: Mooncake agrees with ForwardDiff" begin
    # DifferentiationInterface's per-backend support is a package EXTENSION
    # that only activates once the backend package is actually loaded in
    # this session — `Base.find_package` only confirms it's installed, not
    # loaded, so we `import` it here rather than relying on installation
    # alone (see the ForwardDiff `using` in PracticalBayes.jl for the same
    # gotcha on the default backend).
    if !isnothing(Base.find_package("Mooncake"))
        @eval import Mooncake
        layout, θ0, store0, ref_grad = _reference_grad()
        m = ad_test_model(2.0)
        ldf = LogDensityFunction(m, layout, store0, AutoMooncake(; config=nothing); θ0=θ0)
        _, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
        @test grad ≈ ref_grad atol = 1e-8
    else
        @test_skip "Mooncake not available in this environment"
    end
end

@testset "ad_backends.jl: Enzyme agrees with ForwardDiff" begin
    if !isnothing(Base.find_package("Enzyme"))
        @eval import Enzyme
        layout, θ0, store0, ref_grad = _reference_grad()
        m = ad_test_model(2.0)
        ldf = LogDensityFunction(m, layout, store0, AutoEnzyme(); θ0=θ0)
        _, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
        @test grad ≈ ref_grad atol = 1e-8
    else
        @test_skip "Enzyme not available in this environment"
    end
end

@testset "ad_backends.jl: PolyesterForwardDiff agrees with ForwardDiff" begin
    if !isnothing(Base.find_package("PolyesterForwardDiff"))
        @eval import PolyesterForwardDiff
        layout, θ0, store0, ref_grad = _reference_grad()
        m = ad_test_model(2.0)
        ldf = LogDensityFunction(m, layout, store0, AutoPolyesterForwardDiff(; chunksize=2); θ0=θ0)
        _, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
        @test grad ≈ ref_grad atol = 1e-8
    else
        @test_skip "PolyesterForwardDiff not available in this environment"
    end
end

@testset "ad_backends.jl: latents (DI.Constant) invisible under every available backend" begin
    # Same check as logdensity.jl's ForwardDiff-only version, but sweep every
    # backend that's actually available: gradient dimension/eltype must never
    # reflect the latent, regardless of which AD package computes it.
    @model function with_latent(y)
        mu ~ Normal(0, 1)
        z ~ Normal(0, 1)
        y ~ Normal(mu + z, 1)
    end
    m = with_latent(2.0)
    layout, θ0, store0 = build_layout(m; values=(:z,), init=(; z=0.5))

    backends = Pair{String,Any}["ForwardDiff" => AutoForwardDiff()]
    if !isnothing(Base.find_package("Mooncake"))
        @eval import Mooncake
        push!(backends, "Mooncake" => AutoMooncake(; config=nothing))
    end
    if !isnothing(Base.find_package("Enzyme"))
        @eval import Enzyme
        push!(backends, "Enzyme" => AutoEnzyme())
    end

    for (backend_name, adtype) in backends
        @testset "$backend_name" begin
            ldf = LogDensityFunction(m, layout, store0, adtype; θ0=θ0)
            _, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
            @test length(grad) == 1
            @test eltype(grad) == Float64
        end
    end
end
