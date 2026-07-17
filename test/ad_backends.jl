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

using Distributions: Normal, Exponential, MvNormal
using LinearAlgebra: I
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
        # NB: chunksize=1 is avoided deliberately — PolyesterForwardDiff's own
        # `batch`/`threaded_gradient!` throws `ArgumentError: tuple must be
        # non-empty` for chunksize=1 whenever `dim >= 2` and more than one
        # thread is available (a bug inside PolyesterForwardDiff, unrelated to
        # this package and independent of element type — chunksize 1 is a
        # pathological setting for chunked forward-mode anyway).
        ldf = LogDensityFunction(m, layout, store0, AutoPolyesterForwardDiff(; chunksize=2); θ0=θ0)
        _, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
        @test grad ≈ ref_grad atol = 1e-8
    else
        @test_skip "PolyesterForwardDiff not available in this environment"
    end
end

@testset "ad_backends.jl: PolyesterForwardDiff works with Float32 θ" begin
    # Regression guard: PolyesterForwardDiff's `threaded_gradient!` stores the
    # model's primal return value into a `Ref{eltype(θ)}` via
    # `store_val!(::Ref{T}, ::T)`, so the logjoint's type MUST equal `eltype(θ)`
    # exactly or it `MethodError`s. With a Float32 `θ` this fails the moment the
    # logjoint promotes back to Float64 — which happens for Float64 distribution
    # literals in the model body AND, more subtly, merely for Float64 data
    # (`y`), which we explicitly document as allowed alongside Float32 params.
    # `_logdensity_call` now coerces the result to `eltype(θ)` (logdensity.jl),
    # so both cases must agree with the Float32 ForwardDiff reference here.
    # (Plain ForwardDiff never tripped this, so the Float64 tests above can't
    # catch it — this needs the Float32 + PolyesterForwardDiff combination.)
    if !isnothing(Base.find_package("PolyesterForwardDiff"))
        @eval import PolyesterForwardDiff
        # Float64 data (the documented "data can stay Float64" path) with a
        # Float32 parameter vector — the exact combination that used to throw
        # `store_val!(::Ref{Float32}, ::Float64)`.
        m = ad_test_model(2.0)
        layout, _, store0 = build_layout(m; T=Float32)
        θ0 = zeros(Float32, layout.dim)

        ldf_ref = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
        vref, ref_grad = LogDensityProblems.logdensity_and_gradient(ldf_ref, θ0)
        @test vref isa Float32
        @test ref_grad isa Vector{Float32}

        ldf = LogDensityFunction(m, layout, store0, AutoPolyesterForwardDiff(; chunksize=2); θ0=θ0)
        v, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
        @test v isa Float32
        @test grad isa Vector{Float32}
        @test grad ≈ ref_grad atol = 1.0f-4
    else
        @test_skip "PolyesterForwardDiff not available in this environment"
    end
end

@testset "ad_backends.jl: PolyesterForwardDiff with multi-chunk θ + scalar params" begin
    # Regression guard for the bug that broke every real (i.e. more than a
    # handful of parameters) model under PolyesterForwardDiff.
    #
    # When θ spans MORE THAN ONE chunk, PolyesterForwardDiff hands the model a
    # `StrideArraysCore.StrideArray` rather than a plain `Vector`. `StrideArray`
    # overloads `view`, so `view(θ, range)` returned a `StrideArray` too — and
    # `Bijectors`' `OnlyWrap` (the wrapper for any SCALAR constrained site, here
    # `sigma ~ Exponential(1)`) extracts its element with a zero-argument
    # `getindex`, `x[]`, which `StrideArray` does not implement. The result was
    # `ArgumentError: tuple must be non-empty` from inside StrideArraysCore.
    # `_linked_view` (tilde.jl) now always builds a `Base.SubArray`, so the
    # bijector gets Base indexing semantics whatever `θ`'s type is.
    #
    # The model below is deliberately shaped to trigger it: `dim = 21` with a
    # small chunk size means several chunks, AND it mixes a vector parameter
    # with a constrained scalar. The single-chunk models above cannot catch
    # this — they never see a StrideArray at all.
    if !isnothing(Base.find_package("PolyesterForwardDiff"))
        @eval import PolyesterForwardDiff
        @model function multichunk(y)
            b ~ MvNormal(zeros(20), I)
            sigma ~ Exponential(1)   # scalar + constrained -> OnlyWrap -> `x[]`
            y ~ Normal(sum(b), sigma)
        end
        m = multichunk(2.0)
        layout, θ0, store0 = build_layout(m)
        @test layout.dim == 21

        ldf_ref = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
        _, ref_grad = LogDensityProblems.logdensity_and_gradient(ldf_ref, θ0)

        # chunksize=nothing is what a caller gets by default (and what the
        # EpidemicTrajectories badger benchmark uses); 4 forces several chunks
        # explicitly. Both used to throw.
        for chunksize in (nothing, 4)
            ldf = LogDensityFunction(m, layout, store0, AutoPolyesterForwardDiff(; chunksize); θ0=θ0)
            _, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
            @test grad ≈ ref_grad atol = 1e-8
        end
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
