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

@testset "logdensity.jl: reject_errors turns an exception into -Inf, not a crash" begin
    # `logpdf(Exponential(1), x)` for `x < 0` doesn't throw (it's just -Inf),
    # so to actually exercise the catch path we need a model that genuinely
    # throws for some values of θ — e.g. `log` of a value that can go
    # negative through an identity-linked (unconstrained) parameter.
    @model function maybe_throws(y)
        mu ~ Normal(0, 1)
        w = log(mu)  # throws DomainError for mu <= 0, since `mu` is real-supported/unconstrained
        y ~ Normal(w, 1)
    end
    m = maybe_throws(2.0)
    # `init` forces trace-time `mu=1.0` (a random draw could land <= 0 and
    # throw during `build_layout` itself, before we even get to the actual
    # test of the reject-errors path at a deliberately bad θ).
    layout, θ0, store0 = build_layout(m; init=(; mu=1.0))

    ldf_strict = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    ldf_lenient = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0, reject_errors=true)

    θ_bad = [-1.0]  # mu = -1 <= 0 -> log(mu) throws
    @test_throws DomainError LogDensityProblems.logdensity(ldf_strict, θ_bad)
    @test LogDensityProblems.logdensity(ldf_lenient, θ_bad) == -Inf

    @test_throws DomainError LogDensityProblems.logdensity_and_gradient(ldf_strict, θ_bad)
    val, grad = LogDensityProblems.logdensity_and_gradient(ldf_lenient, θ_bad)
    @test val == -Inf
    @test grad == zeros(1)

    # a good point still works normally under the lenient wrapper
    θ_good = [1.0]
    @test LogDensityProblems.logdensity(ldf_lenient, θ_good) ≈ LogDensityProblems.logdensity(ldf_strict, θ_good)
end
