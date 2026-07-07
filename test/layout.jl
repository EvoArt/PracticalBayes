# Tests that `build_layout` produces the right slot structure, and that
# `EvalMode`'s logdensity computation matches hand-derived values — this is
# the actual correctness check for the bijector/Jacobian math in tilde.jl,
# not just "it runs".

using Distributions: Normal, Exponential, Beta, Bernoulli, logpdf
using Bijectors: Bijectors, with_logabsdet_jacobian
using StableRNGs: StableRNG

@testset "layout.jl: FlatSlot for scalar params, correct log-density" begin
    @model function two_params(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end

    m = two_params(2.0)
    layout, θ0, store0 = build_layout(m; rng=StableRNG(1))
    @test layout.dim == 2
    @test layout.slots.mu isa PracticalBayes.FlatSlot
    @test layout.slots.sigma isa PracticalBayes.FlatSlot
    @test store0 == NamedTuple()

    # Hand-derive the expected log-density at an arbitrary θ, replicating
    # exactly what `_assume`/`tilde` should compute: unconstrained `mu` needs
    # no transform (Normal has full-real support, bijector is identity), but
    # `sigma`'s bijector is exp (Exponential support is (0, Inf)).
    θ = [0.3, -0.7]
    mu_val = θ[layout.slots.mu.range][1]  # identity-transformed
    sigma_dist = Exponential(1)
    sigma_val, sigma_logjac = with_logabsdet_jacobian(
        Bijectors.VectorBijectors.from_linked_vec(sigma_dist), view(θ, layout.slots.sigma.range)
    )
    expected =
        logpdf(Normal(0, 1), mu_val) +
        (logpdf(sigma_dist, sigma_val) + sigma_logjac) +
        logpdf(Normal(mu_val, sigma_val), 2.0)

    mode = EvalMode(layout, θ, store0, m.conditioned)
    _, acc = evaluate(m, mode, Accum(0.0))
    @test logjoint(acc) ≈ expected
end

@testset "layout.jl: indexed family -> FlatArraySlot, uniform linked length enforced" begin
    @model function hmm_like(n)
        z = Vector{paramtype(__mode__)}(undef, n)
        for i in 1:n
            z[i] ~ Normal(0, 1)
        end
        return z
    end

    m = hmm_like(4)
    layout, θ0, _ = build_layout(m)
    @test layout.slots.z isa PracticalBayes.FlatArraySlot
    @test layout.dim == 4
    @test length(θ0) == 4
end

@testset "layout.jl: link/invlink round-trip" begin
    @model function two_params(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    m = two_params(2.0)
    layout, θ0, _ = build_layout(m)
    nt = invlink(layout, θ0)
    @test haskey(nt, :mu)
    @test haskey(nt, :sigma)
    @test nt.sigma > 0  # constrained space: Exponential support

    θ_roundtrip = link(layout, nt)
    @test θ_roundtrip ≈ θ0 atol = 1e-8
end

@testset "layout.jl: ValueSlot for latents assigned to `values`" begin
    # `Bernoulli` has Discrete value support, so tilde.jl's `_is_discrete`
    # classifies `z`'s role as :latent during tracing — this is the case a
    # ValueSlot (not a FlatSlot) exists for.
    @model function with_latent(y)
        p ~ Beta(1, 1)
        z ~ Bernoulli(p)
        y ~ Normal(z, 1)
    end
    m = with_latent(1.0)
    layout, θ0, store0 = build_layout(m; values=(:z,), init=(; z=0.0))
    @test layout.slots.z isa PracticalBayes.ValueSlot
    @test store0.z == 0.0
    @test layout.dim == 1  # only `p` is flat; `z` is in the store
end

@testset "layout.jl: unassigned discrete site errors clearly" begin
    @model function with_latent_unassigned(y)
        p ~ Beta(1, 1)
        z ~ Bernoulli(p)
        y ~ Normal(z, 1)
    end
    m = with_latent_unassigned(1.0)
    @test_throws ArgumentError build_layout(m)
end

@testset "layout.jl: untracked flat params still sampled, hidden from invlink" begin
    @model function many_params(y)
        mu ~ Normal(0, 1)
        nuisance ~ Normal(0, 1)
        y ~ Normal(mu + nuisance, 1)
    end
    m = many_params(2.0)
    layout, θ0, store0 = build_layout(m; untracked=(:nuisance,))
    @test layout.dim == 2  # `nuisance` still occupies flat space...
    @test layout.slots.nuisance isa PracticalBayes.FlatSlot
    @test :nuisance in layout.untracked

    nt = invlink(layout, θ0)
    @test haskey(nt, :mu)
    @test !haskey(nt, :nuisance)  # ...but is hidden from invlink by default

    nt_full = invlink(layout, θ0; include_untracked=true)
    @test haskey(nt_full, :nuisance)

    # the log-density itself is completely unaffected by the flag — this is
    # purely a reporting-level marker, not a change to `EvalMode`/tilde.jl.
    mode = EvalMode(layout, θ0, store0, m.conditioned)
    _, acc = evaluate(m, mode, Accum(0.0))
    @test isfinite(logjoint(acc))
end

@testset "layout.jl: untracked name that isn't a flat site errors clearly" begin
    @model function two_params(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    m = two_params(2.0)
    @test_throws ArgumentError build_layout(m; untracked=(:doesnotexist,))
end

@testset "layout.jl: build_layout respects numeric type T" begin
    @model function two_params(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    m = two_params(2.0)
    layout, θ0_32, _ = build_layout(m; T=Float32)
    @test eltype(θ0_32) === Float32
end
