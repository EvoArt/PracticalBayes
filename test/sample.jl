# Tests for src/sample.jl — the M2 milestone's "sample overloads" gate
# (design plan: "conjugate Normal-Normal posterior within 3 MC-SE of
# analytic"). Exercises PracticalBayes.sample's actual public entry point
# (AbstractMCMC.sample(rng, model::Model, spl::AbstractHMCSampler, N; ...)),
# not the lower-level manual AbstractMCMC.step loop test/turing_comparison.jl
# already covers — this is specifically about the FlexiChains.SymChain
# bundling path.

using Distributions: Normal, Exponential
using AdvancedHMC: NUTS
using StableRNGs: StableRNG
import AbstractMCMC
using Statistics: mean, std
import FlexiChains

@testset "sample.jl: SymChain accuracy — conjugate Normal-Normal posterior" begin
    # Same data-generation procedure as test/turing_comparison.jl's own
    # conjugate-model gate, so this is directly comparable to that
    # independently-verified analytic reference.
    rng = StableRNG(42)
    y = randn(rng, 30) .+ 2.0

    @model function conjugate(y)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
    end

    n = length(y)
    post_var = 1 / (1 + n)
    post_mean = post_var * sum(y)

    m = conjugate(y)
    chn = AbstractMCMC.sample(
        StableRNG(1), m, NUTS(0.8), 2000;
        n_adapts=1000, discard_initial=1000, progress=false,
    )

    @test chn isa FlexiChains.SymChain
    mu_draws = vec(chn[:mu])
    @test length(mu_draws) == 2000

    mcse = std(mu_draws) / sqrt(length(mu_draws))
    @test abs(mean(mu_draws) - post_mean) < 3 * mcse
end

@testset "sample.jl: stats present as Extra keys" begin
    rng = StableRNG(7)
    y = randn(rng, 20)

    @model function m1(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end

    chn = AbstractMCMC.sample(
        StableRNG(2), m1(y), NUTS(0.8), 50;
        n_adapts=25, discard_initial=25, progress=false,
    )
    # AdvancedHMC's per-transition stat NamedTuple (acceptance_rate,
    # n_steps, log_density, ...) must survive the NamedTuple->SymChain
    # bundling as Extra keys, not silently dropped.
    extra_names = Set(FlexiChains.get_name(k) for k in keys(chn) if k isa FlexiChains.Extra)
    @test :acceptance_rate in extra_names
    @test :n_steps in extra_names
    @test :log_density in extra_names
end

@testset "sample.jl: chain_type=nothing returns raw AdvancedHMC transitions" begin
    rng = StableRNG(3)
    y = randn(rng, 15)

    @model function m2(y)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
    end

    raw = AbstractMCMC.sample(
        StableRNG(4), m2(y), NUTS(0.8), 30;
        n_adapts=15, discard_initial=15, chain_type=nothing, progress=false,
    )
    @test raw isa AbstractVector
    @test length(raw) == 30
    # Each element should be a real AdvancedHMC transition (has a phase
    # point `.z.θ`), not something PracticalBayes has already invlink'd.
    @test raw[1].z.θ isa AbstractVector{<:Real}
end
