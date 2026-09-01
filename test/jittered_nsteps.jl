# Tests for src/jittered_nsteps.jl — the opt-in jittered trajectory length for
# the static-HMC samplers (`AdaptiveHMC`, `NUTSthenHMC`, and `static_hmc`).
#
# Three things need to hold, and they are what this file checks:
#   1. The drawn step count is uniform on `1:L` (not fixed, never out of range),
#      and the reported `n_steps` diagnostic matches the trajectory that was
#      actually simulated — AdvancedHMC calls `nsteps` twice per transition, so
#      a careless implementation reports a different length than it integrated.
#   2. Jitter is genuinely OPT-IN: the default is unchanged fixed-length HMC.
#   3. Jitter does not break the sampler — the posterior is still correct, held
#      to the same analytic conjugate-Normal reference test/sample.jl uses.

using Distributions: Normal
using AdvancedHMC: AdvancedHMC, Trajectory, EndPointTS, FixedNSteps, Leapfrog
using StableRNGs: StableRNG
import ADTypes
using Statistics: mean, std
import AbstractMCMC
import FlexiChains

@testset "jittered_nsteps.jl: JitteredNSteps unit behaviour" begin
    tc = PracticalBayes.JitteredNSteps(10)
    @test tc.L == 10
    # Seeded with L so a criterion that somehow reaches `nsteps` before any
    # transition has drawn for it still gives a valid (fixed-L) trajectory.
    @test tc.current == 10

    tau = Trajectory{EndPointTS}(Leapfrog(0.1), tc)
    @test AdvancedHMC.nsteps(tau) == 10
    tc.current = 3
    @test AdvancedHMC.nsteps(tau) == 3

    @test_throws ArgumentError PracticalBayes.JitteredNSteps(0)

    # The criterion selector both samplers share.
    @test PracticalBayes._n_steps_criterion(8, false) isa FixedNSteps
    @test PracticalBayes._n_steps_criterion(8, true) isa PracticalBayes.JitteredNSteps
end

# A conjugate Normal-Normal model with a known analytic posterior, matching
# test/sample.jl's own gate so the accuracy checks below are directly comparable.
@model function _jitter_conjugate(y)
    mu ~ Normal(0, 1)
    y .~ Normal.(mu, 1.0)
end

_jitter_data() = randn(StableRNG(42), 30) .+ 2.0

function _jitter_analytic(y)
    n = length(y)
    post_var = 1 / (1 + n)
    return post_var * sum(y), sqrt(post_var)
end

@testset "jittered_nsteps.jl: n_steps distribution, AdaptiveHMC" begin
    y = _jitter_data()
    m = _jitter_conjugate(y)
    L = 16

    # Fixed (the default) — every iteration is exactly L steps.
    chn = AbstractMCMC.sample(
        StableRNG(1), m, AdaptiveHMC(0.8; n_leapfrog=L), 400;
        n_adapts=200, discard_initial=200, progress=false,
    )
    ns_fixed = vec(chn[FlexiChains.Extra(:n_steps)])
    @test all(==(L), ns_fixed)

    # Jittered — spread over the whole 1:L range.
    chn_j = AbstractMCMC.sample(
        StableRNG(1), m, AdaptiveHMC(0.8; n_leapfrog=L, jitter=true), 400;
        n_adapts=200, discard_initial=200, progress=false,
    )
    ns_jit = vec(chn_j[FlexiChains.Extra(:n_steps)])
    @test all(n -> 1 <= n <= L, ns_jit)
    @test length(unique(ns_jit)) > 1          # actually varying
    @test minimum(ns_jit) <= 3                # reaches the short end
    @test maximum(ns_jit) >= L - 3            # and the long end
    # Uniform on 1:L has mean (L+1)/2; this is the "costs less than fixed L"
    # claim in the docstring, so check it rather than assume it. Loose bound —
    # 400 draws of a discrete uniform, not a precision test.
    @test abs(mean(ns_jit) - (L + 1) / 2) < 2.0
end

@testset "jittered_nsteps.jl: n_steps distribution, NUTSthenHMC" begin
    y = _jitter_data()
    m = _jitter_conjugate(y)
    L = 16
    n_adapts = 200

    # `jitter` must apply ONLY to the post-warm-up HMC leg. With an explicit
    # n_leapfrog the post-warm-up phase is capped at L, while the NUTS warm-up
    # is untouched and free to use its own tree depths.
    chn = AbstractMCMC.sample(
        StableRNG(1), m, NUTSthenHMC(0.8; n_leapfrog=L, jitter=true), 400;
        n_adapts=n_adapts, progress=false,
    )
    # Slice past the warm-up: `discard_initial` is not applied here (the chain
    # keeps all `n_adapts` warm-up iterations), and those are real NUTS tree
    # steps whose counts are `2^depth - 1` and can exceed L. Only the
    # post-switch tail is the jittered HMC phase this test is about.
    ns = vec(chn[FlexiChains.Extra(:n_steps)])[(n_adapts + 1):end]
    @test all(n -> 1 <= n <= L, ns)
    @test length(unique(ns)) > 1

    # Default stays fixed-length, on the same post-warm-up slice.
    chn_f = AbstractMCMC.sample(
        StableRNG(1), m, NUTSthenHMC(0.8; n_leapfrog=L), 400;
        n_adapts=n_adapts, progress=false,
    )
    ns_f = vec(chn_f[FlexiChains.Extra(:n_steps)])[(n_adapts + 1):end]
    @test all(==(L), ns_f)
end

@testset "jittered_nsteps.jl: jitter is opt-in" begin
    @test AdaptiveHMC(0.8; n_leapfrog=8).jitter == false
    @test AdaptiveHMC(0.8; n_leapfrog=8, jitter=true).jitter == true
    @test NUTSthenHMC(0.8; n_leapfrog=8).jitter == false
    @test NUTSthenHMC(0.8; n_leapfrog=8, jitter=true).jitter == true
end

@testset "jittered_nsteps.jl: posterior still correct under jitter" begin
    y = _jitter_data()
    m = _jitter_conjugate(y)
    post_mean, _ = _jitter_analytic(y)

    for spl in (
        AdaptiveHMC(0.8; n_leapfrog=24, jitter=true),
        NUTSthenHMC(0.8; n_leapfrog=24, jitter=true),
    )
        chn = AbstractMCMC.sample(
            StableRNG(1), m, spl, 2000;
            n_adapts=1000, discard_initial=1000, progress=false,
        )
        mu_draws = vec(chn[FlexiChains.Parameter(:mu)])
        @test length(mu_draws) == 2000
        # Same 4x MC-SE bound (rather than the naive 3x) as test/sample.jl, for
        # the reason documented there: autocorrelated HMC draws have a smaller
        # effective sample size than raw N, so `std/sqrt(N)` understates the
        # true MC-SE.
        mcse = std(mu_draws) / sqrt(length(mu_draws))
        @test abs(mean(mu_draws) - post_mean) < 4 * mcse
    end
end

@testset "jittered_nsteps.jl: static_hmc jitter keyword" begin
    # `static_hmc` has jittered since it was written; the keyword makes that
    # switchable without changing the default.
    y = _jitter_data()
    m = _jitter_conjugate(y)
    post_mean, _ = _jitter_analytic(y)

    layout, x0, store0 = build_layout(m)
    ldf = LogDensityFunction(m, layout, store0, ADTypes.AutoForwardDiff(); θ0=x0)
    gradf! = static_gradient(ldf)
    inv_mass = ones(length(x0))

    for jitter in (true, false)
        res = static_hmc(gradf!, x0, inv_mass, 0.25, 12, 2000, 500; seed=3, jitter=jitter)
        mu_draws = vec(res.draws[1, :])
        mcse = std(mu_draws) / sqrt(length(mu_draws))
        @test abs(mean(mu_draws) - post_mean) < 4 * mcse
    end
end
