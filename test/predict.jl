# Tests for src/predict.jl — the M4 milestone (design plan: "prior-predictive
# moments; predict shapes/names; returned round-trip").

using Distributions: Normal, Exponential
using AdvancedHMC: NUTS
using StableRNGs: StableRNG
import AbstractMCMC
using Statistics: mean, std
import FlexiChains

@testset "predict.jl: rand(model) draws a full prior sample" begin
    rng = StableRNG(1)

    @model function m1(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end

    # `.~`'s predictive-sampling support (tilde.jl) needs `y`'s SHAPE, not
    # just "no data" — an array of `missing` at the desired length, matching
    # the standard PPL convention (DynamicPPL's own `predict` docs use the
    # identical `fill(missing, n)` pattern), since a scalar-broadcast
    # `dist_bcast` (`Normal.(mu,sigma)` with scalar mu/sigma) carries no
    # shape information on its own.
    m = m1(fill(missing, 20))
    d = rand(rng, m)
    @test d isa NamedTuple
    @test Set(keys(d)) == Set((:mu, :sigma, :y))
    @test length(d.y) == 20
    @test all(!ismissing, d.y)

    ds = rand(rng, m, 100)
    @test length(ds) == 100
    # Prior-predictive moment check: mu ~ Normal(0,1), so the pooled sample
    # mean of `mu` across many draws should be close to 0.
    mus = [x.mu for x in ds]
    @test abs(mean(mus)) < 3 * std(mus) / sqrt(length(mus))
end

@testset "predict.jl: logjoint/logprior/loglikelihood_at split correctly, match Accum" begin
    rng = StableRNG(2)
    y = randn(rng, 15) .+ 1.0

    @model function m2(y)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
    end
    m = m2(y)
    pt = (mu=0.5,)

    lj = PracticalBayes.logjoint(m, pt)
    lp = PracticalBayes.logprior(m, pt)
    ll = PracticalBayes.loglikelihood_at(m, pt)

    # Independently computed reference: logprior is just logpdf(Normal(0,1), 0.5);
    # loglik is sum(logpdf(Normal(0.5,1), y)).
    @test lp ≈ Distributions.logpdf(Normal(0, 1), 0.5)
    @test ll ≈ sum(Distributions.logpdf(Normal(0.5, 1), yi) for yi in y)
    @test lj ≈ lp + ll

    # Observe site with no data anywhere and predict=false (the default):
    # must error, not silently skip the likelihood term.
    m_nodata = m2(fill(missing, 15))
    @test_throws ArgumentError PracticalBayes.logjoint(m_nodata, pt)
end

@testset "predict.jl: returned gives the model's own return value" begin
    @model function m3(x)
        mu ~ Normal(0, 1)
        return mu^2 + x
    end
    m = m3(1.0)
    @test PracticalBayes.returned(m, (mu=3.0,)) == 10.0
    @test PracticalBayes.returned(m, (mu=-2.0,)) == 5.0
end

@testset "predict.jl: chain_draws + predict — shapes and posterior-predictive accuracy" begin
    rng = StableRNG(3)
    y = randn(rng, 30) .+ 2.0

    @model function conjugate(y)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
    end
    m_train = conjugate(y)
    chn = AbstractMCMC.sample(
        StableRNG(4), m_train, NUTS(0.8), 500;
        n_adapts=250, discard_initial=250, progress=false,
    )

    draws = PracticalBayes.chain_draws(chn)
    @test length(draws) == 500
    @test draws[1] isa NamedTuple
    @test Set(keys(draws[1])) == Set((:mu,))  # AdvancedHMC's Extra stats must NOT leak in

    # Predict 10 new observations per draw, using a fresh model instance with
    # the observed argument replaced by an array of `missing`.
    m_test = conjugate(fill(missing, 10))
    preds = PracticalBayes.predict(StableRNG(5), m_test, draws)
    @test length(preds) == 500
    @test all(p -> length(p.y) == 10, preds)
    @test all(p -> all(!ismissing, p.y), preds)

    # Posterior-predictive mean of y should recover the same posterior mean
    # of mu the chain itself converged to (same analytic target as
    # test/sample.jl's own conjugate gate: post_mean = sum(y)/(1+n)).
    n = length(y)
    post_mean = sum(y) / (1 + n)
    all_pred_y = reduce(vcat, p.y for p in preds)
    mcse = std(all_pred_y) / sqrt(length(all_pred_y))
    @test abs(mean(all_pred_y) - post_mean) < 6 * mcse  # predictive spread is wider than the parameter's own MC-SE (includes observation noise), hence the looser 6x bar
end
