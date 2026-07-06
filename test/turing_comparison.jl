# Speed and accuracy comparisons against Turing.jl, per user request. Turing
# is a `[extras]`-only test dependency (see Project.toml) — these tests are
# skipped (not failed) if it isn't installed, so the base test suite doesn't
# require pulling in the entire Turing stack just to run.
#
# "Accuracy" here means: on an identical model + identical data, both
# packages' posterior summaries agree (same target distribution, sampled
# correctly) — NOT that the two implementations compute bit-identical
# log-densities (their bijector/parameterization conventions can differ in
# additive constants). "Speed" is the requirement-1 headline comparison:
# PracticalBayes' observe-heavy model should be at least as fast as Turing's
# `@addlogprob!` escape hatch, and substantially faster than Turing's
# ordinary `~` on the same data.
#
# NOTE: this file only exercises the LogDensityProblems-level interface
# (`logdensity`/`logdensity_and_gradient`) directly, NOT `sample.jl`/
# `chains.jl` machinery — those are M2 scope and don't exist yet as of M1.
# For "accuracy," we drive AdvancedHMC's own low-level NUTS constructors
# ourselves rather than going through not-yet-written sampling glue.

using Distributions: Normal, Exponential, logpdf
using StableRNGs: StableRNG
using Statistics: mean, std
using ADTypes: AutoForwardDiff
import LogDensityProblems
import AbstractMCMC
import AdvancedHMC

const _HAVE_TURING = !isnothing(Base.find_package("Turing"))

if _HAVE_TURING
    import Turing

    @testset "turing_comparison.jl: accuracy — conjugate Normal-Normal posterior" begin
        # y_i ~ Normal(mu, 1), mu ~ Normal(0, 1): posterior mean has a closed form,
        # and both packages' NUTS chains should recover it within a few MC-SEs.
        rng = StableRNG(42)
        y = randn(rng, 30) .+ 2.0  # true mean ~2

        @model function pb_conjugate(y)
            mu ~ Normal(0, 1)
            y .~ Normal.(mu, 1.0)
        end

        Turing.@model function turing_conjugate(y)
            mu ~ Normal(0, 1)
            for i in eachindex(y)
                y[i] ~ Normal(mu, 1)
            end
        end

        n = length(y)
        post_var = 1 / (1 + n)
        post_mean = post_var * sum(y)

        pb_model = pb_conjugate(y)
        pb_layout, pb_θ0, pb_store0 = build_layout(pb_model)
        pb_ldf = LogDensityFunction(pb_model, pb_layout, pb_store0, AutoForwardDiff(); θ0=pb_θ0)
        pb_ldm = AbstractMCMC.LogDensityModel(pb_ldf)
        pb_samples = AbstractMCMC.sample(
            StableRNG(1), pb_ldm, AdvancedHMC.NUTS(0.8), 2000; initial_params=pb_θ0, progress=false
        )
        # Each element of `pb_samples` is whatever AdvancedHMC's own transition
        # type is; `AbstractMCMC.getparams` on a *transition* (not a state) is
        # the documented way to pull the raw flat vector back out of it.
        pb_mu_draws = [invlink(pb_layout, AbstractMCMC.getparams(pb_ldm, s)).mu for s in pb_samples[500:end]]

        turing_chain = Turing.sample(StableRNG(1), turing_conjugate(y), Turing.NUTS(0.8), 2000; progress=false)
        turing_mu_draws = vec(Array(turing_chain[500:end, [:mu], :]))

        @test abs(mean(pb_mu_draws) - post_mean) < 3 * std(pb_mu_draws) / sqrt(length(pb_mu_draws))
        @test abs(mean(turing_mu_draws) - post_mean) < 3 * std(turing_mu_draws) / sqrt(length(turing_mu_draws))
        @test abs(mean(pb_mu_draws) - mean(turing_mu_draws)) < 0.2
    end

    @testset "turing_comparison.jl: speed — observe-heavy model vs Turing ~ and @addlogprob!" begin
        rng = StableRNG(7)
        y = randn(rng, 10_000)

        @model function pb_many_obs(y)
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            y .~ Normal.(mu, sigma)
        end

        Turing.@model function turing_tilde_many_obs(y)
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            for i in eachindex(y)
                y[i] ~ Normal(mu, sigma)
            end
        end

        Turing.@model function turing_addlogprob_many_obs(y)
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            Turing.@addlogprob! sum(logpdf.(Normal(mu, sigma), y))
        end

        pb_model = pb_many_obs(y)
        pb_layout, pb_θ0, pb_store0 = build_layout(pb_model)
        pb_ldf = LogDensityFunction(pb_model, pb_layout, pb_store0)

        t_pb = @elapsed for _ in 1:200
            LogDensityProblems.logdensity(pb_ldf, pb_θ0)
        end

        turing_tilde_model = turing_tilde_many_obs(y)
        turing_addlogprob_model = turing_addlogprob_many_obs(y)
        turing_vi = Turing.VarInfo(turing_tilde_model)
        turing_vi_add = Turing.VarInfo(turing_addlogprob_model)

        t_turing_tilde = @elapsed for _ in 1:200
            turing_tilde_model(turing_vi)
        end
        t_turing_addlogprob = @elapsed for _ in 1:200
            turing_addlogprob_model(turing_vi_add)
        end

        @info "observe-heavy timing (200 evals)" pb = t_pb turing_tilde = t_turing_tilde turing_addlogprob = t_turing_addlogprob

        # The headline requirement: PracticalBayes' plain `~`/`.~` observe
        # should be roughly as fast as Turing's @addlogprob! escape hatch
        # (allow generous slack — this is a smoke-level regression guard,
        # not a strict benchmark; see bench/observe_overhead.jl for the real
        # microbenchmark), and comfortably faster than Turing's ordinary `~`.
        @test t_pb < 3 * t_turing_addlogprob
        @test t_pb < t_turing_tilde
    end
else
    @testset "turing_comparison.jl" begin
        @test_skip "Turing not available in this environment"
    end
end
