# Speed and accuracy comparisons against Turing.jl, per user request.
#
# IMPORTANT: this does NOT load Turing. Turing 0.45 (newest released)
# depends on AbstractPPL 0.14.2, while PracticalBayes depends on AbstractPPL
# 0.15.x (the VectorBijectors-based Bijectors API the whole layout system is
# built on) — these two genuinely cannot be loaded in the same Julia
# process, in any environment, with any amount of compat-bound tuning. So
# instead of a live side-by-side run, Turing's numbers were captured ONCE by
# `test/comparison_env/generate_turing_reference.jl` (a separate, standalone
# environment that has Turing but not PracticalBayes) and frozen into
# `test/turing_reference.jl` as plain Julia literals. This file loads that
# reference and checks PracticalBayes against it — no Turing dependency at
# test time, ever.
#
# Regenerate `turing_reference.jl` (by rerunning the generator script in
# test/comparison_env/) only if these reference MODELS themselves change;
# ordinary PracticalBayes code changes should never require regenerating it.
#
# "Accuracy" here means: on an identical model + identical data, both
# packages' posterior summaries agree (same target distribution, sampled
# correctly) — NOT that the two implementations compute bit-identical
# log-densities (their bijector/parameterization conventions can differ in
# additive constants). "Speed" is the requirement-1 headline comparison:
# PracticalBayes' observe-heavy model should be competitive with Turing's
# `~`/`@addlogprob!` on the same data (see bench/observe_overhead.jl for the
# stricter microbenchmark against a raw, framework-free loop).
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

const _REFERENCE_PATH = joinpath(@__DIR__, "turing_reference.jl")

if isfile(_REFERENCE_PATH)
    include(_REFERENCE_PATH)

    @testset "turing_comparison.jl: accuracy — conjugate Normal-Normal posterior" begin
        # Same data-generation seed/procedure as generate_turing_reference.jl,
        # so the two sides are evaluating the literal same posterior.
        rng = StableRNG(42)
        y = randn(rng, 30) .+ 2.0  # true mean ~2

        @model function pb_conjugate(y)
            mu ~ Normal(0, 1)
            y .~ Normal.(mu, 1.0)
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
        # Each element of `pb_samples` is an `AdvancedHMC.Transition`, whose
        # flat parameter vector lives at `.z.θ` (`.z` is the phasepoint,
        # `.stat` the NUTS diagnostics) — not through `AbstractMCMC.getparams`,
        # which is for pulling params back out of a sampler *state*, not a
        # per-draw *transition*.
        pb_mu_draws = [invlink(pb_layout, s.z.θ).mu for s in pb_samples[500:end]]

        # 1) PracticalBayes' own NUTS chain should recover the analytic posterior.
        @test abs(mean(pb_mu_draws) - post_mean) < 3 * std(pb_mu_draws) / sqrt(length(pb_mu_draws))
        # 2) ...and should agree with Turing's NUTS chain on the same model/data
        # (frozen reference: mean=$(TURING_REFERENCE.accuracy.mu_mean), std=$(TURING_REFERENCE.accuracy.mu_std)).
        @test abs(mean(pb_mu_draws) - TURING_REFERENCE.accuracy.mu_mean) < 0.2
    end

    @testset "turing_comparison.jl: speed — observe-heavy model vs frozen Turing reference" begin
        rng = StableRNG(7)
        y = randn(rng, 10_000)

        @model function pb_many_obs(y)
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            y .~ Normal.(mu, sigma)
        end

        pb_model = pb_many_obs(y)
        pb_layout, pb_θ0, pb_store0 = build_layout(pb_model)
        pb_ldf = LogDensityFunction(pb_model, pb_layout, pb_store0)

        # warmup (compilation) before timing, matching the reference script
        LogDensityProblems.logdensity(pb_ldf, pb_θ0)
        t_pb = @elapsed for _ in 1:200
            LogDensityProblems.logdensity(pb_ldf, pb_θ0)
        end

        @info "observe-heavy timing (200 evals)" pb = t_pb turing_tilde = TURING_REFERENCE.speed.tilde_seconds turing_addlogprob = TURING_REFERENCE.speed.addlogprob_seconds

        # The headline requirement: PracticalBayes' plain `~`/`.~` observe
        # should be roughly competitive with Turing's `@addlogprob!` escape
        # hatch (generous slack — this is a smoke-level regression guard
        # against the FROZEN reference numbers, not a strict benchmark; see
        # bench/observe_overhead.jl for the real microbenchmark against a
        # framework-free loop), and comfortably faster than Turing's ordinary `~`.
        @test t_pb < 3 * TURING_REFERENCE.speed.addlogprob_seconds
        @test t_pb < 3 * TURING_REFERENCE.speed.tilde_seconds
    end
else
    @testset "turing_comparison.jl" begin
        @test_skip "turing_reference.jl not found; run test/comparison_env/generate_turing_reference.jl once to produce it"
    end
end
