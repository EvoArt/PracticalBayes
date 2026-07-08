# Speed and accuracy comparisons against Turing.jl, per user request.
#
# IMPORTANT: this does NOT load Turing, even though Turing now CAN be
# loaded alongside PracticalBayes (see the top-level Project.toml's
# `Turing = "0.45"` test-only compat entry — this used to be impossible
# because PracticalBayes' own compat bound had drifted to requiring
# AbstractPPL 0.15.x via an unnecessarily wide Bijectors range; Bijectors
# 0.15.24 already has the full VectorBijectors API this package needs and
# only requires AbstractPPL 0.14, which is what Turing/DynamicPPL use too).
# Turing's dependency tree is large and slow to precompile, so it's still
# kept OUT of the default `Pkg.test()` loop: Turing's numbers were captured
# ONCE by `test/comparison_env/generate_turing_reference.jl` (a separate
# environment, still useful for isolating this script's own dependency
# footprint even though it's no longer strictly required for coexistence)
# and frozen into `test/turing_reference.jl` as plain Julia literals. This
# file loads that reference and checks PracticalBayes against it — no
# Turing dependency at ordinary test time.
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
using Statistics: mean, median, std
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

    @testset "turing_comparison.jl: speed — tiny/large × Float64/Float32 vs frozen Turing reference" begin
        # Same shape/precision matrix as bench/suite.jl (tiny=20 obs,
        # large=50_000 obs) and the SAME data-generation procedure as
        # generate_turing_reference.jl, so medians are actually comparable
        # rather than an apples-to-oranges single point.
        @model function pb_many_obs(y)
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            y .~ Normal.(mu, sigma)
        end

        function pb_density_median(T, n; reps=30)
            rng = StableRNG(n + (T == Float32 ? 1 : 0))
            y = T.(randn(rng, n))
            m = pb_many_obs(y)
            layout, θ0, store0 = build_layout(m; T=T)
            ldf = LogDensityFunction(m, layout, store0)
            LogDensityProblems.logdensity(ldf, θ0)  # warmup
            times = [(@elapsed LogDensityProblems.logdensity(ldf, θ0)) for _ in 1:reps]
            return median(times)
        end

        for (n, shapename) in ((20, "tiny"), (50_000, "large")), T in (Float64, Float32)
            t_pb = pb_density_median(T, n)
            ref_tilde = TURING_REFERENCE.speed[(shapename, "$(T)_tilde")].median_s
            ref_add = TURING_REFERENCE.speed[(shapename, "$(T)_addlogprob")].median_s
            @info "density timing, $shapename n=$n T=$T" pb = t_pb turing_tilde = ref_tilde turing_addlogprob = ref_add
            # Generous slack (this is a smoke-level regression guard against
            # frozen reference numbers on a DIFFERENT machine, not a strict
            # benchmark — see bench/suite.jl, run locally, for the real
            # multi-rep comparison): PracticalBayes should be within a small
            # constant factor of Turing's own `~`/`@addlogprob!` at both scales.
            @test t_pb < 5 * ref_add
            @test t_pb < 5 * ref_tilde
        end
    end
else
    @testset "turing_comparison.jl" begin
        @test_skip "turing_reference.jl not found; run test/comparison_env/generate_turing_reference.jl once to produce it"
    end
end
