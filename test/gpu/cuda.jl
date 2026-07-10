# M5 milestone gate (design plan): "regression with y, X::CuArray,
# CUDA.allowscalar(false), logdensity + gradient + short NUTS run; 8-thread
# chain independence."
#
# Entirely gated on `CUDA.functional()` — CUDA is an `[extras]`-only
# dependency (see Project.toml), and even when installed, `CUDA.functional()`
# is `false` on any machine without a real, working GPU driver (confirmed
# directly: `CUDA.CuArray(...)` itself throws `CUDA driver not functional`
# the moment you try to construct one, before any PracticalBayes code even
# runs). This file was developed and CODE-REVIEWED on a machine with NO
# functional GPU — it has never actually been run end-to-end on real
# hardware. Treat it as unverified-by-execution until it's been run
# somewhere `CUDA.functional()` is `true` (a CI runner with a GPU, or a
# contributor's own machine) — this is a real, documented gap, not swept
# under the rug.
#
# The GPU guarantee this tests (see plan.md's "GPU" section): no
# framework-introduced scalar indexing anywhere in the hot path — parameter
# reads are `view(θ, range)` (confirmed directly by grepping tilde.jl: both
# `_assume`/`_assume_index`'s `FlatSlot`/`FlatArraySlot` methods use `view`,
# never a scalar `θ[i]`), accumulation is scalar arithmetic on already-scalar
# values (never indexing INTO a GPU array), and data (`X`/`y`) is read only
# via whole-array operations the model author writes (`X * beta`,
# `Normal.(eta, sigma)` broadcasts, `MvNormal(eta, ...)` array-variate
# construction) — never indexed element-by-element by PracticalBayes itself.
# `CUDA.allowscalar(false)` is the actual enforcement mechanism: it makes
# Julia throw immediately on ANY scalar `getindex`/`setindex!` into a
# CuArray, from ANY code (PracticalBayes' own, or Distributions.jl's, or
# Bijectors') — so this test doesn't just check "does it run", it checks
# "does it run WITHOUT ever touching GPU memory one element at a time",
# which is the actual performance-relevant guarantee.

if !isnothing(Base.find_package("CUDA"))
    @eval import CUDA

    if CUDA.functional()
        @testset "gpu/cuda.jl: regression with X,y::CuArray, allowscalar(false)" begin
            using Distributions: Normal, Exponential, MvNormal
            using ADTypes: AutoForwardDiff
            using AdvancedHMC: NUTS
            import AbstractMCMC
            import LinearAlgebra
            import LogDensityProblems
            using Random: Xoshiro

            CUDA.allowscalar(false)

            # Same vectorized parametrization established elsewhere in this
            # session's work (bench/repl_playground.jl, benchmarks/sweep.jl)
            # as the correct, apples-to-apples pattern: beta ~ MvNormal(...)
            # (never a per-element loop — that pattern allocates a container
            # element-by-element on the CPU inside the model body regardless
            # of X/y's own array type, which would defeat the point of this
            # test), y ~ MvNormal(...) (never .~/scalar broadcast, which
            # would force a CPU-side reduction over the GPU array via
            # Distributions.loglikelihood's generic fallback).
            @model function gpu_regression(X, y)
                k = size(X, 2)
                beta ~ MvNormal(zeros(k), LinearAlgebra.I)
                sigma ~ Exponential(1)
                eta = X * beta
                y ~ MvNormal(eta, sigma^2 * LinearAlgebra.I)
            end

            rng = Xoshiro(1)
            n, k = 200, 5
            X_cpu = randn(rng, n, k)
            true_beta = randn(rng, k)
            y_cpu = X_cpu * true_beta .+ randn(rng, n) .* 0.5

            # Plan's own GPU scope (MVP): θ/position vector on CPU, DATA on
            # GPU. build_layout/θ0 are never touched by CUDA.allowscalar —
            # only X/y (used inside the model body via `X * beta`,
            # `MvNormal(eta, ...)`) are CuArrays.
            X = CUDA.CuArray(X_cpu)
            y = CUDA.CuArray(y_cpu)

            m = gpu_regression(X, y)
            layout, theta0, store0 = build_layout(m)
            ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=theta0)

            # logdensity
            val = LogDensityProblems.logdensity(ldf, theta0)
            @test isfinite(val)

            # gradient
            val2, grad = LogDensityProblems.logdensity_and_gradient(ldf, theta0)
            @test isfinite(val2)
            @test all(isfinite, grad)
            @test length(grad) == length(theta0)

            # short NUTS run
            ldm = AbstractMCMC.LogDensityModel(ldf)
            chn = AbstractMCMC.sample(
                Xoshiro(2), ldm, NUTS(0.8), 50;
                n_adapts=25, discard_initial=25, initial_params=theta0, progress=false,
            )
            @test length(chn) == 50
        end
    else
        @info "gpu/cuda.jl: skipped — CUDA.functional() is false (no working GPU driver on this machine)"
    end
else
    @info "gpu/cuda.jl: skipped — CUDA not installed in this environment"
end

# ===========================================================================
# 8-thread chain independence — this half of the M5 gate does NOT need CUDA
# at all (it's testing AbstractMCMC.MCMCThreads correctness under real
# thread contention, already exercised with 4 chains in test/sample.jl —
# this just raises the count to 8 and checks independence more directly:
# no two chains should produce IDENTICAL draws, which would indicate shared
# mutable state leaking across threads).
# ===========================================================================

@testset "gpu/cuda.jl: 8-thread MCMCThreads — chain independence" begin
    using Distributions: Normal, Exponential
    using AdvancedHMC: NUTS
    using StableRNGs: StableRNG
    import AbstractMCMC
    import FlexiChains

    rng = StableRNG(1)
    y = randn(rng, 20) .+ 1.0

    @model function m_threads(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end

    nchains = min(8, Threads.nthreads())
    if nchains < 2
        @info "gpu/cuda.jl: 8-thread independence test skipped — only $(Threads.nthreads()) Julia thread(s) available (start Julia with -t auto or -t 8+ to exercise this)"
    else
        chns = AbstractMCMC.sample(
            StableRNG(2), m_threads(y), NUTS(0.8), AbstractMCMC.MCMCThreads(), 100, nchains;
            n_adapts=50, discard_initial=50, progress=false,
        )
        @test size(chns[:mu]) == (100, nchains)

        # Independence check: no two chains' full draw sequences are
        # bit-identical (would indicate shared mutable state — e.g. a
        # Layout/LogDensityFunction/DI prep object accidentally reused
        # across threads instead of deepcopy'd, since AbstractMCMC's own
        # MCMCThreads implementation deepcopies model/sampler per thread but
        # relies on PracticalBayes' `sample()` building a genuinely fresh
        # Layout/LogDensityFunction per call, which it does — see
        # src/sample.jl).
        mu_chains = [chns[:mu][:, c] for c in 1:nchains]
        for i in 1:nchains, j in (i+1):nchains
            @test mu_chains[i] != mu_chains[j]
        end

        r = FlexiChains.rhat(chns)
        @test r[FlexiChains.Parameter(:mu)] < 1.05  # looser than the 4-chain gate's 1.01 — shorter chains here (100 vs 500 samples)
    end
end
