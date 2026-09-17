# GPU correctness across precisions and AD backends.
#
# This complements `test/gpu/cuda.jl` (the original M5 milestone gate: one
# model, one backend, Float64). What that file does NOT establish, and this one
# does, is the part that actually matters for a user choosing a configuration:
#
#   * does the GPU path work in Float32 as well as Float64 — the Float32 path
#     being PracticalBayes' own headline feature, and the one where a GPU is
#     most worth having (V100 and later have far more Float32 than Float64
#     throughput, so a GPU result that only holds in Float64 is the less
#     interesting half);
#   * does it work across AD BACKENDS rather than just ForwardDiff — in
#     particular Piste, which is this package's own AD and had never been run
#     against GPU data at all before this file;
#   * do the GPU numbers AGREE with the CPU numbers, which is the only check
#     that distinguishes "it ran" from "it was right". A GPU path that runs and
#     silently returns a wrong gradient is worse than one that throws.
#
# EVERY test here is gated on `CUDA.functional()` and skips cleanly without it
# — see the long comment in `cuda.jl` for why that gate is unavoidable rather
# than a cop-out.
#
# BACKEND FAILURE IS A REPORTED RESULT, NOT A TEST ERROR. Each backend runs
# inside its own guard: a backend that cannot handle GPU data (a real
# possibility for any of them, and the expected outcome for at least some) is
# recorded as a skip with its error message, and does not fail the suite or
# prevent the remaining backends from being tested. What IS a hard failure is a
# backend that returns a WRONG answer — silent numerical divergence from the
# CPU reference is the bug class this file exists to catch.

if isnothing(Base.find_package("CUDA"))
    @info "gpu/cuda_backends.jl: skipped — CUDA not installed in this environment"
else
    @eval import CUDA

    if !CUDA.functional()
        @info "gpu/cuda_backends.jl: skipped — CUDA.functional() is false (no working GPU driver)"
    else
        @testset "gpu/cuda_backends.jl" begin
            using Distributions: Normal, Exponential, MvNormal
            import ADTypes
            using ADTypes: AutoForwardDiff
            import LinearAlgebra
            import LogDensityProblems
            using Random: Xoshiro

            CUDA.allowscalar(false)

            # Vectorized throughout — `beta ~ MvNormal(...)` not a per-element
            # loop, `y ~ MvNormal(...)` not `.~`. On GPU data this is not a
            # style preference but a correctness requirement: either of the
            # alternatives indexes a CuArray element-by-element and trips
            # `allowscalar(false)`. See CLAUDE.md's model-writing notes.
            @model function gpu_reg(X, y)
                pT = PracticalBayes.paramtype(__mode__)
                k = size(X, 2)
                beta ~ MvNormal(zeros(pT, k), LinearAlgebra.I)
                sigma ~ Exponential(one(pT))
                eta = X * beta
                y ~ MvNormal(eta, sigma^2 * LinearAlgebra.I)
            end

            function make_data(T, n, k; seed=1)
                rng = Xoshiro(seed)
                X = T.(randn(rng, n, k))
                b = T.(randn(rng, k))
                y = T.(X * b .+ randn(rng, n) .* T(0.5))
                return X, y
            end

            # Build a log-density for given data + backend. `closed_form=false`
            # because these models are GLM-shaped and would otherwise be
            # silently rerouted to GradMode's analytic gradient — which would
            # mean this file tested GradMode three times over and the AD
            # backends never (the same trap documented in benchmarks/sweep.jl).
            function build(X, y, T, adtype)
                m = gpu_reg(X, y)
                layout, theta0, store0 = build_layout(m; T=T)
                ldf = LogDensityFunction(m, layout, store0, adtype; θ0=theta0, closed_form=false)
                return ldf, theta0
            end

            # -------------------------------------------------------------
            # Float64 and Float32, CPU-vs-GPU agreement.
            #
            # Tolerances differ by precision for a real reason, not to make a
            # flaky test pass: GPU and CPU legitimately produce different
            # roundings of the same reduction (a parallel tree reduction sums
            # in a different ORDER than a serial loop), and that divergence is
            # proportionally far larger in Float32 (~1e-7 eps) than in Float64
            # (~2e-16 eps). Demanding Float64-tight agreement from a Float32
            # GPU reduction would be testing the hardware's rounding, not this
            # package's correctness.
            # -------------------------------------------------------------
            @testset "CPU/GPU agreement — $T" for (T, rtol) in ((Float64, 1e-8), (Float32, 1e-3))
                n, k = 500, 8
                X_cpu, y_cpu = make_data(T, n, k)
                X_gpu, y_gpu = CUDA.CuArray(X_cpu), CUDA.CuArray(y_cpu)

                ldf_cpu, theta0 = build(X_cpu, y_cpu, T, AutoForwardDiff())
                ldf_gpu, theta0g = build(X_gpu, y_gpu, T, AutoForwardDiff())

                @test eltype(theta0) === T
                @test eltype(theta0g) === T   # θ stays on the CPU and keeps its precision

                v_cpu = LogDensityProblems.logdensity(ldf_cpu, theta0)
                v_gpu = LogDensityProblems.logdensity(ldf_gpu, theta0)
                @test isfinite(v_gpu)
                @test isapprox(v_gpu, v_cpu; rtol=rtol)

                _, g_cpu = LogDensityProblems.logdensity_and_gradient(ldf_cpu, theta0)
                _, g_gpu = LogDensityProblems.logdensity_and_gradient(ldf_gpu, theta0)
                @test all(isfinite, g_gpu)
                @test length(g_gpu) == length(theta0)
                @test isapprox(collect(g_gpu), collect(g_cpu); rtol=rtol)
            end

            # -------------------------------------------------------------
            # AD backends on GPU data, each independently contained.
            #
            # A backend failing here is INFORMATION (recorded via @info, with
            # the error), not a suite failure: none of these backends promises
            # GPU support, and finding out which ones actually deliver it is
            # the point of the test. But any backend that DOES run must agree
            # with the CPU reference — that assertion is unconditional.
            # -------------------------------------------------------------
            @testset "AD backends on GPU data" begin
                n, k = 500, 8
                T = Float64
                X_cpu, y_cpu = make_data(T, n, k)
                X_gpu, y_gpu = CUDA.CuArray(X_cpu), CUDA.CuArray(y_cpu)

                # CPU reference, computed once with the most trustworthy path.
                ldf_ref, theta0 = build(X_cpu, y_cpu, T, AutoForwardDiff())
                _, g_ref = LogDensityProblems.logdensity_and_gradient(ldf_ref, theta0)

                # Assembled at runtime so an uninstalled or broken optional
                # backend is a missing entry, never a load-time crash.
                candidates = Any[("forwarddiff", AutoForwardDiff())]

                if !isnothing(Base.find_package("Piste"))
                    try
                        @eval import Piste
                        # Piste's own gradient driver is typed on `Vector{T}`,
                        # which is fine here: PracticalBayes' GPU scope keeps θ
                        # (and hence g) on the CPU and puts only DATA on the
                        # device. What is genuinely being tested is whether
                        # Piste's `Dual` numbers survive being multiplied
                        # against a CuArray — `X * beta` with a CuMatrix and a
                        # `Vector{Dual}` is a host/device type mix that has
                        # never been exercised before this test.
                        push!(candidates, ("piste_fwd", AutoPBForwardDiff()))
                    catch e
                        @info "gpu: Piste present but failed to import; skipping" exception=e
                    end
                end

                if !isnothing(Base.find_package("Mooncake"))
                    try
                        @eval import Mooncake
                        push!(candidates, ("mooncake", ADTypes.AutoMooncake(; config=nothing)))
                    catch e
                        @info "gpu: Mooncake present but failed to import; skipping" exception=e
                    end
                end

                for (name, adtype) in candidates
                    result = try
                        ldf, _ = build(X_gpu, y_gpu, T, adtype)
                        LogDensityProblems.logdensity_and_gradient(ldf, theta0)
                    catch e
                        @info "gpu: backend `$name` does not work on GPU data (recorded, not a failure)" exception=(e, catch_backtrace())
                        nothing
                    end

                    if result !== nothing
                        v, g = result
                        @testset "$name" begin
                            @test isfinite(v)
                            @test all(isfinite, g)
                            # The real assertion: running is not enough, it
                            # must MATCH. A backend silently returning a wrong
                            # gradient on GPU data is precisely the failure
                            # this whole file is built to catch.
                            @test isapprox(collect(g), collect(g_ref); rtol=1e-6)
                        end
                    end
                end
            end

            # -------------------------------------------------------------
            # `allowscalar(false)` is already globally set above, so this
            # verifies the framework guarantee directly: a full gradient at a
            # size big enough to be a realistic GPU workload, with any scalar
            # getindex into device memory — from PracticalBayes, Distributions
            # or Bijectors — throwing rather than silently degrading into a
            # ruinously slow element-by-element device-to-host copy.
            # -------------------------------------------------------------
            @testset "no scalar indexing at scale" begin
                X_cpu, y_cpu = make_data(Float32, 20_000, 32)
                X_gpu, y_gpu = CUDA.CuArray(X_cpu), CUDA.CuArray(y_cpu)
                ldf, theta0 = build(X_gpu, y_gpu, Float32, AutoForwardDiff())
                v, g = LogDensityProblems.logdensity_and_gradient(ldf, theta0)
                @test isfinite(v)
                @test all(isfinite, g)
            end
        end
    end
end
