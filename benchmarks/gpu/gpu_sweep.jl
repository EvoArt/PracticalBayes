# GPU vs CPU benchmark — separate from benchmarks/sweep.jl (the CI-run,
# CPU-only PB-vs-Turing sweep) because GitHub's standard hosted CI runners
# have no GPU, so this can't be part of that always-runs pipeline. This
# script is CUDA.functional()-gated (same pattern as test/gpu/cuda.jl) and
# meant to be run BY HAND on a machine with a real, working GPU — either a
# contributor's own hardware or a manually-triggered GPU-equipped CI runner,
# neither of which this development session has access to (confirmed:
# CUDA.functional() == false here, CuArray construction itself throws
# "CUDA driver not functional"). This script's correctness has been
# reviewed but NOT run end-to-end on real hardware — same caveat as
# test/gpu/cuda.jl.
#
# Usage: julia --project=benchmarks/gpu benchmarks/gpu/gpu_sweep.jl
#
# What this measures: the regression model where GPU throughput SHOULD win
# over CPU — large N (many observations, so `X * beta` and the
# element-wise MvNormal observe are big enough matrix/vector ops to amortize
# GPU kernel-launch overhead), plotted against CPU median gradient time at
# the same N. Small N is included too specifically to show the crossover
# point (GPU loses at small N due to kernel-launch/host-device transfer
# overhead dominating — this is expected and worth recording, not a bug).

import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec(path=normpath(joinpath(@__DIR__, "..", ".."))))
Pkg.instantiate()

import PracticalBayes
import ADTypes
import Distributions
import BenchmarkTools
import Random
import LogDensityProblems
import LinearAlgebra
import JSON3

if isnothing(Base.find_package("CUDA"))
    println("CUDA not installed in this environment — nothing to do. Add it via `Pkg.add(\"CUDA\")` in benchmarks/gpu's own environment.")
    exit(0)
end
import CUDA

if !CUDA.functional()
    println("CUDA.functional() == false — no working GPU driver on this machine. This script must be run somewhere with a real, functional GPU.")
    exit(0)
end

CUDA.allowscalar(false)

# Same vectorized parametrization as benchmarks/sweep.jl and
# bench/repl_playground.jl's fixed version — `beta ~ MvNormal(...)`, never a
# per-element loop (which would allocate its container element-by-element on
# the CPU regardless of X/y's array type, defeating the whole point of a GPU
# comparison), `y ~ MvNormal(...)`, never `.~` (which would force a
# CPU-side reduction via Distributions.jl's generic loglikelihood fallback
# for a non-`Array` argument).
PracticalBayes.@model function gpu_regression(X, y)
    pT = PracticalBayes.paramtype(__mode__)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    sigma ~ Distributions.Exponential(one(pT))
    eta = X * beta
    y ~ Distributions.MvNormal(eta, sigma^2 * LinearAlgebra.I)
end

# N sweep chosen to bracket the expected CPU/GPU crossover: small N should
# favor CPU (GPU kernel-launch overhead dominates), large N should favor GPU
# (the matrix-multiply/vectorized-observe work is big enough to amortize
# that overhead). K (nparams) fixed at a moderate size — this sweep is about
# the N-scaling crossover specifically, not the N-vs-K grid
# benchmarks/sweep.jl already covers on CPU alone.
const NS = (100, 1_000, 10_000, 100_000, 1_000_000)
const K = 20
const BENCH_SAMPLES = 10
const BENCH_EVALS = 1

function make_data(n, k, seed)
    rng = Random.Xoshiro(seed)
    X = randn(rng, n, k)
    true_beta = randn(rng, k) .* 0.5
    y = X * true_beta .+ randn(rng, n) .* 0.5
    return X, y
end

function median_gradient_time_ns(ldf, θ; samples=BENCH_SAMPLES, evals=BENCH_EVALS)
    LogDensityProblems.logdensity_and_gradient(ldf, θ)  # warmup/compile
    trial = BenchmarkTools.@benchmark LogDensityProblems.logdensity_and_gradient($ldf, $θ) samples=samples evals=evals
    return Float64(BenchmarkTools.median(trial).time)
end

function build_ldf(X, y)
    m = gpu_regression(X, y)
    layout, θ0, store0 = PracticalBayes.build_layout(m)
    ldf = PracticalBayes.LogDensityFunction(m, layout, store0, ADTypes.AutoForwardDiff(); θ0=θ0)
    return ldf, θ0
end

function main()
    println("Running CPU vs GPU gradient-time sweep (K=$K, N in $(collect(NS)))")
    rows = Any[]
    for n in NS
        X_cpu, y_cpu = make_data(n, K, 1)

        # CPU
        cpu_ns = try
            ldf, θ0 = build_ldf(X_cpu, y_cpu)
            median_gradient_time_ns(ldf, θ0)
        catch e
            println("  N=$n CPU FAILED — ", sprint(showerror, e))
            NaN
        end

        # GPU: data on-device, θ stays on CPU (plan's own GPU scope — see
        # plan.md's "GPU" section: "θ/position vector on CPU, data on GPU").
        gpu_ns = try
            X_gpu = CUDA.CuArray(X_cpu)
            y_gpu = CUDA.CuArray(y_cpu)
            ldf, θ0 = build_ldf(X_gpu, y_gpu)
            median_gradient_time_ns(ldf, θ0)
        catch e
            println("  N=$n GPU FAILED — ", sprint(showerror, e))
            NaN
        end

        ratio = (isfinite(cpu_ns) && isfinite(gpu_ns) && gpu_ns > 0) ? cpu_ns / gpu_ns : NaN
        println("  N=$n: CPU=$(round(cpu_ns/1e6; digits=3)) ms, GPU=$(round(gpu_ns/1e6; digits=3)) ms, CPU/GPU speedup=$(round(ratio; digits=2))x")

        push!(rows, Dict("N" => n, "K" => K, "cpu_ns" => cpu_ns, "gpu_ns" => gpu_ns, "cpu_over_gpu_ratio" => ratio))
    end

    resdir = joinpath(@__DIR__, "results")
    mkpath(resdir)
    payload = Dict(
        "meta" => Dict("n_values" => collect(NS), "k" => K, "bench_samples" => BENCH_SAMPLES, "bench_evals" => BENCH_EVALS),
        "rows" => rows,
    )
    open(joinpath(resdir, "gpu_sweep.json"), "w") do io
        JSON3.pretty(io, payload)
    end
    println("Saved results to: ", joinpath(resdir, "gpu_sweep.json"))
end

main()
