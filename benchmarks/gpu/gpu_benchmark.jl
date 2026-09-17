# CPU vs GPU gradient benchmark — the full factorial the README's GPU section
# needs, and a Bayesian neural network alongside it.
#
# This supersedes `gpu_sweep.jl` (one model, one backend, Float64, N-sweep
# only). What it adds is everything needed to actually ANSWER "is a GPU worth
# it here", which is a question with at least four interacting axes:
#
#     device     ∈ {cpu, gpu}
#     precision  ∈ {Float64, Float32}
#     N          ∈ small … large   (observations)
#     K          ∈ small … large   (parameters)
#
# crossed with the AD backend and the PPL (PracticalBayes vs Turing). The
# N×K crossing is the point: GPU advantage on a GLM comes almost entirely from
# `X * beta` (an N×K by K matrix-vector product) and the N-length vectorized
# observe, so it is a function of BOTH dimensions and of the arithmetic
# intensity their ratio implies. Sweeping only N — as the old script did —
# cannot distinguish "the GPU wins at large N" from "the GPU wins when there
# is enough work per byte moved", and those have different implications for a
# user deciding how to write a model.
#
# Float32 is a first-class axis rather than an afterthought because it is
# where a GPU is most worth having: a V100 has ~2x the Float32 throughput of
# Float64 (and consumer NVIDIA cards are far more lopsided still, often 32x or
# worse), so a GPU benchmark run only in Float64 measures the least favourable
# case and systematically understates the result.
#
# ON TURING AND THE GPU
# ---------------------
# Turing is benchmarked here on CPU only, and that is a finding rather than an
# omission. DynamicPPL promotes to Float64 internally regardless of the input
# element type (see CLAUDE.md), so the Float32 axis does not exist on the
# Turing side at all; and its VarInfo machinery indexes parameters
# element-by-element, which is exactly what `CUDA.allowscalar(false)` forbids.
# The script still ATTEMPTS the Turing GPU cell rather than assuming — if it
# fails, the recorded error message is the evidence for that claim, which is
# worth more than a footnote asserting it.
#
# ROBUSTNESS
# ----------
# Every backend, every PPL and every device combination runs inside
# `benchmarks/backend_guard.jl`'s `guarded`, and every optional package is
# imported with `try_import`. A backend that is missing, fails to precompile,
# errors on GPU arrays, or blows up only in Float32 costs exactly its own
# cells and nothing else — the sweep runs to completion and writes its JSON
# regardless. On a cluster queue, where a crash means re-queueing and waiting
# for a GPU to free up again, that property is worth more than the few cells
# it might save.
#
# Usage (Slurm — see benchmarks/gpu/slurm/):
#     sbatch benchmarks/gpu/slurm/gpu_benchmark.sbatch

import Pkg
Pkg.activate(@__DIR__)
if get(ENV, "PB_GPU_INSTANTIATE", "0") == "1" ||
   !haskey(Pkg.project().dependencies, "PracticalBayes")
    Pkg.develop(Pkg.PackageSpec(path=normpath(joinpath(@__DIR__, "..", ".."))))
    Pkg.instantiate()
end

import PracticalBayes
import ADTypes
import Distributions
import BenchmarkTools
import Random
import LogDensityProblems
import LinearAlgebra
import JSON3

include(joinpath(@__DIR__, "..", "backend_guard.jl"))

# Optional dependencies, every one of them behind `try_import`. None of these
# is required for the script to produce useful output; each one that loads
# adds columns to the result.
const HAS_CUDA = try_import(:CUDA)
const HAS_PISTE = try_import(:Piste)
const HAS_MOONCAKE = try_import(:Mooncake)
const HAS_ENZYME = try_import(:Enzyme)
const HAS_TURING = try_import(:Turing) && try_import(:DynamicPPL)

const GPU_OK = HAS_CUDA && gpu_available()

if GPU_OK
    # The entire GPU guarantee in one line: any scalar `getindex` into device
    # memory — from this package, Distributions, Bijectors or an AD backend —
    # now throws instead of silently falling back to an element-by-element
    # device-to-host copy that would make the "GPU" timing meaningless. A
    # benchmark that allowed scalar indexing could report a GPU number that is
    # really a very slow CPU number.
    Main.CUDA.allowscalar(false)
end

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# GLM. Vectorized deliberately: `beta ~ MvNormal(...)` rather than a
# per-element loop, `y ~ MvNormal(...)` rather than `.~`. Both alternatives
# index element-by-element, which on GPU data is not merely slower but an
# outright error under `allowscalar(false)`.
PracticalBayes.@model function pb_glm(X, y)
    pT = PracticalBayes.paramtype(__mode__)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    sigma ~ Distributions.Exponential(one(pT))
    eta = X * beta
    y ~ Distributions.MvNormal(eta, sigma^2 * LinearAlgebra.I)
end

# Bayesian neural network — the Turing-docs example, restated in PB.
#
# Included because it stresses something the GLM cannot: a GLM is ONE
# matrix-vector product, so it measures BLAS dispatch nearly as much as it
# measures the PPL. A two-hidden-layer network is a chain of matmuls with
# element-wise nonlinearities between them, which is both a much better proxy
# for the workloads people actually reach for a GPU to run, and a far harder
# test of the AD backends — reverse-mode in particular, which has to tape
# every intermediate.
#
# The weights arrive as ONE flat vector, reshaped inside the model body, for
# a reason specific to this package: a `theta ~ MvNormal(zeros(nweights), I)`
# is a single contiguous slot in the layout, so the hot path reads it with one
# `view(θ, range)`. Declaring each weight matrix as its own `~` statement
# would be more readable but would add a slot, a view and a reshape per
# layer — real overhead on a path that runs once per leapfrog step.
function bnn_nweights(nin, nhidden, nout)
    return nin * nhidden + nhidden +            # layer 1: W1, b1
           nhidden * nhidden + nhidden +        # layer 2: W2, b2
           nhidden * nout + nout                # output:  W3, b3
end

# Split a flat parameter vector into the network's weight matrices and bias
# vectors. `reshape` on a `view` is allocation-free and, importantly,
# differentiable by every backend tested here.
@inline function bnn_unpack(theta, nin, nhidden, nout)
    o = 0
    W1 = reshape(view(theta, o+1 : o+nin*nhidden), nhidden, nin);      o += nin*nhidden
    b1 = view(theta, o+1 : o+nhidden);                                  o += nhidden
    W2 = reshape(view(theta, o+1 : o+nhidden*nhidden), nhidden, nhidden); o += nhidden*nhidden
    b2 = view(theta, o+1 : o+nhidden);                                  o += nhidden
    W3 = reshape(view(theta, o+1 : o+nhidden*nout), nout, nhidden);     o += nhidden*nout
    b3 = view(theta, o+1 : o+nout)
    return W1, b1, W2, b2, W3, b3
end

PracticalBayes.@model function pb_bnn(X, y, nhidden::Int)
    pT = PracticalBayes.paramtype(__mode__)
    nin = size(X, 1)
    nw = bnn_nweights(nin, nhidden, 1)
    # A single flat prior over all weights — one layout slot, one view.
    theta ~ Distributions.MvNormal(zeros(pT, nw), LinearAlgebra.I)
    W1, b1, W2, b2, W3, b3 = bnn_unpack(theta, nin, nhidden, 1)
    # `tanh` rather than `relu`: it is smooth, and a kink in the likelihood
    # surface is a genuine problem for HMC rather than a stylistic choice.
    h1 = tanh.(W1 * X .+ b1)
    h2 = tanh.(W2 * h1 .+ b2)
    out = vec(W3 * h2 .+ b3)
    y ~ PracticalBayes.arraydist(Distributions.BernoulliLogit.(out))
end

if HAS_TURING
    Main.Turing.@model function turing_glm(X, y)
        T = eltype(X)
        k = size(X, 2)
        beta ~ Distributions.MvNormal(zeros(T, k), LinearAlgebra.I)
        sigma ~ Distributions.Exponential(one(T))
        eta = X * beta
        y ~ Distributions.MvNormal(eta, sigma^2 * LinearAlgebra.I)
    end

    # Line-for-line the same computation as `pb_bnn` — same flat prior, same
    # unpacking, same nonlinearities, same observe. Benchmark model parity is
    # not negotiable: if the two model bodies differ, the comparison measures
    # the difference between the models rather than between the PPLs.
    Main.Turing.@model function turing_bnn(X, y, nhidden::Int)
        T = eltype(X)
        nin = size(X, 1)
        nw = bnn_nweights(nin, nhidden, 1)
        theta ~ Distributions.MvNormal(zeros(T, nw), LinearAlgebra.I)
        W1, b1, W2, b2, W3, b3 = bnn_unpack(theta, nin, nhidden, 1)
        h1 = tanh.(W1 * X .+ b1)
        h2 = tanh.(W2 * h1 .+ b2)
        out = vec(W3 * h2 .+ b3)
        y ~ Main.Turing.arraydist(Distributions.BernoulliLogit.(out))
    end
end

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

function make_glm_data(::Type{T}, n, k; seed=1) where {T<:Real}
    rng = Random.Xoshiro(seed)
    X = T.(randn(rng, n, k))
    b = T.(randn(rng, k) .* 0.5)
    y = X * b .+ T.(randn(rng, n) .* 0.5)
    return X, y
end

# Note the orientation: features down the ROWS (nin × n), which is the layout
# the network's `W1 * X` wants. Getting this backwards silently transposes the
# whole computation.
function make_bnn_data(::Type{T}, n, nin; seed=1) where {T<:Real}
    rng = Random.Xoshiro(seed)
    X = T.(randn(rng, nin, n))
    w = T.(randn(rng, nin))
    p = 1 ./ (1 .+ exp.(-(vec(sum(X .* w; dims=1)))))
    y = T.(rand(rng, n) .< p)
    return X, y
end

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

# Every GPU timing must synchronize. CUDA kernel launches are ASYNCHRONOUS:
# without a sync, the host returns as soon as the work is *queued*, and the
# benchmark measures launch overhead rather than execution — which is how GPU
# benchmarks come to report impossible speedups.
function sync_gpu()
    GPU_OK && Main.CUDA.synchronize()
    return nothing
end

const BENCH_SAMPLES = 10

function median_gradient_ns(ldf, theta; on_gpu::Bool, samples=BENCH_SAMPLES)
    LogDensityProblems.logdensity_and_gradient(ldf, theta)   # warm up / compile
    on_gpu && sync_gpu()
    trial = if on_gpu
        BenchmarkTools.@benchmark begin
            LogDensityProblems.logdensity_and_gradient($ldf, $theta)
            Main.CUDA.synchronize()
        end samples = samples evals = 1
    else
        BenchmarkTools.@benchmark LogDensityProblems.logdensity_and_gradient($ldf, $theta) samples = samples evals = 1
    end
    return Float64(BenchmarkTools.median(trial).time)
end

# Sweep axes. Deliberately spanning small→large on BOTH N and K so the
# crossover is inside the grid rather than off its edge: a GPU reliably LOSES
# at small sizes (kernel-launch overhead dominates) and that loss is a result
# worth recording, not a bug to be tuned away.
const NS = (1_000, 10_000, 100_000)
const KS = (10, 100, 500)
const PRECISIONS = (Float64, Float32)

function ad_backends()
    b = Any[("forwarddiff", ADTypes.AutoForwardDiff())]
    HAS_MOONCAKE && push!(b, ("mooncake", ADTypes.AutoMooncake(; config=nothing)))
    HAS_ENZYME && push!(b, ("enzyme", ADTypes.AutoEnzyme(; mode=Main.Enzyme.set_runtime_activity(Main.Enzyme.Reverse))))
    # Piste is PracticalBayes' own AD and has no Turing counterpart, so its
    # cells are PB-only by construction.
    HAS_PISTE && push!(b, ("piste_fwd", PracticalBayes.AutoPBForwardDiff()))
    return b
end

function build_pb(model, T, adtype)
    layout, theta0, store0 = PracticalBayes.build_layout(model; T=T)
    # `closed_form=false`: the GLM here is exactly the shape `gradmode_plan`
    # recognises, so without this every backend would be silently replaced by
    # GradMode's analytic gradient and all the AD columns would be identical.
    ldf = PracticalBayes.LogDensityFunction(model, layout, store0, adtype;
                                            θ0=theta0, closed_form=false)
    return ldf, theta0
end

function build_turing(model, adtype)
    ldf = Main.DynamicPPL.LogDensityFunction(model; adtype=adtype)
    vi = Main.DynamicPPL.VarInfo(Random.Xoshiro(11), model)
    theta0 = Main.DynamicPPL.link(vi, model)[:]
    return ldf, theta0
end

to_gpu(A) = GPU_OK ? Main.CUDA.CuArray(A) : A

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

function run_glm_sweep!(rows)
    for T in PRECISIONS, n in NS, k in KS
        println("\n=== GLM  T=$T  N=$n  K=$k ===")
        X_cpu, y_cpu = make_glm_data(T, n, k)
        X_gpu, y_gpu = (GPU_OK ? (to_gpu(X_cpu), to_gpu(y_cpu)) : (nothing, nothing))

        for (bname, adtype) in ad_backends()
            # --- PB, CPU ---
            pb_cpu = guarded("pb/cpu/$bname") do
                ldf, th = build_pb(pb_glm(X_cpu, y_cpu), T, adtype)
                median_gradient_ns(ldf, th; on_gpu=false)
            end

            # --- PB, GPU ---
            pb_gpu = if GPU_OK
                guarded("pb/gpu/$bname") do
                    ldf, th = build_pb(pb_glm(X_gpu, y_gpu), T, adtype)
                    median_gradient_ns(ldf, th; on_gpu=true)
                end
            else
                skipped("no functional GPU")
            end

            # --- Turing, CPU ---
            # Turing is Float64-internally regardless of input type, so the
            # Float32 row is recorded as a skip with that reason rather than
            # silently reporting a Float64 number in a Float32 column.
            tu_cpu = if !HAS_TURING
                skipped("Turing not loaded")
            elseif T !== Float64
                skipped("DynamicPPL promotes to Float64 internally")
            else
                guarded("turing/cpu/$bname") do
                    ldf, th = build_turing(turing_glm(X_cpu, y_cpu), adtype)
                    median_gradient_ns(ldf, th; on_gpu=false)
                end
            end

            # --- Turing, GPU ---
            # Attempted rather than assumed: if it fails, the recorded message
            # is the evidence for the claim that Turing cannot do this.
            tu_gpu = if !HAS_TURING
                skipped("Turing not loaded")
            elseif !GPU_OK
                skipped("no functional GPU")
            elseif T !== Float64
                skipped("DynamicPPL promotes to Float64 internally")
            else
                guarded("turing/gpu/$bname") do
                    ldf, th = build_turing(turing_glm(X_gpu, y_gpu), adtype)
                    median_gradient_ns(ldf, th; on_gpu=true)
                end
            end

            push!(rows, Dict(
                "model" => "glm", "precision" => string(T), "N" => n, "K" => k,
                "backend" => bname,
                "pb_cpu" => outcome_json(pb_cpu), "pb_gpu" => outcome_json(pb_gpu),
                "turing_cpu" => outcome_json(tu_cpu), "turing_gpu" => outcome_json(tu_gpu),
            ))
            report_row(bname, pb_cpu, pb_gpu, tu_cpu, tu_gpu)
        end
    end
end

# Bayesian neural network. `nhidden` is the size axis here: the parameter
# count grows quadratically in it, so a small range spans a wide range of K.
const BNN_NS = (1_000, 10_000)
const BNN_HIDDEN = (8, 32)
const BNN_NIN = 10

function run_bnn_sweep!(rows)
    for T in PRECISIONS, n in BNN_NS, nh in BNN_HIDDEN
        k = bnn_nweights(BNN_NIN, nh, 1)
        println("\n=== BNN  T=$T  N=$n  hidden=$nh  (K=$k) ===")
        X_cpu, y_cpu = make_bnn_data(T, n, BNN_NIN)
        X_gpu, y_gpu = (GPU_OK ? (to_gpu(X_cpu), to_gpu(y_cpu)) : (nothing, nothing))

        for (bname, adtype) in ad_backends()
            pb_cpu = guarded("pb/cpu/$bname") do
                ldf, th = build_pb(pb_bnn(X_cpu, y_cpu, nh), T, adtype)
                median_gradient_ns(ldf, th; on_gpu=false)
            end

            pb_gpu = if GPU_OK
                guarded("pb/gpu/$bname") do
                    ldf, th = build_pb(pb_bnn(X_gpu, y_gpu, nh), T, adtype)
                    median_gradient_ns(ldf, th; on_gpu=true)
                end
            else
                skipped("no functional GPU")
            end

            tu_cpu = if !HAS_TURING
                skipped("Turing not loaded")
            elseif T !== Float64
                skipped("DynamicPPL promotes to Float64 internally")
            else
                guarded("turing/cpu/$bname") do
                    ldf, th = build_turing(turing_bnn(X_cpu, y_cpu, nh), adtype)
                    median_gradient_ns(ldf, th; on_gpu=false)
                end
            end

            tu_gpu = if !HAS_TURING
                skipped("Turing not loaded")
            elseif !GPU_OK
                skipped("no functional GPU")
            elseif T !== Float64
                skipped("DynamicPPL promotes to Float64 internally")
            else
                guarded("turing/gpu/$bname") do
                    ldf, th = build_turing(turing_bnn(X_gpu, y_gpu, nh), adtype)
                    median_gradient_ns(ldf, th; on_gpu=true)
                end
            end

            push!(rows, Dict(
                "model" => "bnn", "precision" => string(T), "N" => n, "K" => k,
                "nhidden" => nh, "backend" => bname,
                "pb_cpu" => outcome_json(pb_cpu), "pb_gpu" => outcome_json(pb_gpu),
                "turing_cpu" => outcome_json(tu_cpu), "turing_gpu" => outcome_json(tu_gpu),
            ))
            report_row(bname, pb_cpu, pb_gpu, tu_cpu, tu_gpu)
        end
    end
end

# JSON has no Inf/NaN, and failed/skipped cells legitimately carry Inf — so
# the time becomes `null` and the STATUS and MESSAGE are kept alongside it.
# Preserving why a cell is empty is the difference between a result a reader
# can interpret and a hole they have to guess about.
function outcome_json(o::BackendOutcome)
    return Dict(
        "status" => String(o.status),
        "ns" => isfinite(o.time_ns) ? o.time_ns : nothing,
        "message" => o.message,
    )
end

fmt(o::BackendOutcome) = o.status === :ok ? string(round(o.time_ns / 1e6; digits=3), " ms") : String(o.status)

function report_row(bname, pb_cpu, pb_gpu, tu_cpu, tu_gpu)
    speedup = (pb_cpu.status === :ok && pb_gpu.status === :ok && pb_gpu.time_ns > 0) ?
        string(round(pb_cpu.time_ns / pb_gpu.time_ns; digits=2), "x") : "—"
    println("  $(rpad(bname, 12)) PB cpu=$(rpad(fmt(pb_cpu), 12)) gpu=$(rpad(fmt(pb_gpu), 12)) " *
            "speedup=$(rpad(speedup, 8)) | Turing cpu=$(rpad(fmt(tu_cpu), 12)) gpu=$(fmt(tu_gpu))")
end

function main()
    println("="^78)
    println("PracticalBayes GPU benchmark")
    println("  CUDA loaded:      ", HAS_CUDA)
    println("  GPU functional:   ", GPU_OK)
    if GPU_OK
        println("  device:           ", Main.CUDA.name(Main.CUDA.device()))
    end
    println("  Turing available: ", HAS_TURING)
    println("  Piste available:  ", HAS_PISTE)
    println("  Mooncake:         ", HAS_MOONCAKE, "   Enzyme: ", HAS_ENZYME)
    println("="^78)

    rows = Any[]
    # Each sweep is independently guarded: a catastrophic failure inside the
    # GLM sweep must not cost the BNN results, and vice versa.
    try
        run_glm_sweep!(rows)
    catch e
        println("GLM sweep aborted: ", sprint(showerror, e))
    end
    try
        run_bnn_sweep!(rows)
    catch e
        println("BNN sweep aborted: ", sprint(showerror, e))
    end

    resdir = joinpath(@__DIR__, "results")
    mkpath(resdir)
    payload = Dict(
        "meta" => Dict(
            "gpu_functional" => GPU_OK,
            "gpu_name" => GPU_OK ? Main.CUDA.name(Main.CUDA.device()) : nothing,
            "turing_available" => HAS_TURING,
            "julia_version" => string(VERSION),
            "n_values" => collect(NS), "k_values" => collect(KS),
            "bnn_n_values" => collect(BNN_NS), "bnn_hidden" => collect(BNN_HIDDEN),
            "precisions" => string.(PRECISIONS),
            "bench_samples" => BENCH_SAMPLES,
        ),
        "rows" => rows,
    )
    out = joinpath(resdir, "gpu_benchmark.json")
    open(out, "w") do io
        JSON3.pretty(io, payload)
    end
    println("\nSaved results to: ", out)
end

main()
