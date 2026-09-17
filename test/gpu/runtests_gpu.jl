# GPU test entry point — runs ONLY the GPU test files.
#
# Separate from `test/runtests.jl` on purpose. The full suite pulls in Turing
# (a large, slow-to-precompile dependency tree) and takes minutes; on a
# cluster that time is spent inside a GPU allocation, holding a scarce device
# while running tests that have nothing to do with it. This entry point runs
# the GPU files alone, so a GPU reservation is spent on GPU work.
#
# It also makes the gate LOUD. Every GPU testset skips itself when
# `CUDA.functional()` is false, so a suite that "passes" on a GPU-less machine
# has proven nothing whatsoever about the GPU path. This file therefore prints
# the CUDA status up front and, at the end, states explicitly whether the run
# constitutes real GPU verification — because "0 failures" and "0 tests that
# touched a GPU" look identical in a Julia test summary otherwise.
#
# Usage:
#     julia --project=test/gpu_env test/gpu/runtests_gpu.jl

using Test
using PracticalBayes

const CUDA_INSTALLED = !isnothing(Base.find_package("CUDA"))

const CUDA_WORKS = if CUDA_INSTALLED
    try
        @eval import CUDA
        CUDA.functional()
    catch e
        println("CUDA present but unusable: ", sprint(showerror, e))
        false
    end
else
    false
end

println("="^70)
println("PracticalBayes GPU test run")
println("  CUDA installed:      ", CUDA_INSTALLED)
println("  CUDA.functional():   ", CUDA_WORKS)
if CUDA_WORKS
    println("  device:              ", CUDA.name(CUDA.device()))
    println("  capability:          ", CUDA.capability(CUDA.device()))
    try
        println("  driver / runtime:    ", CUDA.driver_version(), " / ", CUDA.runtime_version())
    catch
        # Older/newer CUDA.jl spell these differently; not worth failing over.
    end
end
println("="^70)

if !CUDA_WORKS
    println()
    println("!! NO FUNCTIONAL GPU — every GPU testset below will SKIP.")
    println("!! A passing result from this run does NOT verify the GPU path.")
    println()
end

@testset "PracticalBayes GPU" begin
    include("cuda.jl")
    include("cuda_backends.jl")
end

println()
if CUDA_WORKS
    println("GPU verification: REAL — the tests above ran against ", CUDA.name(CUDA.device()), ".")
else
    println("GPU verification: NONE — the tests above skipped. Re-run on a node with a working GPU.")
end
