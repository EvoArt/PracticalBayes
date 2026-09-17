#!/bin/bash -l
# Run the GPU TEST suite on a real GPU — the "does it actually work" job,
# distinct from the benchmark job next to it.
#
#   sbatch benchmarks/gpu/slurm/gpu_test.sbatch
#
# This is the more important of the two jobs. `test/gpu/cuda.jl` and
# `test/gpu/cuda_backends.jl` both skip themselves cleanly when
# `CUDA.functional()` is false, which means a green test run on a machine
# without a GPU proves nothing about the GPU path at all. Only a run here, on
# a node that really has a device, converts those files from
# code-reviewed-correct into verified.
#
# Short (30 min) and small (one GPU, modest memory) because it is a
# correctness check, not a performance measurement — so it queues quickly and
# can be re-run cheaply after any change to the hot path.

#SBATCH --job-name=pb_gpu_test
#SBATCH --partition=gpuq                 # ADJUST: your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=pb_gpu_test-%j.out
#SBATCH --error=pb_gpu_test-%j.out

set -u

echo "host:    $(hostname)"
echo "gpus:    ${CUDA_VISIBLE_DEVICES:-none}"
echo "started: $(date)"

module load cuda12.2/toolkit/12.2.2 2>/dev/null || echo "NOTE: no CUDA module loaded"
nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1 || echo "NOTE: nvidia-smi unavailable"

REPO="${PB_REPO:-$HOME/pb_src}"
cd "$REPO" || { echo "FATAL: repo not found at $REPO (set PB_REPO)"; exit 1; }

# 8 threads on purpose: `test/gpu/cuda.jl`'s second half is a MULTI-THREADED
# chain-independence test that silently reduces its chain count to
# `Threads.nthreads()`. With one thread it skips itself, and the run looks
# green while having tested nothing.
export JULIA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# `CUDA.functional()` is the gate every GPU test sits behind, so report it
# FIRST. If this prints false, every GPU testset below skipped, and a "passed"
# result means nothing — better to see that plainly at the top of the log than
# to infer it from an absence of failures.
julia --project=test/gpu_env -e '
try
    using CUDA
    println("CUDA.functional() = ", CUDA.functional())
    CUDA.functional() && println("device: ", CUDA.name(CUDA.device()))
catch e
    println("CUDA unavailable: ", sprint(showerror, e))
end
'

julia --project=test/gpu_env test/gpu/runtests_gpu.jl
status=$?

echo "finished: $(date)  (julia exit ${status})"
exit $status
