#!/bin/bash -l
# CPU-vs-GPU benchmark sweep, as a Slurm batch job.
#
# Written for Athena (University of Exeter), but nothing here is
# Athena-specific beyond the module name and the partition: any Slurm cluster
# with an NVIDIA GPU and a Julia install should run it after adjusting the two
# marked lines.
#
#   sbatch benchmarks/gpu/slurm/gpu_benchmark.sbatch
#
# ONE GPU, ONE JOB. `--gres=gpu:1` deliberately requests a single device even
# on nodes that hold two: this benchmark measures single-device gradient time,
# so a second GPU would sit idle while blocking another user. Benchmarks must
# also never share a node with other work — a co-tenant competing for memory
# bandwidth silently corrupts every timing — which is what `--exclusive`
# buys, at the cost of queueing longer.

#SBATCH --job-name=pb_gpu_bench
#SBATCH --partition=gpuq                 # ADJUST: your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=pb_gpu_bench-%j.out
#SBATCH --error=pb_gpu_bench-%j.out

set -u

echo "host:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-none}"
echo "gpus:    ${CUDA_VISIBLE_DEVICES:-none}"
echo "started: $(date)"

# ADJUST: your cluster's CUDA module. The toolkit must be compatible with the
# NODE's driver — `nvidia-smi` reports the maximum CUDA version the driver
# supports, and a toolkit newer than that will leave `CUDA.functional()`
# false no matter what else is configured.
module load cuda12.2/toolkit/12.2.2 2>/dev/null || echo "NOTE: no CUDA module loaded; relying on CUDA.jl's own artifacts"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "NOTE: nvidia-smi unavailable"

REPO="${PB_REPO:-$HOME/pb_src}"
cd "$REPO" || { echo "FATAL: repo not found at $REPO (set PB_REPO)"; exit 1; }

# Pin BLAS threads to the allocation. Left unset, OpenBLAS starts one thread
# per CORE ON THE NODE rather than per core in the allocation — on a 64-core
# GPU node that is heavy oversubscription, and it inflates the CPU baseline
# that every GPU speedup in this sweep is measured against.
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# First run on a new cluster needs the environment built; afterwards this is
# skipped, because re-resolving at startup is both slow on a shared filesystem
# and a known crash trigger (see benchmarks/sweep.jl's own note).
export PB_GPU_INSTANTIATE="${PB_GPU_INSTANTIATE:-0}"

julia --project=benchmarks/gpu benchmarks/gpu/gpu_benchmark.jl
status=$?

echo "finished: $(date)  (julia exit ${status})"

# The JSON is the deliverable, so say plainly whether it exists. A job that
# exits 0 having written nothing is the failure mode worth catching here.
RESULT="$REPO/benchmarks/gpu/results/gpu_benchmark.json"
if [ -f "$RESULT" ]; then
    echo "RESULT: $RESULT ($(wc -c < "$RESULT") bytes)"
else
    echo "RESULT: MISSING — the sweep did not write its JSON"
fi

exit $status
