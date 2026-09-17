# Running on a GPU cluster

This page covers running PracticalBayes' GPU path on a Slurm cluster: what to
check before you start, how to submit the test and benchmark jobs, and the
failure modes that are worth recognising immediately rather than debugging
from scratch.

See [Float32 and GPU usage](float32_gpu.md) for the GPU *programming* model —
what lives on the device, how to write a model that stays off the scalar
indexing path, and why `θ` stays on the host.

## Check the driver first

Almost every "CUDA doesn't work" problem on a shared cluster is one of two
version mismatches, and both are visible in under a minute:

```console
$ nvidia-smi --query-gpu=name,driver_version --format=csv
name, driver_version
Tesla V100-PCIE-32GB, 418.39
```

The driver version caps the CUDA toolkit you can use, and **CUDA.jl cannot
work around a driver that is too old**. Driver 418.x supports CUDA up to
10.1; current CUDA.jl requires a driver supporting **10.2 or newer**. On such
a node you will get, no matter how CUDA.jl is configured:

```
┌ Error: This version of CUDA.jl requires an NVIDIA driver for CUDA 10.2 or
│ higher (yours only supports up to CUDA 10.1.0)
```

followed by `CUDA.functional() == false`. Neither pointing CUDA.jl at a newer
locally-installed toolkit (`CUDA.set_runtime_version!(local_toolkit=true)`)
nor loading a newer toolkit module fixes this: the *toolkit* is not the
constraint, the *driver* is. The options are a node with a newer driver, a
driver upgrade by the cluster administrators, or an old enough CUDA.jl to
match (CUDA.jl 3.x supports CUDA 10.1, but is far behind current releases and
may not work with current PracticalBayes dependencies).

A useful rule when reading cluster docs: the CUDA module list tells you what
toolkits are *installed*, which is not the same question. Always check
`nvidia-smi` on the GPU node itself, from inside a job — login nodes usually
have no GPU and no `nvidia-smi` at all.

## Submitting the jobs

Two batch scripts live in `benchmarks/gpu/slurm/`. Both take the repository
location from `PB_REPO` (default `$HOME/pb_src`) and request **one** GPU:

```console
$ sbatch benchmarks/gpu/slurm/gpu_test.sbatch       # correctness, ~30 min
$ sbatch benchmarks/gpu/slurm/gpu_benchmark.sbatch  # timings, up to 8 h
```

Adjust the two marked lines in each (`--partition` and the `module load`) for
your cluster. Run the test job first: there is no point measuring how fast a
GPU path is before knowing it is correct.

## Reading the test output

The single most important thing about the GPU test suite is that **it skips
itself when there is no GPU**. Every testset is gated on `CUDA.functional()`,
which is what lets the same suite run in CI on GPU-less runners — but it also
means a green run proves nothing on its own.

`test/gpu/runtests_gpu.jl` therefore states plainly what happened:

```
GPU verification: REAL — the tests above ran against Tesla V100-PCIE-32GB.
```

or

```
GPU verification: NONE — the tests above skipped. Re-run on a node with a
working GPU.
```

Check that line before believing a passing result.

## Requesting GPUs politely

The benchmark scripts request `--gres=gpu:1` even on nodes that have two
devices. This is deliberate: the benchmark measures single-device gradient
time, so a second GPU would sit idle while blocking another user. On a
cluster with only a handful of GPU nodes — which is the common case — taking
the whole queue for a benchmark sweep is antisocial and rarely produces a
better measurement.

The benchmark job does ask for `--exclusive`, because a co-tenant process
competing for memory bandwidth silently corrupts timings. That trade (longer
queue wait, trustworthy numbers) is worth making for a benchmark and not for
the test job, which does not request it.

## When a backend fails

Both the GPU tests and the GPU benchmark treat an AD backend failing as a
**result to record, not an error to crash on**. A backend that cannot handle
`CuArray` data is reported and skipped; the remaining backends still run.

In the benchmark's JSON output every cell carries a status and, on failure,
the error message:

```json
"pb_gpu": { "status": "failed", "ns": null,
            "message": "Scalar indexing is disallowed..." }
```

This matters on a cluster, where a crash means re-queueing and waiting for a
GPU to free up again. It also preserves *why* a cell is empty, which is the
difference between a result you can interpret and a hole you have to guess
about.

The one thing that is still a hard failure is a backend that runs and returns
the **wrong** answer: `test/gpu/cuda_backends.jl` checks every working GPU
backend against a CPU reference gradient, because a silently incorrect
gradient is far worse than one that throws.
