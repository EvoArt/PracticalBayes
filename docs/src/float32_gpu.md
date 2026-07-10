# Float32 and GPU usage

PracticalBayes is Float32-first: the parameter vector `θ` can be `Float32`
end-to-end (layout, log-density, gradient, HMC state), which roughly halves
memory traffic on large models and is a prerequisite for GPU data. This page
covers the two together since GPU support builds directly on the Float32
path — a model that isn't correctly Float32-safe won't be GPU-safe either.

## Float32 parameters

`build_layout(model; T=Float32)` makes the *parameter* vector `θ0` Float32,
but that alone is not enough to get a genuinely end-to-end Float32
computation — silent promotion back to Float64 is the main pitfall:

- **Distribution literals need `f0` suffixes.** `Exponential(1.0f0)`, not
  `Exponential(1)` — a bare `1` is an `Int`/`Float64` literal that promotes
  any `Float32` computation touching it back to `Float64`.
- **Vector-valued prior means need `paramtype`, not `zeros(k)`.**
  `zeros(paramtype(__mode__), k)`, not bare `zeros(k)` (which defaults to
  `Float64`) — see `PracticalBayes.paramtype` for how the mode tracks the "current"
  element type. This is easy to get wrong in a way that still runs and still
  produces correct numbers, just silently back in Float64.
- **Data (`X`/`y`) can stay `Float64`** without breaking Float32 propagation
  through the gradient or costing performance — confirmed directly. Only
  parameters and distribution literals need the `f0`/`paramtype` treatment.
- **`AdvancedHMC.NUTS(δ)` fixes its step-size type to `typeof(δ)`.** Passing
  a bare `0.8` (`Float64`) together with a `Float32` `θ0` fails during
  step-size adaptation — use `NUTS(eltype(θ0)(0.8))` or, equivalently,
  `NUTS(0.8f0)`.

```@example
using PracticalBayes
using Distributions
using ADTypes
using LogDensityProblems

PracticalBayes.@model function f32_demo(y)
    pT = PracticalBayes.paramtype(__mode__)
    μ ~ Normal(zero(pT), one(pT))
    σ ~ Exponential(one(pT))
    y .~ Normal.(μ, σ)
end

y32 = Float32.(randn(20) .+ 2)
m = f32_demo(y32)
layout, θ0, store0 = PracticalBayes.build_layout(m; T=Float32)
eltype(θ0)
```

```@example
using PracticalBayes
using Distributions
using ADTypes
using LogDensityProblems

PracticalBayes.@model function f32_demo2(y)
    pT = PracticalBayes.paramtype(__mode__)
    μ ~ Normal(zero(pT), one(pT))
    σ ~ Exponential(one(pT))
    y .~ Normal.(μ, σ)
end

m = f32_demo2(Float32.(randn(20) .+ 2))
layout, θ0, store0 = PracticalBayes.build_layout(m; T=Float32)
ldf = PracticalBayes.LogDensityFunction(m, layout, store0, ADTypes.AutoForwardDiff(); θ0=θ0)
val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
eltype(grad)
```

## GPU

**Scope**: the parameter vector `θ` stays on the CPU; *data* (`X`, `y`) can
live on the GPU (`CUDA.CuArray`). This is deliberately narrower than a
fully-device `θ` (tracked as an open M6 item) — it targets the common case
where the model's per-observation work (a matrix-multiply, a vectorized
observe) is the expensive part, not the parameter count.

**The framework guarantee**: no framework-introduced scalar indexing.
Parameter reads in the hot path are `view(θ, range)`, never `θ[i]` one
element at a time, and the compiler never indexes into model data on your
behalf. This means `CUDA.allowscalar(false)` — CUDA.jl's strict mode that
throws on any scalar GPU-array indexing — is safe to enable, and doing so is
the recommended way to catch accidental scalar indexing in your *own* model
code early.

**What this means for model authors**: use array-variate observes so the
observe statement itself is a single vectorized GPU operation rather than a
loop:

```julia
PracticalBayes.@model function gpu_regression(X, y)
    pT = PracticalBayes.paramtype(__mode__)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    sigma ~ Distributions.Exponential(one(pT))
    eta = X * beta                                    # one GPU matmul
    y ~ Distributions.MvNormal(eta, sigma^2 * LinearAlgebra.I)  # one vectorized observe
end
```

Then build `X`/`y` as `CUDA.CuArray`s before constructing the model — `θ0`
itself stays a plain CPU `Vector`:

```julia
import CUDA
CUDA.allowscalar(false)
X_gpu, y_gpu = CUDA.CuArray(X_cpu), CUDA.CuArray(y_cpu)
m = gpu_regression(X_gpu, y_gpu)
layout, θ0, store0 = PracticalBayes.build_layout(m)  # θ0 is a CPU Vector
ldf = PracticalBayes.LogDensityFunction(m, layout, store0, ADTypes.AutoForwardDiff(); θ0=θ0)
LogDensityProblems.logdensity_and_gradient(ldf, θ0)  # matmul/observe run on GPU, gradient assembled on CPU
```

This exact pattern is exercised by `test/gpu/cuda.jl` (gated on
`CUDA.functional()`) and benchmarked (CPU vs GPU gradient time across an N
sweep) by `benchmarks/gpu/gpu_sweep.jl`, meant to be run by hand on a
machine with a real, working GPU — GitHub's standard hosted CI runners have
none, so this isn't part of the always-on CI benchmark sweep.
