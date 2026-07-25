using AdvancedHMC:
    AdvancedHMC,
    AbstractHMCSampler,
    HMCKernel,
    Trajectory,
    EndPointTS,
    FixedNSteps,
    Leapfrog,
    Hamiltonian
using AdvancedHMC.Adaptation: StanHMCAdaptor, MassMatrixAdaptor, StepSizeAdaptor

"""
    AdaptiveHMC(δ=0.8; n_leapfrog=10, kwargs...)

A fixed-trajectory-length HMC sampler that **adapts its mass matrix and step size
during warm-up**, exactly the way NUTS does, and then holds both fixed for the
sampling phase.

This is the sibling of [`NUTSthenHMC`](@ref): both give you a plain HMC sampler
whose per-iteration cost is a constant `n_leapfrog` leapfrog steps, with a
NUTS-quality metric and step size underneath. The difference is *which* sampler
runs the warm-up. `NUTSthenHMC` uses NUTS' tree-doubling to also *choose*
`n_leapfrog` for you, at the cost of an unpredictable warm-up. `AdaptiveHMC`
never doubles a tree at all — every iteration, warm-up included, is exactly
`n_leapfrog` leapfrog steps — so you must supply the trajectory length yourself,
but the whole run has a flat, predictable cost. Reach for this when you already
know a good `n_leapfrog` (e.g. from a prior `NUTSthenHMC` run, or a `Gibbs` block
you want to keep cheap and constant-time).

Adaptation is AdvancedHMC's `StanHMCAdaptor` — Welford mass-matrix estimation
plus Nesterov dual-averaging on the step size targeting acceptance rate `δ` —
identical to what `NUTS(δ)` builds. It runs for the first `n_adapts` iterations
and then switches off, leaving the learned diagonal metric and step size in
place. (Plain `AdvancedHMC.HMC` adapts *nothing* — its `make_adaptor` returns
`NoAdaptation()` — which is the gap this sampler fills.)

`metric` (`:diagonal`/`:dense`/`:unit` or an `AbstractMetric`), `integrator`,
and `init_ϵ` mean what they do for `AdvancedHMC.NUTS`; a `:diagonal` metric and a
step-size search are the defaults. As with every HMC sampler here, `n_adapts` is
passed at the `sample` call, and the warm-up draws stay in the chain unless you
slice them off:

    chn = sample(model, AdaptiveHMC(0.8; n_leapfrog=16), 2000; n_adapts=1000)
"""
struct AdaptiveHMC{T<:Real,I,M} <: AbstractHMCSampler
    "Target acceptance rate for step-size dual averaging."
    δ::T
    "Fixed number of leapfrog steps per iteration, in both warm-up and sampling."
    n_leapfrog::Int
    "Choice of integrator (`Symbol` or `AbstractIntegrator`); `:leapfrog` searches for ϵ."
    integrator::I
    "Choice of initial metric (`Symbol` for auto-init, or an `AbstractMetric`)."
    metric::M
end

function AdaptiveHMC(δ=0.8; n_leapfrog::Int=10, integrator=:leapfrog, metric=:diagonal)
    T = AdvancedHMC.determine_sampler_eltype(δ, integrator, metric)
    return AdaptiveHMC(T(δ), n_leapfrog, integrator, metric)
end

# Unlike `NUTSthenHMC`, this sampler needs no wrapper state and no `step` methods
# of its own: the kernel is a fixed-`n_leapfrog` leapfrog trajectory in BOTH
# phases, so AdvancedHMC's own generic `step` already does the right thing — it
# adapts while `i <= n_adapts` and freezes afterwards. We only have to fill in
# the four `make_*` hooks its initializer calls, so that (a) the kernel is
# fixed-step HMC rather than NUTS' tree sampler, and (b) the adaptor is the same
# `StanHMCAdaptor` NUTS uses, rather than `HMC`'s `NoAdaptation`.

AdvancedHMC.sampler_eltype(::AdaptiveHMC{T}) where {T} = T

# Metric and step-size handling are identical to any symbol-configured HMC
# sampler; AdvancedHMC's generic `make_metric`/`make_step_size` already dispatch
# on `spl.metric`/`spl.integrator`, so we lean on them unchanged. (No method
# needed here — the fallbacks in abstractmcmc.jl cover `AbstractHMCSampler`.)

# Fixed-length leapfrog trajectory, end-point sample — the same kernel
# `NUTSthenHMC` switches to after warm-up, here used from the first iteration.
function AdvancedHMC.make_kernel(spl::AdaptiveHMC, integrator::AdvancedHMC.AbstractIntegrator)
    return HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(spl.n_leapfrog)))
end

# The piece plain `HMC` omits: a real Stan-style adaptor so warm-up learns the
# mass matrix (Welford) and the step size (dual averaging at target `δ`). The
# metric argument fixes what kind of mass matrix is estimated, so a `:dense`
# request adapts a full covariance, `:diagonal` a diagonal, `:unit` none.
function AdvancedHMC.make_adaptor(
    spl::AdaptiveHMC, metric::AdvancedHMC.AbstractMetric, integrator::AdvancedHMC.AbstractIntegrator
)
    return StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(spl.δ, integrator))
end
