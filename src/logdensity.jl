using LogDensityProblems: LogDensityProblems
using ADTypes: ADTypes, AbstractADType
import DifferentiationInterface as DI

"""
    LogDensityFunction(model, layout, store, adtype)

Implements the `LogDensityProblems` interface for a `Model` given a fixed
`Layout` (from `build_layout`) and constant `store` (latents / other Gibbs
blocks). `adtype === nothing` gives a `LogDensityOrder{0}` object (density
only); otherwise gradients are attached via `DifferentiationInterface`
directly (not `LogDensityProblemsAD`), matching current DynamicPPL practice.

Nothing here hardcodes `Float64`: `logdensity(f, θ)` computes in `eltype(θ)`,
so a `Vector{Float32}` runs in Float32 (SIMD-friendly) and a `CuVector`
computes on-device, provided the constituent `logpdf`/bijector calls used by
the model support that type (a Distributions.jl-level property, not one this
package restricts). `store` values are passed to the evaluator as-is and to
the AD backend as `DI.Constant`, so they are never differentiated and never
force a promotion of `θ`'s element type.
"""
struct LogDensityFunction{M<:Model,L<:Layout,S<:NamedTuple,AD,P}
    model::M
    layout::L
    store::S
    adtype::AD
    prep::P
end

"""
    LogDensityFunction(model, layout, store=NamedTuple(), adtype=nothing; θ0=zeros(Float64, layout.dim))

`θ0` is used only to fix the element type and shape that `DifferentiationInterface`
prepares its gradient tape/config for (`DI.prepare_gradient`); pass a vector of
the type you intend to sample with (e.g. `zeros(Float32, layout.dim)`) if it's
not `Float64`.
"""
function LogDensityFunction(model, layout, store=NamedTuple(), adtype=nothing; θ0=zeros(Float64, layout.dim))
    if adtype === nothing
        # order-0: density only, e.g. for a plain Metropolis-Hastings kernel
        # that never needs a gradient. `prep` is `nothing` and never touched.
        return LogDensityFunction(model, layout, store, nothing, nothing)
    end
    # `DI.prepare_gradient` traces/compiles the AD backend's tape or config
    # ONCE against `θ0`'s type and length; every later `logdensity_and_gradient`
    # call reuses `prep` instead of re-preparing, which matters a lot for
    # tape-based reverse-mode backends (ReverseDiff) where retracing per call
    # would be far more expensive than the gradient itself.
    #
    # `model`, `layout`, and `store` are wrapped in `DI.Constant(...)`: this
    # tells the AD backend "these arguments are non-differentiable, treat them
    # as compile-time constants" — critically, this is what keeps latent
    # values in `store` invisible to the gradient regardless of which AD
    # backend is used (ForwardDiff never even sees them promoted to Dual;
    # reverse-mode backends never build a tape node for them).
    prep = DI.prepare_gradient(_logdensity_call, adtype, θ0, DI.Constant(model), DI.Constant(layout), DI.Constant(store))
    return LogDensityFunction(model, layout, store, adtype, prep)
end

# The actual "run the model and read off the joint log-density" step, shared
# by both the density-only and density+gradient code paths below so there is
# exactly one place this logic lives (any bug fix here fixes both).
@inline function _logdensity_call(θ, model::Model, layout::Layout, store::NamedTuple)
    mode = EvalMode(layout, θ, store, model.conditioned)
    _, acc = evaluate(model, mode, Accum(zero(eltype(θ))))
    return logjoint(acc)
end

LogDensityProblems.dimension(f::LogDensityFunction) = f.layout.dim

# LogDensityProblems uses `capabilities` (dispatched on the TYPE, not the
# value, hence `::Type{<:LogDensityFunction{...}}`) to decide whether a
# sampler needs to wrap this object in its own AD before it can get
# gradients. Since we've already attached gradients ourselves whenever
# `adtype !== nothing`, we report order 1 in that case so AdvancedHMC uses
# our `logdensity_and_gradient` method directly instead of re-differentiating.
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction{<:Any,<:Any,<:Any,AD}}) where {AD}
    return AD === Nothing ? LogDensityProblems.LogDensityOrder{0}() : LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity(f::LogDensityFunction, θ::AbstractVector)
    return _logdensity_call(θ, f.model, f.layout, f.store)
end

function LogDensityProblems.logdensity_and_gradient(f::LogDensityFunction, θ::AbstractVector)
    # `DI.value_and_gradient` reuses `f.prep` (built once at construction, see
    # above) and returns `(value, gradient)` — the exact pair
    # `LogDensityProblems.logdensity_and_gradient` is expected to return, so
    # there's nothing else to do here beyond passing the constants through
    # again (they must match what `prep` was built with).
    return DI.value_and_gradient(
        _logdensity_call, f.prep, f.adtype, θ, DI.Constant(f.model), DI.Constant(f.layout), DI.Constant(f.store)
    )
end
