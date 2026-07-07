using ADTypes: ADTypes, AbstractADType
using LinearAlgebra: LinearAlgebra, Symmetric
using Distributions: Distributions, MvNormal
import DifferentiationInterface as DI

# ===========================================================================
# Point estimation (MAP / MLE) and the Laplace approximation, built directly
# on top of `LogDensityFunction`'s two ingredients — a scalar objective and
# its gradient over the flat unconstrained `θ` — so none of this needs any
# M2/M3 sampling machinery. MAP maximizes the joint log-density
# (`_logdensity_call`, logdensity.jl); MLE maximizes the likelihood term only
# (`_loglikelihood_call`); Laplace is the MAP point plus the Hessian of the
# joint log-density there (`DI.hessian`, same pattern as the existing
# gradient plumbing).
#
# This is a secondary feature — full Bayesian inference (NUTS via
# AbstractMCMC/AdvancedHMC) is the main target of this package, so the
# optimizer itself is NOT reimplemented here: these functions require
# `using Optimization` (plus a solver package, e.g. `OptimizationOptimJL`)
# and hand off to `ext/PracticalBayesOptimizationExt.jl`. There is
# deliberately no built-in fallback optimizer.
# ===========================================================================

"""
    PointEstimate

Result of `maximum_a_posteriori`/`maximum_likelihood`. `theta` is the
optimum in unconstrained space; `constrained` is `invlink(layout, theta)`
(with untracked sites included, since a point estimate is small enough that
there's no reason to hide any of it); `value` is the maximized objective
(logjoint for MAP, loglikelihood for MLE) at `theta`. `layout`/`store` are
carried along so the result can be fed straight into a fresh
`LogDensityFunction` for diagnostics, or into `laplace_approximation`
without rebuilding anything.
"""
struct PointEstimate{T<:Real,V<:AbstractVector{T},L<:Layout,S<:NamedTuple,C<:NamedTuple}
    theta::V
    value::T
    constrained::C
    layout::L
    store::S
    retcode::Symbol
end

"""
    LaplaceApproximation

A Gaussian approximation to the posterior, centered at a MAP estimate with
covariance `-inv(H)` where `H` is the Hessian of the joint log-density at
that point (both in UNCONSTRAINED space — the space where a Gaussian
approximation is actually reasonable; see `laplace_mvnormal` if you want a
`Distributions.MvNormal` built from this directly). `map` is the full
`PointEstimate` the approximation is built around.
"""
struct LaplaceApproximation{T<:Real,M<:AbstractMatrix{T},PE<:PointEstimate{T}}
    covariance::M
    hessian::M
    map::PE
end

"""
    laplace_mvnormal(la::LaplaceApproximation) -> Distributions.MvNormal

Convenience constructor for the unconstrained-space Gaussian approximation
`la` represents. Throws (via `Distributions.MvNormal`) if `la.covariance`
isn't positive-definite — a sign the MAP optimization didn't actually find a
local maximum (or the model is improper at that point), not a bug in this
function.
"""
laplace_mvnormal(la::LaplaceApproximation) = MvNormal(la.map.theta, Symmetric(la.covariance))

# Builds the layout/store shared by all three point-estimation entry points.
# `store` here is USER-supplied fixed values (e.g. nuisance latents held
# constant during optimization); its keys are exactly the names routed to
# the value-store instead of the flat vector, mirroring how `values=` works
# in `build_layout` directly. `untracked` is forwarded as-is (see layout.jl).
function _build_point_layout(model, store, untracked, rng, init)
    layout, θ0, store0 = build_layout(model; values=Tuple(keys(store)), untracked=untracked, rng=rng, init=init)
    return layout, θ0, merge(store0, store)  # user-supplied values win over trace defaults
end

"""
    _external_optimize(negf, negg!, θ0, alg; kwargs...) -> (θ_opt, minimized_value, retcode)

Implemented by `ext/PracticalBayesOptimizationExt.jl`, loaded automatically
once the caller has `using Optimization` (a `[weakdeps]` entry — see
Project.toml). `negf(θ)` is the NEGATED objective (Optimization.jl minimizes;
we maximize) and `negg!(g, θ)` mutates `g` in place with the negated
gradient at `θ`.

Implemented as a `Ref`-held function (`_EXTERNAL_OPTIMIZE_HOOK`, below)
rather than an ordinary generic function the extension adds a method to: a
method with this exact signature defined in both this package and the
extension would be the SAME method to Julia, not a new dispatch, and
overwriting a method during another module's precompilation is forbidden
("Method overwriting is not permitted during Module precompilation").
Assigning into a `Ref` in the extension's `__init__` sidesteps this
entirely — it's a plain value replacement, not a method redefinition.
"""
function _external_optimize(negf, negg!, θ0, alg; kwargs...)
    return _EXTERNAL_OPTIMIZE_HOOK[](negf, negg!, θ0, alg; kwargs...)
end

# Default hook: errors with instructions. Replaced by
# `ext/PracticalBayesOptimizationExt.jl`'s `__init__` when `using Optimization`
# is loaded — there is no other implementation, so point estimation always
# requires it.
const _EXTERNAL_OPTIMIZE_HOOK = Ref{Any}(
    (negf, negg!, θ0, alg; kwargs...) -> error(
        "PracticalBayes' point-estimation functions (maximum_a_posteriori/" *
        "maximum_likelihood/laplace_approximation) require Optimization.jl. Run " *
        "`using Optimization` plus a solver package (e.g. `using OptimizationOptimJL`) " *
        "and pass an algorithm via `optimizer=...` (e.g. `OptimizationOptimJL.BFGS()`).",
    ),
)

# Shared driver for MAP and MLE: prepares one DI gradient tape/config against
# `objective_call` (either `_logdensity_call` or `_loglikelihood_call`, both
# logdensity.jl) and hands off to `_external_optimize` (Optimization.jl
# extension).
function _optimize_point(objective_call, model, layout, store, θ0::AbstractVector, adtype, optimizer; kwargs...)
    prep = DI.prepare_gradient(objective_call, adtype, θ0, DI.Constant(model), DI.Constant(layout), DI.Constant(store))
    fg(θ) = DI.value_and_gradient(objective_call, prep, adtype, θ, DI.Constant(model), DI.Constant(layout), DI.Constant(store))

    # Optimization.jl (like most external optimizers) minimizes; we maximize,
    # so negate. `negg!` recomputes the value via `fg` too (and discards it)
    # rather than needing a separate gradient-only AD call — this runs once
    # per optimizer iteration for a one-off point estimate, not inside any
    # hot loop, so the extra scalar is not worth a second code path.
    negf(θ) = -fg(θ)[1]
    function negg!(g, θ)
        _, grad = fg(θ)
        g .= .-grad
        return g
    end
    θ_hat, negval, retcode = _external_optimize(negf, negg!, θ0, optimizer; kwargs...)
    return θ_hat, -negval, retcode
end

"""
    maximum_a_posteriori(model, optimizer; store=NamedTuple(), untracked=(), rng=Random.default_rng(),
                          init=NamedTuple(), adtype=ADTypes.AutoForwardDiff(), kwargs...) -> PointEstimate

Maximizes the joint log-density (`logprior + loglikelihood`) over the
model's unconstrained parameters — this is exactly `LogDensityFunction`'s
existing objective, so the only new code here is the optimizer call.

Requires `using Optimization` plus a solver package (e.g.
`OptimizationOptimJL`); `optimizer` is an Optimization.jl algorithm object,
e.g. `OptimizationOptimJL.BFGS()`.

- `store`: fixed values for any names that should be held constant during
  optimization (e.g. nuisance latents) instead of optimized over — same role
  as `build_layout`'s `values=`.
- `kwargs...`: forwarded to `Optimization.solve`.
"""
function maximum_a_posteriori(
    model,
    optimizer;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    kwargs...,
)
    layout, θ0, full_store = _build_point_layout(model, store, untracked, rng, init)
    θ_hat, val, retcode = _optimize_point(_logdensity_call, model, layout, full_store, θ0, adtype, optimizer; kwargs...)
    constrained = invlink(layout, θ_hat; include_untracked=true)
    return PointEstimate(θ_hat, val, constrained, layout, full_store, retcode)
end

"""
    maximum_likelihood(model, optimizer; store=NamedTuple(), untracked=(), rng=Random.default_rng(),
                        init=NamedTuple(), adtype=ADTypes.AutoForwardDiff(), kwargs...) -> PointEstimate

Maximizes the likelihood term ONLY (`loglikelihood_(acc)`, ignoring
`logprior`) over the model's unconstrained parameters — i.e. the classical
MLE, as opposed to `maximum_a_posteriori`'s MAP. Every other argument has the
same meaning as `maximum_a_posteriori`.

Priors are still evaluated during tracing (`build_layout` needs a concrete
initial value for every site) but never contribute to the optimized
objective or its gradient — `Accum`'s prior/likelihood split
(accumulator.jl) is what makes this a one-line change from the MAP path
(`_loglikelihood_call` vs `_logdensity_call`, both logdensity.jl).
"""
function maximum_likelihood(
    model,
    optimizer;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    kwargs...,
)
    layout, θ0, full_store = _build_point_layout(model, store, untracked, rng, init)
    θ_hat, val, retcode = _optimize_point(_loglikelihood_call, model, layout, full_store, θ0, adtype, optimizer; kwargs...)
    constrained = invlink(layout, θ_hat; include_untracked=true)
    return PointEstimate(θ_hat, val, constrained, layout, full_store, retcode)
end

"""
    laplace_approximation(model, optimizer; store=NamedTuple(), untracked=(), rng=Random.default_rng(),
                           init=NamedTuple(), adtype=ADTypes.AutoForwardDiff(), kwargs...) -> LaplaceApproximation

Finds the MAP estimate (via `maximum_a_posteriori`, same arguments) and then
computes the Hessian of the joint log-density there via `DI.hessian` — the
same `DifferentiationInterface` machinery `LogDensityFunction` uses for
gradients, just one order higher. The covariance of the Gaussian
approximation is `-inv(H)`, in the SAME unconstrained space `theta` lives in
(not the constrained/reported space — a Gaussian approximation on a
bounded/positive parameter's raw scale usually isn't meaningful).
"""
function laplace_approximation(
    model,
    optimizer;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    kwargs...,
)
    map_est = maximum_a_posteriori(
        model, optimizer; store=store, untracked=untracked, rng=rng, init=init, adtype=adtype, kwargs...
    )
    H = DI.hessian(
        _logdensity_call, adtype, map_est.theta, DI.Constant(model), DI.Constant(map_est.layout), DI.Constant(map_est.store)
    )
    Σ = Matrix(Symmetric(-inv(H)))
    return LaplaceApproximation(Σ, Matrix(H), map_est)
end
