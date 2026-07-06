using ADTypes: ADTypes, AbstractADType
using LinearAlgebra: LinearAlgebra, norm, dot, Symmetric
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
    iterations::Int
    converged::Bool
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
    _external_optimize(negf, negg!, θ0, alg; kwargs...) -> (θ_opt, minimized_value)

Extension point, implemented by `ext/PracticalBayesOptimizationExt.jl` when
the user has `using Optimization` loaded (Optimization.jl is a weak
dependency — see `Project.toml`). `negf(θ)` is the NEGATED objective (since
Optimization.jl minimizes, but we maximize) and `negg!(g, θ)` mutates `g` in
place with the negated gradient at `θ`. Only reached when the caller passes
a non-`nothing` `optimizer`; the default path (`optimizer=nothing`) always
uses the hand-rolled `_lbfgs_maximize` below and never touches this.
"""
function _external_optimize(negf, negg!, θ0, alg; kwargs...)
    return error(
        "optimizer `$(alg)` was passed to a PracticalBayes point-estimation " *
        "function, but the Optimization.jl integration isn't loaded. Run " *
        "`using Optimization` (plus whichever solver package `$(typeof(alg))` " *
        "comes from, e.g. `OptimizationOptimJL`) before passing a non-default " *
        "`optimizer`, or omit `optimizer` to use the built-in L-BFGS.",
    )
end

# Shared driver for MAP and MLE: prepares one DI gradient tape/config against
# `objective_call` (either `_logdensity_call` or `_loglikelihood_call`, both
# logdensity.jl) and runs either the hand-rolled optimizer (default) or hands
# off to `_external_optimize` (Optimization.jl extension) when the caller
# supplied one.
function _optimize_point(objective_call, model, layout, store, θ0::AbstractVector, adtype, optimizer; kwargs...)
    prep = DI.prepare_gradient(objective_call, adtype, θ0, DI.Constant(model), DI.Constant(layout), DI.Constant(store))
    fg(θ) = DI.value_and_gradient(objective_call, prep, adtype, θ, DI.Constant(model), DI.Constant(layout), DI.Constant(store))

    if optimizer === nothing
        res = _lbfgs_maximize(fg, θ0; kwargs...)
        return res.theta, res.value, res.iterations, res.converged
    end

    # Most external optimizers (Optimization.jl included) minimize; we
    # maximize, so negate. `negg!` recomputes the value via `fg` too (and
    # discards it) rather than needing a separate gradient-only AD call —
    # this runs once per optimizer iteration for a one-off point estimate,
    # not inside any hot loop, so the extra scalar is not worth a second
    # code path.
    negf(θ) = -fg(θ)[1]
    function negg!(g, θ)
        _, grad = fg(θ)
        g .= .-grad
        return g
    end
    θ_hat, negval = _external_optimize(negf, negg!, θ0, optimizer; kwargs...)
    return θ_hat, -negval, -1, true
end

"""
    maximum_a_posteriori(model; store=NamedTuple(), untracked=(), rng=Random.default_rng(),
                          init=NamedTuple(), adtype=ADTypes.AutoForwardDiff(), optimizer=nothing,
                          kwargs...) -> PointEstimate

Maximizes the joint log-density (`logprior + loglikelihood`) over the
model's unconstrained parameters — this is exactly `LogDensityFunction`'s
existing objective, so the only new code here is the optimizer loop.

- `store`: fixed values for any names that should be held constant during
  optimization (e.g. nuisance latents) instead of optimized over — same role
  as `build_layout`'s `values=`.
- `optimizer`: `nothing` (default) uses the hand-rolled L-BFGS
  (`_lbfgs_maximize`); pass an Optimization.jl algorithm object (e.g.
  `OptimizationOptimJL.BFGS()`) to use that instead — requires `using
  Optimization` (see `ext/PracticalBayesOptimizationExt.jl`).
- `kwargs...`: forwarded to whichever optimizer runs (`maxiter`/`g_tol`/`m`
  for the built-in L-BFGS; solver-specific keywords for Optimization.jl).
"""
function maximum_a_posteriori(
    model;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    optimizer=nothing,
    kwargs...,
)
    layout, θ0, full_store = _build_point_layout(model, store, untracked, rng, init)
    θ_hat, val, iters, converged = _optimize_point(_logdensity_call, model, layout, full_store, θ0, adtype, optimizer; kwargs...)
    constrained = invlink(layout, θ_hat; include_untracked=true)
    return PointEstimate(θ_hat, val, constrained, layout, full_store, iters, converged)
end

"""
    maximum_likelihood(model; store=NamedTuple(), untracked=(), rng=Random.default_rng(),
                        init=NamedTuple(), adtype=ADTypes.AutoForwardDiff(), optimizer=nothing,
                        kwargs...) -> PointEstimate

Maximizes the likelihood term ONLY (`loglikelihood_(acc)`, ignoring
`logprior`) over the model's unconstrained parameters — i.e. the classical
MLE, as opposed to `maximum_a_posteriori`'s MAP. Every other keyword has the
same meaning as `maximum_a_posteriori`.

Priors are still evaluated during tracing (`build_layout` needs a concrete
initial value for every site) but never contribute to the optimized
objective or its gradient — `Accum`'s prior/likelihood split
(accumulator.jl) is what makes this a one-line change from the MAP path
(`_loglikelihood_call` vs `_logdensity_call`, both logdensity.jl).
"""
function maximum_likelihood(
    model;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    optimizer=nothing,
    kwargs...,
)
    layout, θ0, full_store = _build_point_layout(model, store, untracked, rng, init)
    θ_hat, val, iters, converged = _optimize_point(_loglikelihood_call, model, layout, full_store, θ0, adtype, optimizer; kwargs...)
    constrained = invlink(layout, θ_hat; include_untracked=true)
    return PointEstimate(θ_hat, val, constrained, layout, full_store, iters, converged)
end

"""
    laplace_approximation(model; store=NamedTuple(), untracked=(), rng=Random.default_rng(),
                           init=NamedTuple(), adtype=ADTypes.AutoForwardDiff(), optimizer=nothing,
                           kwargs...) -> LaplaceApproximation

Finds the MAP estimate (via `maximum_a_posteriori`, same keywords) and then
computes the Hessian of the joint log-density there via `DI.hessian` — the
same `DifferentiationInterface` machinery `LogDensityFunction` uses for
gradients, just one order higher. The covariance of the Gaussian
approximation is `-inv(H)`, in the SAME unconstrained space `theta` lives in
(not the constrained/reported space — a Gaussian approximation on a
bounded/positive parameter's raw scale usually isn't meaningful).
"""
function laplace_approximation(
    model;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    optimizer=nothing,
    kwargs...,
)
    map_est = maximum_a_posteriori(
        model; store=store, untracked=untracked, rng=rng, init=init, adtype=adtype, optimizer=optimizer, kwargs...
    )
    H = DI.hessian(
        _logdensity_call, adtype, map_est.theta, DI.Constant(model), DI.Constant(map_est.layout), DI.Constant(map_est.store)
    )
    Σ = Matrix(Symmetric(-inv(H)))
    return LaplaceApproximation(Σ, Matrix(H), map_est)
end

# ===========================================================================
# Hand-rolled L-BFGS, the default optimizer (no dependency beyond LinearAlgebra,
# already a dep). Maximizes `fg(θ) -> (value, gradient)` via the standard
# two-loop recursion for the search direction plus an Armijo backtracking
# line search — textbook L-BFGS (Nocedal & Wright), just phrased as ascent
# (search direction ≈ +H*gradient) instead of the usual descent phrasing,
# since we maximize a log-density rather than minimize a loss.
# ===========================================================================

"""
    _lbfgs_maximize(fg, θ0; maxiter=500, g_tol=1e-8, m=10) -> (theta=..., value=..., gradient=..., iterations=..., converged=...)

`fg(θ) -> (value, gradient)` is the function to MAXIMIZE. `m` is the L-BFGS
history length (number of past (s, y) pairs kept for the two-loop
recursion). Stops when `norm(gradient) < g_tol` (`converged=true`) or after
`maxiter` iterations, or if the backtracking line search fails to find an
improving step (`converged=false` in both of the latter cases — the caller
gets the best point found either way).
"""
function _lbfgs_maximize(fg, θ0::AbstractVector{T}; maxiter=500, g_tol=1e-8, m=10) where {T<:Real}
    θ = copy(θ0)
    f, g = fg(θ)
    s_hist = Vector{T}[]
    y_hist = Vector{T}[]
    rho_hist = T[]
    iter = 0
    converged = false

    while iter < maxiter
        if norm(g) < g_tol
            converged = true
            break
        end

        # Two-loop recursion: builds the search direction `d ≈ H*g` from the
        # last `m` (s, y) pairs without ever forming the (dense) inverse
        # Hessian approximation `H` itself.
        q = copy(g)
        k = length(s_hist)
        alphas = Vector{T}(undef, k)
        for i in k:-1:1
            alphas[i] = rho_hist[i] * dot(s_hist[i], q)
            q .-= alphas[i] .* y_hist[i]
        end
        gamma = k == 0 ? one(T) : dot(s_hist[k], y_hist[k]) / dot(y_hist[k], y_hist[k])
        d = gamma .* q
        for i in 1:k
            beta = rho_hist[i] * dot(y_hist[i], d)
            d .+= (alphas[i] - beta) .* s_hist[i]
        end

        # Armijo backtracking line search along the ascent direction `d`;
        # `dot(g, d)` is the directional derivative at the current point
        # (positive here since `d` is an ascent direction by construction).
        step = one(T)
        c1 = T(1e-4)
        directional = dot(g, d)
        θ_new, f_new, g_new = θ, f, g
        improved = false
        for _ in 1:50
            θ_new = θ .+ step .* d
            f_new, g_new = fg(θ_new)
            if isfinite(f_new) && f_new >= f + c1 * step * directional
                improved = true
                break
            end
            step *= T(0.5)
        end
        improved || break  # line search failed to find any improving step; stop here

        s = θ_new .- θ
        y = g_new .- g
        sy = dot(s, y)
        if sy > 1e-10  # curvature condition; skip the update rather than corrupt the approximation
            push!(s_hist, s)
            push!(y_hist, y)
            push!(rho_hist, one(T) / sy)
            if length(s_hist) > m
                popfirst!(s_hist)
                popfirst!(y_hist)
                popfirst!(rho_hist)
            end
        end
        θ, f, g = θ_new, f_new, g_new
        iter += 1
    end

    return (theta=θ, value=f, gradient=g, iterations=iter, converged=converged)
end
