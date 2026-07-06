module PracticalBayesOptimizationExt

# Loaded automatically (Julia's package-extension mechanism, no Requires.jl
# needed) whenever the user has `using Optimization` in their session —
# Optimization.jl is a `[weakdeps]`-only dependency (see `Project.toml`) so
# it's never pulled in unless the user actually wants it. This overrides
# `PracticalBayes._external_optimize`'s stub (optimize.jl), which otherwise
# just errors with instructions to load this.

using PracticalBayes: PracticalBayes
using Optimization: Optimization

"""
    PracticalBayes._external_optimize(negf, negg!, θ0, alg; kwargs...)

`negf`/`negg!` are already the NEGATED objective/gradient (optimize.jl builds
them that way since Optimization.jl minimizes); we just wrap them in an
`OptimizationFunction`/`OptimizationProblem` and hand off to `alg` — no AD
backend selection needed here since the gradient is supplied directly rather
than through Optimization.jl's own `ADTypes` integration.
"""
function PracticalBayes._external_optimize(negf, negg!, θ0, alg; kwargs...)
    optf = Optimization.OptimizationFunction((u, p) -> negf(u); grad=(g, u, p) -> negg!(g, u))
    prob = Optimization.OptimizationProblem(optf, θ0)
    sol = Optimization.solve(prob, alg; kwargs...)
    return sol.u, sol.objective
end

end # module PracticalBayesOptimizationExt
