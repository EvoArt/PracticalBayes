module PracticalBayesOptimizationExt

# Loaded automatically (Julia's package-extension mechanism, no Requires.jl
# needed) whenever the user has `using Optimization` in their session —
# Optimization.jl is a `[weakdeps]`-only dependency (see `Project.toml`) so
# it's never pulled in unless the user actually wants it. This replaces
# `PracticalBayes._EXTERNAL_OPTIMIZE_HOOK`'s default (optimize.jl), which
# otherwise just errors with instructions to load this.
#
# The hook is a `Ref{Any}` holding a function, not a method the extension
# adds via multiple dispatch: two identically-signatured
# `_external_optimize(negf, negg!, θ0, alg; kwargs...)` methods (this
# extension's and the base package's stub) would be the SAME method to
# Julia, and overwriting a method during another module's precompilation is
# forbidden. Assigning into the Ref in `__init__` is a plain value
# replacement, so it doesn't hit that restriction.

using PracticalBayes: PracticalBayes
using Optimization: Optimization

# `negf`/`negg!` are already the NEGATED objective/gradient (optimize.jl
# builds them that way since Optimization.jl minimizes); this just wraps
# them in an `OptimizationFunction`/`OptimizationProblem` and hands off to
# `alg` — no AD backend selection needed here since the gradient is supplied
# directly rather than through Optimization.jl's own `ADTypes` integration.
function _optimization_jl_optimize(negf, negg!, θ0, alg; kwargs...)
    optf = Optimization.OptimizationFunction((u, p) -> negf(u); grad=(g, u, p) -> negg!(g, u))
    prob = Optimization.OptimizationProblem(optf, θ0)
    sol = Optimization.solve(prob, alg; kwargs...)
    return sol.u, sol.objective, Symbol(sol.retcode)
end

function __init__()
    PracticalBayes._EXTERNAL_OPTIMIZE_HOOK[] = _optimization_jl_optimize
end

end # module PracticalBayesOptimizationExt
