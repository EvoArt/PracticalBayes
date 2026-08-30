module PBTrim

# A PracticalBayes model compiled to a standalone binary with juliac --trim.
#
# Three things a trimmed program must avoid, all established by measurement:
#
#   1. build_layout. It discovers sites by RUNNING the model under TraceMode,
#      through a Dict and dynamic dispatch. Correct for a PPL, uncompilable.
#      Freeze the layout into a const instead: a compiled program has a fixed
#      model, so this costs nothing.  (204 -> 12 verifier errors)
#   2. AD. DifferentiationInterface's ForwardDiff extension leaves
#      GradientConfig and the per-call-site Dual tag types unresolved. Supply
#      the gradient.  (12 -> 2)
#   3. AdvancedHMC. PhasePoint's @argcheck builds its error from an Any, so
#      ArgCheck.build_error is unresolved on the core sampling path. Use
#      static_hmc.  (2 -> ?)

using PracticalBayes
using PracticalBayes: build_layout, LogDensityFunction, static_hmc
using Distributions: Normal, Exponential
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro

@static_model function simple(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

# Built once at precompile time; the binary never calls build_layout.
#
# The seeded RNG is essential, not tidiness. build_layout draws each site's
# initial value from `rng`, so with the default RNG the baked-in theta0 differs
# between every precompile -- the JIT and the binary silently start from
# different points, and an unlucky draw put one build at 500/500 divergences
# while the same code under the JIT had none. A compiled program must be
# reproducible from its source alone.
const _M = simple(2.0)
const _LAYOUT, _THETA0, _STORE0 = build_layout(_M; rng = Xoshiro(20240829))

# No adtype: LogDensityFunction then provides logdensity only, and the gradient
# is supplied below. Central differences here purely to keep this example
# self-contained -- a real program would carry an analytic or generated
# gradient, which is also far cheaper.
const _LDF = LogDensityFunction(_M, _LAYOUT, _STORE0; θ0 = _THETA0)

logp(x::Vector{Float64}) = LogDensityProblems.logdensity(_LDF, x)

function logp_grad!(g::Vector{Float64}, x::Vector{Float64})
    xp = copy(x)
    @inbounds for i in eachindex(x)
        h = 1e-6 * max(1.0, abs(x[i]))
        xp[i] = x[i] + h; fp = logp(xp)
        xp[i] = x[i] - h; fm = logp(xp)
        xp[i] = x[i]
        g[i] = (fp - fm) / (2h)
    end
    return logp(x)
end

function (@main)(args::Vector{String})::Cint
    d = length(_THETA0)
    inv_mass = ones(d)

    # Tuned offline for this model: eps=0.05, L=16 gives zero divergences at
    # 0.999 acceptance. A fixed-L sampler needs its parameters chosen ahead of
    # time, which is exactly what suits a compiled binary -- and eps=0.5 gave
    # 50 divergences in 500 draws, so this is not a formality.
    res = static_hmc(logp_grad!, copy(_THETA0), inv_mass, 0.05, 16, 500, 200; seed = 1)
    s = 0.0
    @inbounds for i in 1:size(res.draws, 2)
        s += res.draws[1, i]
    end
    println(Core.stdout, "draws: ", size(res.draws, 2),
            "  mean mu: ", s / size(res.draws, 2),
            "  divergences: ", res.n_divergent,
            "  accept: ", res.accept_rate)
    return 0
end

end
