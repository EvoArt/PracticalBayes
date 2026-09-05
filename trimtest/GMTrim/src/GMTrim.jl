module GMTrim

# Does GradMode -- PracticalBayes' closed-form GLM gradient -- survive
# `juliac --trim`?
#
# This is the interesting case. The earlier PBTrim experiment had to supply a
# hand-written gradient because AD could not be compiled:
# DifferentiationInterface's ForwardDiff extension leaves `GradientConfig` and
# the per-call-site `Dual` tag types unresolved. GradMode computes the gradient
# in closed form for models `recognize_glm` accepts, with no Duals, no tape and
# no AD package -- so if it trims, recognized models get a compiled binary
# without anyone writing a gradient by hand.

using PracticalBayes
const PB = PracticalBayes
using PracticalBayes: build_layout, LogDensityFunction, static_hmc, GradMode
using Distributions: MvNormal, Exponential
using LinearAlgebra: I
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro

PB.@static_model function gm_normal(X, y)
    pT = PB.paramtype(__mode__); k = size(X, 2)
    beta ~ MvNormal(zeros(pT, k), I)
    sigma ~ Exponential(one(pT))
    y ~ MvNormal(X * beta, sigma^2 * I)
end

# Data and layout are frozen at precompile time: the binary never runs
# build_layout (which discovers sites dynamically and cannot be trimmed), and
# the seeded RNG keeps theta0 reproducible across builds -- with the default
# RNG the baked-in constant differs on every precompile, which silently gave
# one build 500/500 divergences while the same source under the JIT had none.
const _RNG = Xoshiro(20240829)
const _X = randn(_RNG, 200, 3)
const _Y = _X * [0.5, -1.0, 0.3] .+ 0.2 .* randn(_RNG, 200)
const _M = gm_normal(_X, _Y)
const _LAYOUT, _THETA0, _STORE0 = build_layout(_M; rng = Xoshiro(20240829))

# GradMode(), not an AD backend. This is the whole point of the experiment.
const _LDF = LogDensityFunction(_M, _LAYOUT, _STORE0, GradMode(); θ0 = _THETA0)

function logp_grad!(g::Vector{Float64}, x::Vector{Float64})
    lp, grad = LogDensityProblems.logdensity_and_gradient(_LDF, x)
    @inbounds for i in eachindex(g)
        g[i] = grad[i]
    end
    return lp
end

function (@main)(args::Vector{String})::Cint
    d = length(_THETA0)
    g = Vector{Float64}(undef, d)
    lp = logp_grad!(g, _THETA0)
    println(Core.stdout, "logp(theta0) = ", lp)
    println(Core.stdout, "grad[1] = ", g[1])

    res = static_hmc(logp_grad!, copy(_THETA0), ones(d), 0.005, 16, 500, 300; seed = 1)
    s = 0.0
    @inbounds for i in 1:size(res.draws, 2)
        s += res.draws[1, i]
    end
    println(Core.stdout, "draws: ", size(res.draws, 2),
            "  mean beta1: ", s / size(res.draws, 2),
            "  divergences: ", res.n_divergent,
            "  accept: ", res.accept_rate)
    return 0
end

end
