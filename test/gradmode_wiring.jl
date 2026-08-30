# End-to-end tests for the opt-in `GradMode()` backend: recognition happening
# at @model expansion time, the closed-form gradient agreeing with general AD
# through the PUBLIC LogDensityProblems API, and the refusal path.
#
# The refusal tests matter as much as the agreement ones. `GradMode()` is an
# explicit request for a fast path, so a model it cannot handle must ERROR
# rather than silently fall back — otherwise a user would believe they had the
# fast path and never find out they did not.

using Test
using PracticalBayes
using PracticalBayes: gradmode_plan, GradMode, check_gradmode
using Distributions: Distributions, logistic
using LinearAlgebra: LinearAlgebra
using LogDensityProblems: LogDensityProblems
using ADTypes: ADTypes
using Random: Random

const PB = PracticalBayes

PB.@model function gm_logit(X, y)
    pT = PB.paramtype(__mode__); k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    y ~ PB.arraydist(Distributions.BernoulliLogit.(X * beta))
end

PB.@model function gm_normal(X, y)
    pT = PB.paramtype(__mode__); k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    sigma ~ Distributions.Exponential(one(pT))
    y ~ Distributions.MvNormal(X * beta, sigma^2 * LinearAlgebra.I)
end

# nonlinear in the parameters (exp(gamma) scaling) -> must NOT be recognized
PB.@model function gm_nonlinear(X, y)
    pT = PB.paramtype(__mode__); k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    gamma ~ Distributions.Normal(zero(pT), one(pT))
    y ~ PB.arraydist(Distributions.BernoulliLogit.(exp(gamma) .* (X * beta)))
end

function _gm_data(; N=300, P=4, seed=1)
    rng = Random.Xoshiro(seed)
    X = randn(rng, N, P); bt = randn(rng, P) ./ sqrt(P)
    yb = Float64.(rand(rng, N) .< logistic.(X * bt))
    yn = X * bt .+ randn(rng, N) .* 0.5
    return X, yb, yn
end

@testset "gradmode wiring" begin

X, yb, yn = _gm_data()

@testset "recognition happens at @model expansion" begin
    # qualified names (`Distributions.MvNormal`) are what real models use
    @test gradmode_plan(gm_logit(X, yb).f) !== nothing
    @test gradmode_plan(gm_normal(X, yn).f) !== nothing
    @test gradmode_plan(gm_nonlinear(X, yb).f) === nothing
end

@testset "GradMode gradient matches general AD" begin
    for (model, ) in ((gm_logit(X, yb),), (gm_normal(X, yn),))
        lay, θ0, store = build_layout(model)
        fgm = LogDensityFunction(model, lay, store, GradMode(); θ0=θ0)
        fad = LogDensityFunction(model, lay, store, ADTypes.AutoForwardDiff(); θ0=θ0)

        # reports order 1, so a sampler uses our gradient rather than
        # re-differentiating
        @test LogDensityProblems.capabilities(typeof(fgm)) ==
              LogDensityProblems.LogDensityOrder{1}()
        @test LogDensityProblems.dimension(fgm) == lay.dim

        # several NON-ZERO points: at θ=0 many terms vanish and sign errors hide
        for k in 1:5
            θ = randn(Random.Xoshiro(100 + k), lay.dim) .* 0.4
            v1, g1 = LogDensityProblems.logdensity_and_gradient(fgm, θ)
            v2, g2 = LogDensityProblems.logdensity_and_gradient(fad, θ)
            @test isapprox(v1, v2; rtol=1e-8)
            @test isapprox(g1, g2; rtol=1e-6, atol=1e-8)
        end
    end
end

@testset "check_gradmode verifies against general AD" begin
    model = gm_logit(X, yb)
    lay, θ0, store = build_layout(model)
    fad = LogDensityFunction(model, lay, store, ADTypes.AutoForwardDiff(); θ0=θ0)
    plan = gradmode_plan(model.f)
    @test check_gradmode(plan, lay, model.args, fad; n=5, rng=Random.Xoshiro(7))
end

@testset "unrecognized model is refused, not silently slow" begin
    model = gm_nonlinear(X, yb)
    lay, θ0, store = build_layout(model)
    # an explicit fast-path request that cannot be honoured must say so
    @test_throws ArgumentError LogDensityFunction(model, lay, store, GradMode(); θ0=θ0)
    # ...and the same model still works under general AD
    fad = LogDensityFunction(model, lay, store, ADTypes.AutoForwardDiff(); θ0=θ0)
    v, g = LogDensityProblems.logdensity_and_gradient(fad, θ0)
    @test isfinite(v)
    @test length(g) == lay.dim
end

@testset "vector hyper-location (regression: silently wrong gradient)" begin
    # `a ~ MvNormal(mu, I)` with mu a VECTOR parameter. `_gm_hier_params`
    # collapsed it to `mu[1]` via `_gm_scalar`, so every element of `a` was
    # scored against the same hyper-mean and the whole hyper-location gradient
    # was dumped into mu[1] with zeros elsewhere. Value AND gradient were wrong.
    #
    # Found while checking whether a `::Float64` return annotation on
    # `_gm_hier_params` would be safe -- it would NOT have been, and asking
    # that question is what surfaced this.
    PB.@model function gm_vechyper(y, grp, J)
        pT = PB.paramtype(__mode__)
        mu ~ Distributions.MvNormal(zeros(pT, J), LinearAlgebra.I)
        a ~ Distributions.MvNormal(mu, LinearAlgebra.I)
        y ~ Distributions.MvNormal(a[grp], LinearAlgebra.I)
    end
    rng = Random.Xoshiro(11); Nv, Jv = 60, 5
    grpv = rand(rng, 1:Jv, Nv); yv = randn(rng, Nv)
    mv = gm_vechyper(yv, grpv, Jv)
    @test gradmode_plan(mv.f) !== nothing
    lay, θ0, store = build_layout(mv)
    fad = LogDensityFunction(mv, lay, store, ADTypes.AutoForwardDiff(); θ0=θ0)
    @test check_gradmode(gradmode_plan(mv.f), lay, mv.args, fad; n=5, rng=Random.Xoshiro(2))
end

@testset "general AD path is unchanged by the @model plan emission" begin
    # the macro now also emits a `gradmode_plan` method; make sure that has not
    # perturbed the ordinary evaluation path
    model = gm_logit(X, yb)
    lay, θ0, store = build_layout(model)
    f0 = LogDensityFunction(model, lay, store, nothing; θ0=θ0)
    @test LogDensityProblems.capabilities(typeof(f0)) ==
          LogDensityProblems.LogDensityOrder{0}()
    @test isfinite(LogDensityProblems.logdensity(f0, θ0))
end

end  # gradmode wiring
