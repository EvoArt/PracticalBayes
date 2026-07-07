# Tests for the `@model` macro and basic model mechanics: model construction,
# conditioning, and that a model can be evaluated at all under each mode.
# These are deliberately simple models — correctness of the actual
# log-density MATH is checked in layout.jl/logdensity.jl against hand-derived
# values, not here.

using Distributions: Normal, Exponential, Beta, Bernoulli, logpdf
using ADTypes: AutoForwardDiff
import LogDensityProblems

@testset "compiler.jl: @model basics" begin
    @model function simple_normal(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
        return mu
    end

    @testset "constructor produces a Model with the right args/conditioned" begin
        m = simple_normal(1.5)
        @test m isa PracticalBayes.Model
        @test m.args == (; y=1.5)
        @test m.conditioned == NamedTuple()
    end

    @testset "condition/|/decondition" begin
        m = simple_normal(missing)
        m2 = m | (; extra=1)
        @test m2.conditioned == (; extra=1)
        m3 = PracticalBayes.decondition(m2, :extra)
        @test m3.conditioned == NamedTuple()
    end

    @testset "TraceMode discovers all three sites" begin
        m = simple_normal(2.0)
        layout, θ0, store0 = build_layout(m)
        @test layout.dim == 2  # mu (1) + sigma (1, linked/log-scale)
        @test store0 == NamedTuple()
        @test length(θ0) == 2
    end

    @testset "y as `missing` makes it an assumed site instead of observed" begin
        m = simple_normal(missing)
        layout, θ0, _ = build_layout(m)
        # mu, sigma, AND y should all be flat now (y ~ Normal(mu, sigma) with
        # mu/sigma unconstrained -> y itself is unconstrained real, length 1)
        @test layout.dim == 3
    end
end

@testset "compiler.jl: indexed tilde `x[i] ~ dist`" begin
    # `Vector{paramtype(__mode__)}(undef, n)`, NOT `Vector{Float64}(undef, n)`
    # — see the REQUIRED note on `tilde_index`'s docstring (tilde.jl):
    # hardcoding Float64 crashes under any AD backend since the container
    # can't hold the Dual numbers a gradient call needs to write into it.
    @model function indexed_model(n)
        x = Vector{paramtype(__mode__)}(undef, n)
        for i in 1:n
            x[i] ~ Normal(0, 1)
        end
        return x
    end

    m = indexed_model(3)
    layout, θ0, _ = build_layout(m)
    @test layout.dim == 3
    @test layout.slots.x isa PracticalBayes.FlatArraySlot
end

@testset "compiler.jl: indexed tilde differentiates correctly (regression: paramtype)" begin
    # Regression test for a real bug found while benchmarking: a
    # `Vector{Float64}(undef, n)`-declared container crashes under ANY AD
    # backend (can't store a Dual/backend-specific number in a Float64
    # array). `paramtype(__mode__)` is the fix — verify it actually works
    # under gradient computation, not just density-only evaluation.
    @model function indexed_grad_model(n)
        x = Vector{paramtype(__mode__)}(undef, n)
        for i in 1:n
            x[i] ~ Normal(0, 1)
        end
        return x
    end

    m = indexed_grad_model(4)
    layout, θ0, store0 = build_layout(m)
    ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
    val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    # each site is `Normal(0,1)`, identity-linked, so analytic gradient is -θ
    @test grad ≈ -θ0
end

@testset "compiler.jl: `.~` observe sugar" begin
    @model function dot_tilde_model(y)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
        return mu
    end

    m = dot_tilde_model([1.0, 2.0, 3.0])
    layout, θ0, store0 = build_layout(m)
    @test layout.dim == 1  # only `mu` occupies flat space; y is observed data

    mode = EvalMode(layout, θ0, store0, m.conditioned)
    _, acc = evaluate(m, mode, Accum(0.0))
    expected = logpdf(Normal(0, 1), θ0[1]) + sum(logpdf.(Normal.(θ0[1], 1.0), [1.0, 2.0, 3.0]))
    @test logjoint(acc) ≈ expected
end

@testset "compiler.jl: `.~` assume is a clear error, not silently wrong" begin
    # `x` here is a pre-declared local (not a model argument), so this is
    # exactly the ambiguous "looks like data, is actually meant to be
    # unknown" pattern the compiler now rejects at macro-expansion time (see
    # `_dot_tilde_expansion` in compiler.jl) rather than silently computing a
    # logpdf against uninitialized memory at runtime.
    # `@eval` inside an `include`d test file wraps the macro-expansion error
    # in a `LoadError` (Julia's standard behavior for errors during file
    # loading), rather than raising the underlying `ErrorException` directly
    # the way a bare top-level `@eval` in the REPL would.
    @test_throws LoadError @eval @model function bad_dot_assume()
        x = Vector{Float64}(undef, 3)
        x .~ Normal.(zeros(3), 1.0)
        return x
    end
end

@testset "compiler.jl: `:=` conditions on a model-computed local" begin
    # `z` is never a model argument — it's computed from `mu` and `x` partway
    # through the body — but `z := ...` marks it as "has a value already", so
    # `z ~ Normal(0, 1)` should be treated as an OBSERVE using that computed
    # value, not as a declaration of a new parameter. This is the case the
    # user specifically asked for: conditioning on any value that already
    # exists, even one computed inside the model.
    @model function walrus_model(x)
        mu ~ Normal(0, 1)
        z := mu + x
        z ~ Normal(0, 1)
        return mu
    end

    m = walrus_model(0.5)
    layout, θ0, store0 = build_layout(m)
    # Only `mu` should occupy flat space — `z` was never a parameter site at all.
    @test layout.dim == 1
    @test !haskey(layout.slots, :z)

    mode = EvalMode(layout, θ0, store0, m.conditioned)
    _, acc = evaluate(m, mode, Accum(0.0))
    mu_val = θ0[1]
    z_val = mu_val + 0.5
    expected = logpdf(Normal(0, 1), mu_val) + logpdf(Normal(0, 1), z_val)
    @test logjoint(acc) ≈ expected
end

@testset "compiler.jl: keyword-only model arguments (regression: kwargs silently dropped from Model.args)" begin
    # Found while porting the TuringLang docs `coin-flipping` tutorial, whose
    # model is declared `@model function coinflip(; N::Int) ... end`. Before
    # the fix, `_argnames`/the constructor's NamedTuple-building only ever
    # looked at `def[:args]` (positional), never `def[:kwargs]` — so
    # `coinflip(; N=5)` silently produced a `Model` with an EMPTY `args`
    # NamedTuple, `N` was never recognized as "already a concrete value" by
    # the tilde-rewriter, and any tilde site referencing `N` would see it as
    # `UndefVarError` or, worse, silently wrong. `Model.args` is one flat
    # NamedTuple splatted POSITIONALLY into the generated evaluator
    # (`m.f(mode, acc, m.args...)`, model.jl); a positional splat can't bind
    # to a `function f(; k)`-style keyword parameter, so the fix makes the
    # generated evaluator take kwargs positionally too (only the user-facing
    # constructor keeps the real keyword-argument call syntax).
    @model function kwmodel(; N::Int)
        p ~ Beta(1, 1)
        y ~ filldist(Bernoulli(p), N)
        return y
    end
    m = kwmodel(; N=5)
    @test m.args == (; N=5)  # was `NamedTuple()` before the fix

    kwmodel(y::AbstractVector{<:Real}) = kwmodel(; N=length(y)) | (; y)
    y = [1.0, 0.0, 1.0, 1.0, 0.0]
    m2 = kwmodel(y)
    layout, θ0, store0 = build_layout(m2; init=(p=0.5,))
    @test layout.dim == 1  # only `p` — `y`/`N` are both data, not parameters

    ldf = LogDensityFunction(m2, layout, store0, AutoForwardDiff(); θ0=θ0)
    val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    @test isfinite(val)
    @test length(grad) == 1

    # mixed positional + keyword args: both must land in `Model.args`, in
    # positional-args-then-kwargs order (matching the generated evaluator).
    @model function mixedmodel(x; N::Int)
        mu ~ Normal(0, 1)
        z ~ Normal(mu + x, 1)
        return N
    end
    m3 = mixedmodel(1.0; N=3)
    @test m3.args == (x=1.0, N=3)
end
