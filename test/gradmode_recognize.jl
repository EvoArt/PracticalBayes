# Tests for the Stage 2 GLM recognizer (`src/gradmode_recognize.jl`).
#
# The "must reject" set is the important half, and is adversarial by design:
# every case there is a model that LOOKS GLM-shaped but whose true gradient is
# NOT the closed form (nonlinear predictors, centered hierarchical priors,
# unmodelled log-density terms, control flow). A false ACCEPT would produce a
# silently biased posterior rather than a crash, which is the worst failure
# mode this package has — so these cases matter more than the coverage ones.
#
# The "must accept" set guards the opposite regression: over-tightening the
# grammar until nothing is recognized and the whole feature is pointless.
using Test
using PracticalBayes: recognize_glm

# Returns :accept / :reject for a model body, so each case reads as a single
# @test. Recognition returning `nothing` (:reject) is never an error -- it
# just means "use general AD", which is always correct.
function _recog(argnames, body)
    plan = recognize_glm(body, Set{Symbol}(argnames))
    return plan === nothing ? :reject : :accept
end

@testset "gradmode recognizer" begin

@testset "must reject (wrong-gradient risks)" begin

# beta*beta is quadratic in the parameters, not linear.
@test _recog([:y, :X], quote
    beta ~ Normal(0, 1)
    gamma ~ Normal(0, 1)
    y ~ MvNormal(beta * gamma, 1.0)
end) == :reject

# exp() of a parameter inside the predictor is nonlinear.
@test _recog([:y, :x], quote
    beta ~ Normal(0, 1)
    eta = exp(beta) .* x
    y ~ MvNormal(eta, 1.0)
end) == :reject

# A centered hierarchical prior: d/d(mu_a) has chain-rule terms through a.
@test _recog([:y, :grp], quote
    mu_a ~ Normal(0, 5)
    a ~ Normal(mu_a, 1.0)
    y ~ MvNormal(a[grp], 1.0)
end) == :reject

# sigma_a as a prior arg is the same problem.
@test _recog([:y, :x], quote
    s ~ Exponential(1.0)
    beta ~ Normal(0, s)
    y ~ MvNormal(beta .* x, 1.0)
end) == :reject

# A loop is arbitrary control flow -> reject wholesale.
@test _recog([:y, :x, :n], quote
    beta ~ Normal(0, 1)
    for i in 1:n
        y[i] ~ Normal(beta * x[i], 1.0)
    end
end) == :reject

# An unknown function in the predictor must not be assumed linear.
@test _recog([:y, :x], quote
    beta ~ Normal(0, 1)
    eta = mystery(beta, x)
    y ~ MvNormal(eta, 1.0)
end) == :reject

# Unknown distribution: no closed form available.
@test _recog([:y, :x], quote
    beta ~ Gamma(2.0, 3.0)
    y ~ MvNormal(beta .* x, 1.0)
end) == :reject

# Unknown link.
@test _recog([:y, :x], quote
    beta ~ Normal(0, 1)
    y ~ arraydist(Weibull.(beta .* x))
end) == :reject

# Two likelihood sites -> the plan has no way to represent this.
@test _recog([:y, :z, :x], quote
    beta ~ Normal(0, 1)
    y ~ MvNormal(beta .* x, 1.0)
    z ~ MvNormal(beta .* x, 1.0)
end) == :reject

# @addlogprob! injects an arbitrary term we do not model.
@test _recog([:y, :x], quote
    beta ~ Normal(0, 1)
    @addlogprob! -0.5 * beta^2
    y ~ MvNormal(beta .* x, 1.0)
end) == :reject

# Predictor with no parameter at all: nothing to differentiate.
@test _recog([:y, :x], quote
    sigma ~ Exponential(1.0)
    y ~ MvNormal(x, sigma^2)
end) == :reject

# Rebinding a parameter name.
@test _recog([:y, :x], quote
    beta ~ Normal(0, 1)
    beta ~ Normal(0, 2)
    y ~ MvNormal(beta .* x, 1.0)
end) == :reject

# Division by a parameter is nonlinear.
@test _recog([:y, :x], quote
    beta ~ Normal(0, 1)
    eta = x ./ beta
    y ~ MvNormal(eta, 1.0)
end) == :reject

end  # must reject

@testset "must accept (genuinely GLM-shaped)" begin

@test _recog([:y, :X], quote
    beta ~ MvNormal(zeros(2), 1.0)
    sigma ~ Exponential(1.0)
    y ~ MvNormal(X * beta, sigma^2)
end) == :accept

@test _recog([:y, :X], quote
    alpha ~ Flat()
    beta ~ filldist(Flat(), 3)
    y .~ BernoulliLogit.(alpha .+ X * beta)
end) == :accept

@test _recog([:y, :grp, :x], quote
    a ~ MvNormal(zeros(5), 1.0)
    b ~ Normal(0, 1)
    y ~ MvNormal(a[grp] .+ b .* x, 1.0)
end) == :accept

@test _recog([:y, :dist, :N], quote
    d100 = dist ./ 100
    alpha ~ Flat()
    beta ~ Flat()
    y .~ BernoulliLogit.(alpha .+ beta .* d100)
end) == :accept

end  # must accept

@testset "MvNormal prior variance (regression: silently wrong gradient)" begin
    # `MvNormal(zeros(p), 25.0*I)` was ACCEPTED and then differentiated as if
    # the prior were standard normal — the gradient was wrong by a factor of
    # the variance. Caught by check_gradmode, not by recognition, which is
    # exactly what the verification harness is for. All isotropic forms below
    # must be recognized AND carry their variance through to codegen.
    for cov in (:(I), :(25.0 * I), :(I * 25.0), :(4.0), :(pT(25.0) * I))
        @test _recog([:y, :X, :pT], quote
            beta ~ MvNormal(zeros(2), $cov)
            y ~ arraydist(BernoulliLogit.(X * beta))
        end) == :accept
    end
    # a covariance shape codegen cannot interpret must be REJECTED, not
    # silently treated as unit scale
    @test _recog([:y, :X, :S], quote
        beta ~ MvNormal(zeros(2), S)
        y ~ arraydist(BernoulliLogit.(X * beta))
    end) == :reject
end

@testset "cloglog link and @addlogprob! observe form" begin
    # the jolly island `seic_cloglog` shape, both spellings
    @test _recog([:y, :Xmat], quote
        beta ~ MvNormal(zeros(2), I)
        eta = Xmat * beta
        y ~ arraydist(BernoulliCLogLog.(eta))
    end) == :accept

    @test _recog([:y, :Xmat], quote
        beta ~ MvNormal(zeros(2), I)
        eta = Xmat * beta
        @addlogprob! sum(logpdf.(BernoulliCLogLog.(eta), y))
    end) == :accept

    # an @addlogprob! that is NOT the recognized vectorised-sum observe carries
    # an arbitrary log-density term and must reject the whole model
    @test _recog([:y, :Xmat], quote
        beta ~ MvNormal(zeros(2), I)
        @addlogprob! -0.5 * sum(abs2, beta)
        y ~ arraydist(BernoulliLogit.(Xmat * beta))
    end) == :reject

    # a `depends=` annotation implies Gibbs blocking, which GradMode cannot do
    @test _recog([:y, :Xmat], quote
        beta ~ MvNormal(zeros(2), I)
        eta = Xmat * beta
        @addlogprob! sum(logpdf.(BernoulliCLogLog.(eta), y)) depends=(beta,)
    end) == :reject
end

end  # gradmode recognizer
