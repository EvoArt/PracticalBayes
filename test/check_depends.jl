using Test
using PracticalBayes
using Distributions
using Random
using StableRNGs
using LinearAlgebra

# `depends=` is an optimization hint whose one failure mode is SILENT: an
# under-declared annotation makes a Gibbs block skip a term that does affect it,
# dropping a real gradient contribution with no error. `check_depends` is the
# guard, so the load-bearing test here is not that a correct model passes — it is
# that a DELIBERATELY BROKEN one fails. A validator that cannot fail proves
# nothing about the models it approves.

# Two cheap terms with disjoint dependencies, plus a third that couples them.
ll_a(a, y) = -0.5 * sum(abs2, y .- a)
ll_b(b, y) = -0.5 * sum(abs2, y .- b)
ll_ab(a, b, y) = -0.5 * sum(abs2, y .- (a + b))

@model function m_correct(y, fa, fb)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    @addlogprob! fa(a, y)  depends = (:a,)
    @addlogprob! fb(b, y)  depends = (:b,)
end

# `fab` reads BOTH a and b, but only declares `a`. A block sampling `b` alone
# will skip it, and b's gradient silently loses that term.
@model function m_under(y, fa, fab)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    @addlogprob! fa(a, y)     depends = (:a,)
    @addlogprob! fab(a, b, y) depends = (:a,)
end

# The same model with the annotation correct, to confirm the difference is the
# annotation and not the term.
@model function m_fixed(y, fa, fab)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    @addlogprob! fa(a, y)     depends = (:a,)
    @addlogprob! fab(a, b, y) depends = (:a, :b)
end

# No annotations at all: always safe, nothing is ever skipped.
@model function m_none(y, fa, fab)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    @addlogprob! fa(a, y)
    @addlogprob! fab(a, b, y)
end

# Over-declared: names a variable the term does not read. Harmless (it only
# forgoes the speed-up) and must NOT be reported as a failure.
@model function m_over(y, fa, fb)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    @addlogprob! fa(a, y) depends = (:a, :b)
    @addlogprob! fb(b, y) depends = (:b,)
end

const Y = randn(StableRNG(1), 20)

@testset "check_depends" begin
    @testset "a correct model passes" begin
        m = m_correct(Y, ll_a, ll_b)
        spl = Gibbs(:a => NUTS(0.8), :b => NUTS(0.8))
        r = check_depends(m, spl; verbose=false)
        @test r.ok
        @test length(r.rows) == 2
        @test all(row -> row.max_abs_diff == 0.0, r.rows)
    end

    @testset "an UNDER-declared annotation is caught" begin
        # The whole point. If this test ever starts passing trivially, the
        # validator has stopped validating.
        m = m_under(Y, ll_a, ll_ab)
        spl = Gibbs(:a => NUTS(0.8), :b => NUTS(0.8))
        r = check_depends(m, spl; verbose=false)
        @test !r.ok
        # Only the `b` block is wrong: `a` is declared on both terms, so the
        # block owning `a` skips nothing.
        bad = [row for row in r.rows if row.max_abs_diff > 0]
        @test length(bad) == 1
        @test bad[1].names == (:b,)
        @test bad[1].max_abs_diff > 1e-8
        @test :b in bad[1].suspects
    end

    @testset "fixing the annotation fixes the check" begin
        m = m_fixed(Y, ll_a, ll_ab)
        spl = Gibbs(:a => NUTS(0.8), :b => NUTS(0.8))
        @test check_depends(m, spl; verbose=false).ok
    end

    @testset "no annotations at all is safe" begin
        m = m_none(Y, ll_a, ll_ab)
        spl = Gibbs(:a => NUTS(0.8), :b => NUTS(0.8))
        @test check_depends(m, spl; verbose=false).ok
    end

    @testset "over-declaring is not a failure" begin
        m = m_over(Y, ll_a, ll_b)
        spl = Gibbs(:a => NUTS(0.8), :b => NUTS(0.8))
        @test check_depends(m, spl; verbose=false).ok
    end

    @testset "a single block covering everything can never be wrong" begin
        # With one block, every variable is in the flat vector, so no term is
        # skipped whatever the annotations say. The broken model must pass here.
        m = m_under(Y, ll_a, ll_ab)
        spl = Gibbs((:a, :b) => NUTS(0.8))
        @test check_depends(m, spl; verbose=false).ok
    end

    @testset "vector parameters are compared over their whole span" begin
        llv(v, y) = -0.5 * sum(abs2, y .- sum(v))
        llvw(v, w, y) = -0.5 * sum(abs2, y .- (sum(v) + w))
        @model function m_vec(y, f1, f2)
            v ~ PracticalBayes.filldist(Normal(0, 1), 3)
            w ~ Normal(0, 1)
            @addlogprob! f1(v, y)     depends = (:v,)
            @addlogprob! f2(v, w, y)  depends = (:v,)      # under-declared in w
        end
        m = m_vec(Y, llv, llvw)
        spl = Gibbs(:v => NUTS(0.8), :w => NUTS(0.8))
        r = check_depends(m, spl; verbose=false)
        @test !r.ok
        bad = [row for row in r.rows if row.max_abs_diff > 0]
        @test length(bad) == 1 && bad[1].names == (:w,)

        @model function m_vec_ok(y, f1, f2)
            v ~ PracticalBayes.filldist(Normal(0, 1), 3)
            w ~ Normal(0, 1)
            @addlogprob! f1(v, y)     depends = (:v,)
            @addlogprob! f2(v, w, y)  depends = (:v, :w)
        end
        @test check_depends(m_vec_ok(Y, llv, llvw),
                            Gibbs(:v => NUTS(0.8), :w => NUTS(0.8));
                            verbose=false).ok
    end

    @testset "a latent block is skipped, not checked" begin
        struct _NoopKernel <: PracticalBayes.AbstractLatentKernel end
        PracticalBayes.latent_step(rng, ::_NoopKernel, names, c) = (; z=c.values.z)

        @model function m_lat(y, fa)
            a ~ Normal(0, 1)
            z ~ Categorical([0.5, 0.5])
            @addlogprob! fa(a, y) depends = (:a,)
        end
        m = m_lat(Y, ll_a)
        spl = Gibbs(:a => NUTS(0.8), :z => _NoopKernel())
        r = check_depends(m, spl; verbose=false)
        @test r.ok
        # Only the HMC block has a gradient to compare.
        @test length(r.rows) == 1
        @test r.rows[1].names == (:a,)
    end

    @testset "init is honoured, so the check runs where the sampler starts" begin
        m = m_correct(Y, ll_a, ll_b)
        spl = Gibbs(:a => NUTS(0.8), :b => NUTS(0.8))
        @test check_depends(m, spl; init=(; a=0.7, b=-0.3), verbose=false).ok
    end

    @testset "the report prints something usable" begin
        m = m_under(Y, ll_a, ll_ab)
        spl = Gibbs(:a => NUTS(0.8), :b => NUTS(0.8))
        r = check_depends(m, spl; verbose=false)
        s = sprint(show, r)
        @test occursin("MISMATCH", s)
        @test occursin("UNDER-DECLARED", s)
    end
end
