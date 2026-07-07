# Smoke tests for models_batch2.jl — see run_smoke_tests.jl for methodology.

using PracticalBayes
using ADTypes: AutoForwardDiff
import LogDensityProblems

include("models_batch2.jl")

function check_gradient(ldf, θ0; atol=1e-3, h=1e-6)
    val, grad = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    fd_grad = similar(θ0)
    for i in eachindex(θ0)
        θp = copy(θ0); θp[i] += h
        θm = copy(θ0); θm[i] -= h
        fd_grad[i] = (LogDensityProblems.logdensity(ldf, θp) - LogDensityProblems.logdensity(ldf, θm)) / (2h)
    end
    maxerr = maximum(abs.(grad .- fd_grad))
    ok = isfinite(val) && maxerr < atol
    return ok, val, maxerr
end

# Each entry's second element is a zero-arg closure building the fully
# constructed `Model` (data generation + model construction bundled
# together, so every model's own call convention — positional args, a
# NamedTuple wrapper like election88_full_model, whatever — is handled at
# the closure definition site rather than by generic splatting logic here).
#
# `density_only=true` skips the AD/gradient check entirely — used for
# Rate_4/Rate_5, whose `:=` posterior-predictive quantities call
# `rand(Binomial(n, theta))` with `theta` potentially a Dual under AD.
# `Distributions.rand(::Binomial, ::Dual)` genuinely has no sampler (a
# Distributions.jl limitation, not a PracticalBayes one — confirmed via a
# standalone repro) since sampling isn't a differentiable operation in the
# first place; postpred `:=` quantities are meant to be read at a FIXED
# (non-Dual) θ, not differentiated through, so a plain density-only check is
# the correct test for these two models, not a gradient one.
const MODELS = [
    ("Rate_2", () -> Rate_2_model(make_Rate_2_data(1)...), (theta1=0.3, theta2=0.4), false),
    ("Rate_3", () -> Rate_3_model(make_Rate_3_data(1)...), (theta=0.3,), false),
    ("Rate_4", () -> Rate_4_model(make_Rate_4_data(1)...), (theta=0.3, thetaprior=0.3), true),
    ("Rate_5", () -> Rate_5_model(make_Rate_5_data(1)...), (theta=0.3,), true),
    ("GLM_Binomial", () -> GLM_Binomial_model(make_GLM_Binomial_data(1)...), (alpha=0.0, beta1=0.0, beta2=0.0), false),
    ("GLMM_Poisson", () -> GLMM_Poisson_model(make_GLMM_Poisson_data(1)...), (alpha=0.0, beta1=0.0, beta2=0.0, beta3=0.0, sigma=1.0), false),
    ("GLMM1", () -> GLMM1_model(make_GLMM1_data(1)...), (mu_alpha=0.0, sd_alpha=1.0), false),
    ("election88_full", () -> election88_full_model(make_election88_data(1)), NamedTuple(), false),
    ("irt_2pl", () -> irt_2pl_model(make_irt_2pl_data(1)...), (sigma_theta=1.0, sigma_a=1.0, mu_b=0.0, sigma_b=1.0), false),
    ("kidscore_interaction_c", () -> kidscore_interaction_c_model(make_kidscore2_data(1)...), (sigma=10.0,), false),
    ("kidscore_interaction_z", () -> kidscore_interaction_z_model(make_kidscore2_data(1)...), (sigma=10.0,), false),
    ("kidscore_momhs", () -> (d = make_kidscore2_data(1); kidscore_momhs_model(d.N, d.kid_score, d.mom_hs)), NamedTuple(), false),
    ("kidscore_momiq", () -> (d = make_kidscore2_data(1); kidscore_momiq_model(d.N, d.kid_score, d.mom_iq)), NamedTuple(), false),
    ("kilpisjarvi", () -> kilpisjarvi_model(make_kilpisjarvi_data(1)...), (sigma=1.0,), false),
]

function run_smoke_tests()
    n_pass = 0
    for (name, modelfn, init, density_only) in MODELS
        try
            m = modelfn()
            layout, θ0, store0 = build_layout(m; init=init)
            local ok, val
            if density_only
                ldf = LogDensityFunction(m, layout, store0)
                val = LogDensityProblems.logdensity(ldf, θ0)
                ok = isfinite(val)
                println(rpad(name, 26), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  (density-only)  ", ok ? "OK" : "NON-FINITE")
            else
                ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
                ok, val, maxerr = check_gradient(ldf, θ0)
                status = ok ? "OK" : "GRADIENT MISMATCH"
                println(rpad(name, 26), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  max grad err=", round(maxerr; sigdigits=3), "  ", status)
            end
            ok && (n_pass += 1)
        catch e
            println(rpad(name, 26), "  FAILED — ", sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println()
    println(n_pass, " / ", length(MODELS), " models passed")
    return n_pass == length(MODELS)
end

abspath(PROGRAM_FILE) == (@__FILE__) && (run_smoke_tests() || exit(1))
