# Smoke-tests every ported model in models.jl: builds a layout, evaluates
# the log-density, and checks the ForwardDiff gradient against central
# finite differences. This is a STRUCTURAL check (does the model run, is
# every distribution/syntax feature it uses actually supported, does AD flow
# through it correctly) — not a posterior-accuracy check against real
# PosteriorDB data (see models.jl's header for why: no real PosteriorDB
# datasets are fetched here).

using PracticalBayes
using ADTypes: AutoForwardDiff
import LogDensityProblems

include("models.jl")

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

const MODELS = [
    ("Rate_1", Rate_1_model, make_Rate_1_data, (theta=0.3,)),
    ("blr", blr_model, make_blr_data, (sigma=1.0,)),
    ("eight_schools_centered", eight_schools_centered_model, make_eight_schools_data, (tau=1.0, mu=0.0)),
    ("eight_schools_noncentered", eight_schools_noncentered_model, make_eight_schools_data, (tau=1.0, mu=0.0)),
    ("dugongs", dugongs_model, make_dugongs_data, (alpha=2.5, beta=1.5, lambda=0.9, tau=1.0)),
    ("GLM_Poisson", GLM_Poisson_model, make_GLM_Poisson_data, (alpha=0.0, beta1=0.0, beta2=0.0, beta3=0.0)),
    ("kidscore_interaction", kidscore_interaction_model, make_kidscore_data, (sigma=10.0,)),
    ("earn_height", earn_height_model, make_earn_height_data, (sigma=5000.0,)),
    ("wells_dae", wells_dae_model, make_wells_data, NamedTuple()),
]

function run_smoke_tests()
    n_pass = 0
    for (name, modelfn, datafn, init) in MODELS
        data = datafn(1)
        m = modelfn(data...)
        try
            layout, θ0, store0 = build_layout(m; init=init)
            ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
            ok, val, maxerr = check_gradient(ldf, θ0)
            status = ok ? "OK" : "GRADIENT MISMATCH"
            println(rpad(name, 28), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  max grad err=", round(maxerr; sigdigits=3), "  ", status)
            ok && (n_pass += 1)
        catch e
            println(rpad(name, 28), "  FAILED — ", sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println()
    println(n_pass, " / ", length(MODELS), " models passed (logdensity finite + gradient matches finite differences)")
    return n_pass == length(MODELS)
end

abspath(PROGRAM_FILE) == (@__FILE__) && (run_smoke_tests() || exit(1))
