# Smoke tests for models_batch4.jl — see run_smoke_tests.jl for methodology.

using PracticalBayes
using ADTypes: AutoForwardDiff
import LogDensityProblems

include("models_batch4.jl")

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
    ("nes", () -> nes_model(make_nes_data(1)...), (sigma=2.0,)),
    ("pilots", () -> pilots_model(make_pilots_data(1)...), (mu_a=0.0, sigma_a=1.0, mu_b=0.0, sigma_b=1.0)),
    ("rats", () -> rats_model(make_rats_data(1)...), (sigma_y=3.0, sigma_alpha=10.0, sigma_beta=2.0)),
    ("seeds", () -> seeds_model(make_seeds_data(1)...), (tau=1.0,)),
    ("seeds_centered", () -> seeds_centered_model(make_seeds_data(1)...), (sigma=1.0,)),
    ("seeds_stanified", () -> seeds_stanified_model(make_seeds_data(1)...), (sigma=1.0,)),
    ("sesame_one_pred_a", () -> sesame_one_pred_a_model(make_sesame_data(1)...), (sigma=5.0,)),
    ("surgical_model", () -> surgical_model(make_surgical_data(1)...), (sigmasq=1.0,)),
]

function run_smoke_tests()
    n_pass = 0
    for (name, modelfn, init) in MODELS
        try
            m = modelfn()
            layout, θ0, store0 = build_layout(m; init=init)
            ldf = LogDensityFunction(m, layout, store0, AutoForwardDiff(); θ0=θ0)
            ok, val, maxerr = check_gradient(ldf, θ0)
            status = ok ? "OK" : "GRADIENT MISMATCH"
            println(rpad(name, 20), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  max grad err=", round(maxerr; sigdigits=3), "  ", status)
            ok && (n_pass += 1)
        catch e
            println(rpad(name, 20), "  FAILED — ", sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println()
    println(n_pass, " / ", length(MODELS), " models passed")
    return n_pass == length(MODELS)
end

abspath(PROGRAM_FILE) == (@__FILE__) && (run_smoke_tests() || exit(1))
