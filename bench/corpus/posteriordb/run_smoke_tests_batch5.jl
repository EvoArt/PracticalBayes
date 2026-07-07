# Smoke tests for models_batch5.jl — see run_smoke_tests.jl for methodology.

using PracticalBayes
using ADTypes: AutoForwardDiff
import LogDensityProblems

include("models_batch5.jl")

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

radon_hier_init = (mu_alpha=0.0, sigma_alpha=1.0, sigma_y=1.0)

const MODELS = [
    ("radon_county", () -> (d = make_radon_data(1); radon_county_model(d.N, d.J, d.county_idx, d.log_radon)), (mu_a=0.0, sigma_a=1.0, sigma_y=1.0)),
    ("radon_pooled", () -> (d = make_radon_data(1); radon_pooled_model(d.N, d.floor_measure, d.log_radon)), (sigma_y=1.0,)),
    ("radon_hierarchical_intercept_centered", () -> radon_hierarchical_intercept_centered_model(make_radon_data(1)...), radon_hier_init),
    ("radon_hierarchical_intercept_noncentered", () -> radon_hierarchical_intercept_noncentered_model(make_radon_data(1)...), (sigma_alpha=1.0, sigma_y=1.0, mu_alpha=0.0)),
    ("radon_partially_pooled_centered", () -> (d = make_radon_data(1); radon_partially_pooled_centered_model(d.N, d.J, d.county_idx, d.log_radon)), (sigma_y=1.0, sigma_alpha=1.0, mu_alpha=0.0)),
    ("radon_variable_intercept_centered", () -> (d = make_radon_data(1); radon_variable_intercept_centered_model(d.J, d.N, d.county_idx, d.floor_measure, d.log_radon)), (sigma_y=1.0, sigma_alpha=1.0, mu_alpha=0.0)),
    ("radon_variable_intercept_noncentered", () -> (d = make_radon_data(1); radon_variable_intercept_noncentered_model(d.J, d.N, d.county_idx, d.floor_measure, d.log_radon)), (mu_alpha=0.0, sigma_alpha=1.0, sigma_y=0.5)),
    ("radon_county_intercept", () -> (d = make_radon_data(1); radon_county_intercept_model(d.N, d.J, d.county_idx, d.floor_measure, d.log_radon)), (sigma_y=1.0,)),
    ("radon_variable_slope_centered", () -> (d = make_radon_data(1); radon_variable_slope_centered_model(d.J, d.N, d.county_idx, d.floor_measure, d.log_radon)), (sigma_y=1.0, sigma_beta=1.0, mu_beta=0.0)),
    ("radon_variable_intercept_slope_centered", () -> (d = make_radon_data(1); radon_variable_intercept_slope_centered_model(d.N, d.J, d.county_idx, d.floor_measure, d.log_radon)), (sigma_y=1.0, sigma_beta=1.0, sigma_alpha=1.0, mu_alpha=0.0, mu_beta=0.0)),
    ("radon_variable_intercept_slope_noncentered", () -> (d = make_radon_data(1); radon_variable_intercept_slope_noncentered_model(d.N, d.J, d.county_idx, d.floor_measure, d.log_radon)), (sigma_y=1.0, sigma_beta=1.0, sigma_alpha=1.0, mu_alpha=0.0, mu_beta=0.0)),
    ("logistic_regression_rhs", () -> logistic_regression_rhs_model(make_logistic_rhs_data(1)...), (beta0=0.0,)),
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
            println(rpad(name, 44), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  max grad err=", round(maxerr; sigdigits=3), "  ", status)
            ok && (n_pass += 1)
        catch e
            println(rpad(name, 44), "  FAILED — ", sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println()
    println(n_pass, " / ", length(MODELS), " models passed")
    return n_pass == length(MODELS)
end

abspath(PROGRAM_FILE) == (@__FILE__) && (run_smoke_tests() || exit(1))
