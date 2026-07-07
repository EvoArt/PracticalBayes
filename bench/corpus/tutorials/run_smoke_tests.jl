# Smoke-tests every ported model in models.jl — see
# bench/corpus/posteriordb/run_smoke_tests.jl for the methodology (structural
# check: logdensity finite + gradient matches finite differences; not a
# posterior-accuracy check against the original tutorial's real data).

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
    ("coin-flipping", () -> coinflip(make_coinflip_data(1).y), (p=0.5,)),
    ("bayesian-linear-regression", () -> linear_regression(make_linear_regression_data(1)...), (σ²=1.0,)),
    ("bayesian-logistic-regression", () -> logistic_regression(make_logistic_regression_data(1)...), NamedTuple()),
    ("bayesian-poisson-regression", () -> poisson_regression(make_poisson_regression_data(1)...), NamedTuple()),
    ("multinomial-logistic-regression", () -> multinomial_logistic_regression(make_multinomial_data(1)...), NamedTuple()),
    ("bayesian-time-series-analysis", () -> decomp_model(make_decomp_data(1)...), (σ=0.05,)),
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
            println(rpad(name, 34), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  max grad err=", round(maxerr; sigdigits=3), "  ", status)
            ok && (n_pass += 1)
        catch e
            println(rpad(name, 34), "  FAILED — ", sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println()
    println(n_pass, " / ", length(MODELS), " models passed (logdensity finite + gradient matches finite differences)")
    return n_pass == length(MODELS)
end

abspath(PROGRAM_FILE) == (@__FILE__) && (run_smoke_tests() || exit(1))
