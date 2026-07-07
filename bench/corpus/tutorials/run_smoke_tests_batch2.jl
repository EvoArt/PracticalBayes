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

const MODELS = [
    ("probabilistic-pca", () -> pPCA(make_pPCA_data(1)...), NamedTuple()),
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
            println(rpad(name, 22), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  max grad err=", round(maxerr; sigdigits=3), "  ", status)
            ok && (n_pass += 1)
        catch e
            println(rpad(name, 22), "  FAILED — ", sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println()
    println(n_pass, " / ", length(MODELS), " models passed")
    return n_pass == length(MODELS)
end

abspath(PROGRAM_FILE) == (@__FILE__) && (run_smoke_tests() || exit(1))
