# Smoke tests for models_batch3.jl — see run_smoke_tests.jl for methodology.

using PracticalBayes
using ADTypes: AutoForwardDiff
import LogDensityProblems

include("models_batch3.jl")

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

earn_init = (sigma=5000.0,)
mesquite_init = (sigma=1.0,)

const MODELS = [
    ("log10earn_height", () -> (d = make_earn_family_data(1); log10earn_height_model(d.N, d.earn, d.height)), earn_init),
    ("logearn_height_male", () -> logearn_height_male_model(make_earn_family_data(1)...), earn_init),
    ("logearn_interaction", () -> logearn_interaction_model(make_earn_family_data(1)...), earn_init),
    ("logearn_interaction_z", () -> logearn_interaction_z_model(make_earn_family_data(1)...), earn_init),
    ("logearn_logheight_male", () -> logearn_logheight_male_model(make_earn_family_data(1)...), earn_init),
    ("mesquite", () -> mesquite_model(make_mesquite_data(1)...), mesquite_init),
    ("logmesquite", () -> logmesquite_model(make_mesquite_data(1)...), mesquite_init),
    ("logmesquite_logva", () -> (d = make_mesquite_data(1); logmesquite_logva_model(d.N, d.weight, d.diam1, d.diam2, d.canopy_height, d.group)), mesquite_init),
    ("logmesquite_logvas", () -> logmesquite_logvas_model(make_mesquite_data(1)...), mesquite_init),
    ("logmesquite_logvash", () -> (d = make_mesquite_data(1); logmesquite_logvash_model(d.N, d.weight, d.diam1, d.diam2, d.canopy_height, d.total_height, d.group)), mesquite_init),
    ("logmesquite_logvolume", () -> (d = make_mesquite_data(1); logmesquite_logvolume_model(d.N, d.weight, d.diam1, d.diam2, d.canopy_height)), mesquite_init),
    ("wells_dist", () -> (d = make_wells_family_data(1); wells_dist_model(d.N, d.switched, d.dist)), NamedTuple()),
    ("wells_dist100", () -> (d = make_wells_family_data(1); wells_dist100_model(d.N, d.switched, d.dist)), NamedTuple()),
    ("wells_dist100ars", () -> (d = make_wells_family_data(1); wells_dist100ars_model(d.N, d.switched, d.dist, d.arsenic)), NamedTuple()),
    ("wells_dae_c", () -> (d = make_wells_family_data(1); wells_dae_c_model(d.N, d.switched, d.dist, d.arsenic, d.educ)), NamedTuple()),
    ("wells_dae_inter", () -> (d = make_wells_family_data(1); wells_dae_inter_model(d.N, d.switched, d.dist, d.arsenic, d.educ)), NamedTuple()),
    ("wells_daae_c", () -> (d = make_wells_family_data(1); wells_daae_c_model(d.N, d.switched, d.dist, d.arsenic, d.assoc, d.educ)), NamedTuple()),
    ("wells_interaction", () -> (d = make_wells_family_data(1); wells_interaction_model(d.N, d.switched, d.dist, d.arsenic)), NamedTuple()),
    ("wells_interaction_c", () -> (d = make_wells_family_data(1); wells_interaction_c_model(d.N, d.switched, d.dist, d.arsenic)), NamedTuple()),
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
            println(rpad(name, 26), "  dim=", rpad(layout.dim, 4), "  logdensity=", round(val; digits=3), "  max grad err=", round(maxerr; sigdigits=3), "  ", status)
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
