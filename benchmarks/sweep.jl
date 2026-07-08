import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.develop(Pkg.PackageSpec(path=normpath(joinpath(@__DIR__, ".."))))

import PracticalBayes
import Turing
import DynamicPPL
import ADTypes
import Distributions
import BenchmarkTools
import Random
import LogDensityProblems
import StatsFuns
import LinearAlgebra
import CairoMakie
import Enzyme
import JSON3

const ENZYME_MODE = Enzyme.set_runtime_activity(Enzyme.Reverse)
const LIKELIHOODS = (:normal, :poisson, :bernoulli_logit)
const PRECISIONS = (Float64, Float32)
const NS = (50, 200, 1_000, 5_000, 20_000)
const KS = (2, 10, 50, 100, 200)
const BENCH_SAMPLES = 20
const BENCH_EVALS = 1

PracticalBayes.@model function pb_regression_normal(X, y)
    pT = PracticalBayes.paramtype(__mode__)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    sigma ~ Distributions.Exponential(one(pT))
    eta = X * beta
    y ~ Distributions.MvNormal(eta, sigma^2 * LinearAlgebra.I)
end

PracticalBayes.@model function pb_regression_poisson(X, y)
    pT = PracticalBayes.paramtype(__mode__)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    eta = X * beta
    y ~ PracticalBayes.arraydist(PracticalBayes.LogPoisson.(eta))
end

PracticalBayes.@model function pb_regression_bernoulli_logit(X, y)
    pT = PracticalBayes.paramtype(__mode__)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(pT, k), LinearAlgebra.I)
    eta = X * beta
    y ~ PracticalBayes.arraydist(Distributions.BernoulliLogit.(eta))
end

Turing.@model function turing_regression_normal(X, y)
    T = eltype(X)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(T, k), LinearAlgebra.I)
    sigma ~ Distributions.Exponential(one(T))
    eta = X * beta
    y ~ Distributions.MvNormal(eta, sigma^2 * LinearAlgebra.I)
end

Turing.@model function turing_regression_poisson(X, y)
    T = eltype(X)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(T, k), LinearAlgebra.I)
    eta = X * beta
    y ~ Turing.arraydist(Distributions.Poisson.(exp.(eta)))
end

Turing.@model function turing_regression_bernoulli_logit(X, y)
    T = eltype(X)
    k = size(X, 2)
    beta ~ Distributions.MvNormal(zeros(T, k), LinearAlgebra.I)
    eta = X * beta
    y ~ Turing.arraydist(Distributions.BernoulliLogit.(eta))
end

function pb_regression(X, y, likelihood::Symbol)
    likelihood === :normal && return pb_regression_normal(X, y)
    likelihood === :poisson && return pb_regression_poisson(X, y)
    likelihood === :bernoulli_logit && return pb_regression_bernoulli_logit(X, y)
    error("unknown likelihood $likelihood")
end

function turing_regression(X, y, likelihood::Symbol)
    likelihood === :normal && return turing_regression_normal(X, y)
    likelihood === :poisson && return turing_regression_poisson(X, y)
    likelihood === :bernoulli_logit && return turing_regression_bernoulli_logit(X, y)
    error("unknown likelihood $likelihood")
end

function make_regression_data(::Type{T}; n::Int, nparams::Int, likelihood::Symbol, seed::Int) where {T<:Real}
    rng = Random.Xoshiro(seed)
    X = T.(randn(rng, n, nparams))
    true_beta = T.(randn(rng, nparams) .* 0.5)
    eta = X * true_beta
    y = if likelihood === :normal
        eta .+ T.(randn(rng, n) .* 0.5)
    elseif likelihood === :poisson
        T.(rand.(rng, Distributions.Poisson.(exp.(clamp.(eta, T(-10), T(10))))))
    elseif likelihood === :bernoulli_logit
        T.(rand.(rng, Distributions.Bernoulli.(StatsFuns.logistic.(eta))))
    else
        error("unknown likelihood $likelihood")
    end
    return X, y
end

function backend_specs()
    return Dict(
        :forwarddiff => ADTypes.AutoForwardDiff(),
        :mooncake => ADTypes.AutoMooncake(; config=nothing),
        :enzyme => ADTypes.AutoEnzyme(; mode=ENZYME_MODE),
    )
end

function median_gradient_time_ns(ldf, θ; samples::Int=BENCH_SAMPLES, evals::Int=BENCH_EVALS)
    LogDensityProblems.logdensity_and_gradient(ldf, θ)
    trial = BenchmarkTools.@benchmark LogDensityProblems.logdensity_and_gradient($ldf, $θ) samples=samples evals=evals
    return Float64(BenchmarkTools.median(trial).time)
end

function build_ldf_pair(X, y, likelihood::Symbol, adtype, T::Type{<:Real}; seed::Int=1)
    pb_model = pb_regression(X, y, likelihood)
    pb_layout, pb_θ0, pb_store0 = PracticalBayes.build_layout(pb_model; T=T)
    pb_ldf = PracticalBayes.LogDensityFunction(pb_model, pb_layout, pb_store0, adtype; θ0=pb_θ0)

    turing_model = turing_regression(X, y, likelihood)
    turing_ldf = DynamicPPL.LogDensityFunction(turing_model; adtype=adtype)
    turing_vi = DynamicPPL.VarInfo(Random.Xoshiro(seed), turing_model)
    turing_θ0 = DynamicPPL.link(turing_vi, turing_model)[:]

    return pb_ldf, pb_θ0, turing_ldf, turing_θ0
end

function ratio_matrix(results, backend::Symbol, likelihood::Symbol, T::Type, Ns, Ks)
    M = fill(NaN, length(Ks), length(Ns))
    for (i_k, k) in enumerate(Ks), (i_n, n) in enumerate(Ns)
        key = (T, likelihood, n, k)
        cell = get(results, key, nothing)
        cell === nothing && continue
        entry = get(cell, backend, nothing)
        entry === nothing && continue
        pb_ns, tu_ns = entry
        if isfinite(pb_ns) && isfinite(tu_ns) && tu_ns > 0
            M[i_k, i_n] = pb_ns / tu_ns
        end
    end
    return M
end

function fastest_ratio_matrix(results, likelihood::Symbol, T::Type, Ns, Ks)
    M = fill(NaN, length(Ks), length(Ns))
    for (i_k, k) in enumerate(Ks), (i_n, n) in enumerate(Ns)
        key = (T, likelihood, n, k)
        cell = get(results, key, nothing)
        cell === nothing && continue

        pb_times = Float64[]
        tu_times = Float64[]
        for (_, (pb_ns, tu_ns)) in cell
            isfinite(pb_ns) && push!(pb_times, pb_ns)
            isfinite(tu_ns) && push!(tu_times, tu_ns)
        end
        isempty(pb_times) && continue
        isempty(tu_times) && continue

        pb_best = minimum(pb_times)
        tu_best = minimum(tu_times)
        if tu_best > 0
            M[i_k, i_n] = pb_best / tu_best
        end
    end
    return M
end

function heatmap_figure(mats, Ns, Ks, likelihoods, precisions; title::String)
    logmats = Dict{Tuple{DataType,Symbol},Matrix{Float64}}()
    vals = Float64[]

    for T in precisions, lik in likelihoods
        M = mats[(T, lik)]
        LM = fill(NaN, size(M))
        for idx in eachindex(M)
            if isfinite(M[idx]) && M[idx] > 0
                LM[idx] = log2(M[idx])
                push!(vals, LM[idx])
            end
        end
        logmats[(T, lik)] = LM
    end

    maxabs = isempty(vals) ? 1.0 : maximum(abs, vals)
    maxabs = max(maxabs, 1e-9)

    fig = CairoMakie.Figure(size=(1700, 900))
    fig[0, 1:3] = CairoMakie.Label(fig, title, fontsize=22, font=:bold)

    hm = nothing
    for (r, T) in enumerate(precisions), (c, lik) in enumerate(likelihoods)
        ax = CairoMakie.Axis(
            fig[r, c],
            title="$(lik) | $(T)",
            xlabel="N",
            ylabel="NPARAMS",
            xticks=(1:length(Ns), string.(Ns)),
            yticks=(1:length(Ks), string.(Ks)),
            xgridvisible=false,
            ygridvisible=false,
        )

        M = logmats[(T, lik)]
        hm = CairoMakie.heatmap!(
            ax,
            1:length(Ns),
            1:length(Ks),
            M;
            colormap=:RdBu,
            colorrange=(-maxabs, maxabs),
            lowclip=:black,
            highclip=:black,
        )
    end

    CairoMakie.Colorbar(fig[:, 4], hm; label="log2(PB / Turing) median gradient time")
    return fig
end

function main()
    likelihoods = LIKELIHOODS
    precisions = PRECISIONS
    Ns = NS
    Ks = KS

    println("Running sweep")
    println("N values: ", collect(Ns))
    println("NPARAMS values: ", collect(Ks))

    backends = backend_specs()
    results = Dict{Tuple{DataType,Symbol,Int,Int},Dict{Symbol,Tuple{Float64,Float64}}}()

    for T in precisions, lik in likelihoods, n in Ns, k in Ks
        key = (T, lik, n, k)
        cell = Dict{Symbol,Tuple{Float64,Float64}}()
        results[key] = cell

        println("\n--- T=$(T), likelihood=$(lik), N=$(n), K=$(k) ---")
        X, y = make_regression_data(T; n=n, nparams=k, likelihood=lik, seed=1)

        for (bname, adtype) in backends
            pb_ns = Inf
            tu_ns = Inf
            try
                pb_ldf, pb_θ0, turing_ldf, turing_θ0 = build_ldf_pair(X, y, lik, adtype, T; seed=11)
                pb_ns = median_gradient_time_ns(pb_ldf, pb_θ0)
                tu_ns = median_gradient_time_ns(turing_ldf, turing_θ0)
                println("  $(bname): PB=$(round(pb_ns/1e6; digits=3)) ms, Turing=$(round(tu_ns/1e6; digits=3)) ms, ratio=$(round(pb_ns/tu_ns; digits=3))")
            catch e
                println("  $(bname): FAILED — ", sprint(showerror, e)[1:min(end, 220)])
            end
            cell[bname] = (pb_ns, tu_ns)
        end
    end

    figdir = joinpath(@__DIR__, "figures")
    resdir = joinpath(@__DIR__, "results")
    mkpath(figdir)
    mkpath(resdir)

    fig1_mats = Dict((T, lik) => ratio_matrix(results, :forwarddiff, lik, T, Ns, Ks) for T in precisions for lik in likelihoods)
    fig2_mats = Dict((T, lik) => ratio_matrix(results, :mooncake, lik, T, Ns, Ks) for T in precisions for lik in likelihoods)
    fig3_mats = Dict((T, lik) => ratio_matrix(results, :enzyme, lik, T, Ns, Ks) for T in precisions for lik in likelihoods)
    fig4_mats = Dict((T, lik) => fastest_ratio_matrix(results, lik, T, Ns, Ks) for T in precisions for lik in likelihoods)

    fig1 = heatmap_figure(fig1_mats, Ns, Ks, likelihoods, precisions; title="Figure 1 — ForwardDiff gradient median time ratio (PB/Turing)")
    fig2 = heatmap_figure(fig2_mats, Ns, Ks, likelihoods, precisions; title="Figure 2 — Mooncake gradient median time ratio (PB/Turing)")
    fig3 = heatmap_figure(fig3_mats, Ns, Ks, likelihoods, precisions; title="Figure 3 — Enzyme gradient median time ratio (PB/Turing)")
    fig4 = heatmap_figure(fig4_mats, Ns, Ks, likelihoods, precisions; title="Figure 4 — Fastest backend per PPL ratio (PB/Turing)")

    CairoMakie.save(joinpath(figdir, "fig1_forwarddiff.png"), fig1)
    CairoMakie.save(joinpath(figdir, "fig2_mooncake.png"), fig2)
    CairoMakie.save(joinpath(figdir, "fig3_enzyme.png"), fig3)
    CairoMakie.save(joinpath(figdir, "fig4_fastest_per_ppl.png"), fig4)

    rows = Any[]
    for T in precisions, lik in likelihoods, n in Ns, k in Ks
        cell = results[(T, lik, n, k)]
        push!(rows, Dict(
            "precision" => string(T),
            "likelihood" => String(lik),
            "N" => n,
            "NPARAMS" => k,
            "forwarddiff" => Dict("pb_ns" => cell[:forwarddiff][1], "turing_ns" => cell[:forwarddiff][2]),
            "mooncake" => Dict("pb_ns" => cell[:mooncake][1], "turing_ns" => cell[:mooncake][2]),
            "enzyme" => Dict("pb_ns" => cell[:enzyme][1], "turing_ns" => cell[:enzyme][2]),
            "fastest_per_ppl_ratio" => fastest_ratio_matrix(results, lik, T, (n,), (k,))[1, 1],
        ))
    end

    payload = Dict(
        "meta" => Dict(
            "n_values" => collect(Ns),
            "nparams_values" => collect(Ks),
            "likelihoods" => String.(likelihoods),
            "precisions" => string.(precisions),
            "bench_samples" => BENCH_SAMPLES,
            "bench_evals" => BENCH_EVALS,
        ),
        "rows" => rows,
    )

    open(joinpath(resdir, "sweep.json"), "w") do io
        JSON3.pretty(io, payload)
    end

    println("Saved figures to: ", figdir)
    println("Saved results to: ", joinpath(resdir, "sweep.json"))
end

main()
