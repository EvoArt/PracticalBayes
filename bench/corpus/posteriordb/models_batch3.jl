# Third batch of TuringPosteriorDB.jl model ports (see models.jl for the
# overall approach/caveats). This batch is almost entirely MECHANICAL:
# three families of near-duplicate models (earn/logearn regression variants,
# mesquite/logmesquite regression variants, wells logistic-GLM variants),
# each already covered structurally by an earlier port (earn_height in
# models.jl; blr in models.jl; wells_dae_model in models.jl) — every model
# here is the SAME `filldist(Flat(), K) + MvNormal`/`.~ BernoulliLogit.(...)`
# shape with a different covariate transform or design-matrix construction,
# confirming the pattern generalizes rather than needing per-model
# workarounds.

using PracticalBayes
using Distributions
using Distributions: logistic
using LinearAlgebra: I
using Random
using Statistics: mean, std

zscore(x) = (x .- mean(x)) ./ std(x)

# ===========================================================================
# earn/logearn family — all `beta ~ filldist(Flat(), K); sigma ~ FlatPos(0);
# log_earn ~ MvNormal(<linear combination>, sigma^2*I)`, varying only which
# covariate transform (log/z-score/interaction) feeds the linear predictor.
# ===========================================================================
function make_earn_family_data(seed)
    rng = Random.Xoshiro(seed)
    N = 50
    height = randn(rng, N) .* 3 .+ 66
    male = Float64.(rand(rng, Bool, N))
    earn = exp.(8.0 .+ 0.02 .* height .+ 0.3 .* male .+ randn(rng, N) .* 0.4)
    return (N=N, earn=earn, height=height, male=male)
end

@model function log10earn_height_model(N, earn, height; log_earn=log10.(earn))
    beta ~ filldist(Flat(), 2)
    sigma ~ FlatPos(0.0)
    log_earn ~ MvNormal(beta[1] .+ beta[2] .* height, sigma^2 * I)
end

@model function logearn_height_male_model(N, earn, height, male; log_earn=log.(earn))
    beta ~ filldist(Flat(), 3)
    sigma ~ FlatPos(0.0)
    log_earn ~ MvNormal(beta[1] .+ beta[2] .* height .+ beta[3] .* male, sigma^2 * I)
end

@model function logearn_interaction_model(N, earn, height, male; log_earn=log.(earn))
    inter = height .* male
    beta ~ filldist(Flat(), 4)
    sigma ~ FlatPos(0.0)
    log_earn ~ MvNormal(beta[1] .+ beta[2] .* height .+ beta[3] .* male .+ beta[4] .* inter, sigma^2 * I)
end

@model function logearn_interaction_z_model(N, earn, height, male; log_earn=log.(earn))
    z_height = zscore(height)
    inter = z_height .* male
    beta ~ filldist(Flat(), 4)
    sigma ~ FlatPos(0.0)
    log_earn ~ MvNormal(beta[1] .+ beta[2] .* z_height .+ beta[3] .* male .+ beta[4] .* inter, sigma^2 * I)
end

@model function logearn_logheight_male_model(N, earn, height, male; log_earn=log.(earn))
    log_height = log.(height)
    beta ~ filldist(Flat(), 4)
    sigma ~ FlatPos(0.0)
    log_earn ~ MvNormal(beta[1] .+ beta[2] .* log_height .+ beta[3] .* male, sigma^2 * I)
end

# ===========================================================================
# mesquite/logmesquite family — same shape as `blr`, larger design matrices
# (up to 7 covariates), some with log-transformed covariates.
# ===========================================================================
function make_mesquite_data(seed)
    rng = Random.Xoshiro(seed)
    N = 40
    diam1 = abs.(randn(rng, N)) .* 0.5 .+ 1.0
    diam2 = abs.(randn(rng, N)) .* 0.5 .+ 1.0
    canopy_height = abs.(randn(rng, N)) .* 0.3 .+ 1.0
    total_height = abs.(randn(rng, N)) .* 0.3 .+ 1.5
    density = abs.(randn(rng, N)) .* 0.2 .+ 1.0
    group = Float64.(rand(rng, Bool, N))
    weight = abs.(2.0 .+ 0.5 .* diam1 .+ 0.3 .* diam2 .+ randn(rng, N) .* 0.3) .+ 0.5
    return (N=N, weight=weight, diam1=diam1, diam2=diam2, canopy_height=canopy_height, total_height=total_height, density=density, group=group)
end

@model function mesquite_model(N, weight, diam1, diam2, canopy_height, total_height, density, group)
    beta ~ filldist(Flat(), 7)
    sigma ~ FlatPos(0.0)
    weight ~ MvNormal(
        beta[1] .+ beta[2] .* diam1 .+ beta[3] .* diam2 .+ beta[4] .* canopy_height .+
        beta[5] .* total_height .+ beta[6] .* density .+ beta[7] .* group,
        sigma^2 .* I,
    )
end

@model function logmesquite_model(N, weight, diam1, diam2, canopy_height, total_height, density, group; log_weight=log.(weight))
    beta ~ filldist(Flat(), 7)
    sigma ~ FlatPos(0.0)
    log_weight ~ MvNormal(
        beta[1] .+ beta[2] .* log.(diam1) .+ beta[3] .* log.(diam2) .+ beta[4] .* log.(canopy_height) .+
        beta[5] .* log.(total_height) .+ beta[6] .* log.(density) .+ beta[7] .* group,
        sigma^2 .* I,
    )
end

@model function logmesquite_logva_model(N, weight, diam1, diam2, canopy_height, group; log_weight=log.(weight))
    log_canopy_volume = log.(diam1 .* diam2 .* canopy_height)
    log_canopy_area = log.(diam1 .* diam2)
    beta ~ filldist(Flat(), 4)
    sigma ~ FlatPos(0.0)
    log_weight ~ MvNormal(beta[1] .+ beta[2] .* log_canopy_volume .+ beta[3] .* log_canopy_area .+ beta[4] .* group, sigma^2 .* I)
end

@model function logmesquite_logvas_model(N, weight, diam1, diam2, canopy_height, total_height, density, group; log_weight=log.(weight))
    log_canopy_volume = log.(diam1 .* diam2 .* canopy_height)
    log_canopy_area = log.(diam1 .* diam2)
    log_canopy_shape = log.(diam1 ./ diam2)
    log_total_height = log.(total_height)
    log_density = log.(density)
    beta ~ filldist(Flat(), 7)
    sigma ~ FlatPos(0.0)
    log_weight ~ MvNormal(
        beta[1] .+ beta[2] .* log_canopy_volume .+ beta[3] .* log_canopy_area .+ beta[4] .* log_canopy_shape .+
        beta[5] .* log_total_height .+ beta[6] .* log_density .+ beta[7] .* group,
        sigma^2 .* I,
    )
end

@model function logmesquite_logvash_model(N, weight, diam1, diam2, canopy_height, total_height, group; log_weight=log.(weight))
    log_canopy_volume = log.(diam1 .* diam2 .* canopy_height)
    log_canopy_area = log.(diam1 .* diam2)
    log_canopy_shape = log.(diam1 ./ diam2)
    log_total_height = log.(total_height)
    beta ~ filldist(Flat(), 6)
    sigma ~ FlatPos(0.0)
    log_weight ~ MvNormal(
        beta[1] .+ beta[2] .* log_canopy_volume .+ beta[3] .* log_canopy_area .+ beta[4] .* log_canopy_shape .+
        beta[5] .* log_total_height .+ beta[6] .* group,
        sigma^2 .* I,
    )
end

@model function logmesquite_logvolume_model(N, weight, diam1, diam2, canopy_height; log_weight=log.(weight))
    log_canopy_volume = log.(diam1 .* diam2 .* canopy_height)
    beta ~ filldist(Flat(), 2)
    sigma ~ FlatPos(0.0)
    log_weight ~ MvNormal(beta[1] .+ beta[2] .* log_canopy_volume, sigma^2 .* I)
end

# ===========================================================================
# wells family — all `.~ BernoulliLogit.(...)` logistic GLMs (already
# vectorized in the ORIGINAL Turing source, so no `.~`-porting needed),
# varying only which covariates/interactions/centerings feed the linear
# predictor. `wells_dae_model` (models.jl) already covers the base case.
# ===========================================================================
function make_wells_family_data(seed)
    rng = Random.Xoshiro(seed)
    N = 60
    dist = abs.(randn(rng, N)) .* 50 .+ 20
    arsenic = abs.(randn(rng, N)) .* 2 .+ 1
    assoc = Float64.(rand(rng, Bool, N))
    educ = Float64.(rand(rng, 0:12, N))
    true_p = logistic.(0.2 .- 0.01 .* dist .- 0.1 .* arsenic)
    switched = Float64.(rand(rng, N) .< true_p)
    return (N=N, switched=switched, dist=dist, arsenic=arsenic, assoc=assoc, educ=educ)
end

@model function wells_dist_model(N, switched, dist)
    beta ~ filldist(Flat(), 2)
    switched .~ BernoulliLogit.(beta[1] .+ beta[2] .* dist)
end

@model function wells_dist100_model(N, switched, dist)
    c_dist100 = dist ./ 100
    alpha ~ Flat()
    beta ~ Flat()
    switched .~ BernoulliLogit.(alpha .+ c_dist100 .* beta)
end

@model function wells_dist100ars_model(N, switched, dist, arsenic)
    dist100 = dist ./ 100
    x = hcat(dist100, arsenic)
    alpha ~ Flat()
    beta ~ filldist(Flat(), 2)
    switched .~ BernoulliLogit.(alpha .+ x * beta)
end

@model function wells_dae_c_model(N, switched, dist, arsenic, educ)
    c_dist100 = (dist .- mean(dist)) ./ 100
    c_arsenic = arsenic .- mean(arsenic)
    da_inter = c_dist100 .* c_arsenic
    educ4 = educ ./ 4.0
    x = hcat(c_dist100, c_arsenic, da_inter, educ4)
    alpha ~ Flat()
    beta ~ filldist(Flat(), 4)
    switched .~ BernoulliLogit.(alpha .+ x * beta)
end

@model function wells_dae_inter_model(N, switched, dist, arsenic, educ)
    c_dist100 = (dist .- mean(dist)) ./ 100
    c_arsenic = arsenic .- mean(arsenic)
    c_educ4 = (educ .- mean(educ)) ./ 4.0
    da_inter = c_dist100 .* c_arsenic
    de_inter = c_dist100 .* c_educ4
    ae_inter = c_arsenic .* c_educ4
    x = hcat(c_dist100, c_arsenic, da_inter, c_educ4, de_inter, ae_inter)
    alpha ~ Flat()
    beta ~ filldist(Flat(), 6)
    switched .~ BernoulliLogit.(alpha .+ x * beta)
end

@model function wells_daae_c_model(N, switched, dist, arsenic, assoc, educ)
    c_dist100 = (dist .- mean(dist)) ./ 100
    c_arsenic = arsenic .- mean(arsenic)
    da_inter = c_dist100 .* c_arsenic
    educ4 = educ ./ 4.0
    x = hcat(c_dist100, c_arsenic, da_inter, assoc, educ4)
    alpha ~ Flat()
    beta ~ filldist(Flat(), 5)
    switched .~ BernoulliLogit.(alpha .+ x * beta)
end

@model function wells_interaction_model(N, switched, dist, arsenic)
    dist100 = dist ./ 100
    inter = dist100 .* arsenic
    x = hcat(dist100, arsenic, inter)
    alpha ~ Flat()
    beta ~ filldist(Flat(), 3)
    switched .~ BernoulliLogit.(alpha .+ x * beta)
end

@model function wells_interaction_c_model(N, switched, dist, arsenic)
    c_dist100 = (dist .- mean(dist)) ./ 100
    c_arsenic = arsenic .- mean(arsenic)
    inter = c_dist100 .* c_arsenic
    x = hcat(c_dist100, c_arsenic, inter)
    alpha ~ Flat()
    beta ~ filldist(Flat(), 3)
    switched .~ BernoulliLogit.(alpha .+ x * beta)
end
