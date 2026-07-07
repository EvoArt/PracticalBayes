# Second batch of TuringLang docs tutorial ports. `variational-inference`
# isn't ported separately here — its model IS `linear_regression` (models.jl),
# reused verbatim; the tutorial is about the INFERENCE method (ADVI), not a
# new model, and PracticalBayes doesn't implement VI (NUTS/full-Bayes is the
# stated primary target this whole session — see devlog).
#
# `probabilistic-pca` — probabilistic PCA: `filldist(Normal(), D, k)` (a 2D
# filldist, unlike every earlier filldist use which was 1D), and
# `arraydist([MvNormal(m, I) for m in eachcol(genes_mean')])` (an array of
# per-column MvNormals as one observe site). First porting attempt got the
# orientation wrong (dropped the tutorial's `genes_mean'` transpose + the
# `reshape(μ, n_genes, 1)` broadcast, silently producing a genuinely
# incompatible logpdf DimensionMismatch, caught immediately by actually
# running it) — this version matches the original source exactly: `X` is
# `N × D` (`N, D = size(X)`), `genes_mean` is `D × N` (`W::D×k * Z::k×N`),
# so `eachcol(genes_mean')` iterates `N` columns each of length `D`,
# matching `X`'s `N` rows.

using PracticalBayes
using Distributions
using LinearAlgebra: I
using Random

@model function pPCA(X, k::Int)
    N, D = size(X)
    W ~ filldist(Normal(), D, k)
    Z ~ filldist(Normal(), k, N)
    mu ~ MvNormal(fill(0.0, D), I)
    genes_mean = W * Z .+ mu
    return X ~ arraydist([MvNormal(m, I) for m in eachcol(genes_mean')])
end
function make_pPCA_data(seed)
    rng = Random.Xoshiro(seed)
    N, D, k = 20, 4, 2
    true_W = randn(rng, D, k)
    true_Z = randn(rng, k, N)
    X = (true_W * true_Z)' .+ randn(rng, N, D) .* 0.3
    return (X=X, k=k)
end
