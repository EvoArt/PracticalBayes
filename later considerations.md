[DONE 2026-07-06] Allow errors during inference to be treated as rejected samples. — `LogDensityFunction(...; reject_errors=true)`, see devlog.md.

Allow fitting of Frequntist GLM from @model

Interface with Latte.jl, pigeons.jl and paralleltempering.jl

[DONE 2026-07-06] Allow user to specify parameters not to be tracked. Useful for very large models with many "nuisance" parameters. — `build_layout(...; untracked=(...))`, see devlog.md.

Exploit conjugacy. Perhaps some rule based sytem using Symbolics.jl or similar to simplify models.

Build a brms type layer on top (separate package) with a load of tutorials and standard model.

[DONE 2026-07-06] add mle, map, laplace approx. — `maximum_a_posteriori`/`maximum_likelihood`/`laplace_approximation` in `src/optimize.jl`, hand-rolled L-BFGS default + Optimization.jl weak-dep extension, see devlog.md.

Do statistical rethinking, BDA3

Figure out how to combine with Reactant.jl

plotting, prior and posterior predictive vs data.

loglik (tracked optionally) for psisloo etc.

interface with arviz - save in arviz format.

default to pretty printing results. maybe similar to mcelreath package.

residual analysis and plots.

R and python wrappers: performance with simple syntax, means R and python users can benefit from the package without having to know lots of (any?) julia.

Specific ODE and latent variable interfaces that are easy to wrap for R and python use.

handle missing data. [DONE — already handled: bare missing/nothing = "sample
as parameter" for scalar ~, fill(missing, n) shape convention for .~, both
predate this note.]

Docs pass done (2026-07-10): docstring audit across all exported names found
only 6 real gaps (Accum-level logprior/loglikelihood_/logjoint,
PriorMode/FixedMode structs, re-exported SymChain) — the rest of src/ already
had substantive docstrings, contrary to what this note assumed. All 6 gaps
filled. docs/src/ expanded with predictive.md (M4 predict.jl + new M6
pointwise_loglikelihoods) and float32_gpu.md (Float32 usage + GPU scope).
Along the way, found and fixed a real pre-existing bug: index.md's own
flagship example (`arraydist(Normal.(μ, σ))` with scalar μ/σ) was broken and
had been silently failing this whole time — masked by `warnonly=true`.
DocumenterVitepress.jl was tried, but the user said plain Documenter is fine
for now ("you can use documenter for now, if thats easier") — revisit later
if the plain-Documenter site's presentation becomes a real pain point.

epiaware interface https://composableturingidmodels.epiaware.org/dev/

drop @addlogprob for epi produces likelihoods. us ~ instead