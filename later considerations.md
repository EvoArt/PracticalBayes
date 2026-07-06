[DONE 2026-07-06] Allow errors during inference to be treated as rejected samples. — `LogDensityFunction(...; reject_errors=true)`, see devlog.md.

Allow fitting of Frequntist GLM from @model

Interface with Latte.jl, pigeons.jl and paralleltempering.jl

[DONE 2026-07-06] Allow user to specify parameters not to be tracked. Useful for very large models with many "nuisance" parameters. — `build_layout(...; untracked=(...))`, see devlog.md.

Exploit conjugacy. Perhaps some rule based sytem using Symbolics.jl or similar to simplify models.

Build a brms type layer on top (separate package) with a load of tutorials and standard model.

[DONE 2026-07-06] add mle, map, laplace approx. — `maximum_a_posteriori`/`maximum_likelihood`/`laplace_approximation` in `src/optimize.jl`, hand-rolled L-BFGS default + Optimization.jl weak-dep extension, see devlog.md.

Do statistical rethinking, BDA3

Figure out how to combine with Reactant.jl