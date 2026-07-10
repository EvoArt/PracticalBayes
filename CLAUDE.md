# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this package is

PracticalBayes.jl is a from-scratch probabilistic programming package: Turing-style
`@model`/`~` syntax, but a redesigned evaluation core aimed at (1) making observe
statements as cheap as `Turing.@addlogprob!`, (2) first-class support for custom
latent-variable samplers (e.g. FFBS) via `Gibbs`, and (3) a Float32-first parameter
path. It depends on the Turing ecosystem's own packages (AbstractPPL, AdvancedHMC,
Bijectors, Distributions, LogDensityProblems) but **not on DynamicPPL/Turing itself**
for its runtime — compiler and evaluation core are original.

Public repo: `github.com/EvoArt/PracticalBayes` (remote `origin`, branch `master`,
docs on `gh-pages`). CI runs on every push/PR; a separate benchmark-sweep workflow
runs on push to `master` and commits updated figures/README tables back automatically.

## Repository layout — public vs. private split (important)

`.gitignore` deliberately excludes a set of local-only files from version control:
`devlog.md`, `plan.md`, `later considerations.md`, and the entire `/bench` directory.
These exist on disk and are useful working context, but **they are not pushed and
not visible in the public repo**. When asked to "commit" or "push," never assume
these should go along — they're intentionally kept local. `bench/` in particular
contains large exploratory benchmark scripts, REPL playgrounds, and result JSONL
files that are treated as scratch/private, distinct from the *public* benchmark
harness in `benchmarks/` (no leading path overlap; `benchmarks/` is tracked, tested
by CI, and is what generates the README's benchmark table).

Tracked/public structure:
- `src/` — the package itself.
- `test/` — `Pkg.test()` suite, run by CI (`test/runtests.jl` is the entry point;
  add new test files to its `include(...)` list or they silently don't run).
- `test/comparison_env/` — a **separate** Julia environment (own `Project.toml`)
  used to freeze Turing's numbers for `test/turing_comparison.jl` to check against.
  Turing and PracticalBayes historically could not load in the same session; this
  was fixed (see devlog, "self-inflicted compat bound" entry) but the separate
  environment is kept anyway since Turing's dependency tree is large/slow to
  precompile and this keeps it out of the default `Pkg.test()` loop.
- `benchmarks/` — the small, CI-run benchmark sweep (`sweep.jl`) that regenerates
  `README.md`'s benchmark table and figures. Has its own `Project.toml`.
- `docs/` — Documenter.jl source, deployed to `gh-pages` via `.github/workflows/docs.yml`.
- `ext/PracticalBayesOptimizationExt.jl` — weak-dep extension activated by `Optimization.jl`.

Untracked/private (still worth reading, just don't assume they're pushed):
- `devlog.md` — chronological log of design decisions, bugs found, and their fixes.
  Read this before assuming something is undocumented; it's usually the first place
  a past investigation's reasoning was recorded. Newest entries at the top.
- `plan.md` — the original approved architecture plan (milestones M0–M6, key types,
  file skeleton). Still the closest thing to a from-scratch design reference.
- `later considerations.md` — informal backlog / wishlist, not commitments.
- `bench/` — exploratory benchmarking: corpus of ~55 PosteriorDB/tutorial model
  ports, a PB-vs-Turing REPL playground script, report/heatmap generators. See
  devlog for the many real bugs found and fixed while building this out.

## Non-obvious facts worth knowing before making changes

- **`@model`'s compiler does not support `::Val{x}) where {x}`-style destructured
  type-parameter positional arguments** — `_argnames` chokes on them. Use a plain
  `Symbol`/typed positional argument and branch at runtime instead.
- **PracticalBayes and Turing/DynamicPPL export several colliding names**: `@model`,
  `Model`, `LogDensityFunction`, `filldist`, `arraydist`. If a script/test ever needs
  both packages loaded together, use `import` (not `using`) and qualify every call —
  confirmed directly that plain `using` on both makes these names ambiguous and
  errors the moment they're referenced unqualified.
- **Model-writing choice matters a lot for reverse-mode AD (Mooncake/Enzyme)
  performance**: a per-element loop prior (`beta = Vector{paramtype(__mode__)}(undef,k);
  for j in 1:k; beta[j] ~ Normal(0,1); end`) allocates `beta` fresh inside the
  differentiated model body on every call, forcing Mooncake to rebuild a tangent
  (`IdDict` allocation) every gradient call instead of once at prep time — confirmed
  via direct allocation profiling, ~2.8x more allocation than the vectorized
  equivalent (`beta ~ MvNormal(zeros(paramtype(__mode__), k), I)`). This pattern is
  still correct/necessary for genuinely index-varying priors (hierarchical/group
  effects) — it's specifically the IID-over-a-fixed-vector case that has a faster
  alternative. Same logic applies to observes: `.~ Normal.(...)` (scalar-broadcast)
  hits a fast `Distributions.loglikelihood` path, but `.~` over an array of *distinct*
  per-element distributions falls through a slower `sum(logpdf.(...))` fallback —
  use `arraydist(...)`/`MvNormal(...)` there instead.
- **Float32 needs care beyond `build_layout(...; T=Float32)`.** Distribution literals
  need `f0` suffixes (`Exponential(1.0f0)`, not `Exponential(1)`) and vector-valued
  prior means need `zeros(paramtype(__mode__), k)`, not bare `zeros(k)` — otherwise
  the computation silently promotes back to Float64 even with a Float32 `θ0`. Data
  (`X`/`y`) can stay Float64 without a performance penalty or breaking Float32
  propagation through the gradient — confirmed directly.
- **`AdvancedHMC.NUTS(δ)` fixes its internal step-size type to `typeof(δ)`.** Passing
  a bare `0.8` (Float64) with a Float32 `θ0` fails inside step-size adaptation. Use
  `eltype(θ0)(0.8)`.
- **When comparing PB against Turing for performance, match `n_adapts` explicitly.**
  `AdvancedHMC`'s own default (`n_adapts = min(N÷10, 1000)`) and Turing's own default
  (`n_adapts = min(1000, N÷2)`, plus it discards all of it) are materially different
  budgets — this alone can make PB look several times slower on hierarchical models
  when the real cause is under-adaptation, not raw eval speed. Pass `n_adapts`/
  `discard_initial` explicitly and identically on both sides.
- **Turing/DynamicPPL promotes to Float64 internally regardless of input vector
  element type** — Float32 benchmarking is a PB-only lever; don't bother sweeping it
  on the Turing side.
- **The Bijectors/AbstractPPL compat "Turing can't coexist with PB" story is
  resolved** (see devlog "self-inflicted compat bound" entry) — it was PracticalBayes'
  own `Bijectors` compat bound allowing 0.16.x (which first required AbstractPPL 0.15),
  not a real API conflict. `Bijectors = "0.15.24"` (pinned, no 0.16) + `AbstractPPL =
  "0.14, 0.15"` lets Turing 0.45 install directly into the main environment as a
  test-only dependency.
- **`src/sample.jl` exists and is tested** (`AbstractMCMC.sample` overload for `Model`
  + `AdvancedHMC.AbstractHMCSampler`, returns a `FlexiChains.SymChain` by default;
  `chain_type=nothing` gives raw `AdvancedHMC.Transition`s). Verified against the
  analytic conjugate-Normal reference (`test/sample.jl`, within 3 MC-SE, matching
  `test/turing_comparison.jl`'s own reference). Chain output uses **FlexiChains**, not
  MCMCChains (the plan's original choice) — `MCMCChains` was a declared-but-unused dep
  and was removed. Key non-obvious implementation detail:
  `FlexiChains.to_nt_and_stats(nt::NamedTuple)` puts the WHOLE NamedTuple into the
  params half and returns empty stats — merging invlink'd params and AdvancedHMC's
  stat NamedTuple into one flat NamedTuple before bundling loses the params/stats
  split entirely (confirmed: every AdvancedHMC diagnostic showed up as a `Parameter`,
  not an `Extra`, on the first attempt). Fixed with a small `PBTransitionNT(params,
  stats)` wrapper + a `FlexiChains.to_nt_and_stats` method on it — see `sample.jl`'s
  own docstring for the wider explanation.
- **`AbstractMCMC.sample(rng, model, spl, MCMCThreads(), N, nchains; ...)` already
  works, with ZERO extra PracticalBayes code.** `AbstractMCMC.mcmcsample`'s own
  generic `MCMCThreads` implementation calls the single-chain `sample()` method above
  once per thread and combines the resulting `SymChain`s via `chainsstack`/`chainscat`
  (`cat(...; dims=3)`) — `FlexiChain` (DimensionalData-backed) supports this `cat`
  generically. Verified: 4 threaded NUTS chains on a conjugate model combine into one
  correctly-shaped chain with `FlexiChains.rhat` < 1.01 on both parameters (the M2
  milestone's own multi-chain gate) — see `test/sample.jl`.
- **M4 (predictive utilities) is done**: `src/predict.jl` — `rand(model)`/`rand(model,n)`
  (prior-predictive), `logjoint`/`logprior`/`loglikelihood_at(model, nt)` (log-density
  at a fixed point), `returned(model, nt)` (the model body's own return value), and
  `predict(model, draws)` + `chain_draws(chn::SymChain)` (posterior-predictive
  sampling from a chain). All built on `PriorMode`/`FixedMode`, which were ALREADY
  fully implemented since M1 with exactly this use in mind (see their docstrings in
  `modes.jl`) — this milestone was mostly writing the convenience wrappers, except
  for one real gap found and fixed along the way:
  **`.~` observe sites had NO predictive-sampling branch** — `PriorMode`/`FixedMode`
  would crash (`MethodError` on `missing`) instead of drawing fresh values, unlike
  scalar `~` which already handled this correctly. Fixed in `tilde.jl`
  (`_dot_rand`/`_dot_all_missing`) using the standard PPL convention: the observed
  argument must be an `AbstractArray{Missing}` at the desired output SHAPE (e.g.
  `fill(missing, n)`), not a bare `missing` scalar — a scalar-broadcast `dist_bcast`
  (`Normal.(mu,sigma)` with scalar `mu`/`sigma`) carries no shape information on its
  own, so the shape has to come from `y`. This matches DynamicPPL's own `predict`
  convention exactly (its docs use the identical `fill(missing, n)` pattern).

## Known gaps / in-progress areas (as of this writing)

- `bench/suite.jl`'s `manyparam_model`/`poisson_model` use the IID-per-element-loop
  prior pattern described above — confirmed present, not yet fixed. Their Turing-side
  counterparts in `test/comparison_env/generate_turing_reference.jl` have a
  *different* asymmetry (per-observation scalar `~` loops on the observe side, where
  PB's side already uses the faster broadcast form). Net effect on any already-recorded
  benchmark numbers from these two specific models is unclear — don't cite them for a
  PB-vs-Turing performance conclusion until fixed. The main 55-model corpus
  (`bench/corpus/`) was audited and is clean of this pattern.

## Working conventions (see devlog for the reasoning behind each)

- Julia type annotations on function arguments: don't add them unless needed for
  dispatch (house style, not a hard rule for correctness).
- Prefer more inline comments in `src/` than a typical terse house style — this
  codebase's own convention, not a general default.
- When changing `Project.toml` compat bounds: install/test first, derive the bound
  from what actually resolves and works — don't hand-write a bound from package
  browsing and hope.
- A `julia.exe` process that keeps respawning after being killed is very likely VS
  Code's Julia language server restarting itself, not a stray script/benchmark loop
  — check before spending time trying to hunt down a "runaway" process.
- Running `Pkg.test()` or other long Julia processes via the agent's background-bash
  mechanism has been unreliable in this environment recently (empty output files,
  no completion signal). If a background Julia run produces no output, don't keep
  retrying it blindly — a manual, targeted smoke-test script (small model, explicit
  `println`s, run to completion in foreground) has been the more reliable way to get
  a real answer during this session.
- **`test/comparison_env/` also works as a full, reliable test-running environment**
  for the WHOLE package (not just the Turing-reference-generation it was originally
  built for) — `Pkg.develop(path="../..")` PracticalBayes into it plus adding
  `Bijectors`/`AdvancedHMC`/`FlexiChains` (needed directly by some test files, not
  just transitively) makes `include("test/runtests.jl")`'s full `@testset` runnable
  in one foreground `julia --project=test/comparison_env script.jl` call — this
  avoids `Pkg.test()`'s own subprocess machinery, which is part of what's unreliable
  here. Confirmed working end-to-end this way (163/163 passing) when `Pkg.test()`
  itself kept producing no output.
