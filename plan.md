# PracticalBayes.jl — a fast Turing-like PPL

## Context

PracticalBayes.jl (`c:\Users\arn203\.julia\dev\PracticalBayes`, currently empty) is a new Julia PPL with Turing.jl-like syntax but a redesigned evaluation core fixing Turing/DynamicPPL's practical pain points:

1. **`~` observe overhead**: DynamicPPL's generated code constructs a `VarName`, walks the context parent chain, and rebuilds a 3-accumulator tuple on *every* tilde statement — which is why `y ~ Dist` on data is much slower than `Turing.@addlogprob!`. Here, observe statements compile to plain `loglik += logpdf(dist, y)` — identical cost to `@addlogprob!`.
2. **Latent-state sampling**: custom kernels (e.g. iFFBS) plug into a Gibbs scheme; latents are constants during every HMC log-density/gradient call (invisible to AD via `DI.Constant`) and updated once per Gibbs sweep.
3. **GPU + threads**: no framework-introduced scalar indexing in the hot path; multi-chain via `MCMCThreads`.

Verified ecosystem facts (newest installed versions): DynamicPPL 0.41.8 already uses accumulator-only evaluation + flat-vector reads (`OnlyAccsVarInfo`/`InitFromVector`) but keeps per-statement overhead in its generated code; its `@model` compiler (`packages/DynamicPPL/HDqaI/src/compiler.jl`, `generate_tilde` line 466, `build_output` line 696) is the reference to fork. Bijectors 0.15.24 `VectorBijectors` (`packages/Bijectors/xWM2Q/src/vector/interface.jl`) provides `from_linked_vec`/`to_linked_vec`/`linked_vec_length`/`vec_length`/`optic_vec` for all distribution families. AdvancedHMC 0.8.5 runs NUTS on any `AbstractMCMC.LogDensityModel` (`step` at `packages/AdvancedHMC/lUCUf/src/abstractmcmc.jl:131`) and its `setparams!!` (line 38) rebuilds the Hamiltonian — the hook for refreshing HMC state after latent updates. Turing 0.45's `externalsampler`/`Gibbs` (`packages/Turing/4hMHm/src/mcmc/{external_sampler,gibbs}.jl`) is the composition reference.

## Settled decisions (user)

- **Hybrid core**: depend on AbstractPPL (VarName/`@varname`), mirror DynamicPPL's user-facing API (`@model`, `~`, `model | (; y=...)`, `sample`) — own compiler + runtime, **no DynamicPPL dependency**.
- **Static layout**: variable names/shapes fixed given data+args; one trace run discovers the layout; hot-path evaluation is a type-stable function over a flat unconstrained vector. No value-dependent model structure (debug mode asserts site counts).
- **Latents declared in-model** with `~`; sampler spec assigns them to kernels: `Gibbs(@varname(θ) => NUTS(0.8), @varname(z) => IFFBS(...))`.
- **Latent kernel contract**: state + logjoint oracle — kernel receives current values of all variables, data, and a callable joint density; returns new latent values. No auto-derived conditionals.
- **GPU scope (MVP)**: θ/position vector on CPU, data on GPU; observes on CuArray work under `CUDA.allowscalar(false)`. Fully-device θ is a later extension.
- **`.~` supported as sugar**, compiled to vectorized `sum(logpdf.(...))` (not a per-element loop).
- **Reuse**: AbstractMCMC (sample/step/ensembles/bundle_samples), AdvancedHMC (NUTS), LogDensityProblems + ADTypes/DifferentiationInterface directly (not LogDensityProblemsAD), Bijectors.VectorBijectors, MCMCChains, Distributions.

## Architecture

Three layers: **compiler** (`@model` rewrites `~` into calls to one `tilde` function dispatching on an evaluation-mode struct; roles decided in the type domain so branches compile away) → **two-phase runtime** (TraceMode once to build a `Layout`; EvalMode as the type-stable hot path; PriorMode/FixedMode for rand/predict) → **inference glue** (LogDensityFunction → AdvancedHMC via LogDensityModel; own Gibbs; MCMCChains bundling).

### Package skeleton

```
src/
├── PracticalBayes.jl   # module, exports
├── utils.jl            # AST helpers (getargs_tilde-style, trimmed from DynamicPPL utils.jl)
├── compiler.jl         # @model macro
├── model.jl            # Model, |, condition/decondition, evaluate
├── accumulator.jl      # Accum{T}: immutable (logprior, loglik) pair
├── modes.jl            # TraceMode, EvalMode, PriorMode, FixedMode; getcond
├── layout.jl           # SiteRecord, FlatSlot/FlatArraySlot/ValueSlot, Layout, build_layout, link/invlink
├── tilde.jl            # per-mode tilde/tilde_index methods (hot path)
├── logdensity.jl       # LogDensityFunction: LogDensityProblems + DI gradients (DI.Constant store)
├── sample.jl           # AbstractMCMC.sample overloads for Model
├── chains.jl           # bundle_samples → MCMCChains (names via layout meta / optic_vec)
├── gibbs.jl            # Gibbs, GibbsState, block conditioning, setparams!! refresh
├── latent.jl           # AbstractLatentKernel, ModelConditional, latent_step
├── predict.jl          # rand(model), predict, returned, logjoint/logprior/loglikelihood
└── optimize.jl         # maximum_a_posteriori/maximum_likelihood/laplace_approximation
                         # (no built-in optimizer — always requires Optimization.jl)
ext/
└── PracticalBayesOptimizationExt.jl  # loaded when the user has `using Optimization`
test/  runtests.jl, compiler.jl, layout.jl, logdensity.jl, optimize.jl, hmc.jl, gibbs.jl,
       predict.jl, stability.jl, gpu/cuda.jl (gated on CUDA.functional())
bench/ observe_overhead.jl
```

### Errors, untracked params, and point estimation (added post-M1, pre-M2)

Three "later considerations.md" items pulled forward since they're additive
on top of `LogDensityFunction`/`Layout` and need no M2/M3 sampling glue —
see the devlog entry dated 2026-07-06 (later) for full detail:

- **Errors as rejected samples**: `LogDensityFunction(...; reject_errors=true)`
  catches exceptions during model evaluation and reports `-Inf`/zero-gradient
  instead of propagating (off by default).
- **Untracked/nuisance params**: `build_layout(...; untracked=(:name,...))`
  marks flat-vector sites that are still sampled but omitted from
  `invlink`'s output NamedTuple by default (`include_untracked=true` to get
  them back) — a reporting-only flag, invisible to `EvalMode`/`tilde.jl`.
- **MAP/MLE/Laplace**: `maximum_a_posteriori`, `maximum_likelihood`,
  `laplace_approximation` in `optimize.jl`, all built on
  `DifferentiationInterface` directly. Point estimation is a secondary
  feature (full Bayesian NUTS sampling is the main target), so there is NO
  built-in optimizer: `optimizer` is a required positional argument (an
  Optimization.jl algorithm, e.g. `OptimizationOptimJL.BFGS()`), and calling
  any of the three without `using Optimization` (+ a solver package) loaded
  errors with instructions. `Optimization` stays a `[weakdeps]`-only
  dependency + package extension (`ext/PracticalBayesOptimizationExt.jl`).

Deps: AbstractMCMC 5, AbstractPPL 0.15, ADTypes, Accessors, AdvancedHMC 0.8, Bijectors ≥0.15.24, DifferentiationInterface, Distributions, ForwardDiff (default adtype), LogDensityProblems 2, MCMCChains, MacroTools, Random. Test extras: Mooncake or ReverseDiff, JET, StableRNGs, DynamicPPL (logjoint cross-check only), CUDA (gated).

### Key types

```julia
struct Model{F,TArgs<:NamedTuple,TCond<:NamedTuple} <: AbstractPPL.AbstractProbabilisticProgram
    f::F; args::TArgs; conditioned::TCond
end
# role of a name, all in the type domain: in args w/ non-Missing type or in conditioned ⇒ observed; else assumed.
# `|` merges into conditioned ⇒ re-specialization per conditioning *pattern* (type), not per value.

struct Accum{T<:Real}; logprior::T; loglik::T; end   # immutable; per-site "rebuild" is free SSA

# Layout slots (isbits ⇒ type-stable NamedTuple lookup):
#   FlatSlot(range)                          — whole variable in flat θ
#   FlatArraySlot(offset, elsize, dims)      — x[i]-loop family, uniform linked elsize
#   ValueSlot                                — read from mode.store (latents / other Gibbs blocks)
# Bijectors NOT stored: recomputed per-eval via from_linked_vec(dist) — correct when supports
# depend on params (Uniform(0,τ)); only linked *length* must be static (trace asserts).

struct LogDensityFunction{M,L,S,AD,P}  # implements LogDensityProblems order 0/1
    model::M; layout::L; store::S; adtype::AD; prep::P   # prep = DI.prepare_gradient, once
end
# gradient: DI.value_and_gradient(..., θ, DI.Constant(store), ...) ⇒ latents invisible to AD
```

### Tilde expansion (compiler)

`@model function foo(y) ... end` emits an evaluator `_foo_eval(__mode__, __acc__, y)` (returns rewritten to `(retval, __acc__)`) plus constructor `foo(y) = Model(_foo_eval, (; y), (;))`. Site rewrites:

- `x ~ D`, x in argnames: `(x, __acc__) = tilde(__mode__, Val(:x), D, x, __acc__)`
- `x ~ D`, x not in argnames: `(x, __acc__) = tilde(__mode__, Val(:x), D, getcond(__mode__, Val(:x)), __acc__)` — `getcond` folds to the conditioned value or `nothing` at compile time
- `x[I...] ~ D`: `tilde_index(__mode__, Val(:x), x, (I...,), D, __acc__)` — whole-array observed or whole-array assumed (no partial observation in MVP); assume path writes into the pre-declared container
- `y .~ Dbroadcast`: compiles to vectorized accumulation `acc_lik(__acc__, sum(logpdf.(D, y)))`, observe-only (assume via `.~` is a macro-expansion-time error; use `x[i] ~ dist` or an array distribution instead)
- `z := expr` (added after initial design, per user request): marks `z` as "has a concrete value already" without any tilde involved — a plain assignment at runtime, but `z` joins `argnames` for the rest of the body, so a later `z ~ D` (or `z .~ D`) is statically treated as an observe against `z`'s computed value. This is how you condition on a model-internal computed quantity, not just on arguments/conditioning.

Hot-path methods (the whole point):

```julia
# OBSERVE: no VarName, no lookup, no context — same cost as @addlogprob!
@inline tilde(::EvalMode, ::Val, dist, value, acc) = (value, acc_lik(acc, logpdf(dist, value)))
# ASSUME: slot-type dispatch compiles the branch away
@inline _assume(slot::FlatSlot, m, ::Val, dist, acc) = begin
    x, logjac = with_logabsdet_jacobian(from_linked_vec(dist), view(m.θ, slot.range))
    (x, acc_prior(acc, logpdf(dist, x) + logjac))
end
@inline _assume(::ValueSlot, m, ::Val{s}, dist, acc) where {s} =
    (getfield(m.store, s), acc_prior(acc, logpdf(dist, getfield(m.store, s))))  # latent: AD-constant
```

### Layout construction

`build_layout(model; flat=..., values=..., rng, init) -> (Layout, θ0, store0)`: one TraceMode run records sites (name, dist exemplar, linked length, role, init); names assigned to `values` (or discrete) → `ValueSlot`+store; rest → flat slots in visit order; `θ0` built via `to_linked_vec`. Helpers `link(layout, nt)`/`invlink(layout, θ)` serve chains, Gibbs, predict. Discrete variable with no value-block ⇒ informative error ("assign a latent kernel or marginalize").

### Sampling glue

`sample(rng, model, spl::AbstractHMCSampler, N; adtype=AutoForwardDiff(), ...)` builds layout+LDF and delegates to `AbstractMCMC.sample(rng, LogDensityModel(ldf), spl, N; initial_params=θ0, chain_type=MCMCChains.Chains)`; plus the `MCMCThreads` ensemble overload (AbstractMCMC deepcopies per chain, isolating DI prep/tapes). `bundle_samples` dispatches on `LogDensityModel{<:LogDensityFunction}` (no piracy): invlink each draw, name scalars via `vec_length`/`optic_vec` ("x[2,1]"), attach AdvancedHMC stats.

### Gibbs + latent kernels

```julia
struct Gibbs{P<:Tuple} <: AbstractMCMC.AbstractSampler   # Gibbs(vns => NUTS(0.8), vns => kernel, ...)
struct GibbsState{V<:NamedTuple,C<:Tuple}                # values of ALL assumed vars + per-block substates
abstract type AbstractLatentKernel end
struct ModelConditional{M,V}; model::M; values::V; end   # + logjoint(c; overrides...) oracle
latent_step(rng, k::AbstractLatentKernel, vns, c::ModelConditional) -> NamedTuple  # user implements
```

Init: per sampler block, build a block layout (`flat` = block names, `values` = rest) once; layouts + DI preps reused across sweeps. Per sweep, per block:
- **Sampler block**: rebuild LDF with `store` = other blocks' current values (cheap struct); refresh cached Hamiltonian via `AbstractMCMC.setparams!!(LogDensityModel(ldf), substate, link(block_layout, values))` (AdvancedHMC recomputes the phasepoint — solves state invalidation after latent moves); then `step`; invlink result into `values`.
- **Latent block**: `merge(values, latent_step(rng, kernel, vns, ModelConditional(model, values)))`. Never touches AD; runs once per sweep, never inside gradient calls.

Adaptation: `n_adapts` threaded through to component NUTS steps (adaptor lives in the HMCState we carry).

### Predictive

`rand(rng, model[, n])` (PriorMode), `returned(model, chain)` and `predict(rng, model, chain)` (FixedMode: params fixed from chain row; un-conditioned observe sites are *sampled*), `logjoint/logprior/loglikelihood(model, nt)` from `Accum` fields.

### GPU

Guarantee: no framework-introduced scalar indexing — parameter reads are `view(θ, range)`, accumulation is scalar arithmetic, data never indexed by the framework. Model authors use array-variate observes (`y ~ MvNormal(...)`, `product_distribution`, `.~`-sugar) for device data. Tested with `CUDA.allowscalar(false)`.

## Milestones (each gated by runnable verification)

- **M0 — Skeleton**: Project.toml, module, empty tests. `Pkg.test()` green.
- **M1 — Compiler + two-phase eval**: `@model`, condition/`|`, TraceMode/Layout/EvalMode, logjoint. Verify: hand-checked logjoints on 4 models (scalar; constrained w/ Jacobian check; `x[i]~` loop; product-dist observe); `@inferred` + JET clean; `@allocated == 0` after warmup; **`bench/observe_overhead.jl`: 10⁴ scalar observes within ~1.1× of raw `sum(logpdf.(...))`** (the requirement-1 gate); cross-check logjoint vs DynamicPPL (test-only dep).
- **M2 — NUTS end-to-end**: LDF + DI gradients, sample overloads, chains, MCMCThreads. Verify: conjugate Normal–Normal posterior within 3 MC-SE of analytic; rhat < 1.01 over 4 threaded chains; ForwardDiff vs one reverse backend agree to 1e-8; chain names correct for vector/matrix vars.
- **M3 — Gibbs + latents**: block layouts, GibbsState, setparams!! refresh, kernel API. Verify: (a) exact-conditional kernel matches analytic posterior; (b) 2-state Gaussian HMM with FFBS kernel + NUTS matches NUTS on the marginalized (forward-algorithm) model; (c) eltype-checking kernel asserts no Duals reach the store.
- **M4 — Predictive**: rand/predict/returned/logprior/loglikelihood. Verify: prior-predictive moments; predict shapes/names; returned round-trip.
- **M5 — GPU + stress**: `test/gpu/cuda.jl` — regression with `y, X::CuArray`, `CUDA.allowscalar(false)`, logdensity + gradient + short NUTS run; 8-thread chain independence.
- **M6 — Polish (stretch)**: sub-variable conditioning, AbstractPPL `@of` shapes to skip trace, Enzyme in AD matrix, pointwise loglik, fully-device θ.

## Documented restrictions (by design)

- No value-dependent model structure (site count/shapes can't depend on draws); debug mode (off by default) asserts trace/eval site-count match.
- `x[i] ~` families need pre-declared containers and uniform per-element linked length; partial observation of an array unsupported in MVP.
- Conditioning on whole top-level names only (`model | (; x=...)`, not `x[1]`) — but a name need not be a model *argument* to be conditionable: `z := expr` inside the body marks a computed local as "has a value," so `z ~ D` after it is an observe against that computed value, same as if `z` were an argument.
- Recompilation per conditioning pattern / Gibbs partition (fine for MCMC, note for interactive use).
- `DI.Constant` behavior needs an explicit per-backend test (ForwardDiff safe by construction; Mooncake/ReverseDiff in M2/M3 matrix; Enzyme deferred to M6).

## Verification (overall)

Per-milestone gates above; the two headline acceptance tests are (1) the observe-overhead benchmark (M1) proving `~` on data costs the same as `@addlogprob!`, and (2) the HMM Gibbs test (M3) proving a user FFBS kernel runs once per sweep with latents constant during HMC gradients, matching the marginalized-model posterior.
