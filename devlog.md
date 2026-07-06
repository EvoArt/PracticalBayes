# PracticalBayes.jl devlog

Running log of design decisions, motivations, and compromises. Newest entries
at the top. See also the approved design plan at
`C:\Users\arn203\.claude\plans\i-want-to-make-squishy-pike.md` for the full
architecture writeup this log elaborates on.

See "later considerations.md" for some vague future wish list.

## 2026-07-06 (later) — errors-as-rejected-samples, untracked/nuisance params, MAP/MLE/Laplace

Three items pulled off "later considerations.md" in one session, all additive
on top of the existing `LogDensityFunction`/`Layout` machinery — none needed
any M2/M3 sampling infrastructure to exist first.

**Errors as rejected samples** (`src/logdensity.jl`): `LogDensityFunction`
gained a `reject_errors::Bool` field (default `false`, so existing behavior
and tests are unchanged). When `true`, `LogDensityProblems.logdensity`/
`logdensity_and_gradient` catch any exception raised while evaluating the
model body and report `-Inf` density (with an all-zero gradient for the
gradient path) instead of propagating — `InterruptException`/
`OutOfMemoryError` are explicitly re-thrown since those are never "just a bad
draw." This is a thin wrapper around the existing `_logdensity_call`, not a
change to it: the model evaluation code itself doesn't know this feature
exists.

**Untracked / nuisance parameters** (`src/layout.jl`): `Layout` gained an
`untracked::Set{Symbol}` field, populated via a new `untracked=` keyword on
`build_layout` (must name existing flat-vector sites — a value-store name or
an unknown name is a clear `ArgumentError`). An untracked site is otherwise
completely ordinary: it occupies real space in `θ`, is sampled, and fully
participates in the log-density and its gradient. The *only* effect is that
`invlink` skips it by default (a new `include_untracked=false` keyword
overrides this) — so it's a purely reporting-level flag, useful for models
with many per-observation/per-group nuisance parameters nobody wants
materialized into a chain. `EvalMode`/`tilde.jl` never look at this flag at
all, by design — the note in the plan that "the M2 chain path will honor it
later" just means `chains.jl`, once it exists, should call `invlink` the same
way (default-omit) rather than needing any new machinery of its own.

**MAP / MLE / Laplace** (new `src/optimize.jl`): built directly on
`LogDensityFunction`'s two ingredients — a scalar objective and its gradient
over flat `θ` — via `DifferentiationInterface` directly (same pattern as
`logdensity.jl`), not through `LogDensityFunction` itself (a `LogDensityFunction`
always reports the joint; MLE needs a different objective, so `optimize.jl`
calls `DI.prepare_gradient`/`value_and_gradient` on its own two objective
functions):

- `_logdensity_call` (existing) for `maximum_a_posteriori`.
- a new `_loglikelihood_call` (logdensity.jl) — identical evaluation, reads
  `loglikelihood_(acc)` instead of `logjoint(acc)` — for `maximum_likelihood`.
  `Accum`'s prior/likelihood split (accumulator.jl) made this a ~5-line
  addition, exactly as anticipated before starting.
- `laplace_approximation` runs `maximum_a_posteriori` then `DI.hessian` on
  `_logdensity_call` at the MAP point; covariance is `-inv(H)`, reported in
  the SAME unconstrained space `theta` lives in (a Gaussian approximation on
  a constrained parameter's raw/linked scale is the only place it's actually
  reasonable to draw one).

All three share `_build_point_layout` (calls `build_layout` with `values =
keys(store)`, so a caller can hold any subset of names fixed during
optimization — same mechanism Gibbs will eventually use for blocks) and
`_optimize_point` (one `DI.prepare_gradient` call, then either the built-in
optimizer or `_external_optimize`).

Per the user's explicit request, the DEFAULT optimizer is a hand-rolled
L-BFGS (`_lbfgs_maximize`, textbook two-loop recursion + Armijo backtracking
line search, phrased as ascent since we maximize a log-density) requiring no
dependency beyond `LinearAlgebra` (already a dep) — so point estimation works
out of the box with zero new required packages. Optimization.jl is wired in
as a `[weakdeps]`/package-extension (`ext/PracticalBayesOptimizationExt.jl`,
Julia's built-in extension mechanism, no Requires.jl): passing any non-`nothing`
`optimizer=` routes through a `_external_optimize` stub that only resolves to
something useful once the user has `using Optimization` loaded, at which
point the extension's method (dispatching via ordinary multiple dispatch,
not a runtime package check) takes over and builds an `OptimizationFunction`/
`OptimizationProblem` from the already-negated objective/gradient closures
`optimize.jl` computes.

## 2026-07-06 — `:=` for conditioning on model-computed values

The user pointed out a real gap: the initial design only treated a bare
`z ~ dist` as a potential observe when `z` was a model *argument* — a `z`
computed partway through the model body (e.g. `z = mu + x` then later used
in a likelihood) was always treated as a brand-new parameter, with no way to
condition on its computed value. This matters because DynamicPPL/Turing's
actual semantics don't care where a value came from, only whether it's
`missing` at the point of use.

We discussed three options: (a) a runtime "is this local currently bound to
a non-missing value" check at every bare-symbol tilde site, (b) keep args-only
static detection as the zero-cost default and add an explicit marker for the
computed-local case, or (c) require restructuring the model to pass computed
values in from outside. The user picked (b) — keep the fast static path as
default — and specifically suggested the shape: `z := y + mu` to mark a
plain deterministic local, then use `z` normally afterward.

Implementation: `:=` is valid (if semantically unused) Julia syntax that
parses to `Expr(:(:=), lhs, rhs)`, distinct from ordinary `Expr(:(=), ...)`
assignment — so it's an unambiguous marker to intercept in the macro without
colliding with normal code. `_walrus_names` (compiler.jl) does a read-only
scan of the model body collecting every `:=`'d name; that set is unioned into
`argnames` *before* the tilde-rewriting pass runs, so from
`_tilde_expansion`'s point of view a `:=`'d name is indistinguishable from a
model argument — same zero-runtime-cost static branch, no new machinery in
tilde.jl at all. A separate pass, `_rewrite_walrus`, then turns `z := expr`
into plain `z = expr` before the tilde rewrite runs, so at runtime `:=` is
literally just an assignment; all of its "special" behavior is macro-time
bookkeeping. Order matters: names must be collected (`_walrus_names`) using
the *original* body, then the walrus rewrite must happen *before* the tilde
rewrite so `_rewrite_tildes` never has to know `:=` exists.

This also means a `.~` site can now use a `:=`'d name as its LHS (it joins
`argnames`, which is exactly what `_dot_tilde_expansion` checks), extending
`.~`'s conditioning ability to computed locals too, for free.

## 2026-07-06 — Project start, M1 scaffolding

### Motivation

The user wants a Julia PPL with Turing.jl's `@model`/`~` modelling syntax but
without three specific pain points they've hit in practice:

1. **`~` observe overhead.** In DynamicPPL, every tilde statement — including
   plain data observations — goes through a runtime pipeline: a `VarName` is
   constructed, the full parent-context stack is walked, and a 3-accumulator
   tuple (logprior/logjac/loglikelihood) is rebuilt via `map` + NamedTuple
   merge. `Turing.@addlogprob!` skips all of that and is measurably cheaper.
   We want `y ~ Normal(mu, sigma)` on data to cost exactly what
   `@addlogprob! logpdf(Normal(mu, sigma), y)` costs.
2. **Latent-state sampling.** Custom kernels for state-space models (e.g.
   iFFBS for an HMM's discrete/continuous latent states) need to update state
   once per Gibbs sweep, and that update must never be visible to — or
   triggered by — HMC's automatic differentiation of the continuous
   parameters. In practice this means: latents are read as plain constants
   during every leapfrog step's log-density/gradient evaluation.
3. **GPU + threads.** Model evaluation shouldn't introduce framework-level
   scalar indexing that would break under `CUDA.allowscalar(false)`, and
   multi-chain sampling should parallelize over threads for free.

### Why not just fix DynamicPPL?

We surveyed the installed Turing stack (DynamicPPL 0.41.8, AbstractPPL 0.15.3,
AbstractMCMC 5.15.1, AdvancedHMC 0.8.5, Bijectors 0.15.24, LogDensityProblems
2.2.0, Turing 0.45.0) before writing any code. Finding: DynamicPPL 0.41
*already* moved toward the architecture we want at the runtime-representation
level — `OnlyAccsVarInfo` (accumulators only, no parameter dictionary) plus
`InitContext`/`InitFromVector` reads values from a flat vector by
precomputed range, which is close to what we do. But the overhead we care
about lives in the `@model`-*generated code itself*, not the runtime
representation: every tilde site still constructs a `VarName` (even for pure
observes), walks whatever context stack is active, and maps over all three
accumulators regardless of which one actually changes. That's compiler
output we don't control without forking the compiler anyway — so we chose to
write our own `@model` macro and tilde runtime, while still reusing
everything downstream of "produce a log-density and its gradient": AbstractMCMC
for the sampling loop/ensembles, AdvancedHMC for NUTS, Bijectors'
`VectorBijectors` submodule for flat-vector transforms, DifferentiationInterface
+ ADTypes directly for gradients (not LogDensityProblemsAD), and MCMCChains
for output.

### Core design choice: static layout + mode-dispatch, not context stacks

Every `@model`-generated evaluator takes an `AbstractEvalMode` as its first
argument. There are four concrete modes (`TraceMode`, `EvalMode`, `PriorMode`,
`FixedMode`), and every tilde site becomes a call to a generic function
(`tilde`/`tilde_index`/`tilde_dot`) that Julia's multiple dispatch resolves
purely from argument types — no `if mode isa X` branching anywhere, and for
`EvalMode` (the hot path used by every HMC log-density/gradient call), every
type involved is known at compile time for a given model + `Layout`, so the
whole call chain inlines to straight-line arithmetic. This is the mechanism
that replaces DynamicPPL's runtime `AbstractContext` recursion.

The other half of the design is the **two-phase evaluation** split:

- `TraceMode` runs the model exactly *once* to discover its variables
  (name, distribution, unconstrained/"linked" length, role: observed / param
  / latent). This is the only place where dynamic dispatch, allocation, and
  `Dict`-based bookkeeping are allowed.
- `build_layout` (layout.jl) turns that one-shot trace into a `Layout`: an
  isbits NamedTuple mapping each variable name to a slot descriptor
  (`FlatSlot`, `FlatArraySlot`, or `ValueSlot`), fixed for the lifetime of
  that Layout. This is the "static layout" restriction: variable names,
  shapes, and count cannot depend on sampled values, only on the model's
  *arguments* — in exchange, every later evaluation is a plain, type-stable
  function over a flat vector.
- `EvalMode` is that fast path: `getproperty(layout.slots, name)` on a
  concrete-typed field of an isbits NamedTuple is resolved during type
  inference, so the branch between "read a sub-vector of θ and transform it"
  (FlatSlot/FlatArraySlot) vs "read a constant from the store" (ValueSlot,
  used for latents and other Gibbs blocks) compiles away entirely.

### Why `Accum` is an immutable (logprior, loglikelihood) pair

DynamicPPL's split into `LogPriorAccumulator`/`LogJacobianAccumulator`/
`LogLikelihoodAccumulator` (3 separate accumulators, rebuilt via `map` +
NamedTuple merge on every site) was worth keeping the *spirit* of — being
able to read off `logprior`/`loglikelihood`/`logjoint` separately is genuinely
useful — but not the mechanism. We collapsed it to a single immutable
2-field struct (`Accum{T}`, holding `logprior` — which folds in the bijector
Jacobian term — and `loglik`), threaded through every tilde call as a plain
value, not mutated in place. Being immutable is what makes this trivially
correct under every AD backend we plan to test (ForwardDiff, Mooncake,
Enzyme, ReverseDiff): there's no aliasing, no mutation to worry about
differentiating through, just function composition.

### Latents as `ValueSlot`s + `DI.Constant`

A latent (or another Gibbs block's variable, from the point of view of the
block currently being sampled) is represented by a `ValueSlot`: its value
lives in `EvalMode.store` (a NamedTuple), never in the flat vector `θ`. When
`LogDensityFunction` attaches a gradient via DifferentiationInterface, `store`
is passed as `DI.Constant(store)` — this is what guarantees latents are
invisible to every AD backend's gradient computation (ForwardDiff never
promotes them to `Dual`, reverse-mode backends never allocate a tape node
for them), regardless of which backend is in use. A latent kernel (e.g.
iFFBS) updates `store` between HMC steps, in the Gibbs loop, never during a
leapfrog step — satisfying requirement 2 directly from the type of the value,
not from any special-casing in the sampler.

### Numeric-type genericity (added after user feedback)

Initial draft of `build_layout`/`LogDensityFunction` hardcoded `Float64` in a
few places (the initial flat vector, gradient-prep probe vector). Turing's
`VarInfo`-based storage tends to force everything to `Float64` (or whatever
the AD dual type is) because of how its dictionary-of-vectors representation
works, which forecloses `Float32`/SIMD/GPU-native-precision workflows. We
audited every hardcoded `Float64` in the hot path and confirmed: `tilde.jl`
and `EvalMode` never mention a concrete number type at all — they operate on
`eltype(θ)` throughout via `zero(eltype(θ))`, so a `Vector{Float32}` or a
`CuVector{Float32}` position vector flows through untouched, provided the
constituent `logpdf`/bijector calls support that type (a Distributions.jl/
Bijectors.jl property, not one this package restricts). The only places
`Float64` remains as a *default* (not a hardcoded requirement) are: the
initial vector `build_layout` constructs (now parametrized by a `T` keyword),
and `LogDensityFunction`'s AD-prep probe vector `θ0` (now a keyword the
caller can override to match their sampling type).

### `.~` scoped to observe-only for MVP

Turing's `.~` broadcasts a distribution over an array and supports both
assume and observe. We only implemented the observe case
(`y .~ Normal.(μ, σ)` where `y` is already-bound data), compiling it to a
single vectorized `sum(logpdf.(dist_bcast, y))` — never a per-element loop,
so it stays GPU-friendly. Assume via `.~` is deliberately unsupported: giving
each broadcast element its own flat-vector slot reintroduces exactly the
ragged/indexed-family bookkeeping that `x[i] ~ dist` already exists to
handle, and array distributions (`product_distribution`, `MvNormal`) already
cover the "vector of unknowns" case through the ordinary `~` path. Calling
`.~` with an unbound LHS raises a clear error pointing at the alternative.

### Style choices adopted this session (per user feedback, saved to memory)

- Don't annotate Julia function arguments with types unless the annotation
  is load-bearing for dispatch (multiple sibling methods, or extending an
  external package's generic function). Went back through every file and
  stripped decorative single-method annotations (e.g. `build_layout`'s
  keyword args, several mode constructors).
- More inline comments than this project's default house style calls for,
  specifically in the macro/dispatch-heavy files (compiler.jl, tilde.jl,
  modes.jl, layout.jl) — added throughout explaining *why* a given dispatch
  or AST shape does what it does, not just restating identifiers.

### What's implemented so far (M1, not yet run/tested — see below)

`src/accumulator.jl`, `src/modes.jl`, `src/layout.jl`, `src/model.jl`,
`src/tilde.jl`, `src/compiler.jl`, `src/logdensity.jl`, wired up in
`src/PracticalBayes.jl`. This covers: `@model` macro, `Model`/`condition`/`|`,
the four evaluation modes, `build_layout`/`link`/`invlink`, the full tilde
dispatch table (scalar, indexed, dot), and a `LogDensityProblems` +
DifferentiationInterface-backed `LogDensityFunction`.

**Not yet implemented:** NUTS sampling glue (`sample.jl`), MCMCChains
bundling (`chains.jl`), Gibbs + latent kernel machinery (`gibbs.jl`,
`latent.jl`), predictive utilities (`predict.jl`) — these are M2/M3/M4 in the
plan.

### Known compromise: not yet run

This session's Julia process memory is constrained by another running Julia
process, so none of the code above has actually been executed — no
`Pkg.instantiate`, no `include`, no test run. Everything has been written and
reviewed by reading, plus static IDE diagnostics (which flagged and helped
fix two real bugs: an undefined `nameof_container` placeholder in
`tilde.jl`'s indexed-ValueSlot path, and two genuinely-unused mode-argument
bindings in the `.~` PriorMode/FixedMode methods). The package has NOT been
verified to actually load or produce correct log-densities yet — that's the
first thing to do once it's safe to run a Julia process at full weight.
Tests (correctness of logjoint against hand-computed values and against
DynamicPPL, type-stability/`@allocated`, the observe-overhead benchmark
vs `@addlogprob!`, and the AD-backend/Turing-comparison matrix the user
asked for) are written but similarly unexecuted as of this entry.
