# PracticalBayes.jl devlog

Running log of design decisions, motivations, and compromises. Newest entries
at the top. See also the approved design plan at
`C:\Users\arn203\.claude\plans\i-want-to-make-squishy-pike.md` for the full
architecture writeup this log elaborates on.

See "later considerations.md" for some vague future wish list.

## 2026-07-07 (latest) — real bug in indexed tilde under AD, full Turing parity, AD-crossover confirmed

Three things happened in one push, triggered directly by user feedback on
the benchmark methodology ("we should be doing everything we do in our
package in turing as well"; "we should have models where reverse mode beats
forward mode AD"; "we should be testing on discrete likelihoods").

### Real bug found: indexed `x[i] ~ dist` families crash under ANY AD backend

Building a many-parameter (K=200) regression model to test the forward-vs-
reverse-mode crossover immediately hit a `MethodError`/`InexactError`: the
documented pattern for an indexed family, `x = Vector{Float64}(undef, n)`,
cannot hold the `Dual` (or other AD-backend-specific) numbers a gradient
call needs to write into it. This is a REAL bug affecting every existing
test/doc example using this pattern (`test/compiler.jl`, `test/layout.jl`)
— none had ever been run through a gradient, which is exactly why it went
undiscovered until now (a lesson in itself: allocation-profiling and
correctness-checking a code path are not substitutes for exercising it
under every mode it's actually meant to support).

Fix: added `paramtype(mode)` (modes.jl) — `eltype(mode.θ)` for `EvalMode`
(so it tracks Float64/Float32/Dual/whatever the current evaluation actually
needs), `Float64` for the other three (never-differentiated) modes. Model
authors now write `x = Vector{paramtype(__mode__)}(undef, n)`. This is
directly modeled on DynamicPPL's own fix for the identical problem — Turing
models take a `::Type{T}=Float64` argument that DynamicPPL's compiler
rewrites per call to match the current parameter element type
(`promote_model_type_argument`, `packages/DynamicPPL/HDqaI/src/compiler.jl:818-839`)
— but simpler, since our compiler doesn't need to intercept model
*construction* the way DynamicPPL does: `__mode__` is already a real,
in-scope argument to the generated evaluator function, so `paramtype` just
reads the type straight off it. Added a regression test
(`test/compiler.jl`: "indexed tilde differentiates correctly") that
actually calls `logdensity_and_gradient` on an indexed-family model, so this
specific failure mode can never silently regress again. Fixed the two
existing examples in `test/compiler.jl`/`test/layout.jl` that used the
broken pattern (neither was actually broken today, since neither happened
to be differentiated, but both were teaching the wrong thing).

### Full layer parity between the PracticalBayes and Turing benchmark sides

Previously `generate_turing_reference.jl` only measured Turing's
density-only cost (Layer 1) while our own `bench/suite.jl` had gradient
(Layer 2, per AD backend) and NUTS (Layer 3) too — an asymmetry the user
caught directly ("why are we still doing density only for turing?"). Fixed:
`generate_turing_reference.jl` now benchmarks Turing on exactly the same
three layers, via `DynamicPPL.LogDensityFunction(model; adtype=...)` for
gradients (mirrors our own `LogDensityFunction` closely — same
`LogDensityProblems.logdensity_and_gradient` interface) and `Turing.sample`
for NUTS.

### Two new model shapes: many-parameter and discrete-likelihood

Both tiny/large shapes had exactly 2 continuous parameters and a Normal
likelihood — a regime picked, unintentionally, to make forward-mode AD look
uniformly best and to never exercise a non-Gaussian `logpdf`. Added:

- **`manyparam_model`** (K=200 regression coefficients, N=2000 observations)
  on both sides (`bench/suite.jl` / `generate_turing_reference.jl`'s
  `turing_manyparam`). Result — the crossover the user predicted, confirmed
  on BOTH packages: at K=200, ForwardDiff is now the SLOWEST backend by a
  wide margin (PracticalBayes ~25ms median, Turing ~23-38ms across runs),
  while Mooncake (~1.2-1.8ms) and Enzyme (~1.9-2.2ms) are 10-20x faster —
  reverse-mode wins decisively once parameter count is large, exactly as
  expected from AD theory and invisible in the 2-parameter shapes.
  PracticalBayes's Mooncake/Enzyme numbers came out faster than Turing's
  equivalents on this model; PracticalBayes's ForwardDiff came out somewhat
  slower than Turing's — worth another look, plausibly related to
  `FlatArraySlot`'s per-element bijector-transform overhead (200 individual
  `from_linked_vec`/`with_logabsdet_jacobian` calls vs whatever vectorized
  path Turing's `MvNormal` prior takes) rather than a general regression.
- **`poisson_model`** (discrete-likelihood — Poisson-distributed
  observations, NOT a discrete latent/marginalization case) on both sides.
  PracticalBayes tracks a raw hand-written Poisson-loglikelihood loop
  closely (1.04-1.27x, same ballpark as the Normal-likelihood shapes),
  confirming the hot path is equally fast for a discrete pmf as for a
  continuous density.

### New limitation found: Enzyme + Union types + Float32 + this model

`Enzyme gradient (manyparam K=200, T=Float32, ...)` fails with
`IllegalTypeAnalysisException` ("usually indicates the use of a Union
type") — reproducible, Float32-specific, manyparam-model-specific (Float64
works fine on the same model). Not yet root-caused (a Union somewhere in
the generated code's type — possibly from `_assume_index`'s
`ValueSlot`/`FlatArraySlot` dispatch not fully resolving under Enzyme's
stricter type analysis at Float32). `bench/suite.jl`'s per-backend gradient
loops now wrap each backend in a try/catch (report `FAILED` + truncated
error, keep going) so one backend's failure never aborts the whole
benchmark run again — this itself was a real bug found in the same session
(the very first manyparam run crashed the entire suite on this exact error).

### Numbers snapshot (this machine, this session)

Full first-pass PracticalBayes-vs-Turing summary table (tiny/large x
Float64/Float32 x AD backend, plus manyparam/Poisson) is in
`bench/suite.jl`'s and `generate_turing_reference.jl`'s output — an
Artifact table was also produced mid-session but reflects the numbers
BEFORE the manyparam/Poisson/paramtype-fix additions; regenerate before
relying on it for anything beyond the density-only tiny/large comparison.

## Note for later — DynamicPPL.TestUtils.AD.run_ad

DynamicPPL has an AD-backend benchmarking utility
(`DynamicPPL.TestUtils.AD.run_ad`, `packages/DynamicPPL/HDqaI/src/test_utils/ad.jl`,
plus a standalone pretty-printing wrapper in `packages/DynamicPPL/HDqaI/benchmarks/benchmarks.jl`)
that builds a `LogDensityFunction` from a `DynamicPPL.Model` for a given
`AbstractADType`, then uses Chairmarks.jl's `@be` to time
`logdensity`/`logdensity_and_gradient` over a time budget and report
`grad_time`/`primal_time`, optionally checking gradient correctness against
another backend first. Checked whether we could hook into it directly:
NOT reusable as-is — `run_ad`'s signature takes `model::DynamicPPL.Model`
and hard-codes DynamicPPL-internal concepts (`getlogjoint_internal`,
`AbstractTransformStrategy`, `LinkAll`) to build its own `LogDensityFunction`;
there's no path for handing it an arbitrary LogDensityProblems-compatible
object. The underlying idea (time `logdensity`/`logdensity_and_gradient` per
backend, print a table) is generic and simple (~15 lines in their own
`benchmarks.jl`) — we've already effectively built the same pattern
ourselves in `bench/suite.jl`'s `time_reps`/Layer 2. Noting this so a future
session doesn't waste time trying to wire into DynamicPPL's version instead
of just extending our own; if we ever want Chairmarks-quality statistics
(vs our hand-rolled fixed-rep loop) for the AD-comparison layer
specifically, that's the place to look for a reference implementation.

## 2026-07-07 (even later) — going after the full benchmark corpus

User wants (in order): (1) a clean summary table of PracticalBayes-vs-Turing
timings across model sizes/structures and AD backends (building on
`bench/suite.jl` and the `test/comparison_env` reference-freezing approach);
(2) then benchmarking against every model in
https://github.com/JasonPekos/TuringPosteriorDB.jl/tree/main/src/models;
(3) then the same for every tutorial model in
https://github.com/TuringLang/docs/tree/main/tutorials. This is a large,
open-ended expansion of the benchmark corpus beyond the two synthetic
tiny/large models used so far — noting the intent here before starting so
future sessions have the context even if this spans multiple sittings.
Real constraint to work around throughout: Turing still cannot be loaded in
the same process as PracticalBayes (AbstractPPL 0.14 vs 0.15), so every
Turing-side model in this corpus has to be ported/reimplemented in
PracticalBayes' own `@model` syntax by hand and run through the same
freeze-Turing's-numbers-separately pipeline `test/comparison_env` already
established, not a live side-by-side run.

## 2026-07-07 (later) — proper benchmark harness, two more findings

The single-number, single-backend, single-model-size speed comparison from
earlier the same day was methodologically thin — user's own critique: noisy
without repetition, no AD-backend sweep, and for a simple enough model most
wall-clock time may not even be in "PPL overhead" at all. Replaced with
`bench/suite.jl`: a hand-rolled timing harness (NOT BenchmarkTools/
Chairmarks — their adaptive sampling budgets can degenerate to a single
sample once a call is expensive enough, e.g. a real NUTS run, and neither
exposes a portable "minimum N samples regardless of cost" knob) reporting
min/median/mean/std over an explicit fixed rep count, across: two model
shapes (tiny: 2 params/20 obs; large: 2 params/50k obs), Float64 AND Float32,
and three honesty-layered measurements — density-only, density+gradient per
available AD backend, and a short end-to-end NUTS run (explicitly NOT held
to the same "must match a raw loop" bar as the density-only layer, since for
a 2-parameter model this is dominated by AdvancedHMC's own tree-building/
adaptation overhead, not model evaluation — confirmed by the numbers: ~2.3ms
for 200 NUTS samples on the tiny model vs ~860µs for a SINGLE gradient call
on the large model, i.e. inference bookkeeping cost, not model cost,
dominates the tiny case exactly as the user suspected).

Two more findings this pass, both now documented rather than silently
lived-with:

1. **Indexed `x[i] ~ dist` families are the one hot-path allocation left.**
   The container (`x = Vector{Float64}(undef, n)`) is plain user code inside
   the model body, so it reruns — and reallocates — on every single
   `EvalMode` evaluation (every NUTS leapfrog step): ~8KB/call measured for
   n=1000. Considered caching/reusing the buffer; deliberately did NOT
   implement it, because it would only help the no-gradient path (AD
   backends need a fresh Dual-typed/backend-specific buffer per call
   regardless, and NUTS always needs gradients — so a cached buffer wouldn't
   touch the actual sampling hot path) while adding real complexity
   (thread-safety for `MCMCThreads`, cache invalidation). Documented as a
   known limitation on `tilde_index`'s docstring, with the recommended
   workaround (array distributions via plain `~` don't need a container at
   all).
2. **Float32 NUTS needs a Float32-typed `δ`.** `AdvancedHMC.NUTS(δ)` fixes
   its internal step-size type parameter to `typeof(δ)`; passing the default
   `Float64` `0.8` with a `Float32` position vector fails inside step-size
   adaptation (a real error, not silent wrong-answer behavior). `NUTS(0.8f0)`
   fixes it completely — confirmed full end-to-end Float32 NUTS sampling
   works once the types match. This is an AdvancedHMC requirement, not
   something this package introduces or can paper over; documented in
   `plan.md`'s restrictions section and `bench/suite.jl`'s NUTS layer.

Benchmark results as of this session (`bench/suite.jl`, this machine): at
tiny scale (n=20) PracticalBayes matches a raw framework-free loop almost
exactly (ratio ≈1.0, both sub-microsecond). At large scale (n=50,000)
PracticalBayes is actually FASTER than the naive "raw" reference loop
(ratio 0.57–0.78) — the reference recomputes `Normal(mu,sigma)` and does a
fresh broadcast each call, while PracticalBayes's `Distributions.loglikelihood`-
based path avoids that. Both scales confirm the requirement-1 design goal
holds under real repeated measurement, not just a single noisy sample.

## 2026-07-07 — first real test run: found and fixed 7 bugs, dropped the built-in optimizer

Memory freed up enough to actually instantiate and run the package for the
first time (everything up to this point had been reviewed only by reading).
`Pkg.test()` surfaced real problems immediately — recorded here so the
pattern ("static review missed these; only running the code caught them")
isn't lost.

**Julia/dependency resolution.** The `julia` on PATH is a standalone 1.10.9
install, not the juliaup-managed 1.12.4 used when the ecosystem was
originally explored — this didn't matter in the end (1.10 resolves the same
package lines fine), but the FIRST `Pkg.instantiate()` failed with an
"unsatisfiable requirements" error that looked like a real Bijectors/
AbstractPPL incompatibility. Root cause: our own `Bijectors = "0.15.24"`
compat entry was an exact pin instead of a lower bound, artificially forcing
the resolver onto the *old* 0.15.x line (which really is incompatible with
AbstractPPL 0.15) instead of letting it pick 0.16.1 (which works fine and
still has the full `VectorBijectors` API). Fixed to `"0.15.24, 0.16"`. Lesson
(saved to memory): install and verify FIRST, derive compat bounds from what
actually resolved — don't hand-write bounds from package browsing and hope.

**Turing genuinely cannot be loaded alongside this package, ever.** Not a
compat-tuning problem: Turing 0.45 (newest released) depends on AbstractPPL
0.14.2, while this package needs AbstractPPL 0.15.x for the VectorBijectors
API the whole layout system is built on. Confirmed by direct experiment (see
`test/comparison_env/`) — no combination of compat bounds resolves both in
one dependency graph. Fix: `test/comparison_env/generate_turing_reference.jl`
is a one-time, separate-environment script that runs Turing's side of the
accuracy/speed comparisons and freezes the numbers into
`test/turing_reference.jl` (plain literals); `test/turing_comparison.jl`
reads that file and never imports Turing itself. Also had to update it for
Turing 0.45's new FlexiChains-backed `VNChain` return type (`chain[:mu]`
instead of `MCMCChains`-style `chain[range, [:mu], :]` indexing) and the fact
that `Turing.VarInfo` isn't re-exported anymore (use `DynamicPPL.VarInfo`
directly). Mooncake/Enzyme/PolyesterForwardDiff/ReverseDiff/CUDA/
Optimization/OptimizationOptimJL, by contrast, all coexist with this package
just fine — only Turing itself was the blocker, so those stayed in
`[extras]`/`[targets]`.

**Real bugs found by actually running the code:**

1. **`compiler.jl`: `esc()` is not composable.** `esc(:AbstractEvalMode)`
   spliced into a quote that itself gets `esc()`'d at the top level produces
   `Expr(:escape, Expr(:escape, ...))`, which is a syntax error at macro
   expansion ("invalid syntax (escape (outerref ...))") — this broke EVERY
   model definition, immediately, on the very first smoke test. Root cause:
   escaping only needs to happen once per identifier; since the whole
   generated block is already escaped once (so user identifiers like
   `Normal` resolve at the call site), internal names (`AbstractEvalMode`,
   `Accum`, `Model`, `tilde`, `tilde_index`, `tilde_dot`, `getcond`) must be
   reached by module-qualifying them (`PracticalBayes.AbstractEvalMode`) —
   not by escaping them individually. Confirmed the fix pattern with a
   minimal repro before touching the real macro.
2. **`ext/PracticalBayesOptimizationExt.jl`: method-overwrite during
   precompilation.** The original design had both `optimize.jl`'s stub and
   the extension define `_external_optimize(negf, negg!, θ0, alg; kwargs...)`
   — identical signatures, so Julia treats the extension's definition as
   *overwriting* the same method, which precompilation forbids
   ("Method overwriting is not permitted during Module precompilation").
   Fixed by making the hook a `Ref{Any}` holding a function, replaced via
   plain assignment in the extension's `__init__` — a value replacement, not
   a method redefinition, so it doesn't hit the restriction.
3. **`PracticalBayes.jl`: `ForwardDiff` was a `[deps]` entry but never
   actually `using`'d anywhere in source.** `AutoForwardDiff()` is the
   hardcoded default `adtype` in three public functions, but
   DifferentiationInterface's ForwardDiff support is a package extension
   that only activates once ForwardDiff is loaded IN THE SESSION — being a
   transitive dependency isn't enough. `using PracticalBayes` alone used to
   fail the moment any AD path ran; fixed by adding `using ForwardDiff:
   ForwardDiff` to the top-level module file.
4. **Same gotcha in the test suite** for Mooncake/Enzyme/
   PolyesterForwardDiff: `Base.find_package(...)` only confirms a package is
   *installed*, not *loaded* — each backend's testset needed an explicit
   `@eval import X` before using `AutoX()`.
5. **`accumulator.jl`: `Accum{T}`'s implicit default constructor doesn't
   promote.** `build_layout(...; T=Float32)` failed with a `MethodError`
   the first time a `Float64`-typed `logpdf` result (from a `Float64`
   distribution) got accumulated into a `Float32`-seeded `Accum` — the
   auto-generated constructor requires both fields to already be the exact
   same concrete type. Fixed with an explicit `Accum(logprior, loglik) =
   Accum(promote(logprior, loglik)...)` outer constructor (verified by a
   minimal repro that this doesn't infinitely recurse — Julia picks the
   more-specific matching-type constructor first).
6. **`optimize.jl`: `g_tol=1e-8` was stricter than achievable.** A
   1-parameter conjugate-Normal MAP matched the analytic posterior mean to 8
   significant figures (gradient norm ~3e-7) but never satisfied a `1e-8`
   gradient-norm threshold, burning the full 500-iteration budget and
   reporting `converged=false` on an already-correct answer. This class of
   bug (and the built-in L-BFGS entirely) is now moot — see below.
7. **`tilde.jl`: `.~` allocated ~80KB/call and ran ~4x slower than a raw
   loop**, failing the actual core design goal (observe cost parity with a
   framework-free loop). Root cause: `sum(logpdf.(dist, y))` materializes a
   full intermediate `Vector` before summing. Fix: dispatch to
   `Distributions.loglikelihood(dist, y)` (Distributions.jl's own
   purpose-built, allocation-free total-log-likelihood function) whenever
   `dist` is a single `Distribution` — which it always is for the common,
   documented `y .~ Normal.(mu, sigma)` idiom with scalar `mu`/`sigma`
   (Julia's broadcast collapses to a scalar `Normal`, not an array, when
   every argument is scalar). Falls back to the old sum-of-logpdf form only
   for genuine per-observation-distinct distributions (`Normal.(mus, sigma)`
   with a vector `mus`), which `loglikelihood` doesn't support. Result: 16
   bytes/call (was 80,064), and PracticalBayes now runs within ~3.4% of a
   hand-written `sum(logpdf.(Normal(mu,sigma), y))` loop on a 10,000-
   observation model — matching the M1 acceptance target.

**Speed vs Turing, now that both numbers exist.** On the same 10k-observation
two-parameter model: PracticalBayes is ~10% slower than Turing's own `~` and
~4% faster than Turing's `@addlogprob!` escape hatch (frozen reference in
`test/turing_reference.jl`) — essentially parity, not the dramatic win the
original motivating premise assumed. Consistent with the very first
ecosystem survey's finding that DynamicPPL 0.41 already closed most of the
historical `~`-vs-`@addlogprob!` gap itself. The real, and larger, win is
against a raw framework-free loop (~3.4% overhead after bug #7's fix) — that
comparison is the one that actually isolates *this package's* design, since
Turing's number is a moving target of its own implementation's overhead.

**Dropped the hand-rolled L-BFGS entirely** (per user: "we can drop the hand
rolled optimiser, if its too much trouble. full bayes is main target").
`maximum_a_posteriori`/`maximum_likelihood`/`laplace_approximation` now
require `optimizer` as a positional argument and always go through
`Optimization.jl` (still a `[weakdeps]`-only, no-fallback dependency — no
built-in path at all now, matching the "full Bayes is the main target,
point-estimation is a lean, opt-in convenience" framing). Removed the
now-pointless `_lbfgs_maximize`/two-loop-recursion/Armijo-line-search code
(~90 lines) and its `g_tol` tuning problem along with it. `PointEstimate`'s
`iterations`/`converged` fields became a single `retcode::Symbol` (from
Optimization.jl's `sol.retcode`), since there's no longer a builtin
iteration count to report.

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
