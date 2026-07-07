## 2026-07-07 (latest) — NUTS methodology fix, Enzyme re-enabled with subprocess isolation, report/heatmap redesign

Follow-up to the corpus-wide benchmark entry below, triggered by the user
noticing `normal/large` NUTS had apparently gotten slower than Turing between
two runs.

### False regression: unplugged laptop, not a code issue

Investigated via `git log`/diff between the two benchmark commits — none of
the commits in between touched any hot-path file (`src/tilde.jl`,
`src/logdensity.jl`, etc.), all were model-porting/distribution additions.
Root cause turned out to be simpler: the user's laptop was unplugged during
the second run (thermal/power throttling), confirmed directly by the user
rather than inferred. No code change resulted from this — noted here so a
future apparent regression isn't re-investigated from scratch without first
checking power state.

### NUTS benchmark methodology: single compile chain + one timed 500-sample chain

Prior methodology ran multiple full NUTS chains per config and averaged
(`nuts_reps`), at only 200 samples — raised whether this was dominated by
adaptation/warmup rather than steady-state inference. Proposed reporting both
"full run" and "steady-state-only" timings; user rejected this as
overcomplicated ("so long as turing and pb are doing the same thing, it's
fine"), asking instead for 1000 samples everywhere for realism. Given the
corpus is 55 PB models × 2 precisions × 3 AD backends (plus ~20 Turing
models), full multi-rep chains at 1000 samples each was impractical —
resolved via a follow-up ask: one short untimed chain (20-100 samples) purely
to force JIT compilation, then a single timed chain at the real sample count
(500, not 1000 — chosen as a practical compromise once framed as "compile
once, time once" rather than "average over many reps"). Applied identically
to `bench/corpus_bench_worker.jl`'s `bench_nuts` and
`test/comparison_env/corpus_bench_turing.jl`'s NUTS timing block, and to the
two small hand-picked-model benchmarks (`bench/suite.jl`,
`test/comparison_env/generate_turing_reference.jl`) which moved to
`n_samples=1000` (kept `reps=5` there since only 4 total NUTS calls in that
file's small shape matrix — no need for the compile/time split at that
scale).

### Enzyme re-enabled, ReverseDiff dropped

User: "we dont need reversediff. drop it. enzyme would be great though, why
did we drop it?" Re-examined the earlier Enzyme exclusion (see corpus-wide
benchmark entry below): every INDIVIDUAL Enzyme failure across the corpus had
actually been caught and printed cleanly by the existing try/catch — only one
full-sweep process death was ever unexplained (no Julia stack trace, process
just gone), and that run also happened during the same unplugged-laptop
period, so it's plausibly power/thermal, not Enzyme-specific. Resolution
(user's choice after being asked): re-enable Enzyme, but add subprocess
isolation first so any future crash — Enzyme-caused or not — only loses one
model's results, not a multi-hour sweep.

**Subprocess isolation**: `bench/corpus_bench.jl` split into a thin driver
(discovers model names per corpus file, no model construction) that spawns
one `julia --project=bench/bench_env corpus_bench_worker.jl <file> <name>
<corpus>` subprocess per model; the new `bench/corpus_bench_worker.jl`
contains all the actual benchmarking logic (previously in `corpus_bench.jl`
directly) and appends its own results to `history_corpus.jsonl` before
exiting, so a crashed model's siblings are unaffected and already-completed
results are safely on disk regardless of what happens later in the sweep.

**Enzyme runtime activity**: user asked directly ("are we setting enzyme
runtimeactivity in it ADType?") — checked and confirmed bare `AutoEnzyme()`
defaults to runtime-activity analysis OFF, which is the direct, confirmed
cause of the `EnzymeRuntimeActivityError: Detected potential need for
runtime activity` failures seen on several models (any model where a
constant/store value flows into a differentiable computation — latents,
`:=`-computed quantities feeding an observe). Fixed with `AutoEnzyme(;
mode=Enzyme.set_runtime_activity(Enzyme.Reverse))` in both
`corpus_bench_worker.jl` and `corpus_bench_turing.jl`. Does NOT fix the
separate `IllegalTypeAnalysisException`/`EnzymeNoTypeError` failures on a
few other models (genuine Union types hitting Enzyme's static type analysis
— a different root cause).

**ReverseDiff dropped** from both AD-backend lists (redundant with
Mooncake — both reverse-mode, Mooncake is the one this package's own test
suite already leans on).

### Report redesign: PB-Float64/PB-Float32/Turing on one row

User: "in a given row of the report lets have the pb float32, pb float64 and
turing timing on the same row. I want to see if float32 lets us beat turing."
Previously the corpus report table was keyed by precision (separate row per
Float64/Float32), making a Float32-vs-Turing comparison require
cross-referencing two tables. `bench/generate_report.jl`'s corpus section
rewritten to one row per `(corpus, model, backend)` within each layer, with
columns for PB Float64, PB Float32, Turing, plus three ratio columns
(F32/F64, PB-F64/Turing, PB-F32/Turing) — the last two directly answer the
user's question in one glance per row.

### New: `bench/generate_heatmap.jl` — interactive ratio heatmap

User: "each model on the y axis, each column a different comparison vs
turing (logdensity/gradient/nuts), diverging colormap, actual ratio rounded
to one decimal in the cell." New standalone script producing a self-contained
`bench/heatmap.html` (two panels — PB-Float64-vs-Turing and
PB-Float32-vs-Turing — sticky headers/first column, light+dark theme via CSS
custom properties + `prefers-color-scheme`/`data-theme` override per the
artifact-design skill's guidance, log-scaled diverging color from neutral
gray through green (PB faster) or red (Turing faster), clamped at ±1 decade
for saturation).

Hit a real Julia gotcha while writing it: the HTML template was originally a
normal `"""..."""` string (Julia-interpolating), but it embeds JavaScript
containing literal template-literal syntax (`` `hsl(...,${expr}%,...)` ``) —
Julia's parser tried to interpret every `${...}` as its own string
interpolation, producing 5 parse errors. Fixed by switching the template to
`raw"""..."""` (zero Julia interpolation) and injecting the actual JSON data
via a plain string `replace` of a placeholder token (`__DATA_JSON__`) after
the raw string is captured, rather than any form of interpolation.

### New: `bench/run_all.ps1`

User wanted to run benchmarks themselves from a plain PowerShell window
(VSCode's integrated terminal consumes RAM this machine can't spare during a
multi-hour benchmark sweep) rather than have them run in an agent-driven
background process. Small wrapper script: runs the Turing-side benchmark
(`test/comparison_env/corpus_bench_turing.jl`) to completion first, then the
PB-side corpus sweep (`bench/corpus_bench.jl`) — sequential because both
independently append to the same `history_corpus.jsonl` and running them
concurrently would interleave writes for no benefit (they don't share
Julia processes anyway, each already pays its own environment-load cost).

### Current data state: partial, not a full re-run

As of this entry, only 6 of 55 PB corpus models (`GLM_Poisson`, `Rate_1`,
`blr`, `dugongs`, `eight_schools_centered`, `eight_schools_noncentered`) have
fresh results under the new methodology (single-compile+500-sample-timed
NUTS, Enzyme re-enabled, ReverseDiff dropped); the Turing-side sweep hasn't
been re-run at all yet under the new settings. `bench/report.qmd`/`.html` and
`bench/heatmap.html` have been regenerated and DO reflect this partial data
correctly (no crash, ratios computed only where both sides have data for a
given model/backend/precision) — but they are not yet a full, meaningful
PB-vs-Turing comparison across the whole corpus. The user now has
`bench/run_all.ps1` to run both sweeps to completion themselves; report/
heatmap should be regenerated again once that finishes
(`julia --project=. bench/generate_report.jl && quarto render
bench/report.qmd` and `julia --project=. bench/generate_heatmap.jl`).

## 2026-07-07 — corpus-wide benchmark: every model, every AD backend, both precisions, PB vs Turing

User's explicit ask after M3: "for every model we have set up for benchmarking, I want them run — multiple AD backends, Float32 vs Float64, run on Turing with multiple backends too, tracked in report tables."

### `bench/corpus_bench.jl` (new) — benchmarks every corpus model on PracticalBayes

Reuses each `bench/corpus/{posteriordb,tutorials}/run_smoke_tests*.jl` file's own `MODELS` list (already correctness-verified there) rather than redefining model/data pairs — this file is only responsible for timing. Sweeps all 3 layers (logdensity, gradient, short NUTS) × every AD backend available in the running session × Float64/Float32, appending to `bench/results/history_corpus.jsonl` via the same JSONL mechanism as `bench/suite.jl`.

New `bench/bench_env/` — a dedicated environment (`Pkg.develop(path=".")` + `Pkg.add` every AD backend), same pattern as `test/comparison_env/`, since the main `Project.toml` only lists Mooncake/ReverseDiff/Enzyme as test-only `[extras]` (invisible under plain `--project=.`).

**Three real bugs found while getting this to run cleanly, all discovered because a 53-model sweep exercises code paths a handful of hand-picked benchmark shapes never do:**

1. **Julia world-age errors calling into dynamically-`include`d corpus files.** Each corpus file is loaded into its own fresh `Module` at runtime (to avoid name collisions between files reusing helper names like `make_blr_data`) via `_load_corpus_file`. Calling anything built from that — `buildfn()`, and more subtly, every nested dynamic dispatch through the resulting `Model`'s `.f` field (`build_layout`, `evaluate`, `LogDensityFunction`'s constructor) — from a function that had ALREADY started compiling before those methods existed hits `MethodError: ... method too new to be called from this world context`. Fixed by wrapping the entire per-model benchmarking body (not just the outermost `buildfn()` call) in `Base.invokelatest`.
2. **A `StringIndexError` inside error-reporting code silently killed a multi-hour run.** `sprint(showerror, e)[1:min(end, 150)]` slices a Julia `String` by BYTE index; several AD backends' error messages contain multi-byte UTF-8 box-drawing characters (`│`), and byte-slicing can land mid-character. This threw INSIDE a `catch` block's own `println(...)` call — not caught by anything, silently escaping and killing the whole process. First discovered when a run died partway through on a Mooncake error message for `earn_height`, with no useful indication why (the process just stopped, exit code 1, no obvious crash). Fixed with a `_trunc_err` helper using `first(s, n)` (character-counting, always safe) instead of byte-index slicing — applied to both the PracticalBayes-side and Turing-side corpus benchmark scripts.
3. **Enzyme genuinely crashes (not just throws) across this much model diversity.** Even after fixing #2, a full run still died with no Julia-level stack trace at all — the OS process just disappeared (confirmed: not in `ps aux` afterward, exit code 1 from the shell but no Julia error output). This is a real Enzyme stability gap when exposed to this many structurally different model shapes (on top of the `EnzymeRuntimeActivityError`/`IllegalTypeAnalysisException`/`EnzymeNoTypeError` failures already seen and safely caught on individual models earlier this session). Fixed by excluding Enzyme from the corpus-wide sweep entirely (commented out, with an explanation, in both `bench/corpus_bench.jl` and `test/comparison_env/corpus_bench_turing.jl`) — it remains in `bench/suite.jl`'s smaller, hand-picked model sweep, where it's known to behave.

Final PracticalBayes-side run: **490 results** across 55 models (ForwardDiff/Mooncake/ReverseDiff × Float64/Float32 × 3 layers), with 61 individual model/backend/precision combinations gracefully caught and logged as failures (mostly ReverseDiff/Mooncake hitting real, known limitations on specific Float32 type combinations — consistent with earlier findings this session, not new bugs).

### `test/comparison_env/corpus_bench_turing.jl` (new) — Turing-side representative subset

User explicitly scoped this to ~15-20 models (not all 53 — porting every near-duplicate corpus variant to Turing was judged not worth the effort) after being asked directly. Picked one (or two) models per structural family: conjugate scalar, filldist+MvNormal regression, hierarchical MvNormal, nonlinear regression, discrete-likelihood GLM, Flat-prior regression, gather-indexed hierarchical (1-5 simultaneous random effects), UniformScaling+BinomialLogit, non-Flat filldist, regularized horseshoe, 2D matrix observe, plus 6 tutorial models. Models are near-verbatim ports of the SAME PracticalBayes corpus models back to plain Turing syntax.

**Porting bugs found and fixed** (all caught by actually running the models, not by inspection): `logistic` used in several model bodies/data-generators without `using StatsFuns: logistic` (PracticalBayes's environment resolves it transitively via a different path); `Truncated(Normal(0,100); lower=0)` keyword-argument form not supported by this Turing/Distributions version combination — needed the positional `Truncated(dist, lo, hi)` form instead; and the most interesting one — **Turing 0.45/DynamicPPL 0.35+ REMOVED support for `.~` over an array of DIFFERENT distributions** (`ArgumentError: As of v0.35, DynamicPPL does not allow arrays of distributions in .~`), affecting every model using `BernoulliLogit.(...)`/`Poisson.(...)`/`Categorical.(...)` as a `.~` right-hand side (`GLM_Poisson`, `wells_dae`, `seeds`, `logistic_regression_rhs`, `election88_full`, `irt_2pl`, `bayesian-logistic-regression`, `bayesian-poisson-regression`, `multinomial-logistic-regression` — 9 of the 20 models). This is a genuinely notable finding: it's the EXACT same restriction PracticalBayes's own `.~` has always had by design (assume/observe against a per-element-distinct array of distributions needs `product_distribution`, not a bare broadcast) — Turing evidently converged on the same design constraint after this session's ports were largely written, in a version released after most of the reference tutorial/PosteriorDB source was written. Fixed every occurrence by switching to `y ~ product_distribution(Dist.(...))`, mirroring exactly how this session's own PracticalBayes ports already had to handle the analogous restriction.

Final Turing-side run: **235 results** (100 in the last clean run + earlier partial runs), all 20 models, 0 failures.

### Report: `bench/generate_report.jl` extended with a "Model corpus benchmarks" section

New section, keyed by `(corpus, model, layer, precision, backend)` (no "shape" dimension, unlike `bench/suite.jl`'s hand-picked tiny/large models — each corpus model has exactly one data-generation function), one table per layer, PB median vs Turing median with a ratio column where both sides have data for that exact model+backend+precision combination. `bench/report.qmd`/`bench/report.html` regenerated and re-rendered via Quarto.

## 2026-07-07 — M3 complete: Gibbs + latent kernels, three real convergence bugs found and fixed

Milestone M3 (design plan: `C:\Users\arn203\.claude\plans\i-want-to-make-squishy-pike.md`, "M3 implementation plan" section). Motivated directly by the model-corpus pass hitting its boundary at discrete-latent models (HMM/mixtures/capture-recapture) and the user's explicit priority: plug in a custom iFFBS-style latent-trajectory kernel with "nice straightforward syntax," and guarantee it never updates during NUTS gradient calls — structurally, not by convention. `Pkg.test()`: 131/131 passing (up from 124).

### What was built

- **`src/compiler.jl`**: `@addlogprob!(expr)` macro, rewriting to `__acc__ = PracticalBayes.acc_lik(__acc__, expr)` — a prerequisite the HMM gate needed (injecting a hand-written forward-algorithm marginal likelihood into a model body; there was previously no "just add this number to the log-density" primitive).
- **`src/latent.jl`** (new): `AbstractLatentKernel` (supertype for user kernels), `ModelConditional{M,V}` (model + every-other-block's current values, passed to `latent_step`), `logjoint(c::ModelConditional; overrides...)` (a `FixedMode`-based convenience oracle, a new method of the already-exported `logjoint` generic), `latent_step` (the function users implement — full docstring with a worked FFBS example for a 2-state Gaussian HMM).
- **`src/gibbs.jl`** (new): `GibbsBlock`, `Gibbs <: AbstractMCMC.AbstractSampler` (`Gibbs(:mu => NUTS(0.8), :z => MyKernel())`), `GibbsSamplerSub` (caches `layout`+`prep` once, reused every sweep), `GibbsLatentSub`, `GibbsState`, coverage validation, and the systematic-scan `AbstractMCMC.step` sweep loop.
- **`src/logdensity.jl`**: doc-comment (no behavior change) documenting `LogDensityFunction`'s inner constructor as the supported way to reuse a cached `prep` against a new `store` value.

### Design decision, confirmed directly rather than assumed: block identity is `Symbol`/`Tuple{Symbol}`, not `AbstractPPL.VarName`

The original architecture sketch mentioned `@varname(θ) => NUTS(...)` syntax; checked the codebase first and found `AbstractPPL.VarName`/`@varname` are used nowhere (only `Model <: AbstractPPL.AbstractProbabilisticProgram`) — this package only ever conditions on whole top-level names by design, so plain `Symbol`s (matching `build_layout`'s existing `flat=`/`values=` contract) are simpler and consistent with every other user-facing API.

### Three real bugs found via the M3 gates — none were in the initial design, all found by actually running and numerically checking the code, not by review

1. **`_gibbs_init_values` collapsed indexed-latent-family init values to their last scalar.** `TraceMode` records one `SiteRecord` per tilde-site VISIT — for an indexed family like an HMM's `z[t] ~ Categorical(...)` inside a `for t in 1:N` loop, that's N records all named `:z`. The first version of `_gibbs_init_values` just did `pairs[name] = r.init_val` per record with no grouping, so the last record silently won — `values0.z` ended up as a single `Int`, not the length-N trajectory vector. Crashed downstream with a `BoundsError: attempt to access Int64 at index [2]` inside `_assume_index` — a hard crash, not a silent wrong answer, but only once a real indexed-latent-family test (the HMM) exercised it. Fixed by grouping by name (mirroring `build_layout`'s own grouping in layout.jl) before building the initial NamedTuple.

2. **Same fix's first draft used `Vector{Any}` instead of a concretely-typed vector**, which passed a `DifferentiationInterface.PreparationMismatchError` at the FIRST real Gibbs sweep after the latent block ran once (`store.z`'s `Vector{Any}`-vs-`Vector{Int64}` type mismatch between the `prep`-time store and the sweep-time store). This is exactly the prep-caching hazard flagged as an open risk during planning — confirmed it manifests even for `ForwardDiff` (not just reverse-mode backends) whenever the STORE'S OWN NamedTuple field types differ, not just its AD-backend-specific tape. Fixed by building `[r.init_val]` (Julia infers the concrete element type) instead of `Any[]`.

3. **The subtle one: `n_adapts` must be passed as its FINAL value from Gibbs sweep 1 onward, never ramped up (`min(sweep, target)`).** Traced into AdvancedHMC's own source (`packages/AdvancedHMC/4lo5Y/src/sampler.jl:71-86`): `Adaptation.adapt!(h, κ, adaptor, i, n_adapts, ...)` calls `i == 1 && Adaptation.initialize!(adaptor, n_adapts)` — the Stan-style windowed adaptation SCHEDULE is computed once, on a block's first-ever internal step, from WHATEVER `n_adapts` value happens to be passed on that call. The original `test/gibbs.jl` draft used `n_adapts=min(sweep, 500)`, so sweep 1 passed `n_adapts=1`, permanently locking in a window of size 1 for the rest of the run — every later sweep's larger `n_adapts` was ignored for scheduling purposes (only used for the separate `i <= n_adapts` "should we adapt at all" gate, which stayed true). Symptom: NOT a crash, NOT an obviously-frozen chain — `mu` kept moving sweep to sweep (occasional lucky NUTS proposals even at a badly-oversized step size), but with lag-1 autocorrelation of 0.998 (effective sample size ~17 out of 18000 draws), so the posterior mean/std came out badly wrong (`mu` mean off by 0.72 vs the independently-computed analytic reference, `std` too narrow) while LOOKING like ordinary MCMC noise rather than a bug. Diagnosed by tracing `state.subs[1].hmc_state.κ.τ.integrator.ϵ` (frozen at the initial guess) and `state.adaptor.state` (`window(0,0)`) across sweeps, then reading AdvancedHMC's source directly rather than guessing. Fixed the calling convention in `test/gibbs.jl` (pass a fixed `n_adapts` throughout) and added an explicit, prominent doc-comment on `Gibbs`'s `AbstractMCMC.step` method warning about this — a real user writing their own Gibbs loop will hit this identically.

This is the clearest example this session of why every M3 gate was built as a NUMERICALLY CHECKED comparison against an independent reference (closed-form analytic posterior for gate (a); a from-scratch marginalized-model NUTS chain for gate (b)) rather than "does it run without crashing" — bug #3 in particular produces a chain that runs cleanly, produces plausible-looking individual draws, and only reveals itself as wrong when checked against ground truth.

### Verified working

- **Gate (a)**: exact-conditional kernel (`ExactNormalKernel`, closed-form conjugate Normal update) vs. an independently-computed analytic joint posterior (2×2 Gaussian conditioning, `Sigma_prior=[[1,1],[1,2]]`, `H=[0,1]`, `R=0.25`) — posterior means within 3 MC-SE, stds within 15% relative tolerance.
- **Gate (b)**: 2-state Gaussian HMM sampled via `Gibbs(:p_stay=>NUTS, :sigma=>NUTS, :z=>FFBS())` (hand-written forward-filter/backward-sample kernel) vs. NUTS run directly on the same model with `z` marginalized out by hand (forward algorithm via the new `@addlogprob!`) — `p_stay`/`sigma` posterior means agree within 3 combined MC-SE. Forward-algorithm marginal-likelihood math independently verified against brute-force enumeration over all `2^N` discrete paths for small N before trusting it as a test reference.
- **Gate (c)**: a kernel asserting `eltype(c.values.mu) <: Union{Float32,Float64}` (never `ForwardDiff.Dual`) on every call, run for 300 sweeps with real adaptation — never fires, confirming latents structurally never reach a gradient call (the guarantee is lexical: `latent_step`'s only call site is outside the sampler-block branch of the sweep loop, and `store` is always `DI.Constant`-wrapped as defense in depth).

### Also fixed: `test/gibbs.jl`'s own HMM model made the same porting mistake identified earlier this session

The first draft of `hmm_latent` wrote `y[t] ~ Normal(...)` inside a loop over already-observed data — the exact `x[i] ~ dist`-is-assume-only issue documented repeatedly in the model-corpus entries below. Fixed to a vectorized `.~` over the per-timestep mean, same idiom used throughout `bench/corpus/`.

### Known open item, not yet addressed

Prep-reuse-across-sweeps (the `GibbsSamplerSub.prep` caching) is verified correct for `AutoForwardDiff` only (a direct repro confirmed bit-identical gradients when reusing `prep` against a changed-value/same-type `store`). Not yet re-verified for tape-based reverse-mode backends (ReverseDiff/Mooncake/Enzyme) — flagged in `logdensity.jl`'s doc-comment; worth a follow-up test if Gibbs is ever paired with a reverse-mode block.

Also added `StatsFuns` as a real test dependency (`Project.toml` `[extras]`/`[targets]`) — it resolved fine transitively via Distributions in the default environment but wasn't available inside `Pkg.test()`'s isolated test environment, breaking the HMM gate's `logsumexp`/`softmax` imports.

## 2026-07-07 — Float32 performance optimization: coercion fix + speed verification

User reported that using Float32 was not providing expected speedup and was
slightly slower, suggesting implicit Float64 promotion in the hot path.

**Root cause analysis:**
- Traced promotion points via type probes and isolated tests
- P1: `logpdf(dist, value)` returns Float64 if `dist` is Float64-parametrized
  (e.g., `Normal(0.0, 1.0)`), even if `value` is Float32
- P2: Bijectors.jl hardcodes Float64 bounds in constrained transforms
  (e.g., `Exp(minimum(d), 1)` where `minimum(d)` is Float64 even for
  `Exponential{Float32}`), causing bijector output to be Float64 even with
  Float32 input

**Fix implemented (Option A):**
- Added `_to_paramtype` helper in `tilde.jl` to coerce values to
  `eltype(mode.θ)` (the working precision)
- Applied coercion in `_assume(::FlatSlot,...)` and
  `_assume_index(::FlatArraySlot,...)` to both the transformed value `x`
  and the log-jacobian `logjac` from `with_logabsdet_jacobian`
- This ensures constrained bijector outputs are coerced back to the
  parameter type before feeding into `logpdf(dist, x) + logjac`

**Verification:**
- Type probes confirmed Float32 density and gradient types after fix
- Speed benchmarks showed:
  - Density-only (20k params): Float32 still slightly slower (~0.71x)
  - Density + gradient (3k params, ForwardDiff): Float32 faster (~1.24x)
  - Speedup is more pronounced on gradient path (AD overhead) than raw
    density evaluation
- Data type does not need to be Float32: Float32 params + Float64 data is
  actually slightly faster than Float32 params + Float32 data (~0.84x),
  because observe path (logpdf with data) is not differentiated by AD and
  scalar Float64 addition is cheap

**User guidance:**
- Use Float32 for parameters (the things being differentiated by AD)
- Data can stay Float64 without performance penalty
- Distribution literals must use Float32 suffixes (e.g., `Normal(0f0, 1f0)`)
  to avoid P1 promotion; this is now documented in `build_layout` docstring

**Tests added:**
- Float32 density type tests (scalar, positive-constrained, array family)
- Float32 AD gradient type tests
# PracticalBayes.jl devlog

Running log of design decisions, motivations, and compromises. Newest entries
at the top. See also the approved design plan at
`C:\Users\arn203\.claude\plans\i-want-to-make-squishy-pike.md` for the full
architecture writeup this log elaborates on.

See "later considerations.md" for some vague future wish list.

## 2026-07-07 (latest) — note: complex models (mixtures/HMM/capture-recapture/GP/NN/ODE) are targeted next, soon

Explicit forward-looking note per user request, so this doesn't get lost:
the model families deferred at the end of the corpus-porting pass below
(mixtures, HMM, capture-recapture, Gaussian processes, neural networks,
ODEs) are NOT abandoned — we intend to come back and target them soon.
Each needs real new capability, not just more porting effort:

- **Discrete-latent sampling (Gibbs + `AbstractLatentKernel`)** unlocks
  HMM, capture-recapture (M0/Mb/Mh/Mt/Mtbh/Mth, multi_occupancy), and the
  two mixture-model tutorials. This is the design plan's own M3 milestone
  (`i-want-to-make-squishy-pike.md`), never built yet — `build_layout`
  today hard-errors on any unassigned discrete/latent site specifically
  because there's nothing to hand it to. This is probably the highest-
  leverage of the deferred items since it unblocks three whole families at
  once.
- **Gaussian processes** need `AbstractGPs.jl` (or hand-rolled kernel +
  Cholesky code) as a new dependency; not architecturally blocked (a GP
  prior is "just" a differently-parameterized `MvNormal`), just real new
  scope (kernel functions, jitter/numerical-stability handling).
- **Neural networks** need a flat-weight-vector prior + manual
  unpack-into-layers helper and a new dependency (`Flux.jl`/`Lux.jl`); not
  architecturally blocked either — a NN forward pass is arbitrary Julia
  code like any other likelihood.
- **ODEs** need a solver integration (`DifferentialEquations.jl`/
  `OrdinaryDiffEq.jl`) and AD-through-the-solver correctness checking.

Returning to the original task list after this note (see below) — CUDA/GPU
support (design plan milestone M5) is the next item queued up.

## 2026-07-07 — model corpus complete for now: 46 PosteriorDB models + 7 tutorials

Continued the model-porting pass across four more batches (`models_batch2.jl`
through `models_batch5.jl` in `bench/corpus/posteriordb/`, plus
`models_batch2.jl` in `bench/corpus/tutorials/`), stopping deliberately at
the boundary where remaining models need genuinely new capability rather
than more porting effort (user confirmed: stop here rather than start
mixtures/HMM/GP/NN/ODE).

**Batches 2-5 (37 more PosteriorDB models, 46 total)**: covered
posterior-predictive `:=` quantities that call `rand()` (Rate_4/5 — tested
density-only, since `Distributions.rand(::Binomial, ::Dual)` genuinely has
no sampler; sampling isn't a differentiable operation, a Distributions.jl
property, not a PracticalBayes limitation), gather-indexed random effects
(GLMM1's `[year,site]` lookup, election88_full's five simultaneous
group-indexed effects, pilots/rats/radon family's county/group/scenario
indexing), `UniformScaling` as an `MvNormal` covariance, `filldist` over
non-Flat base distributions (`filldist(Normal(mu,sigma), N)`) and over
`Truncated` distributions, `LocationScale`-wrapped `TDist` priors
(`scale*TDist(nu)`, Distributions.jl builtin sugar) for a regularized-
horseshoe logistic regression, and a 2D `filldist` (`filldist(Normal(), D,
k)`) for probabilistic PCA. Two upstream PosteriorDB source bugs found and
worked around (documented in `models_batch5.jl`'s header): two radon
variants write `Normal(fill(...), ...)`/`Normal(mu,sigma)` then index the
result (only sensible for `MvNormal`, clearly a typo in the original Stan
port); a third references an array that's never actually defined anywhere
in the model body and was skipped entirely, not "ported around."

**Porting idiom, reconfirmed multiple times, not a bug**: several tutorial/
PosteriorDB models write `y[i] ~ dist_i` inside a loop over already-observed
data (logistic/poisson/multinomial-logistic-regression tutorials; irt_2pl,
GLMM1 from PosteriorDB). `x[i] ~ dist` is assume-only by design (always
draws into a pre-declared container) — every one of these ports to a
vectorized `.~` instead, including over 2D matrices and arrays of DISTINCT
per-element distributions (`Categorical.(vs)`, matrix-shaped
`BernoulliLogit` arrays) — `tilde_dot`'s `sum(logpdf.(dist_bcast, y))`
fallback handles all of these with no special-casing needed.

**Real porting mistake caught by actually running the model** (not a
package bug): the first `probabilistic-pca` port dropped the original
tutorial's `genes_mean'` transpose, silently producing an incompatible
matrix orientation — caught immediately via `DimensionMismatch` on the
first `build_layout` call, fixed by matching the source's `N×D`/`D×N`
convention exactly. Worth noting as a methodology point: every port in this
corpus was actually RUN (logdensity + gradient-vs-finite-differences), not
just visually reviewed — this is exactly the kind of silent error that
review alone would miss.

**Stopped here, not "done"**: `bench/corpus/posteriordb/` now covers 46 of
~74 models; remaining are capture-recapture (M0/Mb/Mh/Mt/Mtbh/Mth +
multi_occupancy — need hand-coded discrete-latent marginalization),
mixture models (need `MixtureModel`/discrete-latent sampling), HMMs (need
either raw discrete-latent sampling or an external HMM package), a couple
of upstream-broken/stubbed files (diamonds, hier_2pl, soil_incubation — the
last needs `DifferentialEquations.jl` and is empty-bodied upstream anyway).
`bench/corpus/tutorials/` covers 7 of 15; remaining need `AbstractGPs.jl`
(2 GP tutorials), an NN library (bayesian-neural-networks), an ODE solver
(bayesian-differential-equations), or discrete-latent sampling (2 mixture
tutorials, hidden-markov-models) — genuinely new architecture/dependencies,
not more of the same porting pattern. HMM specifically is blocked on real
missing architecture: the design plan's own M3 milestone (Gibbs +
`AbstractLatentKernel` + FFBS-style kernels) was never built this
session — `build_layout` today hard-errors on any discrete/latent site with
no value-block assignment, by design, precisely because there's no Gibbs
mechanism yet to hand it to.

## 2026-07-07 — first PosteriorDB/tutorial model ports, real kwarg-model bug found + fixed

Picked back up the deferred model-corpus work now that Turing-side parity
and the benchmark history/report mechanism are done. Per the earlier plan:
infra first (`filldist`-equivalent, `Flat`/`FlatPos` priors), then a
representative slice of models, then check in before doing the full ~45+
remaining low-effort models.

### `filldist`/`arraydist`/`Flat`/`FlatPos`/`LogPoisson`/`BinomialLogit`

Checked `product_distribution` (the underlying mechanism `filldist` wraps)
against `Bijectors.VectorBijectors` first, before writing anything — it
already works with zero PracticalBayes-side changes, because
`from_linked_vec`/`to_linked_vec` build unconstrained transforms generically
from any `Distribution`'s support and linked length, and `build_layout`
never special-cases scalar vs array-valued sites. Same story for `Flat`/
`FlatPos` (ordinary `Distribution` subtypes — Bijectors derives the
transform from `minimum`/`maximum`) and `Truncated`. So the actual new code
in `src/distributions.jl` is just: the two improper-prior distributions
themselves (ported from Turing's `stdlib/distributions.jl`, since
Distributions.jl doesn't define them), `filldist`/`arraydist` as one-line
`product_distribution` wrappers (ditto, ported from DynamicPPL's
`distribution_wrappers.jl`), and `LogPoisson`/`BinomialLogit` (log/logit-
link reparameterized discrete likelihoods, needed by several GLM-family
PosteriorDB models) — all verified against their reference Distributions.jl
counterparts and via gradient-vs-finite-differences checks. 20 new tests in
`test/distributions.jl`.

### Real bug found: keyword-only `@model` arguments silently dropped

Porting the TuringLang docs `coin-flipping` tutorial (`@model function
coinflip(; N::Int) ... end`) surfaced a genuine compiler bug: `_argnames`
and the constructor's NamedTuple-packaging in `compiler.jl` only ever
iterated `def[:args]` (MacroTools `splitdef`'s positional-argument list),
never `def[:kwargs]`. So `coinflip(; N=5)` silently produced a `Model` with
an EMPTY `args` NamedTuple — `N` was invisible to the tilde-rewriter (any
site referencing it would see an undefined variable or resolve to something
unrelated in an enclosing scope) and never reached the generated evaluator
at all. This had been latent since the compiler was first written; nothing
in the existing test suite used a keyword-only model argument.

Root cause of why the naive fix (just also iterate `def[:kwargs]`) isn't
enough: `Model.args` is one flat `NamedTuple`, splatted **positionally**
into the generated evaluator by `evaluate` (`m.f(mode, acc, m.args...)`,
`model.jl`) — a positional splat can never bind to a `function f(; k)`-style
keyword parameter on the callee. Fix: the generated evaluator now takes
every model argument positionally (original positional args, then original
keyword args, in that fixed order) — `def[:kwargs]` is intentionally NOT
passed through to the evaluator's own `combinedef` call. Only the
user-facing constructor keeps the real keyword-argument call syntax
(`coinflip(; N=5)` still works exactly as written), and packages whatever it
receives into `Model.args` in that same fixed order before handing off.
Added a regression test (`test/compiler.jl`) covering keyword-only args, a
mix of positional + keyword args, and that the resulting `Model.args`
actually contains the keyword value (not just "doesn't crash").

### Model corpus: `bench/corpus/posteriordb/` and `bench/corpus/tutorials/`

Ported 9 models spanning the TuringPosteriorDB family list (conjugate
scalar: `Rate_1`; `filldist` + `MvNormal` array-observe regression: `blr`;
hierarchical `MvNormal` centered/non-centered with `:=` feeding a further
observe: `eight_schools_centered`/`_noncentered`; nonlinear regression with
`:=` used only for reporting: `dugongs`; discrete Poisson GLM with log-link
+ `:=`: `GLM_Poisson`; `Flat`-prior interaction regression:
`kidscore_interaction`; `Flat`/`FlatPos` minimal regression: `earn_height`;
`Flat`-prior logistic GLM via `BernoulliLogit`: `wells_dae`) and 6 models
from the TuringLang docs tutorials (`coin-flipping`, `bayesian-linear-
regression`, `bayesian-logistic-regression`, `bayesian-poisson-regression`,
`multinomial-logistic-regression`, `bayesian-time-series-analysis`). All 15
pass a structural smoke test (`run_smoke_tests.jl` in each corpus
directory): logdensity finite, ForwardDiff gradient matches central finite
differences. No original PosteriorDB/tutorial datasets were fetched (would
need TuringPosteriorDB.jl, a Turing-ecosystem package — same "can't coexist
with PracticalBayes" constraint as everything else Turing-side this
session); each model instead gets synthetic, shape/type-matched data,
sufficient to prove the model STRUCTURE (every distribution/syntax feature
it uses) runs correctly, not to replicate the original posterior exactly.

Porting note, not a bug: three of the six tutorials (`bayesian-logistic-
regression`, `bayesian-poisson-regression`, `multinomial-logistic-
regression`) originally write `y[i] ~ dist` inside a `for` loop over
already-observed data `y`. PracticalBayes' `x[i] ~ dist` is assume-only BY
DESIGN (documented in `compiler.jl`: it always draws into a pre-declared
container) — observing element-by-element against pre-bound data is exactly
what `.~` is for. Ported all three to a vectorized `.~` over the whole
likelihood-parameter array instead (for `multinomial_logistic_regression`,
`Categorical.(vs)` — an array of DISTINCT per-row distributions — exercises
`tilde_dot`'s `sum(logpdf.(dist_bcast, y))` fallback path, not just the
common scalar-broadcast `loglikelihood` fast path).

## 2026-07-07 — automated benchmark history + report

Previously, benchmark results only existed as terminal output and one
hand-written HTML artifact snapshot — no way to tell if a change regressed
performance without manually re-running and eyeballing numbers. Fixed per
user request ("the benchmark report should be an automated thing that gets
updated when the benchmarks are run... tagged by date and latest git
commit, so we can spot regressions").

**Storage**: `bench/suite.jl` and `test/comparison_env/generate_turing_reference.jl`
each got a `record!`/`write_history!` pair that appends one JSON-Lines
record per `TimingResult` — tagged with `package` ("PracticalBayes" or
"Turing"), `layer` (logdensity/gradient/nuts/etc.), `model`/`shape`/
`precision`/`backend`, plus a run-level `timestamp` and git `commit` hash —
to `bench/results/history.jsonl` and `bench/results/history_turing.jsonl`
respectively. Append-only and committed to git, so `git log -p` on either
file doubles as a benchmark changelog and regressions are traceable to the
exact commit that caused them. Hand-rolled JSON (flat objects, no nesting)
rather than adding a JSON dependency just for this — mirrors the project's
existing "don't add a dependency for something this small" bias.

**Report**: explicitly NOT hand-written HTML this time (user: "dont write
html directly. use quarto or plain markdown, that gets rendered as html").
`bench/generate_report.jl` reads both JSONL files back (a small hand-rolled
JSON parser — first attempt naively split on `","` and broke as soon as a
bare numeric field followed a string field with no closing quote to split
on; replaced with a proper left-to-right character scan) and writes
`bench/report.qmd`: a latest-snapshot table per layer (PracticalBayes vs
Turing side by side, with a ratio column), plus a per-series history section
that will start showing multi-row trends once `bench/suite.jl` has been run
more than once. `quarto render bench/report.qmd` turns it into
`bench/report.html` — confirmed this needs nothing beyond the Quarto CLI
itself (no `QuartoNotebookRunner.jl` install needed, since the `.qmd` has no
executable code cells, just YAML frontmatter + prose + Markdown tables;
Quarto bundles what it needs for that case automatically).

Workflow going forward: `julia --project=. bench/suite.jl` (appends to
`history.jsonl`) → `julia --project=test/comparison_env
test/comparison_env/generate_turing_reference.jl` (appends to
`history_turing.jsonl`, only needed when Turing-side numbers might have
moved) → `julia --project=. bench/generate_report.jl` → `quarto render
bench/report.qmd`. `report.html` itself is gitignored (a build artifact,
same treatment as `docs/build/`); `report.qmd`, `generate_report.jl`, and
both `.jsonl` history files are committed.

## 2026-07-07 — real bug in indexed tilde under AD, full Turing parity, AD-crossover confirmed

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
