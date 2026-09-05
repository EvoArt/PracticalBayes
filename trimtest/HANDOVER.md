# Handover: finishing trimmable models

Branch `trim-support`. Read `FINDINGS.md` first, then this.

## Load the skill

Load the `julia-juliac-trim` skill before doing anything else. It covers
juliac, `--trim=safe`, bundling, and diagnosing unresolved-call errors,
which is where most of the time goes.

## What already works

`trimtest/PBTrim/` — a minimal model (mu ~ Normal, sigma ~ Exponential,
y ~ Normal) compiles to **zero verifier errors**, producing a 2.9 MB binary
whose output matches the JIT to all 16 digits. Reproduction steps are at the
bottom of FINDINGS.md. This is the proof the whole thing is possible.

The ladder that got there:

| errors | change |
|---|---|
| 204 | baseline: `sample()` + AdvancedHMC + ForwardDiff |
| 12 | freeze the layout into a `const` so `build_layout` never runs in the binary |
| 2 | supply the gradient instead of using AD |
| **0** | use `static_hmc` instead of AdvancedHMC |

Freezing the layout is by far the biggest single win — bigger than every type
annotation combined.

## Package changes on this branch

**`src/layout.jl`** — `SiteRecord{D,V}` and `Layout{S,M}` gained type
parameters. `build_layout(...; static=true)` freezes records into a Tuple
whose concrete types survive compilation. `@static_model` sets that default
per-model via a trait on the evaluator. `@model` is untouched; no existing
user pays anything.

It is opt-in because it costs ~5 ms compile time **per site record**. Measured:
distinct distribution *types* cost nothing (2 -> 30 types is if anything
cheaper than the Vector); record *count* is what costs. A model whose indexed
family is a loop (`for i in 1:400; x[i] ~ ...`) keeps one record per element
and pays ~2 s; the same model as `x ~ MvNormal(zeros(400), I)` is 3 records
and pays nothing.

**`src/static_hmc.jl`, `src/static_hmc_api.jl`** — a dependency-free HMC/NUTS,
nothing beyond Base. Needed because AdvancedHMC is not trimmable:
`PhasePoint`'s constructor uses `@argcheck`, whose error path builds a message
from an `Any`, leaving `ArgCheck.build_error` unresolved on the core sampling
path.

`static_hmc` is **validated**: sd within 2% of truth across step sizes 0.1-2.0
on a standard normal, matches an analytic conjugate posterior, and is what the
working trimmed binary uses.

`static_nuts` is **NOT correct**. It over-disperses — sd 1.26 against a true
1.0 on a standard normal — once metric adaptation runs. With zero warmup it
gives 0.95, so the fault appears with adaptation, not in the tree. I diagnosed
a p_sum double-counting bug, fixed it, and the output was byte-identical, so
that diagnosis was wrong and the real cause is unknown. Do not use it. Either
fix it properly, or delete it and document tuning as an offline step using
`sample()` — a compiled binary wants fixed parameters anyway, so the second is
defensible.

## The remaining work

`trimtest/GMTrim/` is the same experiment with GradMode supplying the gradient
instead of a hand-written one. It samples correctly under the JIT (recovers
beta1 = 0.4967 against a true 0.5, zero divergences) but `--trim=safe` gives
**162 verifier errors**, down from 282 before commit e3bc3a1.

They cascade from one line:

    src/gradmode_codegen.jl:124    vals = Dict{Symbol,Any}()

`_gm_hier_params` infers `Tuple{Any,Any}` because it does
`_gm_scalar(vals[ps.hyper_loc])`. `_gm_scalar` is fine — it infers `Float64`
given a concrete argument — so the `Any` comes purely from the Dict's value
type. 46 of the 162 errors name `_gm_hier_params` directly; the rest are
getindex/convert/literal_pow cascading from it.

This is the same shape of problem for the third time: `SiteRecord.dist_exemplar`,
then `PriorSite.args`, now `vals`. Each a container typed `Any` whose contents
are concrete at runtime.

**Do not annotate `_gm_scalar(vals[k])::Float64`.** I proposed exactly that and
it was wrong: hyper-*locations* can legitimately be vectors —

    mu ~ MvNormal(zeros(J), I)      # vector parameter
    a  ~ MvNormal(mu, I)            # vector hyper-location

— and checking that assumption turned up a silently wrong gradient in the
vector case, since fixed in e31e113. Hyper-*scale* is scalar-only (the
recognizer enforces it); hyper-location is not.

The other session's assessment, which I agree with: type the container where
the invariant is **established**, not asserted at the use site, and add a
recognizer-side rejection for any site whose linked representation is not
scalar-or-vector, so the invariant is enforced rather than assumed.

**A closed union spelled by hand does NOT work.** I originally suggested
`Dict{Symbol,Union{Float64,Vector{Float64}}}` here. That is wrong, and
practicalbayes-63 measured why -- calling `_gm_read_param` directly on a model
exercising every recognized site kind:

    s     -> Float64
    mu    -> Float64
    beta  -> SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}
    a     -> SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}

Two errors in that suggestion. The vector arm is a **SubArray, not a Vector**:
`_gm_read_param` reads through `_linked_view(theta, slot.range)`, deliberately
a `Base.SubArray` rather than a plain view, because some AD backends overload
`view` to return their own type (PolyesterForwardDiff is the one that bit them
-- see the comment in tilde.jl). And the **element type is not fixed**: this
package is Float32-first, so theta may be `Vector{Float32}`, and under a
ForwardDiff-family backend it is `Vector{<:Dual}`. GradMode is Float64-only in
practice today, but the container type must be honest about what
`_gm_read_param` can return.

Note the failure mode that makes this dangerous: a hand-written union passes a
Float64 test and fails on a Float32 model.

Two directions that would actually work:

1. Parameterise on the element type and store the view type, roughly
   `Dict{Symbol,Union{T,SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}}`
   with `T = eltype(theta)`. Concrete, but it hard-codes the SubArray spelling,
   exactly the thing that silently breaks when the view machinery changes.

2. **Stop using a Dict.** The set of names is known at recognition time
   (`plan.priors` is fixed), so the values can live in a NamedTuple or a plain
   Vector indexed by prior position, built once. That makes the types concrete
   BY CONSTRUCTION rather than by assertion -- the "type where the invariant is
   established" principle applied properly. Probably faster too, since it drops
   the hashing.

Option 2 is what practicalbayes-63 would do, and it is why the estimate is
~1 hour rather than ten minutes: it is a real change to how `vals` is built,
not an annotation.

## A clean verifier pass proves nothing

The single most important thing here. Two separate bugs produced zero verifier
errors and a binary that was completely wrong:

- a trimmed binary ran with 500/500 divergences and acceptance 0.0 while the
  same source under the JIT was clean;
- in a sibling project, a binary passed the verifier and then died at runtime
  on a method that had been trimmed away.

Always run the built binary with `JULIA_LOAD_CODEGEN_LIB=0`, so any JIT
fallback fails loudly, and compare its **output** against the JIT on the same
input. If they do not match you are not done, whatever the verifier said.

## Traps already paid for

1. **`Core.Box`.** A closure that reassigns a captured variable makes Julia box
   it, erasing the type. One boxed variable in an RNG seeder produced 20
   verifier errors. Write plain state-in/state-out functions instead — see
   `_splitmix64`, fixed in c30f459.

2. **Anything baked in at precompile time must come from a seeded source.**
   `build_layout` draws theta0 from `Random.default_rng()`, so a `const` layout
   differs on every build. One unlucky draw gave a binary 500/500 divergences
   from source that was clean under the JIT. See PBTrim's `Xoshiro(20240829)`.

3. **`build_layout` itself cannot be trimmed** — it discovers sites by *running*
   the model under TraceMode, through a Dict and dynamic dispatch. That is
   correct design for a PPL. Freeze the layout into a `const` instead; a
   compiled program has a fixed model, so this costs nothing.

4. **AD cannot be trimmed.** DifferentiationInterface's ForwardDiff extension
   leaves `GradientConfig` and the per-call-site `Dual` tag types unresolved.
   Freezing a prepared AD object does not help — the Dual types live in its
   *type parameters*. This is exactly why GradMode matters: closed form, no Duals.

5. **Not a blocker, despite killing Turing:** FunctionWrappers is absent from
   PracticalBayes' dependency graph entirely (no FunctionWrappers,
   FunctionWrappersWrappers, SciMLBase or Optimization). Do not go looking.
   The AD extension load order is already handled at `src/PracticalBayes.jl:12`.
   Distributions and Bijectors give zero errors.

## The other session

`practicalbayes-63` owns GradMode and is actively editing
`src/gradmode_codegen.jl`, `src/gradmode_recognize.jl`, `src/distributions.jl`.
Message them before editing those files — `SendMessage` to
`practicalbayes-63`, checking `ListAgents` first in case the name has changed.

They are careful and have earned trust: they caught the wrong-gradient bug by
checking an assumption I had asserted without evidence, and they verified my
claims independently rather than taking them on faith. Extend the same
courtesy. They keep `ad_worklog.md` in the repo, which records the trim
findings from their side.

History note: an earlier commit of mine swept in ~1100 lines of their
in-progress work. They rebuilt the history (messages only — not one byte of
code moved, verified by tree hash) and there is a `backup-before-split` tag.
Do not rewrite shared history without asking them.

## Scope

The finish line for the GradMode path is zero verifier errors on GMTrim plus a
binary whose output matches the JIT.

If the `vals` work turns out larger or riskier than an hour, say so rather than
forcing it. The hand-gradient path already works today and is a legitimate
answer for anyone willing to write a gradient. GradMode only fires on models
`recognize_glm` accepts (35 of 69 corpus models), and it is *slower* than
ForwardDiff at small N and P (0.48x at N=200/P=2), so the honest claim is
"trimmable for recognized GLM-shaped models", not for models generally. Do not
overstate it.

Run the full test suite before declaring done. Report what you actually
verified versus what you inferred, and if a number does not reproduce say so
plainly rather than defending it — I had to retract a headline benchmark figure
in this work, and the retraction was more useful than the original claim.
