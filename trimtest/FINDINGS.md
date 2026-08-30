# Trimming a PracticalBayes model: what actually blocks it

Test case: `PBTrim`, a minimal model (`mu ~ Normal`, `sigma ~ Exponential`,
`y ~ Normal(mu, sigma)`) sampled with AdvancedHMC NUTS. Works under the JIT.
Julia 1.12.7, `juliac --trim=safe`.

## Not the blockers

| suspected | reality |
|---|---|
| FunctionWrappers (what killed Turing) | **absent from the dependency graph entirely** — no FunctionWrappers, FunctionWrappersWrappers, SciMLBase or Optimization |
| ForwardDiff extension load order | **already handled** — `src/PracticalBayes.jl:12` does `using ForwardDiff: ForwardDiff` precisely so the DI extension activates |
| ForwardDiff `Dual` MethodError (the Turing failure) | only 8 of 204 errors mention ForwardDiff, and as `GradientConfig` type parameters, not the runtime `Dual` crash |
| Distributions, Bijectors | **0 errors** |
| `sample()`'s `Vector{Any}` (sample.jl:267) | real, but avoidable: driving AdvancedHMC directly through the public API bypasses it and only drops 216 → 212 |

## The actual blocker: `SiteRecord`'s `::Any` fields

`src/layout.jl:11`

```julia
struct SiteRecord
    name::Symbol
    dist_exemplar::Any   # <-- here
    linked_len::Int
    role::Symbol
    init_val::Any        # <-- and here
end
```

`Layout` stores `meta::Vector{SiteRecord}`, and `link`/`invlink` read
`dist_exemplar` on the sampling hot path (layout.jl:317, 336). The clean
statement of the failure is verifier error #8:

```
to_linked_vec(getfield(SiteRecord, :dist_exemplar)::Any)::Any
```

Everything else cascades from it:

| group | count | why |
|---|---|---|
| expression printing (`show_unquoted`, `show_block`, `show_call`, `Base.print`, `Base.show`) | 114 | `Any` could hold an `Expr`, so trim retains the whole recursive printer defensively |
| `SiteRecord` / `to_linked_vec` / `dist_exemplar` | 24 | the root itself |
| AdvancedHMC | 12 | inherits unresolved types from the above |
| `Layout` unresolved parameter | 10 | `build_layout` return type |
| ForwardDiff | 8 | `GradientConfig` parameters |

**204 errors, one cause.** The printing majority is a symptom, not a separate
problem — worth stating because fixing "printing" directly would be chasing the
wrong thing.

## Secondary: `build_layout` return type

```julia
Base.infer_return_type(build_layout, Tuple{typeof(m)})
  → Tuple{Layout, Vector{Float64}, NamedTuple}      # Layout's parameter erased
```

The runtime value IS concrete
(`Layout{@NamedTuple{mu::FlatSlot, sigma::FlatSlot}}`) — inference cannot predict
it because `build_layout` discovers sites by *running the model* under
`TraceMode`, which the source explicitly documents as "the ONE place dynamic
dispatch and allocation are allowed". That design is correct for a PPL; it just
needs a function barrier so the dynamic part is isolated from the compiled part.

Adding a barrier in user code helps only marginally (212 → 204) because the
barrier's own argument types are still unpredictable. The barrier has to be
*inside* the package, where the concrete layout can be passed to a specialised
method.

## The heterogeneity constraint

`SiteRecord` cannot simply be parameterised as `SiteRecord{D,V}` and stored in a
`Vector`, because sites genuinely differ:

```
mu     dist=Normal{Float64}       init=Float64
sigma  dist=Exponential{Float64}  init=Float64
y      dist=Normal{Float64}       init=Float64
```

A `Vector{SiteRecord{D,V}}` cannot hold both `Normal` and `Exponential`. Options:

1. **Tuple of records** — `meta::Tuple{Vararg{SiteRecord}}` stays concrete and
   heterogeneous. Costs compile time on models with many distinct sites.
2. **Union-split on a closed set** — works if the distribution set is bounded.
3. **Keep `Vector{SiteRecord}` for `build_layout`, materialise a typed tuple
   once** into the `Layout` for the hot path, leaving discovery untouched.

(3) looks best: discovery stays dynamic and allocating as designed, and only the
`Layout` that the sampler actually uses becomes concrete.

## Caveat

A clean verifier pass proves nothing. The `test2inf` CLI built with **zero**
verifier errors and then crashed at runtime on a trimmed-away method. Any
trimmed PracticalBayes binary must be exercised with `JULIA_LOAD_CODEGEN_LIB=0`
before being believed.

---

## Reproducing the working trimmed binary (PBTrim)

    cd trimtest
    julia --project=PBTrim -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
    julia --project=PBTrim -e 'using PBTrim; PBTrim.main(String[])'    # JIT reference
    julia --project=juliac-env -e 'using JuliaC; JuliaC.main(ARGS)' -- \
        --output-exe pbtrim --bundle pb_build --trim=safe --experimental ./PBTrim
    JULIA_LOAD_CODEGEN_LIB=0 ./pb_build/bin/pbtrim.exe

Expected: 0 verifier errors, and the binary's output IDENTICAL to the JIT's to
all 16 digits. `juliac-env` is a one-off environment holding JuliaC itself:

    julia -e 'using Pkg; Pkg.activate("juliac-env"); Pkg.add("JuliaC")'

GMTrim is the same experiment with GradMode instead of a hand-supplied
gradient. It currently fails with 162 verifier errors; see the vals section
above.
