# A generic conjugate-Gibbs latent kernel.
#
# Many Gibbs blocks have a closed-form full conditional in a conjugate family:
# given the rest of the state, the block's posterior is a Beta / Dirichlet / etc.
# whose parameters are the prior plus some sufficient statistics ("counts") read
# off the current draw of everything else. Writing such a kernel by hand is always
# the same shape — a struct, a `latent_step` method, a loop that accumulates the
# counts, and a draw — so this file provides the generic engine, leaving the user
# only the part that is actually model-specific: how to compute the counts.
#
# This is the conjugate sibling of the hand-written `FFBS` example in latent.jl:
# both are `AbstractLatentKernel`s that `Gibbs` calls once per sweep, outside every
# gradient step, so they may use plain non-differentiated arithmetic throughout.
#
# It knows nothing about any particular model. The `count` closure receives the
# `ModelConditional` and reaches into `c.values` / `c.model.args` itself — so a
# domain package (e.g. an epidemic-model package) can build ready-made conjugate
# kernels on top of this by supplying count closures that know its data layout,
# without this engine assuming any of it.

using Distributions: Beta, Dirichlet

"""
    ConjugateGibbs(name, family, prior, count; n=nothing, collect=identity)

An `AbstractLatentKernel` (see [`latent_step`](@ref)) that resamples the Gibbs
block `name` from its conjugate full conditional. Once per sweep it calls the
user's `count` function to get the sufficient statistics, adds the `prior`
hyperparameters, and draws from the resulting posterior in `family`.

Arguments:

- `name::Symbol` — the block this kernel owns and returns.
- `family::Symbol` — `:beta` or `:dirichlet` (the conjugate posterior family).
- `prior` — the prior hyperparameters: `(a, b)` for `:beta`, a concentration
  vector for `:dirichlet`.
- `count` — the sufficient-statistics closure. Its arity depends on `n`:
    * `n === nothing` (a single draw): `count(c) -> counts`, where `c` is the
      [`ModelConditional`](@ref) and `counts` is `(successes, failures)` for
      `:beta` or a category-count vector for `:dirichlet`.
    * `n::Integer` (one independent draw per index `1:n`): `count(c, k) -> counts`
      for each `k`, same `counts` shape; the `n` draws are assembled by `collect`.
- `n` — `nothing` for one scalar draw, or the number of independent conjugate
  draws to make (e.g. one per group/season/time).
- `collect` — for the `n`-draw form, maps the `Vector` of per-index draws to the
  block's returned value (default: the vector itself). Ignored when `n === nothing`.

The `count` closure does all model-specific work; this kernel does not touch
`c.values` or `c.model.args` itself, so it imposes no assumptions about the model.

# Example — a conjugate Beta update for a success probability `p`, where the data
# are `n` Bernoulli trials passed as a model argument `y`:

```julia
k = ConjugateGibbs(:p, :beta, (1, 1)) do c   # `do` passes the count closure
    y = c.model.args.y
    s = count(==(1), y)
    (s, length(y) - s)
end
# ... spl = Gibbs(:p => k, :other => NUTS(0.8))
```
"""
struct ConjugateGibbs{C,P,L} <: AbstractLatentKernel
    name::Symbol
    family::Symbol
    prior::P
    count::C
    n::Union{Nothing,Int}
    collect::L
end
function ConjugateGibbs(name::Symbol, family::Symbol, prior, count;
                        n::Union{Nothing,Integer}=nothing, collect=identity)
    family in (:beta, :dirichlet) ||
        error("ConjugateGibbs: unknown conjugate family $(family) (want :beta or :dirichlet)")
    ConjugateGibbs(name, family, prior, count,
                   n === nothing ? nothing : Int(n), collect)
end

# Count-first method so the `do`-block form reads naturally — `do` passes the
# closure as the FIRST positional argument:
#     ConjugateGibbs(:p, :beta, (1, 1)) do c
#         ...counts...
#     end
ConjugateGibbs(count, name::Symbol, family::Symbol, prior; kwargs...) =
    ConjugateGibbs(name, family, prior, count; kwargs...)

# Draw one value of the given family from prior + counts. Dispatch on Val(family)
# keeps the hot path free of a runtime branch on the Symbol.
_conjugate_draw(rng, ::Val{:beta}, prior, counts) =
    rand(rng, Beta(prior[1] + counts[1], prior[2] + counts[2]))
_conjugate_draw(rng, ::Val{:dirichlet}, prior, counts) =
    rand(rng, Dirichlet(prior .+ counts))

function latent_step(rng, k::ConjugateGibbs, block_names, c::ModelConditional)
    block_names == (k.name,) ||
        error("ConjugateGibbs owns block $(k.name), got $(block_names)")
    fam = Val(k.family)
    if k.n === nothing
        draw = _conjugate_draw(rng, fam, k.prior, k.count(c))
        return NamedTuple{(k.name,)}((draw,))
    else
        draws = [_conjugate_draw(rng, fam, k.prior, k.count(c, i)) for i in 1:k.n]
        return NamedTuple{(k.name,)}((k.collect(draws),))
    end
end
