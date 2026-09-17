# Differences from Turing

PracticalBayes uses Turing-style `@model`/`~` syntax, but the evaluation core is
a separate implementation. Most models port across unchanged. This page lists
the places where they differ, from a model author's point of view.

## `.~` accepts per-element distributions

PracticalBayes:

```julia
y .~ Normal.(eta, sigma)      # eta is a vector: one Normal per observation
```

DynamicPPL removed arrays of distributions in `.~` as of v0.35, so the same line
in Turing is rejected with *"does not allow arrays of distributions in `.~`"*.
Turing still supports `.~`, but only when the right-hand side is a **single**
distribution object broadcast over the data — the same distribution for every
datapoint. The moment each observation needs its own parameters (any regression,
where the mean varies with `X*beta`), the Turing model has to be written as:

```julia
y ~ product_distribution(Normal.(eta, sigma))    # Turing
```

Both forms compute the same log density and run at about the same speed; the PB
form is one construction shorter. `product_distribution`/`arraydist` also work
in PracticalBayes, so a model written the Turing way ports over as-is.

## `.~` is observe-only

`.~` accumulates a likelihood against data you already have. It cannot introduce
unknowns:

```julia
x .~ Normal(0, 1)       # error if x is unknown
```

Use an array-valued distribution (`x ~ MvNormal(...)`, `x ~ filldist(...)`) or an
indexed loop (`x[i] ~ dist`) instead. Turing allows `.~` on the assume side.

## Missing data uses a shape-carrying array

For predictive sampling at a `.~` site, pass an array of `missing` at the shape
you want drawn, not a bare scalar:

```julia
predict(model(X, fill(missing, n)), draws)
```

A scalar-broadcast right-hand side (`Normal.(mu, sigma)` with scalar arguments)
carries no length, so the shape has to come from the observed argument. This
matches DynamicPPL's own `predict` convention. Scalar `~` sites accept a bare
`missing` as usual.

## Conditioning is whole-name only

`model | (; y = data)` conditions a top-level variable. Conditioning part of a
vector-valued variable (`model | (; x[3] = 5.0)`) is not supported yet; Turing
allows it.

## Float32 needs explicit literals

PracticalBayes has a Float32 parameter path (Turing promotes to Float64
internally regardless). To keep a model in Float32, distribution literals and
vector-valued prior parameters have to say so:

```julia
sigma ~ Exponential(1.0f0)                      # not Exponential(1)
beta  ~ MvNormal(zeros(paramtype(__mode__), k), I)   # not zeros(k)
```

Data (`X`, `y`) can stay Float64 without breaking Float32 propagation. See
[Float32 and GPU usage](@ref) for the full picture.

## Model arguments cannot destructure type parameters

`f(::Val{x}) where {x}` -style positional arguments are not supported by the
`@model` compiler. Pass a plain `Symbol` or typed argument and branch on it at
runtime.

## Name collisions when both packages are loaded

PracticalBayes and Turing/DynamicPPL both export `@model`, `Model`,
`LogDensityFunction`, `filldist`, and `arraydist`. If a script needs both, use
`import` (not `using`) and qualify every call.

## Chains are FlexiChains, not MCMCChains

`sample` returns a `FlexiChains.SymChain`. Pass `chain_type=nothing` for raw
AdvancedHMC transitions. Turing (as of 0.45) also returns FlexiChains, so this
is only a difference against older Turing code.

### A vector parameter is one chain entry, not many

This is the most common porting trap. MCMCChains flattens `beta` into separate
`beta[1]`, `beta[2]`, ... columns. FlexiChains does not: a vector-valued
parameter stays **one** entry whose elements are the whole vector.

```julia
beta ~ filldist(Normal(0, 2), 3)     # 3-element parameter
b = chn[:beta]                       # (draws × chains) matrix of Vectors
eltype(b)                            # Vector{Float64}, NOT Float64
```

So the natural Turing-style column index reaches for a *chain* that isn't there:

```julia
b[:, 2]     # BoundsError: 200×1 Matrix{Vector{Float64}} at index [1:200, 2]
```

To get the draws for one element, index the draw first and the element second:

```julia
draws = [b[i, 1][2] for i in axes(b, 1)]     # all draws of beta[2], chain 1
```

Or use [`param_draws`](@ref), which does this and handles multiple chains:

```julia
param_draws(chn, :beta, 2)          # all draws of beta[2], every chain
```
