# M4 milestone: predictive utilities built on PriorMode/FixedMode (already
# fully implemented in modes.jl/tilde.jl — see those files' own docstrings
# for the exact per-site semantics each mode uses). This file is just the
# convenience layer that constructs the right mode and reads results back
# into user-facing NamedTuples/arrays.

using Random: Random

"""
    rand([rng,] model::Model) -> NamedTuple
    rand([rng,] model::Model, n::Integer) -> Vector{NamedTuple}

Draw one (or `n` independent) sample(s) from `model`'s prior/prior-predictive
distribution: every assumed site is drawn fresh from its prior; every observe
site uses real data if `model` has any bound (via arguments or conditioning),
otherwise is drawn from the likelihood. Returns a `NamedTuple` mapping every
site name (assumed AND observed) to its value — this is `PriorMode`'s own
`values` accumulator (see modes.jl), not filtered down to just the
parameters, since a common use is generating synthetic `(params..., data...)`
datasets for the model itself.
"""
function Random.rand(rng::Random.AbstractRNG, m::Model)
    mode = PriorMode(rng, m.conditioned)
    evaluate(m, mode, Accum(0.0))
    return mode.values[]
end
Random.rand(m::Model) = rand(Random.default_rng(), m)
Random.rand(rng::Random.AbstractRNG, m::Model, n::Integer) = [rand(rng, m) for _ in 1:n]
Random.rand(m::Model, n::Integer) = rand(Random.default_rng(), m, n)

# Shared by `returned`/`predict`/`logjoint`/`logprior`/`loglikelihood`: builds
# a `FixedMode` at a given NamedTuple of parameter values and runs the model.
# `predict` gates whether un-conditioned observe sites are sampled (true) or
# must already have real data (false, the default — used by the log-density
# accessors, where a missing observation genuinely can't be scored).
function _fixed_evaluate(rng, m::Model, fixed::NamedTuple; predict::Bool=false)
    mode = FixedMode(rng, fixed, m.conditioned; predict=predict)
    retval, acc = evaluate(m, mode, Accum(0.0))
    return retval, acc, mode.values[]
end

"""
    logjoint(model::Model, nt::NamedTuple) -> Real
    logprior(model::Model, nt::NamedTuple) -> Real
    loglikelihood(model::Model, nt::NamedTuple) -> Real

Evaluate the model's log-joint/log-prior/log-likelihood at a fixed point
`nt` (a NamedTuple of parameter values, e.g. one posterior draw). Every
observe site must resolve to real data (via `model`'s own arguments or
conditioning) — there is no `predict=true` escape hatch here, since a
log-density with a missing observation term isn't a well-defined number.

These share their names with the `Accum`-level accessors in accumulator.jl
(`logjoint(acc::Accum)`, etc.) — Julia's multiple dispatch picks the right
method from the argument types, so both are always available un-qualified.
"""
function logjoint(m::Model, nt::NamedTuple; rng=Random.default_rng())
    _, acc, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return logjoint(acc)
end
function logprior(m::Model, nt::NamedTuple; rng=Random.default_rng())
    _, acc, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return logprior(acc)
end
function loglikelihood_at(m::Model, nt::NamedTuple; rng=Random.default_rng())
    _, acc, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return loglikelihood_(acc)
end

"""
    returned(model::Model, nt::NamedTuple; rng=Random.default_rng())

Runs `model` with every assumed site fixed to `nt`'s values (same
`FixedMode`/`predict=false` semantics as `logjoint`) and returns the model
function's own return value — whatever the `@model` body's last expression
evaluates to (a `:=`-computed summary statistic, e.g.), not the log-density.
"""
function returned(m::Model, nt::NamedTuple; rng=Random.default_rng())
    retval, _, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return retval
end

"""
    predict([rng,] model::Model, draws) -> Vector{NamedTuple}

Posterior-predictive sampling: for each parameter draw in `draws` (any
iterable of `NamedTuple`s, e.g. the output of `rand(model, n)`, or built by
hand from a `FlexiChains.SymChain` — see below), fixes the model's assumed
sites to that draw's values and samples every un-conditioned observe site
fresh from the likelihood (`FixedMode(...; predict=true)`). `model` is
typically NOT the same model instance used for inference — pass one with the
observed argument(s) replaced by an array of `missing` at the desired output
shape (the standard convention; see `.~`'s predictive-sampling support in
tilde.jl for why the array must carry shape, not a bare `missing`), e.g.:

```julia
m_train = regression(x_train, y_train)
# ... run `sample` on m_train, get a `chn::SymChain` ...
m_test = regression(x_test, fill(missing, length(x_test)))
draws = chain_draws(chn)  # see below
preds = predict(m_test, draws)
```

Returns a `Vector{NamedTuple}`, one per input draw, containing every site's
value (fixed parameters AND freshly-sampled observations) — same shape as
`rand`'s output.
"""
function predict(rng::Random.AbstractRNG, m::Model, draws)
    return [_fixed_evaluate(rng, m, nt; predict=true)[3] for nt in draws]
end
predict(m::Model, draws) = predict(Random.default_rng(), m, draws)

"""
    chain_draws(chn::FlexiChains.SymChain) -> Vector{NamedTuple}

Flattens every (iteration, chain) draw in `chn` into one `Vector{NamedTuple}`
(all chains pooled together, in chain-major then iteration order), suitable
as the `draws` argument to `predict`/`returned`/`logjoint` etc. Only
`Parameter` keys are included (AdvancedHMC's diagnostic `Extra` keys, e.g.
`acceptance_rate`, are dropped) — a model's `FixedMode` only ever reads
fields it recognizes by name, so including the extras would be harmless but
pointless; this keeps each NamedTuple exactly the shape a model expects.
"""
function chain_draws(chn::FlexiChains.SymChain)
    param_names = FlexiChains.parameters(chn)
    niters, nchains = size(chn[first(param_names)])
    draws = Vector{NamedTuple}(undef, niters * nchains)
    idx = 1
    for c in 1:nchains, i in 1:niters
        draws[idx] = NamedTuple{Tuple(param_names)}(Tuple(chn[p][i, c] for p in param_names))
        idx += 1
    end
    return draws
end

"""
    pointwise_loglikelihoods(model::Model, nt::NamedTuple; flatten=false)
        -> NamedTuple or Vector{Float64}

Per-observation log-likelihood at a fixed point `nt` (e.g. one posterior
draw) — the building block for LOO-CV/WAIC-style model comparison (e.g. via
ParetoSmooth.jl or exporting to ArviZ), which need one log-likelihood value
per observation rather than `loglikelihood_at`'s single summed scalar. Same
"every observe site must resolve to real data" contract as
[`logjoint`](@ref)/`loglikelihood_at` — there is no `predict=true` form of
this, since a per-observation likelihood term is meaningless without a real
observation to score it against.

By default (`flatten=false`) returns a `NamedTuple` mapping each observe
site's name to its own `Vector` of per-observation values, preserving site
identity for models with more than one observe block. Pass `flatten=true` to
get everything concatenated into a single `Vector{Float64}` (in the order
sites were visited during evaluation), which is the shape most LOO-CV/WAIC
tooling expects directly.

This is a genuinely separate, opt-in re-evaluation of the model — it does
NOT reuse any state from sampling — because computing per-observation values
requires `logpdf.(dist, y)` at every `.~` site, which is deliberately NOT
what the hot HMC path computes (see `PointwiseMode`'s own docstring for
why). Expect to call this once per posterior draw you want pointwise values
for, not inside any performance-sensitive loop.

```julia
draws = chain_draws(chn)
pw = [pointwise_loglikelihoods(model, nt) for nt in draws]  # one NamedTuple per draw
loo_matrix = reduce(hcat, pointwise_loglikelihoods(model, nt; flatten=true) for nt in draws)  # n_obs × n_draws
```
"""
function pointwise_loglikelihoods(m::Model, nt::NamedTuple; flatten::Bool=false)
    mode = PointwiseMode(nt, m.conditioned)
    evaluate(m, mode, Accum(0.0))
    pw = mode.pointwise[]
    flatten || return pw
    return reduce(vcat, values(pw))
end
