"""
    Model{F,TArgs<:NamedTuple,TCond<:NamedTuple} <: AbstractPPL.AbstractProbabilisticProgram

`f`: the generated evaluator, called as `f(mode, acc, args...)`.
`args`: the model-call arguments as a NamedTuple (values may be `missing` to
mark a parameter passed positionally, Turing-style).
`conditioned`: extra data bound via `model | (; y = ...)`, merged in on top of
`args` at each tilde site (see `getcond` in modes.jl).

The *role* of a top-level name (observed vs assumed) is determined by
whether it is a key of `args` (with a non-`missing`/non-`nothing` value) or
of `conditioned` — entirely in the type domain, so re-specializing on a new
conditioning pattern is ordinary Julia dispatch, not a runtime branch.
"""
struct Model{F,TArgs<:NamedTuple,TCond<:NamedTuple} <: AbstractPPL.AbstractProbabilisticProgram
    f::F           # the `_name_eval` function the @model macro generated (compiler.jl)
    args::TArgs
    conditioned::TCond
end

# Used by the @model-generated constructor (compiler.jl): `conditioned`
# starts empty and is only ever populated later via `condition`/`|`.
Model(f, args::NamedTuple) = Model(f, args, NamedTuple())

"""
    condition(model, nt::NamedTuple) -> Model
    model | nt -> Model

Returns a new `Model` with `nt` merged into `conditioned`. Re-specializes the
evaluator for the new conditioning pattern (a new `TCond` type), not per
value, so repeated conditioning with the same keys is cheap after the first
compile.
"""
# `merge(m.conditioned, nt)` produces a NamedTuple whose TYPE encodes exactly
# which names are conditioned — that's the "type domain" mentioned above: two
# calls to `condition` with the same set of names (even different values)
# produce the same `TCond` type, so Julia only needs to specialize/compile
# the evaluator once per distinct *set of conditioned names*, not once per
# distinct value.
condition(m::Model, nt::NamedTuple) = Model(m.f, m.args, merge(m.conditioned, nt))
Base.:|(m::Model, nt::NamedTuple) = condition(m, nt)

"""
    decondition(model, names::Symbol...) -> Model

Removes `names` from `conditioned`, reverting those sites to assumed.
"""
function decondition(m::Model, names::Symbol...)
    # `Base.structdiff` builds the NamedTuple of fields in `m.conditioned`
    # EXCLUDING `names` — i.e. the inverse operation of `merge`.
    kept = Base.structdiff(m.conditioned, NamedTuple{names})
    return Model(m.f, m.args, kept)
end

"""
    evaluate(model, mode::AbstractEvalMode, acc::Accum) -> (retval, acc)

Runs the model's evaluator under `mode`, threading `acc` through every tilde
site. This is the single entry point used by all four modes (`TraceMode`,
`EvalMode`, `PriorMode`, `FixedMode`).
"""
@inline function evaluate(m::Model, mode::AbstractEvalMode, acc::Accum)
    # `m.f` is `_name_eval` from compiler.jl, whose fixed calling convention
    # is `(mode, acc, <the model's own positional args>...)`. `m.conditioned`
    # itself isn't passed here — the generated body reads conditioning off
    # `mode.conditioned` (via `getcond`), so it's the CALLER's responsibility
    # to construct `mode` with `mode.conditioned === m.conditioned` (done at
    # each mode-construction site: `build_layout`, `logdensity.jl`'s
    # `_logdensity_call`, etc) — `evaluate` itself doesn't enforce this.
    return m.f(mode, acc, m.args...)
end
