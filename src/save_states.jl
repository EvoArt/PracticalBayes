# Per-variable control over WHERE each Gibbs variable's per-sweep value goes.
#
# The motivating problem: a Gibbs sweep's transition is the whole constrained
# `values` NamedTuple, and `sample` retains one per sweep. That is exactly right
# for scalar/small-vector parameters (and for conjugate blocks whose whole point
# is that you keep the draws) — but a large latent, e.g. a discrete trajectory
# `X` of shape `n_time × n_individuals`, is often megabytes PER SWEEP. Retaining
# all of them materialises the whole run in RAM: a badger-sized `X` (161×2384
# `Int`) is ~3 MB/sweep, so 5000 sweeps is ~15 GB — enough to exhaust memory and
# crash mid-run, while the scalar parameters that were the actual inference
# target occupy kilobytes.
#
# `save_states` lets the user say, per variable, where its per-sweep value
# should go. The sampler ALWAYS keeps the current value live in
# `GibbsState.values` (so conditioning is unaffected — this is purely about the
# OUTPUT, never the algorithm); the only thing `save_states` changes is whether a
# copy of that value is also retained in the chain and/or streamed to disk.
#
#   :chain / true          keep it in the returned chain (the default for any
#                          name not mentioned — unchanged behaviour)
#   :buffer / false        keep it live for conditioning but DROP it from the
#                          chain; nothing is written anywhere
#   (path::AbstractString, every::Integer)
#                          keep it live, and every `every` sweeps append the
#                          buffered draws to `path` (plus a final flush at the
#                          end for the trailing partial buffer). Dropped from the
#                          chain. The concrete writer is provided by a package
#                          extension (JLD2) — see `write_state_chunk!`.

# ── Per-variable disposition (an isbits-friendly sum type so the sweep loop's
#    dispatch on it is a compile-time branch, never a runtime `Any` check) ──────

abstract type SaveDisposition end

"Retain this variable's per-sweep value in the returned chain (the default)."
struct SaveToChain <: SaveDisposition end

"Keep the value live for conditioning but omit it from the chain and disk."
struct SaveToBuffer <: SaveDisposition end

"""
Keep the value live, and write buffered draws to disk every `every` sweeps.

Each flush is written to its OWN file — never appended to a growing one — so no
single file ever has to hold (or be re-read into) the whole run, and a crash
loses at most the last unflushed `every` sweeps. `path` is a template: the
`iters_x_to_y` range of each flush is spliced in before the extension, so
`"X.jld2"` becomes `"X_iters_1_to_100.jld2"`, `"X_iters_101_to_200.jld2"`, ....
"""
struct SaveToDisk <: SaveDisposition
    path::String
    every::Int
end
SaveToDisk(path::AbstractString, every::Integer) = SaveToDisk(String(path), Int(every))

"""
    chunk_path(template::AbstractString, first_iter, last_iter) -> String

Splice `_iters_<first>_to_<last>` into `template` before its file extension:
`chunk_path("dir/X.jld2", 101, 200) == "dir/X_iters_101_to_200.jld2"`. A template
with no extension just gets the suffix appended.
"""
function chunk_path(template::AbstractString, first_iter::Integer, last_iter::Integer)
    base, ext = splitext(template)
    return string(base, "_iters_", first_iter, "_to_", last_iter, ext)
end

# ── Normalising user input to a `NamedTuple{names, Tuple{Vararg{SaveDisposition}}}` ──

_as_disposition(d::SaveDisposition) = d
function _as_disposition(d::Symbol)
    d === :chain && return SaveToChain()
    d === :buffer && return SaveToBuffer()
    throw(ArgumentError("`save_states` value `:$d` not recognised; use :chain, :buffer, " *
                        "`true`/`false`, or `(path, every)` to stream to disk"))
end
_as_disposition(d::Bool) = d ? SaveToChain() : SaveToBuffer()
function _as_disposition(d::Tuple{AbstractString,Integer})
    d[2] > 0 || throw(ArgumentError("`save_states` disk flush interval must be positive, got $(d[2])"))
    SaveToDisk(d[1], d[2])
end
_as_disposition(d) = throw(ArgumentError(
    "`save_states` value $(repr(d)) not recognised; use :chain, :buffer, `true`/`false`, " *
    "or `(path::AbstractString, every::Integer)` to stream to disk"))

"""
    _normalize_save_states(save_states) -> NamedTuple

Turn whatever the user passed (`nothing`, or a NamedTuple/Dict/pairs of
name => disposition) into a NamedTuple mapping each named variable to a
`SaveDisposition`. Names not mentioned default to `SaveToChain` and are simply
absent here — the sweep loop treats "absent" as chain-retained.
"""
_normalize_save_states(::Nothing) = NamedTuple()
_normalize_save_states(nt::NamedTuple) = map(_as_disposition, nt)
function _normalize_save_states(pairs)
    isempty(pairs) && return NamedTuple()
    NamedTuple(k => _as_disposition(v) for (k, v) in pairs)
end

# Is this name kept in the chain? (Default yes; only :buffer/disk drop it.)
_retained_in_chain(save_states::NamedTuple, name::Symbol) =
    !haskey(save_states, name) || save_states[name] isa SaveToChain

# ── The disk-sink interface (implemented by a package extension) ──────────────
#
# The core package takes NO hard dependency on any file format. The concrete
# writer/reader are `Ref`-held functions a package extension assigns into (in its
# `__init__`), NOT methods the extension adds by dispatch — the same pattern
# `_external_optimize` uses (see optimize.jl): a fallback method and an
# extension's method sharing one signature would be a method-OVERWRITE during
# precompile, which Julia forbids. Assigning into a `Ref` is plain value
# replacement, so it sidesteps that. Loading `using JLD2` activates the bundled
# sink; any other backend can assign these two hooks instead.

"""
    write_state_chunk!(disp::SaveToDisk, name::Symbol, chunk::AbstractVector,
                       first_iter::Integer, last_iter::Integer)

Write `chunk` — the buffered per-sweep values of variable `name` for sweeps
`first_iter:last_iter`, in order — to its OWN file, named via
[`chunk_path`](@ref)`(disp.path, first_iter, last_iter)`. Never appends to a
previous flush's file. Dispatches through `_WRITE_STATE_CHUNK_HOOK`, which a
backend extension (JLD2) fills in; unset, it errors with instructions.
"""
write_state_chunk!(disp::SaveToDisk, name::Symbol, chunk::AbstractVector,
                   first_iter::Integer, last_iter::Integer) =
    _WRITE_STATE_CHUNK_HOOK[](disp, name, chunk, first_iter, last_iter)

const _WRITE_STATE_CHUNK_HOOK = Ref{Any}(
    (disp, name, chunk, first_iter, last_iter) ->
        error("streaming `save_states` variable `$name` to `$(disp.path)` needs a disk backend. " *
              "Load one first — e.g. `using JLD2` — which activates PracticalBayes's JLD2 sink; " *
              "then the `(path, every)` form works. (Or use `:buffer` to drop it from output entirely.)"),
)

"""
    read_states(template::AbstractString) -> Vector

Read back every per-sweep value streamed by
`sample(...; save_states = (name = (template, every),))`, across all its
per-flush files, stitched into one vector in sweep order. `template` is the same
path template passed to `save_states` (e.g. `"X.jld2"`); the reader discovers the
`*_iters_x_to_y.*` files itself and orders them by first iteration.

Note this materialises the whole run in memory — the point of streaming was to
avoid that during sampling, so for a large latent prefer reading one flush file
at a time (each is an ordinary file the backend can open directly). Dispatches
through `_READ_STATES_HOOK`, filled in by the same backend extension that wrote
the files.
"""
read_states(template::AbstractString) = _READ_STATES_HOOK[](template)

const _READ_STATES_HOOK = Ref{Any}(
    template ->
        error("reading streamed states from `$template` needs the matching backend loaded — " *
              "e.g. `using JLD2` for files written by the bundled JLD2 sink."),
)
