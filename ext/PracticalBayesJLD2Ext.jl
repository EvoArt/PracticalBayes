module PracticalBayesJLD2Ext

# Loaded automatically (Julia's package-extension mechanism) whenever the user
# has `using JLD2` in their session. Provides the disk backend for
# `sample(...; save_states = (X = ("X.jld2", every),))`: without it, that form
# errors with instructions to load a backend (see save_states.jl).
#
# The two entry points are assigned into PracticalBayes's `Ref`-held hooks in
# `__init__` (NOT added as dispatch methods) — see save_states.jl for why (a
# fallback + extension method on one signature would be a forbidden precompile
# method-overwrite; the same reason `_external_optimize` is a Ref hook).
#
# ONE FILE PER FLUSH. Every flush writes its buffered chunk to its OWN file,
# named `<base>_iters_<first>_to_<last><ext>` (via `chunk_path`), and never
# touches a previous flush's file. This means:
#   * memory stays flat — only the current buffer is ever in RAM, and no file
#     ever has to hold (or be re-read to extend) the whole run;
#   * a crash mid-run loses at most the last unflushed `every` sweeps, and every
#     already-written file is complete and independently readable;
#   * `read_states` finds the per-flush files by their `_iters_x_to_y` names and
#     stitches them back in sweep order.
# Values are stored exactly as produced — a discrete trajectory's chunk is a
# `Vector{Matrix{Int}}`, one entry per sweep.

using PracticalBayes: PracticalBayes, SaveToDisk, chunk_path
using JLD2: JLD2

function _jld2_write_state_chunk!(disp::SaveToDisk, name::Symbol, chunk::AbstractVector,
                                  first_iter::Integer, last_iter::Integer)
    path = chunk_path(disp.path, first_iter, last_iter)
    dir = dirname(path)
    isempty(dir) || isdir(dir) || mkpath(dir)
    # The sweep loop buffers into a `Vector{Any}` (heterogeneous, off the hot
    # path); narrow to the concrete element type here so the on-disk and
    # read-back arrays are `Vector{Matrix{Int}}` etc., not `Vector{Any}`.
    # `[x for x in chunk]` recovers the concrete `eltype` from the values.
    states = [x for x in chunk]
    JLD2.jldopen(path, "w") do f
        f["varname"] = String(name)
        f["first_iter"] = Int(first_iter)
        f["last_iter"] = Int(last_iter)
        f["states"] = states
    end
    nothing
end

# Discover a template's per-flush files (`<base>_iters_<a>_to_<b><ext>`), order
# them by first iteration, and concatenate their `states` in sweep order.
function _jld2_read_states(template::AbstractString)
    base, ext = splitext(template)
    dir = dirname(base)
    dir = isempty(dir) ? "." : dir
    stem = basename(base)
    # Match "<stem>_iters_<first>_to_<last><ext>" and capture <first> for sorting.
    pat = Regex("^" * _regex_escape(stem) * raw"_iters_(\d+)_to_\d+" *
                _regex_escape(ext) * raw"$")
    hits = Tuple{Int,String}[]
    for fn in readdir(dir)
        m = match(pat, fn)
        m === nothing && continue
        push!(hits, (parse(Int, m.captures[1]), joinpath(dir, fn)))
    end
    isempty(hits) && error("no streamed-state files found for template `$template` in `$dir` " *
                           "(looked for `$(stem)_iters_x_to_y$(ext)`)")
    sort!(hits; by = first)

    out = nothing
    for (_, file) in hits
        states = JLD2.jldopen(file, "r") do f
            f["states"]
        end
        if out === nothing
            out = similar(states, 0)
        end
        append!(out, states)
    end
    return out
end

# Escape a literal string for embedding in a Regex (filenames may contain `.`).
_regex_escape(s::AbstractString) = replace(s, r"([.\\^$|?*+()\[\]{}])" => s"\\\1")

function __init__()
    PracticalBayes._WRITE_STATE_CHUNK_HOOK[] = _jld2_write_state_chunk!
    PracticalBayes._READ_STATES_HOOK[] = _jld2_read_states
end

end # module PracticalBayesJLD2Ext
