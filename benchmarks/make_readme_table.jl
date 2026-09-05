import Pkg
Pkg.activate(@__DIR__)
# Conditional for the same reason as sweep.jl: `Pkg.instantiate()` at startup
# reproducibly crashes Julia 1.12.7 here, inside libjulia-codegen/LLVM -- a bare
# `RaiseException` with nothing useful on stdout, before a line of this script
# runs. It is only needed when the environment does not exist yet, so skip it
# when it does. Set PB_SWEEP_INSTANTIATE=1 to force it on a fresh checkout.
if get(ENV, "PB_SWEEP_INSTANTIATE", "0") == "1" ||
   !haskey(Pkg.project().dependencies, "JSON3")
    Pkg.instantiate()
end

import JSON3

const README_PATH = normpath(joinpath(@__DIR__, "..", "README.md"))
const SWEEP_PATH = joinpath(@__DIR__, "results", "sweep.json")

const MARKER_START = "<!-- BENCH:START -->"
const MARKER_END = "<!-- BENCH:END -->"

function ratio(pb_ns, tu_ns)
    if !isfinite(pb_ns) || !isfinite(tu_ns) || tu_ns <= 0
        return "n/a"
    end
    return string(round(pb_ns / tu_ns; digits=3))
end

function build_lookup(rows)
    bykey = Dict{Tuple{String,String,Int,Int},Any}()
    for row in rows
        key = (String(row["precision"]), String(row["likelihood"]), Int(row["N"]), Int(row["NPARAMS"]))
        bykey[key] = row
    end
    return bykey
end

function backend_ratios(row, backend::String)
    cell = row[backend]
    # sweep.jl writes JSON null (read back here as `nothing`) for a backend
    # that failed on this cell (e.g. Mooncake on some Float32+bernoulli_logit
    # combos) — map to NaN so the existing isfinite check in ratio() below
    # handles it the same way as any other non-finite value.
    pb_ns = cell["pb_ns"] === nothing ? NaN : Float64(cell["pb_ns"])
    tu_ns = cell["turing_ns"] === nothing ? NaN : Float64(cell["turing_ns"])
    return ratio(pb_ns, tu_ns)
end

# Piste has no Turing counterpart (Turing cannot use it), so a PB/Turing ratio
# is meaningless. What IS meaningful is how it compares to the fastest backend
# PracticalBayes could otherwise use on the same model -- which is also the
# question a user picking a backend actually has.
function make_piste_table(lookup, backend::String, nvals::Vector{Int}, kvals::Vector{Int},
                          likelihoods::Vector{String}, precisions::Vector{String})
    io = IOBuffer()
    println(io, "| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |")
    println(io, "|---|---:|---:|---:|---:|---:|")
    corners = [(first(nvals), first(kvals)), (first(nvals), last(kvals)),
               (last(nvals), first(kvals)), (last(nvals), last(kvals))]
    for lik in likelihoods, prec in precisions
        cells = String[]
        for (n, k) in corners
            row = get(lookup, (prec, lik, n, k), nothing)
            if row === nothing
                push!(cells, "n/a"); continue
            end
            pb = row[backend]["pb_ns"]
            pb = pb === nothing ? NaN : Float64(pb)
            # the best of PB's Turing-comparable backends on this same cell
            best = Inf
            for b in ("forwarddiff", "mooncake", "enzyme")
                v = row[b]["pb_ns"]
                v === nothing && continue
                vv = Float64(v)
                isfinite(vv) && vv < best && (best = vv)
            end
            push!(cells, (isfinite(pb) && isfinite(best) && best > 0) ?
                          string(round(pb / best; digits=3)) : "n/a")
        end
        println(io, "| `", lik, "` | `", prec, "` | ", join(cells, " | "), " |")
    end
    return String(take!(io))
end

function make_backend_table(lookup, backend::String, nvals::Vector{Int}, kvals::Vector{Int}, likelihoods::Vector{String}, precisions::Vector{String})
    n_small, n_large = first(nvals), last(nvals)
    k_small, k_large = first(kvals), last(kvals)

    lines = String[]
    push!(lines, "### $(uppercasefirst(backend))")
    push!(lines, "")
    push!(lines, "| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |")
    push!(lines, "|---|---:|---:|---:|---:|---:|")

    for lik in likelihoods, prec in precisions
        r1 = get(lookup, (prec, lik, n_small, k_small), nothing)
        r2 = get(lookup, (prec, lik, n_small, k_large), nothing)
        r3 = get(lookup, (prec, lik, n_large, k_small), nothing)
        r4 = get(lookup, (prec, lik, n_large, k_large), nothing)

        c1 = r1 === nothing ? "n/a" : backend_ratios(r1, backend)
        c2 = r2 === nothing ? "n/a" : backend_ratios(r2, backend)
        c3 = r3 === nothing ? "n/a" : backend_ratios(r3, backend)
        c4 = r4 === nothing ? "n/a" : backend_ratios(r4, backend)

        push!(lines, "| `$(lik)` | `$(prec)` | $(c1) | $(c2) | $(c3) | $(c4) |")
    end

    push!(lines, "")
    return join(lines, "\n")
end

function update_readme_table(new_block::String)
    text = read(README_PATH, String)
    i1 = findfirst(MARKER_START, text)
    i2 = findfirst(MARKER_END, text)
    i1 === nothing && error("Missing marker $(MARKER_START) in README")
    i2 === nothing && error("Missing marker $(MARKER_END) in README")
    i1start = first(i1)
    i1end = last(i1)
    i2start = first(i2)

    replacement = string(MARKER_START, "\n", new_block, "\n", MARKER_END)
    updated = text[1:i1start-1] * replacement * text[i2start + length(MARKER_END):end]
    write(README_PATH, updated)
end

function main()
    payload = JSON3.read(read(SWEEP_PATH, String))
    meta = payload["meta"]
    rows = payload["rows"]

    nvals = [Int(x) for x in meta["n_values"]]
    kvals = [Int(x) for x in meta["nparams_values"]]
    likelihoods = [String(x) for x in meta["likelihoods"]]
    precisions = [String(x) for x in meta["precisions"]]

    lookup = build_lookup(rows)

    legend = "Ratios are median gradient time `PracticalBayes / Turing` (`< 1` means PracticalBayes faster)." *
             " Corner cells come from the 5x5 sweep grid: N in $(nvals), NPARAMS in $(kvals).\n"

    block = join([
        legend,
        make_backend_table(lookup, "forwarddiff", nvals, kvals, likelihoods, precisions),
        make_backend_table(lookup, "mooncake", nvals, kvals, likelihoods, precisions),
        make_backend_table(lookup, "enzyme", nvals, kvals, likelihoods, precisions),
        "
### Piste (PracticalBayes' own AD)

" *
        "[Piste](https://github.com/EvoArt/Piste.jl) is the only backend here that survives " *
        "`juliac --trim`, i.e. the only one that lets a model compile to a standalone " *
        "binary. Turing cannot use it, so there is no PB/Turing ratio to give; these ratios " *
        "are instead `Piste / fastest of ForwardDiff, Mooncake, Enzyme` on the same cell, " *
        "with `< 1` meaning Piste is faster.

" *
        "Read the two tables together. **Forward mode** wins at small problems (roughly 20x " *
        "faster at N=50, P=2, where the other backends' per-call setup dominates) and loses " *
        "badly at large P, because its cost is O(P) by construction -- the 100x+ figures in " *
        "the bottom-right are that, not a defect. **Reverse mode** is the one to use on " *
        "parameter-heavy models; it costs one sweep regardless of P.

" *
        "#### Forward mode

" *
        make_piste_table(lookup, "piste_fwd", nvals, kvals, likelihoods, precisions),
        "
#### Reverse mode

" *
        make_piste_table(lookup, "piste_rev", nvals, kvals, likelihoods, precisions),
    ], "\n")

    update_readme_table(block)
    println("Updated README benchmark tables from ", SWEEP_PATH)
end

main()
