import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

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
    pb_ns = Float64(cell["pb_ns"])
    tu_ns = Float64(cell["turing_ns"])
    return ratio(pb_ns, tu_ns)
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
    ], "\n")

    update_readme_table(block)
    println("Updated README benchmark tables from ", SWEEP_PATH)
end

main()
