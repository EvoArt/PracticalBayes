# Reads bench/results/history.jsonl (PracticalBayes) and
# bench/results/history_turing.jsonl (Turing, generated separately by
# test/comparison_env/generate_turing_reference.jl — see that file for why
# the two packages can never run in the same process) and renders
# bench/report.qmd: plain Markdown, no embedded code execution, so
# `quarto render bench/report.qmd` needs nothing beyond the Quarto CLI
# itself (no Julia/Python/R notebook engine).
#
# Run after any benchmark run that calls `write_history!()`:
#   julia --project=. bench/generate_report.jl
#   quarto render bench/report.qmd
#
# Deliberately hand-rolled JSON parsing (mirrors the hand-rolled JSON
# *writing* in bench/suite.jl / generate_turing_reference.jl) rather than
# adding a JSON dependency — every line is one flat object, no nesting,
# only strings/numbers, so a tiny regex-based parser is enough.

using Dates: now

const HERE = @__DIR__
const PB_HISTORY = joinpath(HERE, "results", "history.jsonl")
const TURING_HISTORY = joinpath(HERE, "results", "history_turing.jsonl")
const OUT_QMD = joinpath(HERE, "report.qmd")

# ===========================================================================
# Minimal JSONL parsing — each line is `{"k1":v1,"k2":v2,...}` with only
# string values (quoted) or numeric/null values (bare), no nesting. A simple
# left-to-right scan (rather than splitting on `","`, which breaks as soon as
# a bare numeric field follows a string field with no closing quote to split
# on) — good enough for the flat records this suite ever writes, not a
# general JSON parser.
# ===========================================================================

function _parse_json_line(line::AbstractString)
    s = strip(line)
    record = Dict{String,Any}()
    i, n = 1, lastindex(s)
    @assert s[1] == '{' && s[n] == '}'
    i = nextind(s, 1)
    while i <= n && s[i] != '}'
        # key: always a quoted string
        @assert s[i] == '"'
        key_start = nextind(s, i)
        key_end = findnext('"', s, key_start) - 1
        key = s[key_start:key_end]
        i = nextind(s, key_end + 1)  # skip closing quote
        @assert s[i] == ':'
        i = nextind(s, i)
        # value: quoted string, or bare token up to next ',' or '}'
        if s[i] == '"'
            val_start = nextind(s, i)
            val_end = findnext('"', s, val_start) - 1
            value = s[val_start:val_end]
            i = nextind(s, val_end + 1)
        else
            val_start = i
            stop = something(findnext(c -> c == ',' || c == '}', s, i), n + 1)
            raw = s[val_start:stop-1]
            value = raw == "null" ? missing : parse(Float64, raw)
            i = stop
        end
        record[key] = value
        if i <= n && s[i] == ','
            i = nextind(s, i)
        end
    end
    return record
end

function load_jsonl(path)
    isfile(path) || return Dict{String,Any}[]
    return [_parse_json_line(line) for line in eachline(path) if !isempty(strip(line))]
end

# ===========================================================================
# Formatting helpers
# ===========================================================================

_fmt_s(x) = x === missing ? "—" : (x < 1e-3 ? string(round(x * 1e6; digits=2), " µs") : string(round(x * 1e3; digits=3), " ms"))
_fmt_commit(c) = c == "unknown" ? "unknown" : first(c, 8)

key(r) = (r["layer"], r["model"], r["shape"], r["precision"], r["backend"])

function latest_by_key(records)
    out = Dict{Tuple,Dict{String,Any}}()
    for r in records
        k = key(r)
        # records are appended in run order, so the last one wins == latest run
        out[k] = r
    end
    return out
end

# History of a single (layer, model, shape, precision, backend) series across
# commits, in file order (oldest first, since history.jsonl is append-only).
function series_for(records, k)
    return [r for r in records if key(r) == k]
end

# ===========================================================================
# Build the report
# ===========================================================================

pb_records = load_jsonl(PB_HISTORY)
turing_records = load_jsonl(TURING_HISTORY)

pb_latest = latest_by_key(pb_records)
turing_latest = latest_by_key(turing_records)

io = IOBuffer()

println(io, "---")
println(io, "title: \"PracticalBayes vs Turing.jl — Benchmark Report\"")
println(io, "subtitle: \"Auto-generated from bench/results/*.jsonl — do not edit by hand\"")
println(io, "format:")
println(io, "  html:")
println(io, "    toc: true")
println(io, "    theme: cosmo")
println(io, "    embed-resources: true")
println(io, "---")
println(io)

if isempty(pb_records)
    println(io, "No PracticalBayes benchmark history found at `bench/results/history.jsonl`.")
    println(io, "Run `julia --project=. bench/suite.jl` first, then regenerate this report.")
else
    latest_pb_run = pb_records[end]
    println(io, "Latest PracticalBayes run: **", latest_pb_run["timestamp"], "**, commit `", _fmt_commit(latest_pb_run["commit"]), "`.")
    if !isempty(turing_records)
        latest_turing_run = turing_records[end]
        println(io, "  ")
        println(io, "Latest Turing reference: **", latest_turing_run["timestamp"], "**, commit `", _fmt_commit(latest_turing_run["commit"]), "`.")
    end
    println(io)
    println(io, "Timings report **first-call** (includes JIT compilation) separately from **steady-state median** — see `bench/suite.jl` for methodology (hand-rolled fixed-repetition timing, not BenchmarkTools/Chairmarks, so expensive configurations still get a real sample size instead of degenerating to N=1).")
    println(io)

    # -----------------------------------------------------------------
    # Section: latest snapshot, one table per layer, PB vs Turing side by side
    # -----------------------------------------------------------------
    println(io, "## Latest snapshot")
    println(io)

    layers_present = unique(r["layer"] for r in pb_records)
    for layer in layers_present
        println(io, "### Layer: `", layer, "`")
        println(io)
        println(io, "| Model | Shape | Precision | Backend | PB first-call | PB median | Turing first-call | Turing median | PB/Turing (median) |")
        println(io, "|---|---|---|---|---:|---:|---:|---:|---:|")
        pb_keys = sort(collect(k for k in keys(pb_latest) if k[1] == layer))
        for k in pb_keys
            pb_r = pb_latest[k]
            t_r = get(turing_latest, k, nothing)
            model, shape, precision, backend = k[2], k[3], k[4], k[5]
            pb_first = _fmt_s(pb_r["first_call_s"])
            pb_med = _fmt_s(pb_r["median_s"])
            if t_r === nothing
                t_first, t_med, ratio = "—", "—", "—"
            else
                t_first = _fmt_s(t_r["first_call_s"])
                t_med = _fmt_s(t_r["median_s"])
                ratio = string(round(pb_r["median_s"] / t_r["median_s"]; digits=2), "×")
            end
            println(io, "| ", model, " | ", shape, " | ", precision, " | ", backend, " | ", pb_first, " | ", pb_med, " | ", t_first, " | ", t_med, " | ", ratio, " |")
        end
        println(io)
    end

    # -----------------------------------------------------------------
    # Section: regression history — steady-state median per commit, for
    # every series that has more than one recorded run.
    # -----------------------------------------------------------------
    all_keys = unique(key(r) for r in pb_records)
    multi_run_keys = [k for k in all_keys if length(series_for(pb_records, k)) > 1]

    println(io, "## History (regression tracking)")
    println(io)
    if isempty(multi_run_keys)
        println(io, "Only one recorded run so far for every series — history will populate as `bench/suite.jl` is run again over time (each run appends to `bench/results/history.jsonl`, one line per result).")
        println(io)
    else
        println(io, "Series with more than one recorded run (steady-state median, PracticalBayes side):")
        println(io)
        for k in sort(multi_run_keys)
            layer, model, shape, precision, backend = k
            println(io, "### `", layer, "` / ", model, " / ", shape, " / ", precision, " / ", backend)
            println(io)
            println(io, "| Timestamp | Commit | Median | First-call |")
            println(io, "|---|---|---:|---:|")
            for r in series_for(pb_records, k)
                println(io, "| ", r["timestamp"], " | `", _fmt_commit(r["commit"]), "` | ", _fmt_s(r["median_s"]), " | ", _fmt_s(r["first_call_s"]), " |")
            end
            println(io)
        end
    end

    println(io, "## Reproducing")
    println(io)
    println(io, "```")
    println(io, "julia --project=. bench/suite.jl                                    # PracticalBayes side, appends to bench/results/history.jsonl")
    println(io, "julia --project=test/comparison_env test/comparison_env/generate_turing_reference.jl  # Turing side, appends to bench/results/history_turing.jsonl")
    println(io, "julia --project=. bench/generate_report.jl                          # regenerate this file")
    println(io, "quarto render bench/report.qmd                                      # render to HTML")
    println(io, "```")
end

write(OUT_QMD, take!(io))
println("Wrote report to: ", OUT_QMD)
