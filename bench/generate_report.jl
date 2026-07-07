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
const PB_CORPUS_HISTORY = joinpath(HERE, "results", "history_corpus.jsonl")
const TURING_CORPUS_HISTORY = joinpath(HERE, "results", "history_corpus_turing.jsonl")
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
    # Section: latest snapshot, one table per layer. One ROW per
    # (model, shape, backend) with PB-Float64, PB-Float32, and Turing (always
    # Float64 — Turing/DynamicPPL promotes internally regardless of literal
    # precision, confirmed elsewhere in this repo) as separate COLUMNS, so
    # the Float32-vs-Float64-vs-Turing comparison is readable at a glance in
    # one row instead of scattered across separate rows.
    # -----------------------------------------------------------------
    println(io, "## Latest snapshot")
    println(io)

    layers_present = unique(r["layer"] for r in pb_records)
    for layer in layers_present
        println(io, "### Layer: `", layer, "`")
        println(io)
        println(io, "| Model | Shape | Backend | PB Float64 | PB Float32 | Turing | F32/F64 | PB(F64)/Turing | PB(F32)/Turing |")
        println(io, "|---|---|---|---:|---:|---:|---:|---:|---:|")
        row_keys = sort(unique((k[2], k[3], k[5]) for k in keys(pb_latest) if k[1] == layer))
        for (model, shape, backend) in row_keys
            k64 = (layer, model, shape, "Float64", backend)
            k32 = (layer, model, shape, "Float32", backend)
            r64 = get(pb_latest, k64, nothing)
            r32 = get(pb_latest, k32, nothing)
            t_r = get(turing_latest, k64, nothing)  # Turing side is always run at "Float64" precision label
            pb64 = r64 === nothing ? "—" : _fmt_s(r64["median_s"])
            pb32 = r32 === nothing ? "—" : _fmt_s(r32["median_s"])
            t_med = t_r === nothing ? "—" : _fmt_s(t_r["median_s"])
            f32_vs_f64 = (r64 !== nothing && r32 !== nothing) ? string(round(r32["median_s"] / r64["median_s"]; digits=2), "×") : "—"
            pb64_vs_t = (r64 !== nothing && t_r !== nothing) ? string(round(r64["median_s"] / t_r["median_s"]; digits=2), "×") : "—"
            pb32_vs_t = (r32 !== nothing && t_r !== nothing) ? string(round(r32["median_s"] / t_r["median_s"]; digits=2), "×") : "—"
            println(io, "| ", model, " | ", shape, " | ", backend, " | ", pb64, " | ", pb32, " | ", t_med, " | ", f32_vs_f64, " | ", pb64_vs_t, " | ", pb32_vs_t, " |")
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

end

# ===========================================================================
# Section: model corpus (bench/corpus/{posteriordb,tutorials}/) — every
# ported model, every AD backend, both precisions, three layers. Keyed by
# (corpus, model, layer, precision, backend) — no "shape" field here (each
# corpus model has exactly one data-generation function, unlike bench/suite.jl's
# hand-picked tiny/large shapes).
# ===========================================================================

pb_corpus = load_jsonl(PB_CORPUS_HISTORY)
turing_corpus = load_jsonl(TURING_CORPUS_HISTORY)

corpus_key(r) = (r["corpus"], r["model"], r["layer"], r["precision"], r["backend"])
function latest_by_corpus_key(records)
    out = Dict{Tuple,Dict{String,Any}}()
    for r in records
        out[corpus_key(r)] = r  # last write wins == latest run, same as latest_by_key
    end
    return out
end

if !isempty(pb_corpus)
    pb_corpus_latest = latest_by_corpus_key(pb_corpus)
    turing_corpus_latest = latest_by_corpus_key(turing_corpus)

    println(io, "## Model corpus benchmarks")
    println(io)
    latest_pb_corpus_run = pb_corpus[end]
    println(io, "Latest PracticalBayes corpus run: **", latest_pb_corpus_run["timestamp"], "**, commit `", _fmt_commit(latest_pb_corpus_run["commit"]), "`, ", length(unique(r["model"] for r in pb_corpus)), " distinct models.")
    if !isempty(turing_corpus)
        latest_turing_corpus_run = turing_corpus[end]
        println(io, "  ")
        println(io, "Latest Turing corpus run (representative subset): **", latest_turing_corpus_run["timestamp"], "**, commit `", _fmt_commit(latest_turing_corpus_run["commit"]), "`, ", length(unique(r["model"] for r in turing_corpus)), " distinct models.")
    end
    println(io)
    println(io, "Every model in `bench/corpus/posteriordb/` and `bench/corpus/tutorials/` (see `bench/corpus_bench.jl`), across ForwardDiff/Mooncake/Enzyme (ReverseDiff dropped — redundant with Mooncake), Float64 and Float32, on three layers (logdensity, gradient, a 1000-sample NUTS run). Turing only covers a representative subset (~20 models spanning each structural family — porting all ~55 to Turing as well was judged not worth the effort vs. the coverage gained). Each model runs in its own subprocess (`bench/corpus_bench_worker.jl`), so one model crashing (Enzyme has genuinely done this on certain type combinations) only loses that model's results, not the whole sweep.")
    println(io)

    for layer in ("logdensity", "gradient", "nuts")
        println(io, "### Corpus layer: `", layer, "`")
        println(io)
        println(io, "| Corpus | Model | Backend | PB Float64 | PB Float32 | Turing | F32/F64 | PB(F64)/Turing | PB(F32)/Turing |")
        println(io, "|---|---|---|---:|---:|---:|---:|---:|---:|")
        row_keys = sort(unique((k[1], k[2], k[5]) for k in keys(pb_corpus_latest) if k[3] == layer))
        for (corpus, model, backend) in row_keys
            k64 = (corpus, model, layer, "Float64", backend)
            k32 = (corpus, model, layer, "Float32", backend)
            r64 = get(pb_corpus_latest, k64, nothing)
            r32 = get(pb_corpus_latest, k32, nothing)
            t_r = get(turing_corpus_latest, k64, nothing)
            pb64 = r64 === nothing ? "—" : _fmt_s(r64["median_s"])
            pb32 = r32 === nothing ? "—" : _fmt_s(r32["median_s"])
            t_med = t_r === nothing ? "—" : _fmt_s(t_r["median_s"])
            f32_vs_f64 = (r64 !== nothing && r32 !== nothing && r64["median_s"] > 0) ? string(round(r32["median_s"] / r64["median_s"]; digits=2), "×") : "—"
            pb64_vs_t = (r64 !== nothing && t_r !== nothing && t_r["median_s"] > 0) ? string(round(r64["median_s"] / t_r["median_s"]; digits=2), "×") : "—"
            pb32_vs_t = (r32 !== nothing && t_r !== nothing && t_r["median_s"] > 0) ? string(round(r32["median_s"] / t_r["median_s"]; digits=2), "×") : "—"
            println(io, "| ", corpus, " | ", model, " | ", backend, " | ", pb64, " | ", pb32, " | ", t_med, " | ", f32_vs_f64, " | ", pb64_vs_t, " | ", pb32_vs_t, " |")
        end
        println(io)
    end
else
    println(io, "## Model corpus benchmarks")
    println(io)
    println(io, "No corpus benchmark history found. Run `julia --project=<env-with-AD-backends> bench/corpus_bench.jl` (see that file's header for the `bench/bench_env` environment used to get every AD backend) to populate `bench/results/history_corpus.jsonl`.")
    println(io)
end

println(io, "## Reproducing")
println(io)
println(io, "```")
println(io, "julia --project=. bench/suite.jl                                    # PracticalBayes side, appends to bench/results/history.jsonl")
println(io, "julia --project=test/comparison_env test/comparison_env/generate_turing_reference.jl  # Turing side, appends to bench/results/history_turing.jsonl")
println(io, "julia --project=bench/bench_env bench/corpus_bench.jl               # PracticalBayes corpus, appends to bench/results/history_corpus.jsonl")
println(io, "julia --project=test/comparison_env test/comparison_env/corpus_bench_turing.jl  # Turing corpus subset, appends to bench/results/history_corpus_turing.jsonl")
println(io, "julia --project=. bench/generate_report.jl                          # regenerate this file")
println(io, "quarto render bench/report.qmd                                      # render to HTML")
println(io, "```")

write(OUT_QMD, take!(io))
println("Wrote report to: ", OUT_QMD)
