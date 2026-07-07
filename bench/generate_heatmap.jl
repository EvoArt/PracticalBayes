# Generates bench/heatmap.html: a self-contained interactive heatmap of
# PracticalBayes/Turing speed ratios across every corpus model (rows) ×
# comparison (columns: logdensity, gradient per backend, NUTS) × precision
# pairing (PB-Float64-vs-Turing, PB-Float32-vs-Turing). Reads the same
# bench/results/history_corpus*.jsonl files as generate_report.jl (reuses
# its hand-rolled JSONL parser verbatim — see that file for why it's
# hand-rolled rather than a JSON dependency).
#
# Run after bench/corpus_bench.jl / corpus_bench_turing.jl:
#   julia --project=. bench/generate_heatmap.jl

const HERE = @__DIR__
const PB_CORPUS_HISTORY = joinpath(HERE, "results", "history_corpus.jsonl")
const TURING_CORPUS_HISTORY = joinpath(HERE, "results", "history_corpus_turing.jsonl")
const OUT_HTML = joinpath(HERE, "heatmap.html")

# ===========================================================================
# JSONL parsing — verbatim copy of generate_report.jl's parser (kept as a
# separate copy rather than a shared include so each generator script stays
# independently runnable).
# ===========================================================================

function _parse_json_line(line::AbstractString)
    s = strip(line)
    record = Dict{String,Any}()
    i, n = 1, lastindex(s)
    @assert s[1] == '{' && s[n] == '}'
    i = nextind(s, 1)
    while i <= n && s[i] != '}'
        @assert s[i] == '"'
        key_start = nextind(s, i)
        key_end = findnext('"', s, key_start) - 1
        key = s[key_start:key_end]
        i = nextind(s, key_end + 1)
        @assert s[i] == ':'
        i = nextind(s, i)
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
# Build the ratio matrix: rows = models, columns = one per
# (layer, backend) combination that appears in the PB data. Two ratio
# variants per cell: PB-Float64/Turing and PB-Float32/Turing (both < 1
# means PracticalBayes is faster).
# ===========================================================================

pb = load_jsonl(PB_CORPUS_HISTORY)
turing = load_jsonl(TURING_CORPUS_HISTORY)

corpus_key(r) = (r["corpus"], r["model"], r["layer"], r["precision"], r["backend"])
function latest_by_key(records)
    out = Dict{Tuple,Dict{String,Any}}()
    for r in records
        out[corpus_key(r)] = r
    end
    return out
end

pb_latest = latest_by_key(pb)
turing_latest = latest_by_key(turing)

# Columns: every (layer, backend) pair present in the PB data, ordered
# logdensity, then gradient (ForwardDiff, Mooncake, Enzyme), then nuts.
_layer_order = Dict("logdensity" => 0, "gradient" => 1, "nuts" => 2)
_backend_order = Dict("none" => 0, "ForwardDiff" => 1, "Mooncake" => 2, "Enzyme" => 3, "ReverseDiff" => 4)
columns = sort(
    unique((k[3], k[5]) for k in keys(pb_latest));
    by=c -> (_layer_order[c[1]], get(_backend_order, c[2], 99)),
)
col_label(layer, backend) = backend == "none" ? layer : "$layer\n$backend"

# Rows: every distinct model that has a Turing counterpart for at least one
# column (models with no Turing data at all wouldn't have any ratio to
# show) — but include EVERY PB model, marking missing cells rather than
# dropping rows, so the heatmap also communicates "we don't have a Turing
# comparison for this model" as a genuine (blank) state, not a silent gap.
models = sort(unique((k[1], k[2]) for k in keys(pb_latest)))

# `cells[(corpus,model)][(layer,backend)] = (ratio64, ratio32)` — `missing`
# where no comparison is available.
function ratio_pair(corpus, model, layer, backend)
    k64 = (corpus, model, layer, "Float64", backend)
    k32 = (corpus, model, layer, "Float32", backend)
    t = get(turing_latest, k64, nothing)  # Turing always benchmarked at "Float64" label
    r64 = get(pb_latest, k64, nothing)
    r32 = get(pb_latest, k32, nothing)
    ratio64 = (t !== nothing && r64 !== nothing && t["median_s"] > 0) ? r64["median_s"] / t["median_s"] : missing
    ratio32 = (t !== nothing && r32 !== nothing && t["median_s"] > 0) ? r32["median_s"] / t["median_s"] : missing
    return ratio64, ratio32
end

# ===========================================================================
# Emit JSON data (hand-rolled, matching this project's existing convention)
# directly into the HTML page.
# ===========================================================================

_num(x) = x === missing ? "null" : string(round(x; digits=3))

io = IOBuffer()
println(io, "{")
println(io, "  \"columns\": [", join(("\"" * col_label(l, b) * "\"" for (l, b) in columns), ", "), "],")
println(io, "  \"rows\": [")
for (i, (corpus, model)) in enumerate(models)
    cells64 = String[]
    cells32 = String[]
    for (layer, backend) in columns
        r64, r32 = ratio_pair(corpus, model, layer, backend)
        push!(cells64, _num(r64))
        push!(cells32, _num(r32))
    end
    comma = i == length(models) ? "" : ","
    println(
        io,
        "    {\"model\": \"", corpus, "/", model, "\", \"f64\": [", join(cells64, ", "), "], \"f32\": [",
        join(cells32, ", "), "]}", comma,
    )
end
println(io, "  ]")
println(io, "}")
data_json = String(take!(io))

# ===========================================================================
# HTML page: a static (no server, no external CDN) heatmap. Two stacked
# grids (Float64 and Float32), diverging color scale centered on 1.0×
# (PB == Turing), log-scaled so a 10× slowdown and a 10× speedup get equal
# visual weight. Each cell shows the ratio to one decimal place; a
# click/toggle switches which precision's ratio is showing full-size (both
# are always in the DOM, one just dims) so the two can be compared without
# losing your place in the row/column layout.
# ===========================================================================

html_template = raw"""
<title>PracticalBayes vs Turing — Speed Ratio Heatmap</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b6b6b; --grid-line: #e2e2e2;
    --bad: 0, 68%; --good: 152, 55%; --neutral-l: 96%;
    --accent: #2f6f4f;
  }
  :root[data-theme="dark"] {
    --bg: #14161a; --fg: #e8e8e8; --muted: #9a9a9a; --grid-line: #2a2d33;
    --neutral-l: 18%;
    --accent: #6fcf9a;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14161a; --fg: #e8e8e8; --muted: #9a9a9a; --grid-line: #2a2d33;
      --neutral-l: 18%;
      --accent: #6fcf9a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem clamp(1rem, 4vw, 3rem) 4rem; background: var(--bg); color: var(--fg);
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  h1 { font-size: 1.4rem; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 0.2rem; }
  .sub { color: var(--muted); font-size: 0.9rem; margin: 0 0 1.6rem; max-width: 62ch; line-height: 1.5; }
  .legend { display: flex; align-items: center; gap: 0.6rem; font-size: 0.78rem; color: var(--muted); margin-bottom: 1.2rem; }
  .legend-bar { width: 180px; height: 10px; border-radius: 3px; background: linear-gradient(90deg,
    hsl(var(--bad), 55%), hsl(var(--bad), 80%), hsl(0,0%,var(--neutral-l)), hsl(var(--good), 80%), hsl(var(--good), 45%)); }
  .panel { margin-bottom: 3rem; }
  .panel h2 { font-size: 1rem; font-weight: 600; margin: 0 0 0.15rem; }
  .panel .note { color: var(--muted); font-size: 0.82rem; margin: 0 0 0.9rem; }
  .grid-wrap { overflow-x: auto; border: 1px solid var(--grid-line); border-radius: 8px; }
  table { border-collapse: collapse; font-size: 0.78rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
  th, td { padding: 0; }
  th { font-weight: 600; font-size: 0.72rem; color: var(--muted); text-align: center; padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--grid-line); position: sticky; top: 0; background: var(--bg); white-space: pre-line; line-height: 1.2; }
  th.rowhead-col { text-align: left; position: sticky; left: 0; z-index: 2; }
  td.rowhead { position: sticky; left: 0; background: var(--bg); font-family: ui-monospace, "JetBrains Mono", Consolas, monospace; font-size: 0.76rem; padding: 0.35rem 0.9rem 0.35rem 0.6rem; border-bottom: 1px solid var(--grid-line); border-right: 1px solid var(--grid-line); white-space: nowrap; }
  td.cell { text-align: center; font-family: ui-monospace, "JetBrains Mono", Consolas, monospace; padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--grid-line); min-width: 3.6rem; }
  td.cell.empty { color: var(--muted); opacity: 0.35; }
  tr:hover td.rowhead, tr:hover td.cell:not(.empty) { filter: brightness(1.08); }
  footer { color: var(--muted); font-size: 0.78rem; margin-top: 2rem; }
</style>

<h1>PracticalBayes vs Turing — Speed Ratio Heatmap</h1>
<p class="sub">Ratio = PracticalBayes median / Turing median, per model (rows) and layer/backend (columns). Below 1.0 (green) means PracticalBayes is faster; above 1.0 (red) means Turing is faster. Log-scaled color so a 10&times; win and a 10&times; loss get equal visual weight. Blank cells: no comparable Turing result for that model/layer/backend.</p>
<div class="legend">
  <span>0.1&times; (PB 10&times; faster)</span>
  <div class="legend-bar"></div>
  <span>10&times; (Turing 10&times; faster)</span>
</div>

<div class="panel">
  <h2>PracticalBayes Float64</h2>
  <p class="note">PB running in Float64 vs Turing (always Float64).</p>
  <div class="grid-wrap"><table id="grid-f64"></table></div>
</div>

<div class="panel">
  <h2>PracticalBayes Float32</h2>
  <p class="note">PB running in Float32 (parameter vector only — see bench/corpus_bench_worker.jl) vs Turing (always Float64). This is the column to check for a Float32 crossover.</p>
  <div class="grid-wrap"><table id="grid-f32"></table></div>
</div>

<footer>Generated by <code>bench/generate_heatmap.jl</code> from <code>bench/results/history_corpus*.jsonl</code>. Regenerate after any corpus benchmark run.</footer>

<script>
const DATA = __DATA_JSON__;

function colorFor(ratio) {
  if (ratio === null || ratio === undefined) return null;
  // log-scale, clamp at +-1 decade (10x) for color saturation purposes
  const logr = Math.log10(Math.max(0.05, Math.min(20, ratio)));
  const t = Math.max(-1, Math.min(1, logr));  // -1 = PB 10x faster, +1 = Turing 10x faster
  const neutralL = getComputedStyle(document.documentElement).getPropertyValue('--neutral-l').trim();
  const L = parseFloat(neutralL);
  if (t < 0) {
    // green side
    const k = -t; // 0..1
    return `hsl(152, ${Math.round(35 + 45*k)}%, ${Math.round(L + (88-L)*(1-k))}%)`;
  } else if (t > 0) {
    const k = t;
    return `hsl(0, ${Math.round(35 + 45*k)}%, ${Math.round(L + (88-L)*(1-k))}%)`;
  }
  return `hsl(0, 0%, ${Math.round(L)}%)`;
}

function buildGrid(tableId, key) {
  const table = document.getElementById(tableId);
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  const corner = document.createElement('th');
  corner.className = 'rowhead-col';
  corner.textContent = 'Model';
  headRow.appendChild(corner);
  for (const col of DATA.columns) {
    const th = document.createElement('th');
    th.textContent = col;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (const row of DATA.rows) {
    const tr = document.createElement('tr');
    const rh = document.createElement('td');
    rh.className = 'rowhead';
    rh.textContent = row.model;
    tr.appendChild(rh);
    const vals = row[key];
    for (const v of vals) {
      const td = document.createElement('td');
      if (v === null) {
        td.className = 'cell empty';
        td.textContent = '—';
      } else {
        td.className = 'cell';
        td.style.background = colorFor(v);
        td.style.color = '#111';
        td.textContent = v.toFixed(1) + '×';
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

buildGrid('grid-f64', 'f64');
buildGrid('grid-f32', 'f32');
</script>
"""

html = replace(html_template, "__DATA_JSON__" => data_json)
write(OUT_HTML, html)
println("Wrote heatmap to: ", OUT_HTML)

