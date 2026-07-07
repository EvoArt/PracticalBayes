# Driver: benchmarks EVERY model in bench/corpus/{posteriordb,tutorials}/,
# spawning a FRESH `julia` SUBPROCESS per model via corpus_bench_worker.jl.
#
# Subprocess-per-model, not one shared in-process loop, is deliberate: a
# single model can crash the ENTIRE process outright (confirmed: Enzyme has
# produced an OS-level process death — not a catchable Julia exception — on
# at least one model in earlier runs of this corpus). Running each model in
# its own subprocess means one crash only loses that one model's results;
# every other model still gets benchmarked and its results are already
# safely on disk (each worker call appends to
# bench/results/history_corpus.jsonl itself, immediately after finishing).
# The cost is per-model process-startup/precompilation overhead, which is
# fine here since this is a background/offline benchmark run, not something
# run on every commit.
#
# Run with every optional AD backend available (Mooncake/Enzyme; ReverseDiff
# dropped per user request — redundant with Mooncake) using the dedicated
# `bench/bench_env` environment (Pkg.develop(path=".") + every AD backend as
# a real dep — see that directory's Project.toml):
#   julia --project=bench/bench_env bench/corpus_bench.jl

using Dates: now

const HERE = @__DIR__
const WORKER = joinpath(HERE, "corpus_bench_worker.jl")

const CORPUS_FILES = [
    ("posteriordb", joinpath(HERE, "corpus", "posteriordb", "run_smoke_tests.jl")),
    ("posteriordb", joinpath(HERE, "corpus", "posteriordb", "run_smoke_tests_batch2.jl")),
    ("posteriordb", joinpath(HERE, "corpus", "posteriordb", "run_smoke_tests_batch3.jl")),
    ("posteriordb", joinpath(HERE, "corpus", "posteriordb", "run_smoke_tests_batch4.jl")),
    ("posteriordb", joinpath(HERE, "corpus", "posteriordb", "run_smoke_tests_batch5.jl")),
    ("tutorials", joinpath(HERE, "corpus", "tutorials", "run_smoke_tests.jl")),
    ("tutorials", joinpath(HERE, "corpus", "tutorials", "run_smoke_tests_batch2.jl")),
]

# Discovering each file's model NAMES (not building the models themselves —
# that happens inside the worker subprocess) needs the same
# `_load_corpus_file`/`_normalize_models` logic the worker uses. Loaded here
# too (rather than shared) since this is a small, self-contained piece and
# keeping the driver a single standalone file (no relative `include` of the
# worker's internals) is simpler than factoring out a third shared file for
# ~15 lines of logic.
function _normalize_models(raw_models)
    out = String[]
    for entry in raw_models
        push!(out, entry[1])  # model name is always the first element, 3- or 4-tuple alike
    end
    return out
end

function _model_names(path)
    mod = Module(Symbol("Discover_", replace(basename(path), r"[^A-Za-z0-9]" => "_")))
    Core.eval(mod, :(eval(x) = Core.eval($mod, x)))
    Core.eval(mod, :(include(x) = Base.include($mod, x)))
    Base.include(mod, path)
    return _normalize_models(Base.eval(mod, :MODELS))
end

function run_corpus_bench(; julia_cmd=Base.julia_cmd())
    n_ok, n_fail = 0, 0
    for (corpus, path) in CORPUS_FILES
        println("\n", "="^100, "\nDiscovering models in: ", path, "\n", "="^100)
        names = try
            _model_names(path)
        catch e
            println("FAILED TO DISCOVER MODELS in ", path, ": ", sprint(showerror, e)[1:min(end, 300)])
            continue
        end
        for name in names
            println("  benchmarking ", corpus, "/", name, " (subprocess) ...")
            cmd = `$julia_cmd --project=$(Base.active_project()) $WORKER $path $name $corpus`
            proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=false)
            wait(proc)
            if proc.exitcode == 0
                n_ok += 1
            else
                n_fail += 1
                println("  ", corpus, "/", name, ": subprocess exited with code ", proc.exitcode, " (result for this model is missing, everything else is unaffected)")
            end
        end
    end
    println("\nDone: ", n_ok, " models completed, ", n_fail, " models failed/crashed (see above).")
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_corpus_bench()
