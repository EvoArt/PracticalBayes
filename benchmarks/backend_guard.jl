# Backend robustness: never let one AD backend take down a whole sweep.
#
# WHY THIS FILE EXISTS
# --------------------
# A benchmark sweep is a long, expensive, mostly-unattended job (the CPU sweep
# is the best part of an hour; the GPU sweep runs on a cluster queue where a
# crash means re-queueing and waiting again). The AD backends it measures are
# NOT equally reliable, and their failures are not hypothetical:
#
#   * Mooncake genuinely fails on some Float32 + bernoulli_logit cells (see
#     devlog) — a real, reproducible rule failure, not flakiness;
#   * Enzyme's failures often arrive as an LLVM-level error, and on a bad day
#     as a SEGFAULT that takes the whole Julia process with it;
#   * CUDA.jl is non-functional on any machine without a working driver, and a
#     driver/toolkit mismatch (exactly the case on Athena: driver 418.39 caps
#     at CUDA 10.1 while the cluster module is 12.2) makes it fail at
#     `import`/init time, not at call time;
#   * Piste is this package's own AD and is deliberately narrow in scope — it
#     is expected to fail on inputs it never claimed to support.
#
# Each of those needs a DIFFERENT containment strategy, which is the whole
# point of this file: a plain `try`/`catch` around the benchmark call only
# catches the second-cheapest of the four failure modes.

"""
    BackendOutcome

The result of trying one backend on one benchmark cell. Always one of three
things, never an exception escaping into the sweep loop:

  * `ok`      — `time_ns` is a real measurement;
  * `failed`  — the backend threw; `message` says how, `time_ns` is `Inf`;
  * `skipped` — the backend was never attempted (not loaded, or disabled);
                `time_ns` is `Inf`.

`Inf` (rather than `NaN` or a missing key) is the sentinel on both failure
paths ON PURPOSE: every downstream consumer — ratio matrices, "fastest
backend" reductions, the JSON writer — already treats `Inf` as "infinitely
slow, i.e. lost", so a failed backend degrades to "did not win" instead of
poisoning a cell with `NaN` or throwing a `KeyError`.
"""
struct BackendOutcome
    status::Symbol      # :ok | :failed | :skipped
    time_ns::Float64
    message::String
end

BackendOutcome(status::Symbol, time_ns::Real) = BackendOutcome(status, Float64(time_ns), "")

ok(t) = BackendOutcome(:ok, t, "")
failed(msg) = BackendOutcome(:failed, Inf, msg)
skipped(msg) = BackendOutcome(:skipped, Inf, msg)

"""
    guarded(f, label; rethrow_fatal=true) -> BackendOutcome

Run `f()` and turn ANY failure into a `BackendOutcome` rather than an
exception. Returns `ok(f())` when `f` returns a number.

`InterruptException` and `OutOfMemoryError` are re-thrown by default
(`rethrow_fatal`). Swallowing Ctrl-C would make a long sweep impossible to
stop, and swallowing an OOM is actively harmful: the process is already in a
degraded state, so every SUBSEQUENT cell would produce a bogus number rather
than an honest failure. Everything else — a `MethodError` from a backend that
cannot handle the input, an LLVM error from Enzyme, an `InexactError` from a
Float32 edge case — is contained and recorded.
"""
function guarded(f, label; rethrow_fatal::Bool=true)
    try
        return ok(f())
    catch e
        if rethrow_fatal && e isa Union{InterruptException,OutOfMemoryError}
            rethrow()
        end
        # Truncated: an Enzyme or CUDA stacktrace-bearing error message can run
        # to many kilobytes, and the sweep log has one line per cell per
        # backend. The first 200 characters reliably contain the exception type
        # and the salient message, which is what a later reader needs to tell
        # "unsupported input" from "the backend is broken".
        msg = sprint(showerror, e)
        msg = length(msg) > 200 ? msg[1:200] * "…" : msg
        println("  $(label): FAILED — ", msg)
        return failed(msg)
    end
end

"""
    try_import(mod::Symbol) -> Bool

Attempt `import mod` at RUNTIME, returning whether it succeeded.

A top-level `import Enzyme` in a sweep script is a single point of failure for
the entire run: if the package fails to precompile — a genuinely common event
for Enzyme/CUDA on a cluster, where a driver mismatch or a half-written depot
cache surfaces exactly here — the script dies at load, before a single
benchmark has run, and every other backend's numbers are lost along with it.

Importing inside a `try` defers that failure to a per-backend `skipped`
outcome, so a broken Enzyme costs you the Enzyme column and nothing else.
"""
function try_import(mod::Symbol)
    try
        @eval Main import $mod
        return true
    catch e
        msg = sprint(showerror, e)
        println("NOTE: `import $mod` failed — its backends will be reported as skipped.")
        println("      ", length(msg) > 300 ? msg[1:300] * "…" : msg)
        return false
    end
end

"""
    gpu_available() -> Bool

True only when CUDA is installed AND `CUDA.functional()`.

Both halves matter and they fail differently: CUDA not being installed is a
`Base.find_package` miss, while an installed-but-unusable CUDA (no device, or
a driver too old for the toolkit) can THROW from `CUDA.functional()` itself on
a badly mismatched setup rather than politely returning `false` — so even this
check needs its own guard.
"""
function gpu_available()
    isnothing(Base.find_package("CUDA")) && return false
    try
        return Main.CUDA.functional()
    catch
        return false
    end
end
