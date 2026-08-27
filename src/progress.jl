using Logging: Logging
using ProgressLogging: ProgressLogging
using LoggingExtras: LoggingExtras
using TerminalLoggers: TerminalLoggers
using ConsoleProgressMonitor: ConsoleProgressMonitor
using UUIDs: uuid4

# ===========================================================================
# Progress reporting for the `sample` loops.
#
# Both `sample` methods in sample.jl run their OWN iteration loop rather than
# AbstractMCMC's `mcmcsample`, so they get none of AbstractMCMC's progress
# handling for free — a `progress=true` passed by a user landed in `kwargs...`
# and was silently ignored. This file supplies the equivalent.
#
# The mechanism is ProgressLogging: `@withprogress`/`@logprogress` emit log
# RECORDS at a special `ProgressLevel`, which render as a live bar only if the
# active logger understands them (VSCode's and Pluto's do; the plain REPL's
# `ConsoleLogger` does not, and would either drop them or spam text). So, like
# AbstractMCMC, we install a `TerminalLogger` fallback for the duration of the
# run when the current logger can't handle progress levels, and tee everything
# else through to the user's own logger untouched.
# ===========================================================================

"""
    default_progress()

Whether [`sample`](@ref) shows a progress meter when `progress` is not passed.

`true` at an interactive REPL/IDE session, `false` otherwise — so scripts, CI
and test suites (whose output is usually redirected to a file, where a live bar
is just noise) stay quiet without having to pass `progress=false` everywhere.
Honours the `CI` environment variable, which the major CI providers set.
"""
default_progress() = isinteractive() && !haskey(ENV, "CI")

# Whether `logger` will actually render progress records, rather than drop them.
# `ProgressLevel` sits BELOW `Debug`, so a logger that hasn't opted in reports a
# `min_enabled_level` above it and the records vanish.
_has_progress_level(logger) =
    Logging.min_enabled_level(logger) <= ProgressLogging.ProgressLevel

# The fallback progress renderer, used only when the active logger can't render
# progress itself. `TerminalLogger` draws the bar in place on a normal terminal;
# IJulia can't handle its escape sequences, so notebooks get the
# ConsoleProgressMonitor renderer instead (same choice AbstractMCMC makes).
function _progress_logger()
    if isdefined(Main, :IJulia) && Main.IJulia.inited
        return ConsoleProgressMonitor.ProgressLogger()
    else
        return TerminalLoggers.TerminalLogger()
    end
end

# Run `f` with a logger that renders OUR progress records, while every other log
# record (including progress records from other modules) still reaches the
# user's own logger. The two `EarlyFilteredLogger`s partition the stream on
# exactly that condition, so nothing is duplicated and nothing is swallowed.
function _with_progress_logger(f)
    logger = Logging.current_logger()
    _has_progress_level(logger) && return f()

    # Bound once here: `@__MODULE__` inside the `do` blocks would parse as a
    # macro consuming the rest of the expression.
    mod = @__MODULE__
    ours = LoggingExtras.EarlyFilteredLogger(_progress_logger()) do log
        log._module === mod && log.level == ProgressLogging.ProgressLevel
    end
    theirs = LoggingExtras.EarlyFilteredLogger(logger) do log
        log._module !== mod || log.level != ProgressLogging.ProgressLevel
    end
    return Logging.with_logger(f, LoggingExtras.TeeLogger(ours, theirs))
end

# ---------------------------------------------------------------------------
# The handle the sample loops actually use.
#
# `ProgressLogging.@withprogress`/`@logprogress` are macros wrapping a lexical
# block, which doesn't fit a loop split across a burn-in phase and a sampling
# phase (and, for Gibbs, an initial step taken before the loop starts). So we
# emit the underlying progress records directly, keyed by a `uuid` — exactly
# what those macros expand to — behind a tiny struct with three calls:
# `_progress_start!`, `_progress_update!`, `_progress_finish!`.
#
# `NoProgress` is the `progress=false` branch: every call is a no-op that
# inlines away, so a disabled meter costs nothing in the loop.
# ---------------------------------------------------------------------------

struct NoProgress end

mutable struct ProgressBar
    name::String
    uuid::Base.UUID
    total::Int
    # Records are only emitted when the completed fraction has moved by at least
    # `min_delta`; a 1e-5-per-sweep bar would otherwise flood the logger with far
    # more records than a terminal can redraw.
    min_delta::Float64
    last::Float64
end

_make_progress(progress::Bool, name::AbstractString, total::Integer) =
    progress ? ProgressBar(String(name), uuid4(), Int(total), 1 / 200, -1.0) : NoProgress()

@inline _progress_start!(::NoProgress) = nothing
@inline _progress_update!(::NoProgress, ::Integer) = nothing
@inline _progress_finish!(::NoProgress) = nothing

function _progress_start!(p::ProgressBar)
    ProgressLogging.@logprogress p.name nothing _id = p.uuid
    p.last = 0.0
    return nothing
end

function _progress_update!(p::ProgressBar, i::Integer)
    frac = p.total <= 0 ? 1.0 : i / p.total
    # Always emit the final update, so the bar reaches 100% before it is closed.
    if frac - p.last >= p.min_delta || i >= p.total
        ProgressLogging.@logprogress p.name frac _id = p.uuid
        p.last = frac
    end
    return nothing
end

function _progress_finish!(p::ProgressBar)
    # A "done" payload is what closes the bar; without it the renderer leaves a
    # stale partial bar on screen for the rest of the session.
    ProgressLogging.@logprogress p.name "done" _id = p.uuid
    return nothing
end

# Wrap a whole sampling run: install the fallback logger if needed, open the
# bar, guarantee it is closed even if the sampler throws. `f` receives the
# progress handle to pass `_progress_update!`.
function _run_with_progress(f, progress::Bool, name::AbstractString, total::Integer)
    p = _make_progress(progress, name, total)
    p isa NoProgress && return f(p)
    return _with_progress_logger() do
        _progress_start!(p)
        try
            f(p)
        finally
            _progress_finish!(p)
        end
    end
end
