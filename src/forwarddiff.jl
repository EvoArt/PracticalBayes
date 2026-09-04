# =============================================================================
# PBForwardDiff -- a chunked forward-mode AD owned by this package.
#
# WHY THIS EXISTS: TRIMMABLE GENERIC MODELS
# -----------------------------------------
# `juliac --trim=safe` compiles a PB model to a standalone native binary with no
# Julia install at the target -- a capability Turing structurally cannot match
# (DynamicPPL reaches FunctionWrappers, which is untrimmable). The blocker was
# always AD: every off-the-shelf backend fails under trim, each because its
# derivative code is built at RUNTIME in some form the trimmer cannot see:
#
#   ForwardDiff          builds clean, then MethodErrors on the Dual objective
#   FastDifferentiation  RuntimeGeneratedFunctions (an Expr in a type parameter)
#   Enzyme               Enzyme.Compiler.thunk(...)::Any -- calls the compiler
#   Mooncake             reaches Compiler.InferenceState -- needs the compiler
#   Differ.jl            0 verifier errors, then dies: ContextualInterpreter
#
# The rule those measurements produced: the derivative must exist as ordinary
# compiled Julia, needing nothing from the compiler or an interpreter at any
# point the trimmed binary runs.
#
# ForwardDiff is instructive because duals are NOT what breaks it -- the
# machinery around them is: a `GradientConfig` built at runtime, per-call-site
# `Tag` types that multiply the method table, and dynamically chosen chunk
# sizes. So the dual used here has no tag, no config, and a chunk width fixed
# as a TYPE PARAMETER. Everything about it is known at compile time.
#
# THE ENGINE LIVES IN Piste.jl
# ----------------------------
# The dual-number arithmetic, the distribution rules and the chunked gradient
# driver are all in `Piste` (a groomed ski run: a slope, trimmed). It is a
# standalone, general-purpose package with no dependency on this one -- nothing
# about trimmable forward-mode AD is Bayes-specific, and keeping it separate
# means it can be used, tested and released on its own. This file is only the
# glue that presents it as a PracticalBayes AD backend.
#
# Verified: 0 verifier errors under `--trim=safe`, and the binary reproduces the
# JIT's gradient exactly when run with `JULIA_LOAD_CODEGEN_LIB=0` (a clean
# verifier pass alone proves nothing -- that is exactly how ForwardDiff fails).
#
# WHAT IT COSTS
# -------------
# This is forward mode, so a K-parameter gradient costs ceil(K/N) evaluations.
# It is competitive with ForwardDiff (within ~15% at matched chunk width, and
# the two agree to 0.0), and like ForwardDiff it loses badly to reverse mode at
# large K -- at K=200 Mooncake is ~20x faster. Use it when you want a trimmed
# binary, or on a small-K model; use Mooncake/Enzyme otherwise.
#
# WHY IT WORKS ON MODELS AT ALL
# -----------------------------
# `Piste.Dual <: Real` is load-bearing: `Distributions.logpdf` is generic over
# `Real`, which is the same door ForwardDiff walks through. Three methods beyond
# plain arithmetic were needed to make the whole distribution suite work, each
# found by measurement rather than guessed (`SpecialFunctions._logabsgamma`,
# two-argument `atan`, and `isinteger`) -- see Piste's own docstring for the
# details and for what to do when a new one is missing.
# =============================================================================

using Piste: Piste, value_and_gradient!

# =============================================================================
# The user-facing backend
# =============================================================================

"""
    AutoPBForwardDiff(; chunk=8)

PracticalBayes' own forward-mode AD backend — **the one that survives
`juliac --trim=safe`**.

Pass it where any other AD backend would go:

```julia
ldf = LogDensityFunction(model, layout, store, AutoPBForwardDiff(); θ0=θ0)
```

`chunk` is the number of directional derivatives computed per pass, so a
`K`-parameter gradient costs `ceil(K/chunk)` model evaluations. It is a **type
parameter, not a field** — that is deliberate and is a large part of why this
trims: the chunk width, and therefore every tuple size in the hot loop, is known
at compile time. 8 is a good default on current hardware (four Float64 lanes ×
two, matching common SIMD widths); very small models can be faster with 4.

Choosing between backends:

- **Want a trimmed standalone binary?** This is currently the only option that
  works. ForwardDiff builds with zero verifier errors and then dies at runtime;
  Mooncake and Enzyme need the Julia compiler inside the binary.
- **Otherwise, and `K` is large?** Use `AutoMooncake()` or `AutoEnzyme()`.
  This is forward mode: cost grows with the parameter count, and around K=200
  reverse mode is roughly 20× faster.
- **Otherwise, and `K` is small?** Any of them; this is within ~15% of
  ForwardDiff and agrees with it to the last bit.

Float32 works: the dual is generic in its element type, so a `Float32` `θ0`
propagates without promoting.
"""
struct AutoPBForwardDiff{C} end

AutoPBForwardDiff(; chunk::Int=8) = AutoPBForwardDiff{chunk}()

"""
    chunksize(::AutoPBForwardDiff) -> Int

The compile-time chunk width, recovered from the type parameter.
"""
@inline chunksize(::AutoPBForwardDiff{C}) where {C} = C

function Base.show(io::IO, ::AutoPBForwardDiff{C}) where {C}
    print(io, "AutoPBForwardDiff(chunk=", C, ")")
end
