# Validating `depends=` annotations.
#
# `@addlogprob! expr depends=(:a, :b)` is a pure optimization hint: under a Gibbs
# block owning none of the named variables, `expr` is a constant w.r.t. that
# block's parameters, so the evaluator skips it. Skipping is exact -- but ONLY if
# the annotation is right. Under-declaring is silent: the term still gets skipped
# for a block that owns an undeclared variable, and that block's gradient loses a
# real contribution with no error, no warning, and a log density that merely
# differs by what looks like a constant.
#
# `check_depends` is the guard. It compares, per block, the gradient the model
# computes as written against the gradient with nothing skipped.
#
# HOW, AND WHY IT IS NOT "STRIP THE ANNOTATIONS"
# ----------------------------------------------
# `depends=` is consumed at macro-expansion time and baked into `model.f`, so
# there is no way to rebuild the same `Model` without it. But nothing needs to
# be: `addlogprob_needed` decides purely on whether any declared name is a
# `FlatSlot` in the CURRENT layout (tilde.jl). So building a second layout for
# the same model, in which every variable is flat, makes every term's gate fold
# to `true` -- the same model, evaluated with nothing skipped.
#
# The two evaluations differ in dimension (the reference layout carries every
# variable in `theta`), so the comparison is over the block's OWN coordinates,
# which is exactly the subvector a Gibbs block's sampler would move.

"""
    DependsReport

The result of [`check_depends`](@ref). `ok` is the headline: `true` when no
block's gradient changed. `rows` holds one entry per checked block with the
worst absolute gradient discrepancy found and, when non-zero, the model
variables that are the likely culprits.
"""
struct DependsReport
    ok::Bool
    rows::Vector{NamedTuple{(:block, :names, :max_abs_diff, :logp_diff, :suspects),
                            Tuple{Int,Tuple{Vararg{Symbol}},Float64,Float64,Vector{Symbol}}}}
end

_depends_rows() = NamedTuple{(:block, :names, :max_abs_diff, :logp_diff, :suspects),
                             Tuple{Int,Tuple{Vararg{Symbol}},Float64,Float64,Vector{Symbol}}}[]

function Base.show(io::IO, r::DependsReport)
    println(io, "DependsReport: ", r.ok ? "OK" : "MISMATCH",
            " (", length(r.rows), " block(s) checked)")
    for row in r.rows
        println(io, "  block ", row.block, " ", row.names,
                "  max|dgrad| = ", row.max_abs_diff,
                "  dlogp = ", row.logp_diff,
                "  ", row.max_abs_diff == 0 ? "ok" : "MISMATCH")
        if !isempty(row.suspects)
            println(io, "      likely undeclared: ", join(row.suspects, ", "))
        end
    end
    if !r.ok
        println(io, "A non-zero gradient difference means a `depends=` annotation is")
        println(io, "UNDER-DECLARED: a term that does affect this block is being skipped.")
    end
end

"""
    check_depends(model, spl::Gibbs; init=NamedTuple(), adtype=AutoForwardDiff(),
                  rng=Random.default_rng(), atol=0.0, verbose=true) -> DependsReport

Verify that every `depends=` annotation in `model` is safe under the blocking
`spl`, by comparing each HMC-family block's gradient with the model as written
against the same gradient computed with no term skipped. They must agree
exactly; any difference means a term that DOES affect the block is being
skipped, i.e. an under-declared `depends=`.

This is the check for the one hazard `depends=` carries. Over-declaring is
harmless -- it only forgoes a speed-up -- and is not reported as a failure.

Returns a [`DependsReport`](@ref); `report.ok` is `false` if any block
mismatched. Latent blocks are skipped: they never build a `LogDensityFunction`
and never touch AD, so there is no gradient to compare.

`atol` exists only for backends whose gradients are not bit-reproducible across
two calls; leave it at `0.0` unless you have a concrete reason, since a genuine
under-declaration is usually far larger than any tolerance.

# Example

```julia
spl = Gibbs((:a, :b) => NUTS(0.8), :z => MyKernel())
report = check_depends(model, spl)
report.ok || error("bad depends= annotation")
```

Note this checks the annotations AT ONE POINT in parameter space -- the initial
values, or `init` if given. A dependency that happens to be inactive there (a
term multiplied by a parameter that starts at zero, a branch not taken) would
pass. Run it at more than one `init` for a model with regime-dependent
structure.
"""
function check_depends(
    model::Model,
    spl::Gibbs;
    init::NamedTuple=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    rng::Random.AbstractRNG=Random.default_rng(),
    atol::Real=0.0,
    verbose::Bool=true,
)
    # The same starting values the sampler itself would use, so the check runs at
    # a point the model actually visits.
    records = _validate_gibbs_coverage(model, spl)
    values0 = _gibbs_init_values(records, init)
    all_names = keys(values0)

    # The reference layout: every variable that CAN be flat, is. Discrete/latent
    # sites cannot go in the flat vector at all, so they stay in the store --
    # correct and harmless here, since a name that cannot belong to an HMC block
    # is not one an under-declared annotation could rob of a gradient.
    ref_flat = Tuple(k for k in all_names if _flat_eligible(records, k))
    ref_other = Tuple(k for k in all_names if !(k in ref_flat))
    ref_init = NamedTuple{ref_flat}(Tuple(values0[k] for k in ref_flat))
    ref_layout, ref_θ0, _ = build_layout(
        model; flat=ref_flat, values=ref_other, init=ref_init, rng=rng
    )
    ref_store = NamedTuple{ref_other}(Tuple(values0[k] for k in ref_other))
    f_ref = LogDensityFunction(model, ref_layout, ref_store, adtype; θ0=ref_θ0)
    lp_ref, g_ref = LogDensityProblems.logdensity_and_gradient(f_ref, ref_θ0)

    rows = _depends_rows()
    ok = true

    for (i, block) in enumerate(spl.blocks)
        _is_latent_block(block) && continue          # no gradient to compare

        own = block.names
        other = Tuple(k for k in all_names if !(k in own))
        own_init = NamedTuple{own}(Tuple(values0[k] for k in own))

        # Exactly `_init_block_sub`'s layout, so the `depends=` gates fold as
        # they do during sampling.
        layout, θ0, _ = build_layout(model; flat=own, values=other, init=own_init, rng=rng)
        store = NamedTuple{other}(Tuple(values0[k] for k in other))
        f_real = LogDensityFunction(model, layout, store, adtype; θ0=θ0)
        lp_real, g_real = LogDensityProblems.logdensity_and_gradient(f_real, θ0)

        maxdiff = 0.0
        for n in own
            ra, rb = _slot_range(layout, n), _slot_range(ref_layout, n)
            length(ra) == length(rb) || error(
                "check_depends: slot length mismatch for `$n` -- this is a bug in " *
                "the check, not in your model.")
            for (a, b) in zip(ra, rb)
                d = abs(g_real[a] - g_ref[b])
                d > maxdiff && (maxdiff = d)
            end
        end

        bad = maxdiff > atol
        bad && (ok = false)
        suspects = bad ? _undeclared_candidates(records, own, model, values0,
                                                layout, g_real, ref_layout, g_ref,
                                                atol) : Symbol[]
        push!(rows, (block=i, names=own, max_abs_diff=maxdiff,
                     logp_diff=Float64(lp_real - lp_ref), suspects=suspects))
    end

    report = DependsReport(ok, rows)
    verbose && show(stdout, report)
    return report
end

# A name can live in the flat vector unless its site is discrete/latent -- the
# same rule `build_layout`'s default applies.
function _flat_eligible(records, name::Symbol)
    for r in records
        r.name === name || continue
        r.role == :observed && return false
        r.role == :latent && return false
    end
    return true
end

# The index span a name occupies in a layout's flat vector. A name is either a
# plain `FlatSlot` or a `FlatArraySlot` (an indexed family, occupying one
# contiguous span of `prod(dims) * elsize`).
_slot_range(layout::Layout, name::Symbol) = _span(getproperty(layout.slots, name), name)

_span(s::FlatSlot, ::Symbol) = s.range
_span(s::FlatArraySlot, ::Symbol) = (s.offset + 1):(s.offset + prod(s.dims) * s.elsize)
_span(::ValueSlot, name::Symbol) = error(
    "check_depends: `$name` is in the value store where the flat vector was " *
    "expected. A discrete/latent variable cannot belong to an HMC block, so this " *
    "should have been caught by Gibbs coverage validation.")

# Attribution hint for the message: which of the block's own names a term is
# actually skipping.
#
# The annotations are consumed at macro-expansion time and survive only as
# `Val{(:a, :b)}` TYPES inside the generated evaluator, which are not reliably
# reachable from the lowered AST -- an IR walk for them is fragile and was
# dropped. Instead the culprit is found the same way the check itself works: for
# each of the block's names, compare the block's gradient against one computed
# with that name ALSO visible to the gates. A name whose presence changes the
# gradient is one some term reads but does not declare.
#
# This is only ever a hint attached to an already-failing check, so it is allowed
# to come back empty rather than raise.
function _undeclared_candidates(records, own::Tuple{Vararg{Symbol}},
                                model::Model, values0::NamedTuple,
                                base_layout::Layout, base_g::AbstractVector,
                                ref_layout::Layout, ref_g::AbstractVector,
                                atol::Real)
    suspects = Symbol[]
    for n in own
        try
            ra, rb = _slot_range(base_layout, n), _slot_range(ref_layout, n)
            length(ra) == length(rb) || continue
            any(abs(base_g[a] - ref_g[b]) > atol for (a, b) in zip(ra, rb)) &&
                push!(suspects, n)
        catch
            continue
        end
    end
    return suspects
end
