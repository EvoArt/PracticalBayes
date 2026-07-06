using Bijectors: Bijectors
using Bijectors.VectorBijectors: from_linked_vec, to_linked_vec, linked_vec_length, vec_length, optic_vec
using Bijectors: with_logabsdet_jacobian

"""
    SiteRecord

Metadata captured once during `TraceMode` for a single top-level model
variable. Never touched on the hot (`EvalMode`) path.
"""
struct SiteRecord
    name::Symbol
    dist_exemplar::Any  # a representative Distribution instance for this site
    linked_len::Int     # length of the unconstrained ("linked") representation
    role::Symbol        # :observed, :param, or :latent
    init_val::Any       # constrained-space initial value
end

abstract type AbstractSlot end

"""
    FlatSlot(range)

The variable occupies `range` within the flat unconstrained parameter vector `θ`.
"""
struct FlatSlot <: AbstractSlot
    range::UnitRange{Int}
end

"""
    FlatArraySlot(offset, elsize, dims)

An indexed family (`x[i] ~ dist` for `i` in some pre-declared container `x`)
where every element shares the same linked length `elsize`. The linked
sub-vector for element with linear index `k` (1-based, matching
`LinearIndices(dims)`) is `offset + (k-1)*elsize .+ (1:elsize)`.
"""
struct FlatArraySlot <: AbstractSlot
    offset::Int
    elsize::Int
    dims::Dims
end

# e.g. offset=10, elsize=3: element 1 -> 11:13, element 2 -> 14:16, etc.
# (1-based `k`, matching Julia's default array indexing.)
@inline function elem_range(s::FlatArraySlot, k::Int)
    lo = s.offset + (k - 1) * s.elsize
    return (lo + 1):(lo + s.elsize)
end

"""
    ValueSlot()

The variable's constrained value is read from `mode.store` (a NamedTuple),
not from the flat vector. Used for latents assigned to a non-HMC kernel and
for variables belonging to other blocks in a Gibbs sweep. Values behind a
`ValueSlot` are constants w.r.t. AD (passed to `DifferentiationInterface` as
`DI.Constant`), so a kernel updating them never triggers a HMC gradient call.
"""
struct ValueSlot <: AbstractSlot end

"""
    Layout{S<:NamedTuple}

`slots`: NamedTuple mapping variable name -> `AbstractSlot`, isbits so that
`getproperty(layout.slots, name)` is fully inferred and the assume-path branch
(`FlatSlot` vs `FlatArraySlot` vs `ValueSlot`) compiles away per call site.
`dim`: total length of the flat unconstrained vector.
`meta`: `Vector{SiteRecord}` in trace-visitation order, used for naming chains,
building initial vectors, and error messages — not used in `EvalMode`.
`untracked`: names (a subset of the flat-slot names in `slots`) that should
still be sampled — they occupy real space in `θ` and are fully part of the
model — but are excluded by `invlink` (and therefore from chain-bundling,
once `chains.jl` exists) as a memory/reporting optimization for models with
many "nuisance" parameters nobody wants in the output. This is purely a
reporting-level flag: `EvalMode`/`tilde.jl` never look at it, so an untracked
site's gradient contribution and posterior correlation with everything else
is completely unaffected.
"""
struct Layout{S<:NamedTuple}
    slots::S
    dim::Int
    meta::Vector{SiteRecord}
    untracked::Set{Symbol}
end

"""
    build_layout(model; flat=nothing, values=(), untracked=(), rng=Random.default_rng(), init=NamedTuple()) -> (layout, θ0, store0)

Runs `model` once under `TraceMode` to discover its variables, then
partitions assumed (non-conditioned) sites into flat-vector slots vs
value-store slots.

- `flat`: `Tuple` of `Symbol`s to place in the flat vector, or `nothing` to
  mean "every assumed site not listed in `values`".
- `values`: `Tuple` of `Symbol`s to place in the constant store instead
  (latents / other Gibbs blocks). Discrete-distribution sites MUST appear
  here (or be conditioned) — there is no flat-vector encoding for them.
- `untracked`: `Tuple` of `Symbol`s (must be a subset of the flat-vector
  names) to mark as nuisance parameters: they still occupy real space in
  `θ`, are still sampled, and still fully participate in the log-density and
  its gradient — the ONLY effect is that `invlink` (and therefore chain
  output, once `chains.jl` exists) omits them by default. Useful for very
  large models with many per-observation/per-group parameters nobody wants
  materialized in the output chain.

Returns the `Layout`, an initial flat vector `θ0` (via `to_linked_vec` on each
site's initial constrained value), and an initial store NamedTuple `store0`.

`T` fixes the element type of `θ0` (default `Float64`). This only affects the
*initial* vector handed to the sampler — the hot path (`tilde.jl`/`EvalMode`)
never hardcodes a numeric type: it is parametric on `eltype(θ)` throughout, so
a caller can subsequently swap in a `Vector{Float32}` (for SIMD) or a
`CuVector{Float32}` (for GPU) of the same length and everything downstream
(logpdf, bijector transforms, `Accum`) will compute in that type. Distributions.jl
itself is generally eltype-generic (`logpdf(Normal(0f0,1f0), 0f0)::Float32`);
the exceptions are distributions whose parameters are stored/promoted as
`Float64` internally, which is a Distributions.jl-level limitation, not one
introduced by this package.
"""
function build_layout(
    model;
    flat=nothing,
    values=(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    T::Type{<:Real}=Float64,
)
    # Step 1: run the model exactly once under TraceMode. Every tilde method
    # specialized on TraceMode (tilde.jl) pushes a SiteRecord as it goes, so
    # afterwards `tmode.sites` is the complete, ordered list of variables this
    # model has for this particular `model.conditioned`/`init`. This is the
    # ONE place dynamic dispatch and allocation are allowed — everything built
    # from `records` below is precomputed once and reused for every future
    # (fast, type-stable) `EvalMode` call.
    tmode = TraceMode(rng, model.conditioned, init)
    evaluate(model, tmode, Accum(zero(T)))
    records = SiteRecord[r for r in tmode.sites]

    # Step 2: figure out which names go in the flat vector vs the value-store.
    # `assumed` = everything that isn't already data (observed sites don't get
    # a slot at all — they're read straight from args/conditioned every time).
    assumed = filter(r -> r.role != :observed, records)
    value_names = Set{Symbol}(values)
    if flat === nothing
        # default: every assumed name not explicitly sent to the value-store
        flat_names = Set{Symbol}(r.name for r in assumed if !(r.name in value_names))
    else
        flat_names = Set{Symbol}(flat)
    end

    # Discrete/latent sites have no continuous flat-vector encoding (there's
    # no sensible "unconstrained bijector" for e.g. a categorical draw), so
    # they MUST be routed to the value-store (and updated by a user kernel via
    # Gibbs) rather than ending up in `flat_names` by default.
    for r in assumed
        if r.role == :latent && !(r.name in value_names) && !(r.name in flat_names)
            throw(ArgumentError(
                "site `$(r.name)` has a discrete/latent distribution and is neither " *
                "assigned to a value-block (`values=(:$(r.name),...)`) nor conditioned. " *
                "Assign it to a latent kernel via Gibbs, or condition it.",
            ))
        end
    end

    slot_pairs = Pair{Symbol,AbstractSlot}[]
    store_pairs = Pair{Symbol,Any}[]
    theta_chunks = Vector{T}[]  # concatenated at the end to build θ0
    offset = 0                  # running length already assigned in the flat vector

    # Step 3: group SiteRecords by name, preserving the order each name was
    # FIRST seen in. A name can appear more than once only for an indexed
    # family (`x[i] ~ dist` inside a loop produces one SiteRecord per `i`) —
    # grouping recovers that family so we can emit one FlatArraySlot for it
    # instead of many separate FlatSlots.
    seen_order = Symbol[]
    by_name = Dict{Symbol,Vector{SiteRecord}}()
    for r in assumed
        if !haskey(by_name, r.name)
            push!(seen_order, r.name)
            by_name[r.name] = SiteRecord[]
        end
        push!(by_name[r.name], r)
    end

    # Step 4: emit one slot per name, in first-seen order (this fixes the
    # flat vector's layout for the lifetime of this Layout — the whole point
    # of "static layout": names/order/sizes never change after this point).
    for name in seen_order
        recs = by_name[name]
        if name in value_names
            # Value-store slot: no flat-vector space consumed. If this name
            # was visited more than once (shouldn't normally happen for a
            # ValueSlot, but harmless if it does), the LAST recorded value
            # wins as the initial store value.
            push!(store_pairs, name => recs[end].init_val)
            push!(slot_pairs, name => ValueSlot())
            continue
        end
        # All elements of an indexed family must have the same unconstrained
        # ("linked") length — e.g. every `x[i] ~ Normal(...)` contributes 1
        # scalar, but mixing that with an occasional `x[i] ~ MvNormal(...)`
        # of a different size would break the fixed offset arithmetic in
        # `elem_range`. We check this explicitly rather than let it silently
        # miscompute.
        lens = Set(r.linked_len for r in recs)
        length(lens) == 1 || throw(ArgumentError(
            "site `$name` has non-uniform linked length across its indexed family " *
            "($(collect(lens))); split into separate names or use an array distribution.",
        ))
        elsize = only(lens)
        if length(recs) == 1
            # Ordinary scalar/vector/matrix-valued parameter: one contiguous
            # range in θ.
            push!(slot_pairs, name => FlatSlot((offset + 1):(offset + elsize)))
            push!(theta_chunks, T.(to_linked_vec(recs[1].dist_exemplar)(recs[1].init_val)))
            offset += elsize
        else
            # Indexed family: one FlatArraySlot covering all elements
            # contiguously; `elem_range` computes each element's sub-range
            # on demand during evaluation (see tilde.jl `tilde_index`).
            dims = (length(recs),)
            push!(slot_pairs, name => FlatArraySlot(offset, elsize, dims))
            for r in recs
                push!(theta_chunks, T.(to_linked_vec(r.dist_exemplar)(r.init_val)))
            end
            offset += elsize * length(recs)
        end
    end

    slots = NamedTuple(slot_pairs)  # isbits NamedTuple -> concrete-typed lookup in EvalMode
    store0 = NamedTuple(store_pairs)
    θ0 = isempty(theta_chunks) ? T[] : reduce(vcat, theta_chunks)

    untracked_names = Set{Symbol}(untracked)
    for name in untracked_names
        haskey(slots, name) && slots[name] isa ValueSlot && throw(ArgumentError(
            "site `$name` is in the value-store (`values=(:$name,...)`), not the flat vector; " *
            "`untracked` only applies to flat-vector sites.",
        ))
        haskey(slots, name) || throw(ArgumentError(
            "`untracked` name `$name` is not one of this layout's flat-vector sites: $(seen_order)",
        ))
    end
    layout = Layout(slots, offset, records, untracked_names)
    return layout, θ0, store0
end

"""
    link(layout, nt::NamedTuple) -> θ::Vector{Float64}

Maps a NamedTuple of constrained values (keyed by the flat-slot variable
names in `layout`) to the flat unconstrained vector, using each site's
`dist_exemplar` from `layout.meta` for the transform. Used by Gibbs to
refresh a component sampler's state after other blocks moved.
"""
function link(layout::Layout, nt::NamedTuple)
    chunks = Vector{Float64}[]
    # Re-derive the by-name grouping from `layout.meta` (this is cheap, O(number
    # of sites), and keeps Layout itself from needing to store this Dict) —
    # same grouping `build_layout` computed, just reconstructed here so `link`
    # doesn't need extra fields threaded through Layout.
    by_name = Dict{Symbol,Vector{SiteRecord}}()
    for r in layout.meta
        push!(get!(by_name, r.name, SiteRecord[]), r)
    end
    # `pairs(layout.slots)` iterates in the same order slots were inserted in
    # `build_layout` — i.e. the same order chunks must be concatenated in to
    # match θ's layout.
    for (name, slot) in pairs(layout.slots)
        slot isa ValueSlot && continue  # ValueSlot vars aren't part of θ at all
        recs = by_name[name]
        val = getfield(nt, name)
        if slot isa FlatSlot
            push!(chunks, to_linked_vec(recs[1].dist_exemplar)(val))
        else # FlatArraySlot
            for (i, r) in enumerate(recs)
                push!(chunks, to_linked_vec(r.dist_exemplar)(val[i]))
            end
        end
    end
    return isempty(chunks) ? Float64[] : reduce(vcat, chunks)
end

"""
    invlink(layout, θ; include_untracked=false) -> NamedTuple

Inverse of `link`: maps a flat unconstrained vector back to a NamedTuple of
constrained values for every flat-slot variable in `layout`. Used for
MCMCChains bundling, `predict`, and Gibbs block updates.

Names in `layout.untracked` (see `build_layout`'s `untracked` keyword) are
omitted unless `include_untracked=true` — they're still fully present in `θ`
and the log-density, this just controls whether they're materialized into
the returned NamedTuple (and, downstream, into a chain).
"""
function invlink(layout::Layout, θ::AbstractVector; include_untracked=false)
    by_name = Dict{Symbol,Vector{SiteRecord}}()
    for r in layout.meta
        push!(get!(by_name, r.name, SiteRecord[]), r)
    end
    pairs_out = Pair{Symbol,Any}[]
    for (name, slot) in pairs(layout.slots)
        slot isa ValueSlot && continue  # nothing to invert — these never lived in θ
        !include_untracked && name in layout.untracked && continue
        recs = by_name[name]
        if slot isa FlatSlot
            # `from_linked_vec(dist)` builds the bijector mapping an
            # unconstrained sub-vector back to `dist`'s constrained value
            # space; `with_logabsdet_jacobian` returns `(value, logjac)` — we
            # only need `value` here (the Jacobian only matters inside
            # `tilde.jl`'s accumulation, not for reporting constrained draws).
            x, _ = with_logabsdet_jacobian(from_linked_vec(recs[1].dist_exemplar), view(θ, slot.range))
            push!(pairs_out, name => x)
        else # FlatArraySlot
            elems = map(1:length(recs)) do i
                x, _ = with_logabsdet_jacobian(from_linked_vec(recs[i].dist_exemplar), view(θ, elem_range(slot, i)))
                x
            end
            push!(pairs_out, name => elems)
        end
    end
    return NamedTuple(pairs_out)
end
