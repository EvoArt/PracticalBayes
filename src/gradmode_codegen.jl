# =============================================================================
# Stage 2, part 2: closed-form value+gradient evaluation from a `GLMPlan`.
#
# `gradmode_value_and_gradient!(plan, layout, args, theta, g, ws)` computes the
# joint log-density and its gradient analytically, for a model that
# `recognize_glm` accepted. Stage 1 measured this shape of gradient at
# 5.45x-8.55x faster than Mooncake with the real Layout indirection included.
#
# STRUCTURE OF THE COMPUTATION
# ----------------------------
# Every recognized model has the same three-part shape, which is exactly why
# a closed form exists:
#
#   1. read parameters out of theta (through the real Layout slots/bijectors)
#   2. eta = sum of linear terms;  loglik = sum_i f(eta_i, y_i)
#   3. gradient: dL/d(param) = (dEta/d(param))' * (dL/dEta), plus prior terms
#
# The only per-link knowledge needed is `dL/dEta`, which for a GLM is a
# one-liner (see `_gm_link_loglik_and_dEta!`). The chain back to each
# parameter is then determined entirely by the term kind, NOT by the link:
#   :intercept  -> sum(w)
#   :matvec     -> X' * w
#   :scaled     -> dot(x, w)
#   :indexed    -> scatter-add w into groups
# That factorisation is what makes this general across links rather than one
# hardcoded model.
#
# WHY THIS IS AN INTERPRETER, NOT A MACRO
# ---------------------------------------
# It walks `plan.terms` at run time rather than generating specialized code
# per model. The loops are over a handful of terms whose bodies are the O(N)
# BLAS/SIMD work, so the dispatch overhead is negligible against the matvec —
# Stage 1 confirmed the achievable time is dominated by `mul!`. Generating
# specialized code is a possible later optimization, not a prerequisite, and
# keeping it an interpreter makes it far easier to verify.
#
# SAFETY: nothing here is trusted on its own. `check_gradmode` (below) compares
# this gradient against a general-AD gradient at randomized points, mirroring
# `check_depends.jl`'s role for `depends=` annotations. Recognition plus
# codegen is an optimization; the check is what licenses using it.
#
# STATUS: not yet wired into `@model`/`LogDensityFunction`. Callable and
# tested directly.
# =============================================================================

"""
    GradMode()

Opt-in AD type selecting the closed-form analytical gradient for models
`recognize_glm` accepts. Pass it where you would pass an `ADTypes` backend:

    ldf = LogDensityFunction(model, layout, store, GradMode(); θ0=θ0)

**Opt-in on purpose — it is NOT always faster.** Measured against the fastest
general backend per cell on logistic regression (see the factorial sweep in
the project notes): ~4.9-5.7x at N=5000-20000 with P=50-200, but **0.48-0.96x
(i.e. SLOWER) at N<=1000 with P<=10**, where ForwardDiff's forward-mode cost
is proportional to a tiny P and this path's fixed per-call overhead dominates.
Use it for large-N, moderate-to-large-P models; use ForwardDiff for small ones.

If the model is not recognized, construction throws with the reason rather
than silently falling back — an opt-in request for a fast path should tell you
when it cannot be honoured, otherwise you would think you had it and not.

Correctness is not assumed: verify with `check_gradmode` before relying on it
for real inference.
"""
struct GradMode end

"""
    gradmode_plan(f) -> Union{GLMPlan,Nothing}

The `GLMPlan` recognized for a model evaluator at `@model` expansion time, or
`nothing` when the model was not recognized. The `@model` macro emits a method
of this for every model it compiles; this fallback covers evaluators built by
other means.
"""
gradmode_plan(::Any) = nothing

"""
    GradModeWorkspace(N, dim)

Pre-allocated scratch for `gradmode_value_and_gradient!`: `eta` and the
per-observation weight vector `w = dL/dEta`, both length `N`. Held across
calls so the hot path allocates nothing (an HMC leapfrog calls this on every
step, so per-call allocation would show up immediately).
"""
struct GradModeWorkspace{T}
    eta::Vector{T}
    w::Vector{T}
end
GradModeWorkspace{T}(N::Int) where {T} = GradModeWorkspace{T}(zeros(T, N), zeros(T, N))

"""
    GradModePrep(plan, ws)

What `LogDensityFunction` stores in its `prep` slot for a `GradMode()` backend,
in place of a `DifferentiationInterface` tape/config. Holds the recognized plan
and the reusable scratch buffers, so the hot path allocates only the returned
gradient vector.
"""
struct GradModePrep{P<:GLMPlan,W}
    plan::P
    ws::W
end

"""
    gradmode_value_and_gradient!(plan, layout, args, theta, g, ws) -> value

Closed-form log-joint and gradient for a recognized GLM. Writes the gradient
into `g` (which must be `length(theta)`) and returns the log-joint.

`args` is the model's argument NamedTuple (`Model.args`), from which data
operands are looked up by the names the recognizer recorded.
"""
function gradmode_value_and_gradient!(plan::GLMPlan, layout, args, theta, g, ws)
    T = eltype(theta)
    fill!(g, zero(T))

    # --- 1. parameters, read through the real Layout ------------------------
    # `_gm_read_param` goes through the same bijector call `_assume` uses, so
    # constrained parameters (e.g. Exponential -> exp) are handled identically
    # and the log-Jacobian is accumulated the same way.
    # `vals` maps a site name to its CONSTRAINED value. `Any` because the
    # values are heterogeneous: a scalar site gives a `Float64`, a vector site
    # gives a `SubArray` (NOT a `Vector` — `_linked_view` deliberately
    # constructs a `Base.SubArray` to dodge AD backends that overload `view`;
    # see its docstring in tilde.jl). The element type is not fixed either:
    # `theta` may be Float32 (this package is Float32-first) or a Dual under a
    # ForwardDiff-family backend.
    #
    # KNOWN COST, not an oversight: this `Any` is why `_gm_hier_params` infers
    # `Tuple{Any,Any}`, which blocks `juliac --trim=safe` (162 verifier errors
    # cascade from here) and costs some dynamic dispatch on the prior/scale
    # arithmetic. The fix is NOT to annotate at the use site — hyper-locations
    # can legitimately be vectors, and asserting `::Float64` there is what
    # would have reintroduced the wrong gradient fixed in e31e113. The right
    # fix is to stop using a Dict: the name set is fixed at recognition time
    # (`plan.priors`), so these could live in a NamedTuple or a
    # position-indexed Vector, concrete by construction rather than by
    # assertion. Would also need a recognizer-side rejection for any site whose
    # linked representation is not scalar-or-vector, so the invariant is
    # ENFORCED rather than assumed — skipping that half is how the
    # vector-hyper-location bug (e31e113) happened in the first place.
    #
    # PARKED, deliberately (decision 2026-09-04). The only thing it unlocks is
    # `--trim=safe` for GradMode-recognized models, and trimming already works
    # today for anyone willing to hand-write a gradient (204 -> 0 errors, 2.9 MB
    # binary matching the JIT to 16 digits — see trimtest/FINDINGS.md). So this
    # is convenience, not capability. The dispatch cost it also removes is
    # small: profiling puts the closed form at 79.6% BLAS `gemv!` at
    # N=20000/P=200, so the prior arithmetic is not where the time goes.
    #
    # Do NOT gate this on `@static_model` if it is ever picked up. That was
    # considered and rejected: `vals` threads through ten functions, so a
    # conditional version means either duplicating the whole gradient path (two
    # copies that can drift, in the file where a mistake silently biases a
    # posterior) or parameterising the container — which IS the general fix, at
    # which point gating buys nothing. `@static_model` is opt-in because it
    # costs ~5ms compile time per site record; this has no such tradeoff and is
    # strictly better for every model, so it should be unconditional or not at
    # all. See trimtest/HANDOVER.md for the trim-side context.
    vals = Dict{Symbol,Any}()
    lp = zero(T)
    for ps in plan.priors
        v, ljac = _gm_read_param(ps, layout, theta)
        vals[ps.name] = v
        lp += (_gm_is_hier(ps) ? _gm_hier_logpdf(ps, v, vals) : _gm_prior_logpdf(ps, v)) + ljac
    end

    # --- 2. forward: eta ----------------------------------------------------
    eta = ws.eta
    fill!(eta, zero(T))
    for t in plan.terms
        _gm_accum_term!(eta, t, vals, args)
    end

    # --- 3. likelihood value and dL/dEta ------------------------------------
    y = getfield(args, plan.response)
    sigma = _gm_obs_scale(plan, vals)
    ll = _gm_link_loglik_and_dEta!(ws.w, plan.link, eta, y, sigma)
    lp += ll

    # --- 4. reverse: push w back through each term to its parameter ---------
    for t in plan.terms
        _gm_pullback_term!(g, t, ws.w, vals, args, layout)
    end

    # --- 5. prior gradient contributions ------------------------------------
    for ps in plan.priors
        if _gm_is_hier(ps)
            _gm_hier_grad!(g, ps, vals[ps.name], vals, layout)
        else
            _gm_prior_grad!(g, ps, vals[ps.name], layout, theta)
        end
    end

    # the observation scale (sigma) gets its own likelihood-side derivative
    _gm_scale_grad!(g, plan, vals, eta, y, layout, theta)
    return lp
end

# --- parameter reading ------------------------------------------------------

# Reads one prior's parameter out of theta via its Layout slot, using the same
# bijector path as `_assume(::FlatSlot, ...)` in tilde.jl so constrained
# parameters and their log-Jacobians behave identically.
function _gm_read_param(ps::PriorSite, layout, theta)
    slot = getproperty(layout.slots, ps.name)
    slot isa FlatSlot || error("gradmode: expected FlatSlot for $(ps.name)")
    d = _gm_exemplar(ps, layout)
    x, ljac = with_logabsdet_jacobian(from_linked_vec(d), _linked_view(theta, slot.range))
    return x, ljac
end

# The distribution instance for a site, taken from the Layout's own recorded
# exemplar (built during TraceMode) rather than re-evaluated from the AST —
# so the values used here are exactly the ones the normal path would use.
function _gm_exemplar(ps::PriorSite, layout)
    for rec in layout.meta
        rec.name === ps.name && return rec.dist_exemplar
    end
    error("gradmode: no layout record for $(ps.name)")
end

# --- prior value/gradient ---------------------------------------------------

_gm_prior_logpdf(ps::PriorSite, v) = _gm_flatlike(ps.dist) ? 0.0 : logpdf_of(ps, v)

# Flat/FlatPos/Uniform are constant on their support: they contribute nothing
# to the gradient, and their (improper or constant) log-density contributes
# nothing that varies with theta.
_gm_flatlike(d::Symbol) = d in (:Flat, :FlatPos, :Uniform)

function logpdf_of(ps::PriorSite, v)
    d = ps.dist
    # hierarchical (centered): location and/or scale come from `vals`, not the
    # AST, so this is handled by `_gm_hier_logpdf` with the resolved numbers.
    _gm_is_hier(ps) && error("gradmode: hierarchical prior needs _gm_hier_logpdf")
    if d === :Normal
        mu, sd = _gm_num(ps.args, 1, 0.0), _gm_num(ps.args, 2, 1.0)
        return sum(@. -0.5*((v-mu)/sd)^2 - log(sd) - 0.5*log(2pi))
    elseif d === :MvNormal
        # isotropic covariance `c*I`; `c` recovered from the AST at
        # recognition time (see _gm_mvnormal_var). Ignoring it silently
        # differentiates a scaled prior as if it were standard normal.
        c = _gm_mvnormal_var_of(ps)
        return sum(@. -0.5*v^2/c) - 0.5*length(v)*(log(2pi) + log(c))
    elseif d === :Exponential
        th = _gm_num(ps.args, 1, 1.0)
        return sum(@. -v/th - log(th))
    elseif d === :Cauchy
        mu, sc = _gm_num(ps.args, 1, 0.0), _gm_num(ps.args, 2, 1.0)
        return sum(@. -log(pi*sc*(1 + ((v-mu)/sc)^2)))
    end
    error("gradmode: no closed-form logpdf for $(d)")
end

# d(logpdf)/dx for each whitelisted prior, written into g at the slot range.
# The chain through the bijector is handled by `_gm_bij_chain`.
function _gm_prior_grad!(g, ps::PriorSite, v, layout, theta)
    _gm_flatlike(ps.dist) && return nothing
    slot = getproperty(layout.slots, ps.name)
    d = ps.dist
    dv = if d === :Normal
        mu, sd = _gm_num(ps.args, 1, 0.0), _gm_num(ps.args, 2, 1.0)
        @. -(v-mu)/sd^2
    elseif d === :MvNormal
        c = _gm_mvnormal_var_of(ps)
        @. -v/c
    elseif d === :Exponential
        th = _gm_num(ps.args, 1, 1.0)
        fill(-1/th, size(v))
    elseif d === :Cauchy
        mu, sc = _gm_num(ps.args, 1, 0.0), _gm_num(ps.args, 2, 1.0)
        @. -2*(v-mu)/(sc^2 + (v-mu)^2)
    else
        error("gradmode: no closed-form prior gradient for $(d)")
    end
    _gm_add_chained!(g, slot.range, dv, v, ps, layout)
    return nothing
end

# --- hierarchical (centered) priors ----------------------------------------
#
# `alpha ~ Normal(mu, sigma)` where mu and/or sigma are themselves parameters.
# All three gradients are needed, and omitting the hyperparameter ones is the
# classic silent-bias bug (the posterior for mu/sigma would be wrong while
# alpha's looked fine):
#
#   dL/dalpha = -(alpha - mu)/sigma^2
#   dL/dmu    =  sum((alpha - mu)/sigma^2)
#   dL/dsigma =  sum(((alpha-mu)^2/sigma^2 - 1)/sigma)
#
# Each hyperparameter's own bijector chain is then applied by
# `_gm_add_hyper!` (sigma is positive, so d/d(log sigma) = sigma * d/d sigma).

_gm_is_hier(ps::PriorSite) = ps.hyper_loc !== nothing || ps.hyper_scale !== nothing

# Resolved (mu, sigma) for a hierarchical site: from `vals` when the argument
# is a parameter, otherwise the literal from the AST.
# `mu` may be SCALAR (a hyper-mean broadcast over the site, `fill(mu_a, J)`)
# or a VECTOR (`a ~ MvNormal(mu, I)` with `mu` itself a vector parameter).
# `_gm_scalar` collapses the vector case to `mu[1]`, which silently uses one
# element for every observation — a wrong gradient, not a slow one. So the
# vector case is kept as a vector and the arithmetic below broadcasts.
#
# Found while investigating an unrelated type-inference question: annotating
# this `::Float64` (the obvious fix for the inference problem) would have been
# WRONG precisely because of this case, and checking whether the annotation
# was safe is what surfaced the bug.
function _gm_hier_params(ps::PriorSite, vals)
    mu = ps.hyper_loc === nothing ?
        (ps.dist === :MvNormal ? 0.0 : _gm_num(ps.args, 1, 0.0)) :
        _gm_hyper_value(vals[ps.hyper_loc])
    if ps.hyper_scale === nothing
        sd = ps.dist === :MvNormal ? sqrt(_gm_mvnormal_var_of(ps)) : _gm_num(ps.args, 2, 1.0)
    else
        s = _gm_scalar(vals[ps.hyper_scale])
        # What the argument MEANS differs by distribution, and getting it
        # backwards is a silent factor error in the gradient:
        #   Normal(mu, s)        -> s is a standard deviation
        #   MvNormal(mu, s^2*I)  -> s^2 is a variance, so sigma = s
        #   MvNormal(mu, s*I)    -> s is the variance, so sigma = sqrt(s)
        # The recognizer recorded whether the expression was squared
        # (`_gm_scale_is_variance`) precisely so this does not have to guess.
        sq = _gm_scale_is_variance(get(ps.args, 2, nothing))
        sd = ps.dist === :Normal ? s : (sq ? s : sqrt(s))
    end
    return mu, sd
end

function _gm_hier_logpdf(ps::PriorSite, v, vals)
    mu, sd = _gm_hier_params(ps, vals)
    n = length(v)
    #  is scalar or elementwise;  handles both without
    # allocating a broadcast temporary.
    ss = 0.0
    @inbounds for k in eachindex(v); ss += (v[k] - _gm_mu_at(mu, k))^2; end
    return -0.5*ss/sd^2 - n*(log(sd) + 0.5*log(2pi))
end

# One element of a hyper-location that may be a scalar or a vector.
@inline _gm_mu_at(mu::Real, k) = mu
@inline _gm_mu_at(mu::AbstractArray, k) = @inbounds mu[k]

# A hyperparameter value: scalars stay scalar, length-1 vectors collapse (a
# scalar site is stored as a 1-element view), longer vectors stay vectors.
@inline function _gm_hyper_value(v)
    v isa AbstractArray || return v
    return length(v) == 1 ? @inbounds(v[1]) : v
end

function _gm_hier_grad!(g, ps::PriorSite, v, vals, layout)
    mu, sd = _gm_hier_params(ps, vals)
    s2 = sd*sd
    # elementwise in both cases: broadcasting a scalar mu is a no-op, and a
    # vector mu lines up with v element for element
    dv = [-(v[k] - _gm_mu_at(mu, k))/s2 for k in eachindex(v)]
    slot = getproperty(layout.slots, ps.name)
    _gm_add_chained!(g, slot.range, dv, v, ps, layout)

    # hyper-location. For a SCALAR hyper-mean every element contributes to the
    # same partial, so the contributions sum:  dL/dmu = sum((x - mu)/sigma^2).
    # For a VECTOR hyper-mean each element has its OWN partial and they must
    # NOT be summed:  dL/dmu[k] = (x[k] - mu[k])/sigma^2. Collapsing the vector
    # case into a single sum (which is what happened before `_gm_hyper_value`
    # kept vectors intact) put the whole sum into mu[1] and zero elsewhere.
    if ps.hyper_loc !== nothing
        if mu isa AbstractArray
            _gm_add_hyper_vec!(g, ps.hyper_loc, v, mu, s2, vals, layout)
        else
            acc = 0.0
            @inbounds for x in v; acc += (x - mu); end
            _gm_add_hyper!(g, ps.hyper_loc, acc/s2, vals, layout)
        end
    end
    # hyper-scale: dL/dsigma = sum((x-mu)^2/sigma^2 - 1)/sigma. The scale is
    # always scalar (the recognizer only accepts scalar scale forms), so this
    # sum is correct in both cases.
    if ps.hyper_scale !== nothing
        ss = 0.0
        @inbounds for k in eachindex(v); ss += (v[k] - _gm_mu_at(mu, k))^2; end
        dsd = (ss/s2 - length(v))/sd
        _gm_add_hyper!(g, ps.hyper_scale, dsd, vals, layout)
    end
    return nothing
end

# Add a derivative w.r.t. a hyperparameter's CONSTRAINED value into g, chained
# through that hyperparameter's own bijector. Note this ADDS: a hyperparameter
# also has its own prior contribution, and may be shared by several sites.
function _gm_add_hyper!(g, name::Symbol, dval, vals, layout)
    slot = getproperty(layout.slots, name)
    i = first(slot.range)
    d = _gm_exemplar_by_name(name, layout)
    if _gm_positive_support(d)
        g[i] += dval * _gm_scalar(vals[name])   # x = exp(z), dx/dz = x
    else
        g[i] += dval
    end
    return nothing
end

# Vector hyper-location: each element of  gets its OWN partial, written
# into its own slot position, rather than one summed value.
function _gm_add_hyper_vec!(g, name::Symbol, v, mu, s2, vals, layout)
    slot = getproperty(layout.slots, name)
    d = _gm_exemplar_by_name(name, layout)
    pos = _gm_positive_support(d)
    mv = vals[name]
    o = first(slot.range) - 1
    @inbounds for k in eachindex(v)
        dk = (v[k] - mu[k])/s2
        g[o + k] += pos ? dk * mv[k] : dk
    end
    return nothing
end

function _gm_exemplar_by_name(name::Symbol, layout)
    for rec in layout.meta
        rec.name === name && return rec.dist_exemplar
    end
    error("gradmode: no layout record for $(name)")
end

# Adds `dv` (a derivative w.r.t. the CONSTRAINED value) into g at `range`,
# chained through the bijector to unconstrained space, plus the derivative of
# the log-Jacobian term itself.
#
# For an unconstrained (identity) parameter this is just `g[range] += dv`.
# For a positive parameter under exp: x = exp(z), dx/dz = x, and the logjac
# term is `z`, whose derivative is 1.
function _gm_add_chained!(g, range, dv, v, ps, layout)
    if _gm_is_positive(ps, layout)
        vv = collect(v)
        for (k, i) in enumerate(range)
            g[i] += dv[k] * vv[k] + 1.0
        end
    else
        for (k, i) in enumerate(range)
            g[i] += dv[k]
        end
    end
    return nothing
end

# Whether this site's support is positive (so the linked representation is a
# log). Read off the Layout's exemplar rather than assumed from the dist name,
# so a Truncated/derived distribution is classified by its actual support.
function _gm_is_positive(ps::PriorSite, layout)
    d = _gm_exemplar(ps, layout)
    return _gm_positive_support(d)
end
_gm_positive_support(d) = try
    minimum(d) >= 0 && isfinite(minimum(d)) && !isfinite(maximum(d))
catch
    false
end

# --- linear terms -----------------------------------------------------------

# Add one term's contribution into eta.
function _gm_accum_term!(eta, t::LinTerm, vals, args)
    if t.kind === :intercept
        a = _gm_scalar(vals[t.param])
        @inbounds @simd for i in eachindex(eta); eta[i] += a; end
    elseif t.kind === :matvec
        X = getfield(args, t.data)
        b = vals[t.param]
        LinearAlgebra.mul!(eta, X, b, true, true)   # eta .+= X*b
    elseif t.kind === :scaled
        x = getfield(args, t.data)
        a = _gm_scalar(vals[t.param])
        @inbounds @simd for i in eachindex(eta); eta[i] += a*x[i]; end
    elseif t.kind === :indexed
        v = vals[t.param]
        idx = getfield(args, t.index)
        @inbounds for i in eachindex(eta); eta[i] += v[idx[i]]; end
    elseif t.kind === :data
        x = getfield(args, t.data)
        @inbounds @simd for i in eachindex(eta); eta[i] += x[i]; end
    else
        error("gradmode: unknown term kind $(t.kind)")
    end
    return nothing
end

# Push dL/dEta back to this term's parameter. This is the transpose of
# `_gm_accum_term!` and is where the reverse-matvec (the whole point) happens.
function _gm_pullback_term!(g, t::LinTerm, w, vals, args, layout)
    t.param === nothing && return nothing     # :data has no parameter
    slot = getproperty(layout.slots, t.param)
    r = slot.range
    if t.kind === :intercept
        s = zero(eltype(w)); @inbounds @simd for i in eachindex(w); s += w[i]; end
        g[first(r)] += s
    elseif t.kind === :matvec
        X = getfield(args, t.data)
        LinearAlgebra.mul!(view(g, r), transpose(X), w, true, true)  # g[r] .+= X'w
    elseif t.kind === :scaled
        x = getfield(args, t.data)
        s = zero(eltype(w)); @inbounds @simd for i in eachindex(w); s += x[i]*w[i]; end
        g[first(r)] += s
    elseif t.kind === :indexed
        idx = getfield(args, t.index)
        o = first(r) - 1
        @inbounds for i in eachindex(w); g[o + idx[i]] += w[i]; end
    end
    return nothing
end

# --- links ------------------------------------------------------------------

"""
    _gm_link_loglik_and_dEta!(w, link, eta, y, sigma) -> loglik

Compute the log-likelihood and write `dL/dEta` into `w`. This is the ONLY
link-specific code in the whole gradient: everything downstream depends on
the term structure, not on the link.

  BernoulliLogit  dL/dEta = y - logistic(eta)
  LogPoisson      dL/dEta = y - exp(eta)
  Normal/MvNormal dL/dEta = (y - eta)/sigma^2
"""
function _gm_link_loglik_and_dEta!(w, link::Symbol, eta, y, sigma)
    T = eltype(eta)
    ll = zero(T)
    if link === :BernoulliLogit
        @inbounds @simd for i in eachindex(eta)
            p = _gm_logistic(eta[i])
            yi = y[i] > 0
            ll += yi ? log(p) : log1p(-p)
            w[i] = (yi ? one(T) : zero(T)) - p
        end
    elseif link === :LogPoisson
        # The -log(y!) normaliser is a constant w.r.t. theta, so it does not
        # affect the gradient — but it DOES affect the value, and the value is
        # compared against a general-AD reference in `check_gradmode`. It is
        # added once, outside the hot loop, via `_gm_logfactorial_sum` (which
        # needs no SpecialFunctions dependency: see its own comment).
        @inbounds @simd for i in eachindex(eta)
            m = exp(eta[i])
            ll += y[i]*eta[i] - m
            w[i] = y[i] - m
        end
        ll -= _gm_logfactorial_sum(y)
    elseif link === :BernoulliCLogLog
        # complementary log-log: p = 1 - exp(-exp(eta)).
        # With m = exp(eta):  log p = log1p(-exp(-m)),  log(1-p) = -m.
        # dL/deta = y * m*exp(-m)/p  -  (1-y)*m
        # The y=0 branch is exactly -m, which is why this is written as two
        # branches rather than via a shared `p` — it avoids both a cancellation
        # and a division when y=0.
        @inbounds for i in eachindex(eta)
            m = exp(eta[i])
            if y[i] > 0
                em = exp(-m)
                p = -expm1(-m)          # 1 - exp(-m), accurate for small m
                ll += log(p)
                w[i] = m*em/p
            else
                ll += -m
                w[i] = -m
            end
        end
    elseif link === :Normal || link === :MvNormal
        s2 = sigma*sigma
        @inbounds @simd for i in eachindex(eta)
            r = y[i] - eta[i]
            ll += -0.5*r*r/s2
            w[i] = r/s2
        end
        ll += -length(eta)*(log(sigma) + 0.5*log(2pi))
    else
        error("gradmode: no closed-form link derivative for $(link)")
    end
    return ll
end

@inline _gm_logistic(x) = inv(one(x) + exp(-x))

# sum(log(y_i!)) for integer counts. Constant w.r.t. theta (it only shifts the
# value, never the gradient), so this runs once per call outside the hot loop
# and its cost is irrelevant. Computed by direct summation rather than
# `loggamma` specifically to avoid taking a SpecialFunctions dependency for a
# term that never touches the gradient — counts in practice are small, and the
# loop is O(sum(y)) only for the largest count seen.
function _gm_logfactorial_sum(y)
    s = 0.0
    @inbounds for v in y
        k = Int(v)
        for j in 2:k; s += log(j); end
    end
    return s
end

# --- observation scale (the `sigma` of a Normal likelihood) -----------------

# The scale parameter of a Normal/MvNormal observe, if the model has one.
# Recognized models write `MvNormal(pred, sigma^2 * I)`; the scale is whichever
# prior-bound parameter appears in that argument.
function _gm_obs_scale(plan::GLMPlan, vals)
    plan.link in (:Normal, :MvNormal) || return 1.0
    plan.scale === nothing && return 1.0
    return _gm_scalar(vals[plan.scale])
end

# d(loglik)/d(sigma) = sum(r^2)/sigma^3 - N/sigma, chained through the log
# bijector (sigma is positive, so the linked value is log sigma and
# d/d(log sigma) = sigma * d/d(sigma)).
function _gm_scale_grad!(g, plan::GLMPlan, vals, eta, y, layout, theta)
    (plan.link in (:Normal, :MvNormal) && plan.scale !== nothing) || return nothing
    s = _gm_scalar(vals[plan.scale])
    ss = zero(eltype(eta))
    @inbounds @simd for i in eachindex(eta)
        r = y[i] - eta[i]; ss += r*r
    end
    dsigma = ss/(s^3) - length(eta)/s
    slot = getproperty(layout.slots, plan.scale)
    g[first(slot.range)] += dsigma * s     # chain through exp
    return nothing
end

# --- small helpers ----------------------------------------------------------

@inline _gm_scalar(v) = v isa AbstractArray ? v[1] : v

# The isotropic variance of an MvNormal prior site (1.0 when unspecified).
function _gm_mvnormal_var_of(ps::PriorSite)::Float64
    length(ps.args) >= 2 || return 1.0
    v = _gm_mvnormal_var(ps.args[2])
    v === nothing && error("gradmode: unrecognized MvNormal covariance for $(ps.name)")
    return v
end

# Numeric literal from a prior's argument list, or a default when absent.
#
# Sees through numeric CASTS (`pT(10)`, `Float32(2.5)`) and `zero(pT)`/`one(pT)`,
# which is how this package's own models are written (parameters are
# Float32-first, so distribution literals are routinely cast). Treating
# `pT(10)` as unparseable and silently falling back to the DEFAULT was a real
# bug: `Normal(zero(pT), pT(10))` was evaluated with sd=1 instead of 10, giving
# a wrong log-density and a wrong prior gradient. Caught by check_gradmode.
#
# Returns `nothing` when the argument is present but not a recognizable
# constant, so callers can distinguish "absent, use default" from "present but
# not understood" — the latter must reject rather than guess.
# The `::Float64` return annotation is load-bearing, not decoration.
# `PriorSite.args` is a `Vector{Any}` of unevaluated AST, so `_gm_const`'s walk
# below necessarily infers `Any` — and without this annotation that `Any`
# propagates into every arithmetic operation downstream (broadcasts, `sum`,
# `literal_pow`, `log`), leaving them as unresolved dynamic calls. This is the
# right place to put the boundary: it is exactly where AST walking ends and
# floating-point arithmetic begins, so annotating here concretises all the
# callers while the dynamic walk beneath stays dynamic.
#
# Measured effect (reported by the trim work on a gm_normal GLM): `_gm_num`
# accounted for 200 of 282 `--trim=safe` verifier errors, always as `::Any`.
function _gm_num(args, i, default)::Float64
    length(args) >= i || return default
    v = _gm_const(args[i])
    v === nothing && error("gradmode: non-constant prior argument $(args[i])")
    return v
end

# A compile-time numeric constant, seeing through casts and zero/one.
function _gm_const(a)::Union{Float64,Nothing}
    a isa Number && return float(a)
    if a isa Expr && a.head == :call && length(a.args) == 2
        f = _gm_head(a.args[1])
        f === :zero && return 0.0
        f === :one && return 1.0
        # a numeric cast such as `pT(10)` / `Float64(2.5)`
        a.args[2] isa Number && return float(a.args[2])
    end
    return nothing
end

# =============================================================================
# VERIFICATION
# =============================================================================

"""
    check_gradmode(plan, layout, args, ldf; n=8, rtol=1e-6, atol=1e-8, rng)

Compare the closed-form gradient against a general-AD gradient (`ldf`, a
`LogDensityFunction` with an `adtype`) at `n` randomized non-zero points.
Returns `true` when every point agrees.

This is the safety net that licenses using `GradMode` at all, and mirrors
`check_depends.jl`'s role for `depends=` annotations. Recognition is only a
syntactic hint; a recognizer bug, a wrong derivative in the tables above, or a
model shape that is accepted but subtly different would all show up here.

Points are drawn AWAY from zero on purpose: at theta = 0 many terms vanish
(`beta = 0` kills every matvec contribution) and sign errors hide.
"""
function check_gradmode(plan::GLMPlan, layout, args, ldf;
                        n::Int=8, rtol=1e-6, atol=1e-8, rng=Random.default_rng(), verbose::Bool=false)
    N = length(getfield(args, plan.response))
    ws = GradModeWorkspace{Float64}(N)
    g = zeros(Float64, layout.dim)
    ok = true
    for k in 1:n
        theta = randn(rng, layout.dim) .* 0.5
        v_ref, g_ref = LogDensityProblems.logdensity_and_gradient(ldf, theta)
        v_an = gradmode_value_and_gradient!(plan, layout, args, theta, g, ws)
        dv = abs(v_an - v_ref)
        dg = maximum(abs, g .- g_ref)
        good = isapprox(v_an, v_ref; rtol=rtol, atol=atol) &&
               isapprox(g, g_ref; rtol=rtol, atol=atol)
        good || (ok = false)
        verbose && println("  point $k: dvalue=$(dv)  maxdgrad=$(dg)  ", good ? "OK" : "MISMATCH")
    end
    return ok
end
