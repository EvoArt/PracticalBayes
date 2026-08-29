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
    vals = Dict{Symbol,Any}()
    lp = zero(T)
    for ps in plan.priors
        v, ljac = _gm_read_param(ps, layout, theta)
        vals[ps.name] = v
        lp += _gm_prior_logpdf(ps, v) + ljac
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
        _gm_prior_grad!(g, ps, vals[ps.name], layout, theta)
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
    if d === :Normal
        mu, sd = _gm_num(ps.args, 1, 0.0), _gm_num(ps.args, 2, 1.0)
        return sum(@. -0.5*((v-mu)/sd)^2 - log(sd) - 0.5*log(2pi))
    elseif d === :MvNormal
        return sum(@. -0.5*v^2) - 0.5*length(v)*log(2pi)
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
        @. -v
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

# Numeric literal from a prior's argument list, or a default when absent.
function _gm_num(args, i, default)
    length(args) >= i || return default
    a = args[i]
    a isa Number && return float(a)
    return default
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
