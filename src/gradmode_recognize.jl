# =============================================================================
# Stage 2 (in progress): compile-time recognition of GLM-shaped models.
#
# GOAL
# ----
# Decide, at `@model` macro-expansion time, whether a model body has the shape
#
#     <params> ~ <closed-form priors>
#     eta = <linear predictor built from params and data>
#     y ~ <link>.(eta)            (or `y .~ <link>.(eta)`)
#
# for which a closed-form gradient exists. Stage 1 measured that gradient at
# 5.45x-8.55x faster than Mooncake, integrated through the real Layout, so the
# payoff is established; this file is about firing on it SAFELY.
#
# WHY THIS IS A SYNTACTIC WHITELIST, NOT AN INFERENCE PASS
# -------------------------------------------------------
# The crux problem is that `eta = X*beta` is arbitrary user code sitting
# between the tilde sites, so in general "is this expression linear in the
# parameters?" is undecidable. We do NOT attempt to answer that in general.
# Instead we accept a small, explicitly enumerated grammar of expressions
# whose linearity is manifest from the syntax alone, and reject everything
# else. A model outside the grammar is not "partially recognized" — it is
# rejected WHOLE and falls back to general AD.
#
# THE SAFETY ARGUMENT (read this before extending the grammar)
# ------------------------------------------------------------
# A wrong gradient does not crash. It silently biases the posterior, which is
# the worst failure mode this package has. Three rules follow, and every
# addition to this file must preserve all three:
#
#   1. ALL-OR-NOTHING. `recognize_glm` returns `nothing` unless it understands
#      EVERY statement in the body. There is no partial-credit path.
#   2. CONSERVATIVE BY CONSTRUCTION. Any expression head, function name, or
#      statement form not on the whitelist is an automatic reject. New syntax
#      appearing in a user's model can therefore only ever cause a fallback to
#      general AD (slow but correct), never a wrong fast path.
#   3. VERIFIED AT RUNTIME ANYWAY. Recognition alone is not trusted — the
#      analytical gradient is checked against a general-AD gradient by
#      `check_gradmode` (mirroring `check_depends.jl`'s existing role for
#      `depends=` annotations). Recognition is an optimization hint; the check
#      is what makes it safe to ship.
#
# STATUS: recognizer only. This file currently performs NO code generation and
# is NOT yet wired into `@model` — nothing in the package calls it. It is
# deliberately inert so it can be developed and tested against the corpus
# without any risk to the existing evaluation path.
# =============================================================================

"""
    LinTerm

One additive term of a recognized linear predictor. `eta` is represented as a
sum of these, so `alpha .+ X*beta .+ a[grp]` becomes three `LinTerm`s.

`kind` is one of:
- `:intercept`  — a bare scalar parameter added to every row (`alpha`)
- `:matvec`     — `X * beta`, a data matrix times a parameter vector
- `:scaled`     — `x .* beta` / `beta .* x`, data vector times a SCALAR param
- `:indexed`    — `a[grp]` / `view(a, grp)`, a parameter vector gathered by a
                  data index vector (the hierarchical group-effect case)
- `:data`       — a term with no parameter dependence at all (a pure offset)

`param` names the parameter involved (`nothing` for `:data`), `data` names the
data operand (`nothing` for `:intercept`), and `index` names the index vector
for `:indexed`.
"""
struct LinTerm
    kind::Symbol
    param::Union{Symbol,Nothing}
    data::Union{Symbol,Nothing}
    index::Union{Symbol,Nothing}
end

"""
    PriorSite

A recognized prior `name ~ dist(args...)`. `dist` is the distribution's head
symbol as written (`:Normal`, `:MvNormal`, `:Exponential`, ...) and `args` the
unevaluated argument expressions — enough for the (not-yet-written) codegen
step to emit the closed-form logpdf and its derivative, and enough for the
recognizer to reject a distribution it has no closed form for.
"""
struct PriorSite
    name::Symbol
    dist::Symbol
    args::Vector{Any}
end

"""
    GLMPlan

The result of successfully recognizing a model. Carries everything codegen
would need: the priors in declaration order, the additive decomposition of the
linear predictor, the response variable, and the link.

This is a *description*, not code — building it has no effect on evaluation.
"""
struct GLMPlan
    priors::Vector{PriorSite}
    eta_var::Union{Symbol,Nothing}   # `nothing` when the predictor is inline
    terms::Vector{LinTerm}
    response::Symbol
    link::Symbol
    dotted::Bool          # `y .~ link.(eta)` vs `y ~ arraydist(link.(eta))`
    # For a Normal/MvNormal likelihood, the parameter supplying the
    # observation scale (`sigma` in `MvNormal(pred, sigma^2 * I)`), or
    # `nothing` when the scale is a constant. It needs its own likelihood-side
    # derivative, unlike the predictor parameters, so it is tracked separately.
    scale::Union{Symbol,Nothing}
end

# --- the whitelists -------------------------------------------------------
#
# Priors we have a closed-form logpdf AND derivative for. Deliberately short:
# adding an entry is a commitment to getting its derivative right, so entries
# are added only alongside a test.
#
# `filldist(D, n)` and `Truncated(D, lo, hi)`/`truncated(...)` are accepted as
# WRAPPERS around a whitelisted scalar distribution: `filldist` is IID
# replication (the logpdf is a sum of identical terms, derivative unchanged
# per element) and a truncation only adds a CONSTANT normaliser, so it does
# not change d(logpdf)/dx anywhere in the interior of the support. Both are
# therefore safe to see through — but only when the inner distribution is
# itself on the whitelist, which `_gm_match_prior` enforces recursively.
#
# Every entry here needs a known closed-form d(logpdf)/dx:
#   Normal/MvNormal  -(x-mu)/sigma^2
#   Exponential      -1/theta            (plus the log-Jacobian of exp)
#   Cauchy           -2(x-mu)/(sigma^2+(x-mu)^2)
#   Flat/FlatPos     0  (improper, contributes NOTHING to value or gradient)
#   Uniform          0  in the interior of the support
# `Flat`/`FlatPos`/`Uniform` are the reason so much of the corpus is
# reachable: they are extremely common as weak priors and are the easiest
# derivatives in the set.
const GRADMODE_PRIORS = Set([
    :Normal, :MvNormal, :Exponential, :Cauchy, :Flat, :FlatPos, :Uniform,
])
const GRADMODE_PRIOR_WRAPPERS = Set([:filldist, :Truncated, :truncated])

# Likelihood links with a one-line d(loglik)/d(eta). These are the GLM cases
# where the whole point of the exercise lives:
#   BernoulliLogit:   y - logistic(eta)
#   LogPoisson:       y - exp(eta)
#   BernoulliCLogLog: y*m*exp(-m)/p - (1-y)*m,  m = exp(eta)   [survival/hazard]
#   Normal/MvNormal:  (y - eta)/sigma^2
const GRADMODE_LINKS = Set([
    :BernoulliLogit, :LogPoisson, :BernoulliCLogLog, :Normal, :MvNormal,
])

"""
    recognize_glm(body, argnames) -> Union{GLMPlan,Nothing}

Attempt to recognize `body` (the model body as written, BEFORE tilde
rewriting) as a GLM. `argnames` is the set of names that are data — model
arguments and `:=`-bound names — exactly the set `@model` already computes for
its data-vs-parameter decision.

Returns a `GLMPlan` on success, or `nothing` if anything at all is
unrecognized. `nothing` is not an error: it means "use general AD", which is
always correct.
"""
function recognize_glm(body, argnames_in)
    # copied because derived data names (`x = hcat(...)`) are added as we go,
    # and the caller's set must not be mutated
    argnames = Set{Symbol}(argnames_in)
    priors = PriorSite[]
    params = Set{Symbol}()       # names bound by a `~` prior so far
    etas = Dict{Symbol,Vector{LinTerm}}()   # name => decomposition, for `eta = ...`
    response = nothing; link = nothing; eta_var = nothing; dotted = false
    eta_terms = LinTerm[]
    scale = nothing

    for stmt in _gm_statements(body)
        # skip line-number nodes and bare literals
        stmt isa LineNumberNode && continue

        # --- prior / likelihood site: `lhs ~ rhs` or `lhs .~ rhs`
        if _gm_is_tilde(stmt)
            lhs, rhs, isdot = _gm_tilde_parts(stmt)

            # A tilde whose LHS is data is the LIKELIHOOD.
            if lhs isa Symbol && lhs in argnames
                response !== nothing && return nothing   # two likelihoods: reject
                lk, ev, tms = _gm_match_link(rhs, isdot, etas, params, argnames)
                lk === nothing && return nothing
                sc = _gm_scale_param(rhs, params)
                sc === :__reject__ && return nothing
                link = lk; eta_var = ev; eta_terms = tms
                response = lhs; dotted = isdot; scale = sc
                continue
            end

            # Otherwise it is a PRIOR binding a new parameter name.
            isdot && return nothing                      # `.~` assume unsupported
            lhs isa Symbol || return nothing             # indexed priors: not yet
            lhs in params && return nothing              # rebinding: reject
            ps = _gm_match_prior(lhs, rhs, params)
            ps === nothing && return nothing
            push!(priors, ps); push!(params, lhs)
            continue
        end

        # --- assignment (`=` or `:=`): either pure data preprocessing, or the
        # construction of a linear predictor.
        #
        # Data preprocessing (`dist100 = dist ./ 100`, `x = hcat(a,b,c)`) is
        # accepted and the new name is recorded as DATA: it involves no
        # parameters, so it is a constant with respect to the gradient no
        # matter what it computes. This is what lets models that massage their
        # inputs before the tilde sites still be recognized.
        if stmt isa Expr && (stmt.head == :(=) || stmt.head == :(:=))
            lhs, rhs = stmt.args[1], stmt.args[2]
            lhs isa Symbol || return nothing
            if _gm_param_free(rhs, params)
                push!(argnames, lhs)     # a derived data value
                continue
            end
            terms = _gm_linear_terms(rhs, params, argnames, etas)
            terms === nothing && return nothing
            etas[lhs] = terms
            continue
        end

        # --- `@addlogprob! sum(logpdf.(Dist.(<pred>), y))` as the LIKELIHOOD.
        #
        # This vectorised-sum idiom is a common hand-optimisation (it avoids a
        # per-row tilde site) and is exactly how the jolly island epi models
        # write their observes, so recognizing it is what lets those models use
        # the fast path at all. It is accepted ONLY in this precise shape — a
        # sum of a broadcast logpdf over a whitelisted link applied to a
        # recognized linear predictor. Any other `@addlogprob!` expression is
        # an arbitrary log-density term we cannot differentiate in closed form,
        # and rejects the model.
        # (the macro name may be qualified — `PracticalBayes.@addlogprob!` —
        # in which case `args[1]` is an Expr, not a Symbol; `_gm_head` handles
        # both, same as for qualified function names)
        if stmt isa Expr && stmt.head == :macrocall &&
           _gm_head(stmt.args[1]) == Symbol("@addlogprob!")
            aargs = stmt.args[3:end]
            # a `depends=` annotation implies Gibbs blocking, which GradMode
            # does not support (it derives the whole model at once)
            length(aargs) == 1 || return nothing
            response !== nothing && return nothing
            lk, ev, tms, resp = _gm_match_addlogprob(aargs[1], etas, params, argnames)
            lk === nothing && return nothing
            link = lk; eta_var = ev; eta_terms = tms
            response = resp; dotted = true; scale = nothing
            continue
        end

        # Anything else (loops, control flow, other calls, ...) is outside the
        # grammar. Reject the WHOLE model — rule 1 above.
        return nothing
    end

    response === nothing && return nothing
    isempty(eta_terms) && return nothing
    # The predictor must actually depend on at least one parameter, otherwise
    # there is no gradient to compute and this is not a model we should claim.
    any(t -> t.param !== nothing, eta_terms) || return nothing
    return GLMPlan(priors, eta_var, eta_terms, response, link, dotted, scale)
end

# --- qualified-name handling -----------------------------------------------

"""
    _gm_head(ex) -> Union{Symbol,Nothing}

The bare name of a callee, seeing through module qualification:
`Distributions.MvNormal` -> `:MvNormal`, `MvNormal` -> `:MvNormal`.

Necessary because real models qualify almost everything
(`Distributions.MvNormal`, `PracticalBayes.arraydist`), and matching only bare
symbols recognizes essentially nothing outside a toy test.

Only the FINAL component is used. That is safe here because the name is then
checked against a whitelist and the derivative is verified against general AD
by `check_gradmode` — a same-named function from an unexpected module would be
caught there, not silently trusted.
"""
function _gm_head(ex)
    ex isa Symbol && return ex
    if ex isa Expr && ex.head == :. && length(ex.args) == 2
        q = ex.args[2]
        q isa QuoteNode && q.value isa Symbol && return q.value
        q isa Symbol && return q
    end
    return nothing
end

# --- statement flattening --------------------------------------------------

# Flatten a `begin ... end` block into its statements. A body that is not a
# block is treated as a single statement.
function _gm_statements(body)
    if body isa Expr && body.head == :block
        return body.args
    end
    return Any[body]
end

# --- tilde helpers ---------------------------------------------------------

function _gm_is_tilde(x)
    x isa Expr || return false
    if x.head == :call && length(x.args) == 3 && x.args[1] === :~
        return true
    end
    # `.~` parses as Expr(:call, :.~, lhs, rhs) or as a dotted-call form
    if x.head == :call && length(x.args) == 3 && x.args[1] === :.~
        return true
    end
    return false
end

function _gm_tilde_parts(x)
    isdot = x.args[1] === :.~
    return x.args[2], x.args[3], isdot
end

# --- prior matching --------------------------------------------------------

# `name ~ Dist(args...)`, where `Dist` is on the closed-form whitelist, or a
# whitelisted wrapper (`filldist`/`Truncated`) around such a distribution.
#
# `params` is the set of names already bound by an earlier prior. Any prior
# whose ARGUMENTS mention an earlier parameter is rejected: that is a
# hierarchical/centered prior (`a ~ Normal(mu_a, sigma_a)`), whose gradient
# has extra chain-rule terms flowing back into `mu_a`/`sigma_a`. Those terms
# are real and omitting them would produce a silently WRONG gradient, so
# until codegen handles them explicitly this shape must fall back to general
# AD. (The non-centered form `a_raw ~ Normal(0,1); a = mu .+ s .* a_raw` is
# unaffected — its prior args are constants.)
function _gm_match_prior(name, rhs, params)
    rhs isa Expr || return nothing
    rhs.head == :call || return nothing
    d = _gm_head(rhs.args[1])                 # sees through `Distributions.Normal`
    d === nothing && return nothing

    if d in GRADMODE_PRIOR_WRAPPERS
        length(rhs.args) >= 2 || return nothing
        inner = _gm_match_prior(name, rhs.args[2], params)
        inner === nothing && return nothing
        # the wrapper's OWN extra arguments (n / lo / hi) must be constant too
        for a in rhs.args[3:end]
            _gm_param_free(a, params) || return nothing
        end
        return PriorSite(name, inner.dist, inner.args)
    end

    d in GRADMODE_PRIORS || return nothing
    for a in rhs.args[2:end]
        _gm_param_free(a, params) || return nothing
    end
    # MvNormal's covariance argument: codegen implements ONLY the unit-scale
    # case (`MvNormal(mu, I)`), whose gradient is `-v`. A scaled covariance
    # (`MvNormal(mu, 25.0 * I)`) has gradient `-v/25` — silently using the
    # unit form there produces a WRONG gradient, not a slow one, so anything
    # that is not recognizably plain `I` is rejected.
    #
    # This was a real bug caught by `check_gradmode`, not by recognition:
    # `MvNormal(zeros(p), 25.0*I)` was accepted and then differentiated as if
    # the prior were standard normal. Exactly the failure mode the verification
    # harness exists for.
    # Accepted forms are plain `I` (unit) and `c * I` / `I * c` with a NUMERIC
    # literal `c` (variance `c`, so the gradient is `-v/c`). `c` must be a
    # literal because codegen reads it at recognition time; a symbolic scale
    # would need evaluating in the user's scope, which recognition cannot do.
    if d === :MvNormal && length(rhs.args) >= 3
        _gm_mvnormal_var(rhs.args[3]) === nothing && return nothing
    end
    return PriorSite(name, d, collect(rhs.args[2:end]))
end

"""
    _gm_mvnormal_var(ex) -> Union{Float64,Nothing}

The scalar variance of an `MvNormal` covariance argument, or `nothing` if the
expression is not a recognized isotropic form. `I` -> 1.0, `25.0 * I` -> 25.0.
"""
function _gm_mvnormal_var(ex)
    _gm_head(ex) === :I && return 1.0
    # `MvNormal(mu, 4.0)` — a bare scalar variance (Distributions treats this
    # as an isotropic covariance), and `MvNormal(mu, pT(4.0))`.
    ex isa Number && return float(ex)
    (v = _gm_cast_literal(ex)) !== nothing && return v
    if ex isa Expr && ex.head == :call && _gm_head(ex.args[1]) in (:*, :.*) &&
       length(ex.args) == 4 - 1
        a, b = ex.args[2], ex.args[3]
        _gm_head(a) === :I && b isa Number && return float(b)
        _gm_head(b) === :I && a isa Number && return float(a)
        # `pT(25.0) * I` — a precision-cast literal, extremely common in this
        # package's own models since parameters are Float32-first.
        _gm_head(b) === :I && (v = _gm_cast_literal(a)) !== nothing && return v
        _gm_head(a) === :I && (v = _gm_cast_literal(b)) !== nothing && return v
    end
    return nothing
end

# `pT(25.0)` / `Float32(25.0)` -> 25.0, for any one-argument numeric cast.
function _gm_cast_literal(ex)
    (ex isa Expr && ex.head == :call && length(ex.args) == 2 &&
     ex.args[2] isa Number) || return nothing
    return float(ex.args[2])
end

# True when `ex` mentions none of `params` anywhere in its tree.
function _gm_param_free(ex, params)
    ex isa Symbol && return !(ex in params)
    ex isa Expr || return true
    for a in ex.args
        _gm_param_free(a, params) || return false
    end
    return true
end

# --- likelihood matching ---------------------------------------------------

# Recognize the observe site and return `(link, eta_name_or_nothing, terms)`.
#
# The linear predictor may be either a NAMED variable assigned earlier
# (`eta = X*beta; y ~ ...(eta)`) or written INLINE at the observe site
# (`y ~ MvNormal(X*beta, ...)`). The corpus overwhelmingly uses the inline
# form, so supporting only the named one recognizes almost nothing. Both
# route through `_gm_linear_terms`, so the same grammar and the same safety
# properties apply either way.
#
# Accepts:
#   y .~ Link.(<pred>)                   (dotted broadcast)
#   y ~ arraydist(Link.(<pred>))         (the arraydist wrapper form)
#   y ~ MvNormal(<pred>, ...)            (vector normal, predictor as the mean)
function _gm_match_link(rhs, isdot, etas, params, argnames)
    fail = (nothing, nothing, LinTerm[])
    rhs isa Expr || return fail

    # resolve a predictor expression that may be a name or an inline expression
    function pred(ex)
        if ex isa Symbol && haskey(etas, ex)
            return (ex, copy(etas[ex]))
        end
        t = _gm_linear_terms(ex, params, argnames, etas)
        t === nothing && return (nothing, nothing)
        return (nothing, t)
    end

    # unwrap `arraydist(...)`
    if rhs.head == :call && _gm_head(rhs.args[1]) === :arraydist && length(rhs.args) == 2
        return _gm_match_link(rhs.args[2], true, etas, params, argnames)
    end

    # `Link.(pred)` — a dotted call parses as Expr(:., :Link, :(tuple(pred)))
    if rhs.head == :. && length(rhs.args) == 2
        f = _gm_head(rhs.args[1])
        f === nothing && return fail
        f in GRADMODE_LINKS || return fail
        tup = rhs.args[2]
        (tup isa Expr && tup.head == :tuple && length(tup.args) == 1) || return fail
        ev, tms = pred(tup.args[1])
        tms === nothing && return fail
        return (f, ev, tms)
    end

    # `MvNormal(pred, sigma^2*I)` / `Normal(pred, sigma)` — predictor is the mean.
    if rhs.head == :call && _gm_head(rhs.args[1]) in (:MvNormal, :Normal) && length(rhs.args) >= 2
        ev, tms = pred(rhs.args[2])
        tms === nothing && return fail
        return (_gm_head(rhs.args[1]), ev, tms)
    end

    return fail
end

"""
    _gm_match_addlogprob(ex, etas, params, argnames) -> (link, eta_var, terms, response)

Match `sum(logpdf.(Dist.(<pred>), y))` — the vectorised-sum observe idiom —
returning `(nothing, nothing, LinTerm[], nothing)` if `ex` is anything else.

Deliberately rigid. `@addlogprob!` can carry ANY expression, including terms
with no closed-form derivative, so this accepts one exact shape and rejects
everything else rather than trying to interpret the expression generally.
"""
function _gm_match_addlogprob(ex, etas, params, argnames)
    fail = (nothing, nothing, LinTerm[], nothing)
    (ex isa Expr && ex.head == :call && _gm_head(ex.args[1]) === :sum &&
     length(ex.args) == 2) || return fail
    inner = ex.args[2]
    # `logpdf.(dists, y)` parses as Expr(:., :logpdf, :(tuple(dists, y)))
    (inner isa Expr && inner.head == :. && length(inner.args) == 2 &&
     _gm_head(inner.args[1]) === :logpdf) || return fail
    tup = inner.args[2]
    (tup isa Expr && tup.head == :tuple && length(tup.args) == 2) || return fail
    dists, resp = tup.args
    # response must be a data name
    (resp isa Symbol && resp in argnames) || return fail
    # `Dist.(<pred>)`
    (dists isa Expr && dists.head == :. && length(dists.args) == 2) || return fail
    f = _gm_head(dists.args[1])
    (f !== nothing && f in GRADMODE_LINKS) || return fail
    dtup = dists.args[2]
    (dtup isa Expr && dtup.head == :tuple && length(dtup.args) == 1) || return fail

    pex = dtup.args[1]
    if pex isa Symbol && haskey(etas, pex)
        return (f, pex, copy(etas[pex]), resp)
    end
    tms = _gm_linear_terms(pex, params, argnames, etas)
    tms === nothing && return fail
    return (f, nothing, tms, resp)
end

"""
    _gm_scale_param(rhs, params) -> Union{Symbol,Nothing}

The parameter supplying the observation scale in a Normal/MvNormal observe,
e.g. `sigma` in `MvNormal(eta, sigma^2 * I)`. Returns `nothing` when the scale
argument mentions no parameter (a fixed constant).

Rejects (by returning `:__reject__`, which the caller turns into an overall
reject) any scale expression mentioning MORE than one parameter, or one
parameter in a form other than `sigma`/`sigma^2`: those have different
derivatives and must not be silently treated as the simple case.
"""
function _gm_scale_param(rhs, params)
    (rhs isa Expr && rhs.head == :call && _gm_head(rhs.args[1]) in (:MvNormal, :Normal) &&
     length(rhs.args) >= 3) || return nothing
    sc = rhs.args[3]
    found = Symbol[]
    _gm_collect_params!(found, sc, params)
    isempty(found) && return nothing
    length(found) == 1 || return :__reject__
    s = found[1]
    # accept only `s`, `s^2`, `s^2 * I`, `s^2 .* I` (and the `I *`/`I .*` order)
    _gm_scale_shape_ok(sc, s) || return :__reject__
    return s
end

function _gm_collect_params!(out, ex, params)
    if ex isa Symbol
        ex in params && !(ex in out) && push!(out, ex)
        return out
    end
    ex isa Expr || return out
    for a in ex.args; _gm_collect_params!(out, a, params); end
    return out
end

function _gm_scale_shape_ok(ex, s)
    ex === s && return true
    if ex isa Expr && ex.head == :call
        f = ex.args[1]
        if f === :^ && length(ex.args) == 3 && ex.args[2] === s && ex.args[3] == 2
            return true
        end
        if f in (:*, :.*) && length(ex.args) == 3
            a, b = ex.args[2], ex.args[3]
            (_gm_head(a) === :I && _gm_scale_shape_ok(b, s)) && return true
            (_gm_head(b) === :I && _gm_scale_shape_ok(a, s)) && return true
        end
    end
    if ex isa Expr && ex.head == :. && length(ex.args) == 2
        tup = ex.args[2]
        if ex.args[1] === :* && tup isa Expr && tup.head == :tuple && length(tup.args) == 2
            a, b = tup.args
            (_gm_head(a) === :I && _gm_scale_shape_ok(b, s)) && return true
            (_gm_head(b) === :I && _gm_scale_shape_ok(a, s)) && return true
        end
    end
    return false
end

# --- linear-predictor decomposition ---------------------------------------

"""
    _gm_linear_terms(ex, params, argnames, etas) -> Union{Vector{LinTerm},Nothing}

Decompose `ex` into a sum of `LinTerm`s, or return `nothing` if it is not
manifestly a linear predictor.

This is the heart of the safety argument. Every accepted form is one where
linearity in the parameters is visible from the syntax:
  - `a + b`, `a .+ b`      → concatenate both sides' terms
  - `X * beta`             → `:matvec`, X data, beta parameter
  - `x .* beta`            → `:scaled`, only when beta is a SCALAR parameter
  - `a[idx]`, `view(a,idx)`→ `:indexed`, parameter gathered by data index
  - a bare parameter       → `:intercept`
  - a bare data name       → `:data` (pure offset)
Anything else returns `nothing`.
"""
function _gm_linear_terms(ex, params, argnames, etas)
    # bare symbol: parameter (intercept), data (offset), or an earlier eta
    if ex isa Symbol
        haskey(etas, ex) && return copy(etas[ex])
        ex in params && return [LinTerm(:intercept, ex, nothing, nothing)]
        ex in argnames && return [LinTerm(:data, nothing, ex, nothing)]
        return nothing
    end

    ex isa Expr || return nothing

    # `view(a, idx)` — the hierarchical group-effect gather
    if ex.head == :call && _gm_head(ex.args[1]) === :view && length(ex.args) == 3
        a, idx = ex.args[2], ex.args[3]
        (a isa Symbol && idx isa Symbol) || return nothing
        (a in params && idx in argnames) || return nothing
        return [LinTerm(:indexed, a, nothing, idx)]
    end

    # `a[idx]` — parameter gathered by a data index vector (group effects),
    # OR `beta[1]` — a single scalar element of a parameter vector, which acts
    # exactly like an intercept. Both are linear; they differ only in whether
    # the index is a data vector or a literal.
    if ex.head == :ref && length(ex.args) == 2
        a, idx = ex.args[1], ex.args[2]
        a isa Symbol || return nothing
        a in params || return nothing
        idx isa Integer && return [LinTerm(:intercept, a, nothing, nothing)]
        (idx isa Symbol && idx in argnames) || return nothing
        return [LinTerm(:indexed, a, nothing, idx)]
    end

    if ex.head == :call
        f = ex.args[1]

        # addition (scalar or broadcast), n-ary
        if f === :+ || f === :.+
            out = LinTerm[]
            for a in ex.args[2:end]
                t = _gm_linear_terms(a, params, argnames, etas)
                t === nothing && return nothing
                append!(out, t)
            end
            return out
        end

        # multiplication: exactly the two GLM shapes
        if (f === :* || f === :.*) && length(ex.args) == 3
            return _gm_mul_term(f, ex.args[2], ex.args[3], params, argnames)
        end
    end

    # broadcast-fused `a .+ b` parses as Expr(:., :+, :(tuple(a,b)))
    if ex.head == :. && length(ex.args) == 2
        f = ex.args[1]
        tup = ex.args[2]
        (tup isa Expr && tup.head == :tuple) || return nothing
        if f === :+
            out = LinTerm[]
            for a in tup.args
                t = _gm_linear_terms(a, params, argnames, etas)
                t === nothing && return nothing
                append!(out, t)
            end
            return out
        end
        if f === :* && length(tup.args) == 2
            return _gm_mul_term(:.*, tup.args[1], tup.args[2], params, argnames)
        end
    end

    return nothing
end

"""
    _gm_mul_term(op, l, r, params, argnames) -> Union{Vector{LinTerm},Nothing}

One multiplication term of a linear predictor. Exactly one operand must be a
parameter and the other pure data; anything else (param*param — nonlinear —
or data*data — not a parameter term at all) is rejected.

The parameter operand may be a bare name (`beta`) or a literal element of a
parameter vector (`beta[2]`), the latter being the dominant corpus style
(`beta[1] .+ beta[2] .* height`). `*` yields a `:matvec`, `.*` a `:scaled`.
"""
function _gm_mul_term(op, l, r, params, argnames)
    # a parameter operand: `beta` or `beta[k]` with literal k
    function as_param(ex)
        ex isa Symbol && ex in params && return ex
        if ex isa Expr && ex.head == :ref && length(ex.args) == 2 &&
           ex.args[1] isa Symbol && ex.args[1] in params && ex.args[2] isa Integer
            return ex.args[1]
        end
        return nothing
    end
    as_data(ex) = (ex isa Symbol && ex in argnames) ? ex : nothing

    kind = op === :* ? :matvec : :scaled
    pl, pr = as_param(l), as_param(r)
    dl, dr = as_data(l), as_data(r)
    # exactly one side parameter, the other data
    if pl !== nothing && dr !== nothing && pr === nothing
        return [LinTerm(kind, pl, dr, nothing)]
    end
    if pr !== nothing && dl !== nothing && pl === nothing
        return [LinTerm(kind, pr, dl, nothing)]
    end
    return nothing
end
