using MacroTools: MacroTools, @capture, postwalk, splitdef, combinedef

"""
    @model function name(args...) ... end

Defines a PracticalBayes model. Expands to:

  * an evaluator `_name_eval(__mode__, __acc__, args...)` whose body is the
    user's code with every `~` site rewritten (see below), and every
    `return expr` rewritten to `return (expr, __acc__)` (a bare fallthrough
    end-of-function also returns `(nothing, __acc__)`);
  * a constructor `name(args...) = Model(_name_eval, (; args...))`.

Tilde-site rewriting:

  * `x ~ D` where `x` is a bare `Symbol` becomes
    `(x, __acc__) = tilde(__mode__, Val(:x), D, x, __acc__)` if `x` is one of
    the model's argument names (so it may be data), or
    `(x, __acc__) = tilde(__mode__, Val(:x), D, getcond(__mode__, Val(:x)), __acc__)`
    otherwise (`x` is a parameter unless conditioned via `model | (; x=...)`).
  * `x[i...] ~ D` becomes
    `(x[i...], __acc__) = tilde_index(__mode__, Val(:x), x, (i...,), D, __acc__)`
    (the container `x` must already exist, e.g. `x = Vector{Float64}(undef, n)`).
  * `x .~ D` (broadcast observe/assume over an already-bound collection `x`)
    becomes a single vectorized accumulation via `tilde_dot`, not a loop.

`z := expr` declares `z` as a plain deterministic local — exactly an ordinary
Julia assignment at runtime (no tilde machinery, no accumulator involvement),
but it also marks `z` as "data available" for the REST of the model body,
exactly like a model argument. This is what lets you condition on a
computed-in-model quantity: e.g.

    @model function f(x)
        mu ~ Normal(0, 1)
        z := mu + x        # z is a plain local, computed once
        z ~ Normal(0, 1)   # observe: conditions on z's computed value
    end

Without `:=`, a bare `z ~ dist` where `z` is not a model argument is always
treated as a new parameter (see below) — `:=` is the escape hatch for "this
name already has a value by the time it's used, so treat it like an
argument," decided entirely at macro-expansion time (no runtime cost beyond
the assignment itself).

No `VarName` is constructed and no context stack is walked at any tilde site
— the mode argument alone determines behavior, via ordinary Julia dispatch.
This is the key difference from DynamicPPL's `@model`: here, dispatch on the
*type* of `__mode__` (and, for the assume path, on the *type* of a slot
looked up from an isbits NamedTuple) does all the work that DynamicPPL does
at runtime via its `AbstractContext` chain.
"""
macro model(expr)
    # `splitdef` (MacroTools) parses `function name(args...; kwargs...) body end`
    # into a Dict with keys :name, :args, :kwargs, :whereparams, :body, ... —
    # much less fragile than hand-rolling AST pattern matching for every
    # variant of function-definition syntax (short form, where-clauses, etc).
    def = splitdef(expr)
    name = def[:name]
    # `:=`'d names are folded into the same `argnames` set used for the
    # data-vs-parameter decision — from the tilde-rewriter's point of view,
    # "declared as a model argument" and "already computed via `z := expr`
    # earlier in the body" are the same thing: a name that's guaranteed to
    # already hold a concrete value by the time it's referenced.
    argnames = union(_argnames(def), _walrus_names(def[:body]))

    eval_name = Symbol("_", name, "_eval")
    # Passes, in order: (1) turn `z := expr` into plain `z = expr` — by the
    # time this runs `argnames` has already recorded `z`, so the special
    # syntax has done its job and reduces to an ordinary assignment; (2) turn
    # every `~`/`.~` into a call into tilde.jl; (3) make every `return expr`
    # also return the threaded accumulator. `_rewrite_returns` must run last
    # so it also catches any `return` the user wrote *around* a tilde
    # statement (rare, but legal Julia).
    body = _rewrite_walrus(def[:body])
    body = _rewrite_tildes(body, argnames)
    body = _rewrite_returns(body)

    # Build the evaluator's `Dict` in the same shape `splitdef` produces, so
    # `combinedef` can turn it back into a function-definition Expr. We prepend
    # `__mode__` and `__acc__` as the first two positional arguments — every
    # generated evaluator has this fixed calling convention, checked by
    # `evaluate(model, mode, acc)` in model.jl (`m.f(mode, acc, m.args...)`).
    #
    # `AbstractEvalMode`/`Accum` are referenced fully module-qualified
    # (`PracticalBayes.AbstractEvalMode`) rather than escaped: the whole
    # generated block gets wrapped in one `esc(quote ... end)` at the bottom
    # (so user identifiers like `Normal` resolve at the macro call site), and
    # `esc` is not composable — escaping a piece that's already inside a
    # fully-escaped quote produces invalid syntax ("escape (escape ...)").
    # Module-qualifying is the standard way to reach into our own package's
    # internals from inside an escaped quote without re-escaping.
    eval_def = Dict(
        :name => eval_name,
        :args => [:(__mode__::PracticalBayes.AbstractEvalMode), :(__acc__::PracticalBayes.Accum), def[:args]...],
        :kwargs => def[:kwargs],
        :whereparams => def[:whereparams],
        :body => quote
            $(body)
            return (nothing, __acc__)  # reached only if the user's body falls through without an explicit `return`
        end,
    )
    eval_fn = combinedef(eval_def)

    # The user-facing constructor keeps the original argument list (no
    # __mode__/__acc__) and just packages the call args into a NamedTuple for
    # `Model.args`. Missing/positional data vs parameters is NOT decided here
    # — it's decided per tilde-site by `_tilde_expansion` below, using
    # `argnames` (static, macro-time) plus `getcond` (runtime, but folds away
    # for a concrete conditioning-pattern type).
    ctor_args = def[:args]
    ctor_kwargs = def[:kwargs]
    nt_pairs = [:($(_argsym(a)) = $(_argsym(a))) for a in ctor_args]

    ctor_def = Dict(
        :name => name,
        :args => ctor_args,
        :kwargs => ctor_kwargs,
        :whereparams => def[:whereparams],
        :body => quote
            return PracticalBayes.Model($(eval_name), (; $(nt_pairs...)))
        end,
    )
    ctor_fn = combinedef(ctor_def)

    # `esc(...)` wraps the *entire* generated code so it is spliced into the
    # macro call site's environment rather than into PracticalBayes' own
    # module scope — necessary so that e.g. `Normal` in the user's model body
    # resolves via the user's `using Distributions`, not ours.
    return esc(quote
        $(eval_fn)
        $(ctor_fn)
    end)
end

# Pulls the bare argument name out of either `x` or `x::T` argument syntax.
# `splitdef`'s :args entries are Exprs like `:(x::Int)` for typed args.
_argsym(a) = a isa Symbol ? a : (a isa Expr && a.head == :(::) ? a.args[1] : error("unsupported arg form: $a"))
_argexpr(a) = _argsym(a)

# The set of the model's own positional argument names, e.g. for
# `@model function foo(y, n) ... end` this is `Set([:y, :n])`. Used only to
# decide, at macro-expansion time, whether a bare-symbol tilde LHS *could* be
# data (passed in as an argument, possibly `missing`) vs is definitely a
# parameter (any other name — e.g. one only ever assigned inside the body).
function _argnames(def)
    names = Symbol[]
    for a in get(def, :args, [])
        push!(names, _argsym(a))
    end
    return Set(names)
end

# Collects every name declared via `z := expr` anywhere in the model body,
# by walking the (pre-rewrite) body Expr tree with a plain recursive scan —
# deliberately NOT `postwalk`-and-replace like the other passes, since this
# is read-only (we're only gathering names here, not producing a new Expr;
# the actual `z := expr` -> `z = expr` rewrite happens separately in
# `_rewrite_walrus`, after this set has already been folded into `argnames`).
# `:=` parses as `Expr(:(:=), lhs, rhs)`, distinct from Julia's `Expr(:(=), ...)`
# plain assignment, so this can't accidentally pick up ordinary `z = expr`.
function _walrus_names(body)
    names = Symbol[]
    MacroTools.postwalk(body) do x
        if x isa Expr && x.head == :(:=) && x.args[1] isa Symbol
            push!(names, x.args[1])
        end
        x  # always return `x` unchanged — this postwalk is used only for its side effect
    end
    return Set(names)
end

# Turns every `z := expr` into a plain `z = expr` — by the time this runs,
# `_walrus_names` has already recorded `z` into `argnames`, so all `:=` needs
# to do at THIS point is stop being special syntax and become a normal
# variable assignment that the rest of the function body can use.
function _rewrite_walrus(body)
    return postwalk(body) do x
        if x isa Expr && x.head == :(:=)
            :($(x.args[1]) = $(x.args[2]))
        else
            x
        end
    end
end

"""
    _rewrite_tildes(body, argnames) -> Expr

Walks `body` and rewrites every `~` expression. `argnames` is the set of the
model's positional argument names, used to decide whether a bare-symbol LHS
is potentially-observed (in argnames) or purely a parameter (not in argnames).
"""
function _rewrite_tildes(body, argnames)
    # `postwalk` visits every subexpression bottom-up and replaces it with
    # whatever the closure returns (or leaves it alone via the `else x` arm).
    # `@capture(x, lhs_ ~ rhs_)` is MacroTools pattern matching: it succeeds
    # only when `x` is literally an Expr of the form `lhs ~ rhs`, binding
    # `lhs`/`rhs` to the matched sub-Exprs. This is far more robust than
    # checking `x.head == :call && x.args[1] == :~` by hand.
    return postwalk(body) do x
        if @capture(x, lhs_ ~ rhs_)
            _tilde_expansion(lhs, rhs, argnames)
        elseif @capture(x, lhs_ .~ rhs_)
            _dot_tilde_expansion(lhs, rhs, argnames)
        else
            x  # not a tilde statement — leave it untouched
        end
    end
end

# Rewrites a single `lhs ~ rhs` site. `rhs` is left completely alone (it's
# just the distribution expression, evaluated as-is at eval time — this is
# where e.g. `Normal(mu, sigma)` gets constructed on every call, same as
# DynamicPPL). Two shapes of `lhs` are supported; anything else is a macro
# expansion-time (not runtime) error, so mistakes show up immediately.
function _tilde_expansion(lhs, rhs, argnames)
    if lhs isa Symbol
        # Plain `x ~ D`. Whether `x`'s value comes from the model's own call
        # arguments (`x` itself, which might be `missing`/`nothing` meaning
        # "actually a parameter") or from conditioning (`getcond`) is decided
        # HERE, at macro-expansion time, by simple Set membership — this is
        # what lets the generated code skip DynamicPPL's runtime
        # `inargnames`/context-walk check entirely.
        s = lhs
        valueexpr = s in argnames ? s : :(PracticalBayes.getcond(__mode__, $(Val)($(QuoteNode(s)))))
        return quote
            ($(s), __acc__) = PracticalBayes.tilde(__mode__, $(Val)($(QuoteNode(s))), $(rhs), $(valueexpr), __acc__)
        end
    elseif lhs isa Expr && lhs.head == :ref
        # Indexed `x[i, j, ...] ~ D`: `lhs.args[1]` is the container
        # expression, `lhs.args[2:end]` are the index expressions (as written
        # by the user — could be loop variables, literals, etc). We require
        # the container itself to be a bare Symbol (not e.g. `foo().x[i]`) so
        # that `Val(:x)` unambiguously names one layout slot family.
        container = lhs.args[1]
        idx = lhs.args[2:end]
        container isa Symbol || error("indexed tilde LHS must index a bare variable, got: $lhs")
        s = container
        # `tilde_index` both returns the drawn/observed value AND mutates
        # `container[k] = x` itself (see tilde.jl), so the LHS-assignment
        # target here is just a throwaway — we don't need `container[idx...]`
        # on the left because the container is already updated in place.
        return quote
            (_pb_tmp_, __acc__) = PracticalBayes.tilde_index(
                __mode__, $(Val)($(QuoteNode(s))), $(container), ($(idx...),), $(rhs), __acc__
            )
        end
    else
        error("unsupported tilde LHS: $lhs")
    end
end

# `y .~ Dbroadcast` — MVP only supports this as an OBSERVE (`y` already bound
# to data); see the big comment block at the top of tilde.jl for why assume
# via `.~` is deliberately unsupported. Note this returns only `__acc__`
# (no drawn value to assign back), unlike the two functions above.
#
# Unlike plain `~`, we can't tell "data" from "parameter" at runtime here by
# checking for `nothing`/`missing` — a `.~` LHS is typically a pre-declared
# container (e.g. `x = Vector{Float64}(undef, n)`), which is never `nothing`
# even when it's meant to hold UNKNOWN values. So instead of a runtime check,
# we reject the ambiguous case statically: `.~` is only allowed when `lhs` is
# one of the model's own argument names (so it's unambiguously either real
# data or `missing`/`nothing` data, exactly like plain `~`'s argnames check).
# Anything else is almost certainly the "assume via .~" pattern we don't
# support, so we fail at macro-expansion time with a clear message rather
# than silently computing a logpdf against uninitialized memory at runtime.
function _dot_tilde_expansion(lhs, rhs, argnames)
    lhs isa Symbol || error("`.~` LHS must be a bare variable naming an existing collection, got: $lhs")
    lhs in argnames || error(
        "`.~` assume (unknown LHS) is not supported: `$lhs` is not one of this model's " *
        "arguments, so it can't be observed data. Use `x[i] ~ dist` for indexed families " *
        "of unknowns, or an array distribution (`product_distribution`, `MvNormal`) via plain `~`.",
    )
    s = lhs
    return quote
        (__acc__) = PracticalBayes.tilde_dot(__mode__, $(Val)($(QuoteNode(s))), $(lhs), $(rhs), __acc__)
    end
end

"""
    _rewrite_returns(body) -> Expr

Rewrites every `return expr` in `body` to `return (expr, __acc__)`, so the
generated evaluator always returns `(retval, acc)`. Does not descend into
nested function definitions.
"""
function _rewrite_returns(body)
    # Same postwalk-and-replace trick as `_rewrite_tildes`. We don't need to
    # special-case nested function/closure definitions here because a nested
    # `function ... return ... end` would introduce its own local `return`,
    # which Julia scopes to that inner function anyway — rewriting it to
    # reference the outer `__acc__` would be wrong, but in practice model
    # bodies don't define nested named functions with their own returns, and
    # if they do the rewritten inner return simply captures the outer
    # `__acc__` by closure, which is still a plain accumulator value, not a
    # bug — just something to be aware of if you see surprising behavior.
    return postwalk(body) do x
        if x isa Expr && x.head == :return
            retexpr = isempty(x.args) ? :(nothing) : x.args[1]
            return :(return ($(retexpr), __acc__))
        else
            x
        end
    end
end
