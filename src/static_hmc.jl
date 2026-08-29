# A self-contained HMC/NUTS implementation for trimmable binaries.
#
# ## Why this exists alongside AdvancedHMC
#
# AdvancedHMC is the right sampler for interactive use and is what `sample()`
# drives. It is not compilable by `juliac --trim`: `PhasePoint`'s constructor
# uses `@argcheck`, whose error path builds a message from an `Any`, leaving
# `ArgCheck.build_error` unresolved. That is on the core sampling path, so it
# cannot be avoided by choosing different options.
#
# Measured on a two-site model, verifier errors under `--trim=safe`:
#
#   baseline (sample(), AdvancedHMC, ForwardDiff)      204
#   + layout frozen into a const (no build_layout)      12
#   + gradient supplied instead of AD                    2   <- both ArgCheck
#   + this sampler instead of AdvancedHMC                0
#
# Nothing here depends on any package: Base only. It deliberately implements
# the same algorithms AdvancedHMC does (Hoffman & Gelman 2014 Alg. 6 for NUTS,
# Betancourt 2017 for multinomial sampling and the generalised U-turn), so a
# model can be developed against `sample()` and then compiled with this without
# the posterior changing.
#
# ## What it does not do
#
# No Gibbs, no multiple chains, no vectorised (matrix-valued) phase points, no
# progress reporting, no chain object. A trimmed binary wants a fixed model, a
# fixed sampler and an array of draws; anything richer belongs on the
# interactive path.

"""
    StaticHMCResult

Draws and diagnostics from [`static_nuts`](@ref) or [`static_hmc`](@ref).

`draws` is `n_params x n_draws`. Divergences and acceptance are reported
separately for warmup and sampling: warmup divergences are expected while dual
averaging walks the step size down, and only `n_divergent` (post-warmup) bears
on whether the draws can be trusted.
"""
struct StaticHMCResult
    draws::Matrix{Float64}
    logp::Vector{Float64}
    step_size::Float64
    inv_mass::Vector{Float64}
    n_divergent::Int          # post-warmup
    accept_rate::Float64      # post-warmup
    n_divergent_warmup::Int
    accept_rate_warmup::Float64
    mean_depth::Float64       # post-warmup, NUTS only (0.0 for fixed-L HMC)
end

const _STATIC_MAX_DEPTH = 10
const _STATIC_DELTA_MAX = 1000.0

# --- dual averaging (Nesterov, as used by Stan and AdvancedHMC) -------------
mutable struct _DualAverage
    mu::Float64
    log_eps_bar::Float64
    h_bar::Float64
    counter::Int
    gamma::Float64
    t0::Float64
    kappa::Float64
    delta::Float64
end

_DualAverage(eps0::Float64, delta::Float64) =
    _DualAverage(log(10.0 * eps0), 0.0, 0.0, 0, 0.05, 10.0, 0.75, delta)

function _adapt!(da::_DualAverage, accept_prob::Float64)
    da.counter += 1
    m = Float64(da.counter)
    eta = 1.0 / (m + da.t0)
    da.h_bar = (1.0 - eta) * da.h_bar + eta * (da.delta - accept_prob)
    log_eps = da.mu - sqrt(m) / da.gamma * da.h_bar
    mk = m^(-da.kappa)
    da.log_eps_bar = mk * log_eps + (1.0 - mk) * da.log_eps_bar
    return exp(log_eps)
end

_final_eps(da::_DualAverage) = exp(da.log_eps_bar)

# --- Welford variance, for the diagonal metric ------------------------------
mutable struct _Welford
    n::Int
    mean::Vector{Float64}
    m2::Vector{Float64}
end

_Welford(d::Int) = _Welford(0, zeros(d), zeros(d))

function _accumulate!(w::_Welford, x::Vector{Float64})
    w.n += 1
    @inbounds for i in eachindex(x)
        delta = x[i] - w.mean[i]
        w.mean[i] += delta / w.n
        w.m2[i] += delta * (x[i] - w.mean[i])
    end
    return w
end

function _variance(w::_Welford)
    d = length(w.mean)
    out = Vector{Float64}(undef, d)
    n = w.n
    @inbounds for i in 1:d
        # Stan's regularisation toward 1: with few samples the raw variance is
        # a poor metric, and a bad metric is worse than an uninformative one.
        v = n > 1 ? w.m2[i] / (n - 1) : 1.0
        out[i] = (n / (n + 5.0)) * v + 1e-3 * (5.0 / (n + 5.0))
    end
    return out
end

_reset!(w::_Welford) = (w.n = 0; fill!(w.mean, 0.0); fill!(w.m2, 0.0); w)

# --- leapfrog ---------------------------------------------------------------
@inline function _kinetic(p::Vector{Float64}, inv_mass::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(p)
        s += p[i] * p[i] * inv_mass[i]
    end
    return 0.5 * s
end

@inline function _leapfrog!(x::Vector{Float64}, p::Vector{Float64},
                            g::Vector{Float64}, eps::Float64,
                            inv_mass::Vector{Float64}, gradf!::F) where {F}
    @inbounds for i in eachindex(p)
        p[i] += 0.5 * eps * g[i]
    end
    @inbounds for i in eachindex(x)
        x[i] += eps * inv_mass[i] * p[i]
    end
    lp = gradf!(g, x)
    @inbounds for i in eachindex(p)
        p[i] += 0.5 * eps * g[i]
    end
    return lp
end

# --- RNG --------------------------------------------------------------------
# xoshiro256++ with Box-Muller, so the sampler needs nothing from Random and a
# run is reproducible from its seed alone.
mutable struct _Rng
    s0::UInt64
    s1::UInt64
    s2::UInt64
    s3::UInt64
    has_spare::Bool
    spare::Float64
end

function _Rng(seed::Int)
    # SplitMix64 to expand the seed into the four words xoshiro needs.
    z = UInt64(seed) + 0x9e3779b97f4a7c15
    nxt = () -> begin
        z += 0x9e3779b97f4a7c15
        y = z
        y = (y ⊻ (y >> 30)) * 0xbf58476d1ce4e5b9
        y = (y ⊻ (y >> 27)) * 0x94d049bb133111eb
        return y ⊻ (y >> 31)
    end
    return _Rng(nxt(), nxt(), nxt(), nxt(), false, 0.0)
end

@inline _rotl(x::UInt64, k::Int) = (x << k) | (x >> (64 - k))

@inline function _next(r::_Rng)
    result = _rotl(r.s0 + r.s3, 23) + r.s0
    t = r.s1 << 17
    r.s2 ⊻= r.s0
    r.s3 ⊻= r.s1
    r.s1 ⊻= r.s2
    r.s0 ⊻= r.s3
    r.s2 ⊻= t
    r.s3 = _rotl(r.s3, 45)
    return result
end

@inline _rand_uniform(r::_Rng) = Float64(_next(r) >> 11) * (1.0 / 9007199254740992.0)

function _rand_normal(r::_Rng)
    if r.has_spare
        r.has_spare = false
        return r.spare
    end
    u1 = _rand_uniform(r)
    u1 < 1e-300 && (u1 = 1e-300)
    u2 = _rand_uniform(r)
    mag = sqrt(-2.0 * log(u1))
    r.spare = mag * sin(2π * u2)
    r.has_spare = true
    return mag * cos(2π * u2)
end

# --- initial step size ------------------------------------------------------
"""
Heuristic initial step size: double or halve until the acceptance probability
of a single leapfrog step crosses 0.5. Same procedure as Hoffman & Gelman's
`FindReasonableEpsilon` and AdvancedHMC's `find_good_stepsize`.
"""
function _find_initial_eps(x0::Vector{Float64}, gradf!::F,
                           inv_mass::Vector{Float64}, rng::_Rng) where {F}
    d = length(x0)
    eps = 1.0
    x = copy(x0)
    g = Vector{Float64}(undef, d)
    lp = gradf!(g, x)
    p = Vector{Float64}(undef, d)

    # One trial leapfrog at the current eps, returning its acceptance prob.
    trial = function (e::Float64)
        copyto!(x, x0)
        lp0 = gradf!(g, x)
        @inbounds for i in 1:d
            p[i] = _rand_normal(rng) / sqrt(inv_mass[i])
        end
        h0 = -lp0 + _kinetic(p, inv_mass)
        lp1 = _leapfrog!(x, p, g, e, inv_mass, gradf!)
        h1 = isfinite(lp1) ? -lp1 + _kinetic(p, inv_mass) : Inf
        dh = h0 - h1
        return isfinite(dh) ? min(1.0, exp(dh)) : 0.0
    end

    # Pick a direction from the first trial, then move that way until the
    # acceptance probability crosses 0.5. Deciding the direction ONCE is what
    # makes this terminate: alternating doubling and halving on each iteration
    # can oscillate forever around the crossing point.
    a = trial(eps)
    grow = a > 0.5
    for _ in 1:100
        if grow
            a > 0.5 || break
            eps < 1e5 || break
            eps *= 2.0
        else
            a <= 0.5 || break
            eps > 1e-10 || break
            eps *= 0.5
        end
        a = trial(eps)
    end
    return eps
end

# --- NUTS tree --------------------------------------------------------------
# The subtree's edge states, its multinomially sampled proposal, and the running
# totals the U-turn checks and acceptance statistic need.
struct _Tree
    x_minus::Vector{Float64}
    p_minus::Vector{Float64}
    g_minus::Vector{Float64}
    x_plus::Vector{Float64}
    p_plus::Vector{Float64}
    g_plus::Vector{Float64}
    x_prop::Vector{Float64}
    lp_prop::Float64
    log_weight::Float64      # logsumexp of -H over the subtree
    p_sum::Vector{Float64}
    sum_alpha::Float64
    n_alpha::Int
    diverged::Bool
    stopped::Bool            # a U-turn was found INSIDE this subtree
end

@inline function _logaddexp(a::Float64, b::Float64)
    a == -Inf && return b
    b == -Inf && return a
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

# Generalised U-turn (Betancourt 2017): check the summed momentum against both
# ends rather than the endpoint separation, which is what makes this correct
# for a non-identity metric.
@inline function _no_u_turn(p_sum::Vector{Float64}, p_minus::Vector{Float64},
                            p_plus::Vector{Float64}, inv_mass::Vector{Float64})
    a = 0.0
    b = 0.0
    @inbounds for i in eachindex(p_sum)
        v = inv_mass[i] * p_sum[i]
        a += v * p_minus[i]
        b += v * p_plus[i]
    end
    return a > 0.0 && b > 0.0
end

function _build_tree(x::Vector{Float64}, p::Vector{Float64}, g::Vector{Float64},
                     log_u::Float64, depth::Int, direction::Int, eps::Float64,
                     inv_mass::Vector{Float64}, h0::Float64, gradf!::F,
                     rng::_Rng) where {F}
    if depth == 0
        xc = copy(x); pc = copy(p); gc = copy(g)
        lp = _leapfrog!(xc, pc, gc, direction * eps, inv_mass, gradf!)
        h = isfinite(lp) ? -lp + _kinetic(pc, inv_mass) : Inf
        dh = h0 - h
        diverged = !isfinite(dh) || dh < -_STATIC_DELTA_MAX
        lw = isfinite(-h) ? -h : -Inf
        a = isfinite(dh) ? min(1.0, exp(dh)) : 0.0
        return _Tree(xc, pc, gc, copy(xc), copy(pc), copy(gc), copy(xc), lp,
                     lw, copy(pc), a, 1, diverged, false)
    end

    left = _build_tree(x, p, g, log_u, depth - 1, direction, eps, inv_mass,
                       h0, gradf!, rng)
    (left.diverged || left.stopped) && return left

    right = if direction == -1
        _build_tree(left.x_minus, left.p_minus, left.g_minus, log_u, depth - 1,
                    direction, eps, inv_mass, h0, gradf!, rng)
    else
        _build_tree(left.x_plus, left.p_plus, left.g_plus, log_u, depth - 1,
                    direction, eps, inv_mass, h0, gradf!, rng)
    end

    # Progressive multinomial sampling between the two subtrees.
    lw = _logaddexp(left.log_weight, right.log_weight)
    x_prop = left.x_prop
    lp_prop = left.lp_prop
    if !right.diverged && right.log_weight > -Inf
        if log(_rand_uniform(rng)) < right.log_weight - lw
            x_prop = right.x_prop
            lp_prop = right.lp_prop
        end
    end

    d = length(x)
    p_sum = Vector{Float64}(undef, d)
    @inbounds for i in 1:d
        p_sum[i] = left.p_sum[i] + right.p_sum[i]
    end

    x_minus, p_minus, g_minus, x_plus, p_plus, g_plus = if direction == -1
        right.x_minus, right.p_minus, right.g_minus, left.x_plus, left.p_plus, left.g_plus
    else
        left.x_minus, left.p_minus, left.g_minus, right.x_plus, right.p_plus, right.g_plus
    end

    stopped = right.diverged || right.stopped ||
              !_no_u_turn(p_sum, p_minus, p_plus, inv_mass)

    return _Tree(x_minus, p_minus, g_minus, x_plus, p_plus, g_plus,
                 x_prop, lp_prop, lw, p_sum,
                 left.sum_alpha + right.sum_alpha,
                 left.n_alpha + right.n_alpha,
                 right.diverged, stopped)
end
