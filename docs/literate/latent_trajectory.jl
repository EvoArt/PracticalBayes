# # Sampling a latent-trajectory model
#
# In this tutorial we fit a model that combines continuous parameters with a
# large discrete latent variable — a hidden state trajectory — using a `Gibbs`
# sampler that updates the parameters with NUTS and the trajectory with a custom
# latent kernel.
#
# This is the pattern PracticalBayes' [`AbstractLatentKernel`](@ref) interface is
# built for: a block of variables that should not go through HMC (here, a
# discrete infection-state matrix) is updated by a sampler you supply, once per
# Gibbs sweep, while the continuous parameters are differentiated as usual and
# the latent variable is held constant during their gradient steps.
#
# The epidemic model itself — the simulator, the trajectory log-likelihood, and
# the forward-filtering/backward-sampling (iFFBS) kernel — comes from
# [EpidemicTrajectories.jl](https://github.com/EvoArt/EpidemicTrajectories). We
# focus here on wiring it into a PracticalBayes model. The model is a two-state
# susceptible/infected epidemic: each animal is susceptible or infected and moves
# between the two states each day; we observe an imperfect test on some days and
# never see the infection state directly.
#
# We will:
#
# 1. simulate data from known parameters,
# 2. reserve the hidden trajectory as a discrete latent variable,
# 3. write a latent kernel that resamples it,
# 4. compose a `Gibbs` sampler and run it, and
# 5. plot the posterior against the values we simulated from.

using PracticalBayes
using EpidemicTrajectories
using Distributions
using Random
using AdvancedHMC: NUTS
import AbstractMCMC
using StableRNGs: StableRNG
using Statistics: mean
using FlexiChains
using CairoMakie

# ## Simulate the data
#
# We simulate ten pens of eight animals over eighty days from known parameters:
# the external and within-pen forces of infection ``\alpha`` and ``\beta``, the
# mean infectious period ``m``, the initial infection frequency ``\nu``, and the
# test sensitivity ``\theta``.

true_pars = (; α = 0.01, β = 0.02, m = 6.0)
true_ν = 0.10
true_θ = 0.80

n_pens, n_per_pen, n_times = 10, 8, 80
n_ind = n_pens * n_per_pen
group = repeat(1:n_pens; inner = n_per_pen)

ss = SI
rates = TwoStateSI()

rng = StableRNG(2024)
states, data = simulate_trajectory(rng, ss, rates, true_pars, group,
                                   [1 - true_ν, true_ν]; n_times = n_times)

rams = DiagnosticTest(; sensitivity = p -> p.θ, specificity = p -> 1.0, positive_code = 1)
R = simulate_observations(rng, (rams,), (; θ = true_θ), ss, states)[1]

observed_days = 1:6:n_times
Rmask = fill(-1, size(R))
Rmask[:, observed_days] .= R[:, observed_days]

# ## Reserve the hidden trajectory as a discrete latent variable
#
# The hidden state matrix `X` (animals × time) is represented as a matrix-valued
# discrete distribution. PracticalBayes routes a discrete `~` site to the value
# store — a latent block updated by a kernel, held constant during HMC gradients
# — instead of putting it in the parameter vector. Its density is supplied
# through `@addlogprob!` in the model body, so the distribution itself
# contributes nothing.

struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_ind::Int
    n_times::Int
end
Base.size(d::TrajectoryLatent) = (d.n_ind, d.n_times)
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
Distributions.rand(rng::Random.AbstractRNG, d::TrajectoryLatent) = zeros(Int, d.n_ind, d.n_times)

# ## Write the latent kernel
#
# A latent kernel is a subtype of [`AbstractLatentKernel`](@ref) with one method,
# [`latent_step`](@ref), returning a `NamedTuple` of new values for its block. It
# reads the current values of the other blocks from `c.values` and, here, calls
# EpidemicTrajectories' `ffbs_sweep!` to resample the whole trajectory.

struct iFFBS{RB<:RateBundle} <: PracticalBayes.AbstractLatentKernel
    ss::StateSpace
    rates::RB
    group::Vector{Int}
    tests::Tuple{DiagnosticTest}
    results::Tuple{Matrix{Int}}
end

function PracticalBayes.latent_step(rng, k::iFFBS, block_names, c::ModelConditional)
    pars = (; α = c.values.α, β = c.values.β, m = c.values.m̃ + 1.0, θ = c.values.θ)
    X = copy(c.values.X)
    d = EpidemicTrajectories.make_data(X, k.group)
    model = (; state_space = k.ss, rates = k.rates, pars = pars)
    EpidemicTrajectories.ffbs_sweep!(rng, model, d, k.tests, k.results;
                                     initial_prob = [1 - c.values.ν, c.values.ν])
    return (; X = d.states)
end

# ``\nu`` and ``\theta`` have closed-form conditional posteriors, so we give them
# conjugate kernels too — the same one-struct, one-method pattern.

struct NuKernel <: PracticalBayes.AbstractLatentKernel end
function PracticalBayes.latent_step(rng, ::NuKernel, block_names, c::ModelConditional)
    x1 = @view c.values.X[:, 1]
    n_inf = count(==(1), x1)
    return (; ν = rand(rng, Beta(1 + n_inf, 1 + length(x1) - n_inf)))
end

struct ThetaKernel <: PracticalBayes.AbstractLatentKernel
    results::Matrix{Int}
end
function PracticalBayes.latent_step(rng, k::ThetaKernel, block_names, c::ModelConditional)
    X, R = c.values.X, k.results
    n_pos = n_inf = 0
    for t in axes(R, 2), i in axes(R, 1)
        (R[i, t] < 0 || X[i, t] != 1) && continue
        n_inf += 1
        R[i, t] == 1 && (n_pos += 1)
    end
    return (; θ = rand(rng, Beta(1 + n_pos, 1 + n_inf - n_pos)))
end

# ## The model
#
# The continuous parameters get priors; the hidden trajectory is the latent
# variable; and the trajectory log-likelihood is added with `@addlogprob!`. The
# latent `X` is read here as a constant, so gradients flow only through the
# parameters. We reparameterise the infectious period as ``m = \tilde m + 1`` so
# the recovery probability stays below one.

@model function cattle_model(Rmask, group, ss, rates, n_ind, n_times)
    α ~ Gamma(1, 1)
    β ~ Gamma(1, 1)
    m̃ ~ Gamma(2, 4)
    m := m̃ + 1.0
    ν ~ Beta(1, 1)
    θ ~ Beta(1, 1)

    X ~ TrajectoryLatent(n_ind, n_times)

    pars = (; α = α, β = β, m = m, θ = θ)
    data = EpidemicTrajectories.make_data(X, group)
    model = (; state_space = ss, rates = rates, pars = pars)
    @addlogprob! EpidemicTrajectories.trajectory_loglik(pars, model, data)
end

# ## Compose the sampler and run
#
# Each variable is assigned to exactly one `Gibbs` block: NUTS for the continuous
# transmission parameters, the conjugate kernels for ``\nu`` and ``\theta``, and
# the iFFBS kernel for the hidden trajectory. `sample` runs the sweeps and
# returns a chain; `discard_initial` drops burn-in.

m = cattle_model(Rmask, group, ss, rates, n_ind, n_times)

spl = Gibbs(
    (:α, :β, :m̃) => NUTS(0.8),
    :ν => NuKernel(),
    :θ => ThetaKernel(Rmask),
    :X => iFFBS(ss, rates, group, (rams,), (Rmask,)),
)

X0, _ = simulate_trajectory(StableRNG(99), ss, rates, true_pars, group,
                            [1 - true_ν, true_ν]; n_times = n_times)
init = (; X = copy(X0), α = 0.05, β = 0.05, m̃ = 4.0, ν = 0.1, θ = 0.7)

chn = AbstractMCMC.sample(StableRNG(7), m, spl, 1000;
                          init = init, n_adapts = 400, discard_initial = 500)

# ## Check the recovery
#
# We compare the posterior means to the values we simulated from. The infectious
# period is ``m = \tilde m + 1``.

posterior_m = vec(chn[:m̃]) .+ 1.0
recovered = (α = vec(chn[:α]), β = vec(chn[:β]), m = posterior_m,
             ν = vec(chn[:ν]), θ = vec(chn[:θ]))
truths = (α = true_pars.α, β = true_pars.β, m = true_pars.m, ν = true_ν, θ = true_θ)

for name in (:α, :β, :m, :ν, :θ)
    println(rpad(name, 3), " mean ", round(mean(getfield(recovered, name)); digits = 4),
            "  (truth ", getfield(truths, name), ")")
end

# ## Plot
#
# We plot each parameter's posterior density with a line at the value we
# simulated from.

fig = Figure(size = (900, 500))
for (i, name) in enumerate((:α, :β, :m, :ν, :θ))
    ax = Axis(fig[fldmod1(i, 3)...]; title = string(name), ylabel = "density")
    density!(ax, getfield(recovered, name))
    vlines!(ax, [getfield(truths, name)]; color = :firebrick, linewidth = 2)
end
fig

# Every posterior concentrates around the value we simulated from. The `Gibbs`
# sampler updated the continuous parameters with NUTS and the hidden trajectory
# with our iFFBS kernel, and the latent trajectory never entered a gradient
# calculation — exactly what the latent-kernel interface is for.
