module PracticalBayes

using Random: Random, AbstractRNG
using AbstractPPL: AbstractPPL
# `ForwardDiff` is the hardcoded default `adtype` for `maximum_a_posteriori`/
# `maximum_likelihood`/`laplace_approximation` (optimize.jl). DifferentiationInterface
# resolves `AutoForwardDiff()` via a package extension that only activates
# once ForwardDiff is actually loaded IN THE SESSION — being merely a
# `[deps]` entry in Project.toml is not enough, since Julia doesn't eagerly
# load a dependency's code just because it's declared; `using` here forces
# it, so `using PracticalBayes` alone is sufficient for the AD defaults to work.
using ForwardDiff: ForwardDiff

# AdvancedHMC is a hard dependency (the HMC/NUTS samplers a Gibbs block wraps).
# Re-export its samplers and AbstractMCMC's `sample` so `using PracticalBayes`
# alone is enough to build a model, pick a sampler, and run it — no second
# `using AdvancedHMC`/`using AbstractMCMC` needed at the call site.
using AbstractMCMC: AbstractMCMC, sample
using AdvancedHMC: HMC, NUTS, HMCDA

include("accumulator.jl")
include("distributions.jl")
include("modes.jl")
include("layout.jl")
include("model.jl")
include("tilde.jl")
include("compiler.jl")
# Stage 2 GLM recognizer. Loaded so it is testable, but INERT: nothing in the
# package calls `recognize_glm` and it generates no code. Wiring it into
# `@model` is a later, separate step gated on a runtime gradient check.
include("gradmode_recognize.jl")
include("gradmode_codegen.jl")
include("logdensity.jl")
include("optimize.jl")
include("latent.jl")
include("conjugate.jl")
include("save_states.jl")
include("gibbs.jl")
include("check_depends.jl")
include("nuts_to_hmc.jl")
include("adaptive_hmc.jl")
include("progress.jl")
include("sample.jl")
include("predict.jl")

export @model, @static_model, Model, condition, decondition, @addlogprob!
export Flat, FlatPos, filldist, arraydist, LogPoisson, BinomialLogit
export build_layout, link, invlink, Layout
export Accum, logjoint, logprior, loglikelihood_
export LogDensityFunction
export GradMode, check_gradmode
export AbstractEvalMode, TraceMode, EvalMode, PriorMode, FixedMode, PointwiseMode
export evaluate, paramtype
export maximum_a_posteriori, maximum_likelihood, laplace_approximation
export PointEstimate, LaplaceApproximation, laplace_mvnormal
export AbstractLatentKernel, ModelConditional, latent_step
export ConjugateGibbs
export Gibbs, GibbsState
export check_depends, DependsReport
export SaveToChain, SaveToBuffer, SaveToDisk, write_state_chunk!, read_states
export SymChain
# Re-exported so `using PracticalBayes` is self-contained for running inference:
export sample, HMC, NUTS, HMCDA, NUTSthenHMC, AdaptiveHMC
export default_progress
export returned, predict, chain_draws, loglikelihood_at, pointwise_loglikelihoods

end # module PracticalBayes
