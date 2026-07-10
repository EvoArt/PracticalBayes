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

include("accumulator.jl")
include("distributions.jl")
include("modes.jl")
include("layout.jl")
include("model.jl")
include("tilde.jl")
include("compiler.jl")
include("logdensity.jl")
include("optimize.jl")
include("latent.jl")
include("gibbs.jl")
include("sample.jl")
include("predict.jl")

export @model, Model, condition, decondition, @addlogprob!
export Flat, FlatPos, filldist, arraydist, LogPoisson, BinomialLogit
export build_layout, link, invlink, Layout
export Accum, logjoint, logprior, loglikelihood_
export LogDensityFunction
export AbstractEvalMode, TraceMode, EvalMode, PriorMode, FixedMode, PointwiseMode
export evaluate, paramtype
export maximum_a_posteriori, maximum_likelihood, laplace_approximation
export PointEstimate, LaplaceApproximation, laplace_mvnormal
export AbstractLatentKernel, ModelConditional, latent_step
export Gibbs, GibbsState
export SymChain
export returned, predict, chain_draws, loglikelihood_at, pointwise_loglikelihoods

end # module PracticalBayes
