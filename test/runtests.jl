using Test
using PracticalBayes

@testset "PracticalBayes.jl" begin
    include("compiler.jl")
    include("layout.jl")
    include("logdensity.jl")
    include("distributions.jl")
    include("ad_backends.jl")
    include("optimize.jl")
    include("gibbs.jl")
    include("check_depends.jl")
    include("gradmode_recognize.jl")
    include("gradmode_wiring.jl")
    include("sample.jl")
    include("jittered_nsteps.jl")
    include("predict.jl")
    include("turing_comparison.jl")
    include("gpu/cuda.jl")
end
