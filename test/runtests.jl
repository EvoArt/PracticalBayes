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
    include("sample.jl")
    include("predict.jl")
    include("turing_comparison.jl")
end
