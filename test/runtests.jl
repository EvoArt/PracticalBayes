using Test
using PracticalBayes

@testset "PracticalBayes.jl" begin
    include("compiler.jl")
    include("layout.jl")
    include("logdensity.jl")
    include("ad_backends.jl")
    include("turing_comparison.jl")
end
