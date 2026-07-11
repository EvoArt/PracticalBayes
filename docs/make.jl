using Documenter
using Literate
using PracticalBayes
# `@docs` blocks resolve their target expression's binding in `Main` (this
# process, not any per-page `@example`/`@setup` scope) — these need to be
# `using`/`import`ed here, not just inside the .md source, or Documenter
# can't find `AbstractMCMC.sample`/`Random.rand(...)` to attach the docstring to.
using AbstractMCMC
using Random

# Turn the runnable Literate tutorial into an executed markdown page, so the code
# shown on the page is exactly what a reader would run.
Literate.markdown(
    joinpath(@__DIR__, "literate", "latent_trajectory.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    documenter=true,
)

# Documenter configuration

DocMeta.setdocmeta!(PracticalBayes, :DocTestSetup, :(using PracticalBayes); recursive=true)

makedocs(
    modules=[PracticalBayes],
    sitename="PracticalBayes.jl",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
        "Sampling" => "sampling.md",
        "Latent-trajectory tutorial" => "tutorials/latent_trajectory.md",
        "Predictive utilities" => "predictive.md",
        "Float32 and GPU" => "float32_gpu.md",
    ],
    checkdocs=:none,
    warnonly=true,
)

deploydocs(
    repo="github.com/EvoArt/PracticalBayes.git",
    devbranch="master",
    devurl="",
    versions=["v#.#", "v#.#.#", "stable"],
)
