using Documenter
using PracticalBayes

# Documenter configuration

DocMeta.setdocmeta!(PracticalBayes, :DocTestSetup, :(using PracticalBayes); recursive=true)

makedocs(
    modules=[PracticalBayes],
    sitename="PracticalBayes.jl",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
        "Sampling" => "sampling.md",
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
