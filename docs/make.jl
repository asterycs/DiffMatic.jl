using Documenter
using DocumenterCitations

using DiffMatic

DocMeta.setdocmeta!(DiffMatic, :DocTestSetup, :(using DiffMatic); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "bibliography.bib"))

makedocs(
    sitename = "DiffMatic",
    format = Documenter.HTML(),
    modules = [DiffMatic],
    pages = [
        "Quick Start" => "index.md",
        "Detailed Usage" => ["usage.md", "examples.md"],
        "API Reference" => "api.md",
    ],
    plugins = [bib],
    checkdocs = :exports,
    doctest = false, # doctests are run separately
)

deploydocs(repo = "https://github.com/asterycs/DiffMatic.jl")
