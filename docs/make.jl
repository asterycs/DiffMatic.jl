using Documenter
using DiffMatic

DocMeta.setdocmeta!(DiffMatic, :DocTestSetup, :(using DiffMatic); recursive = true)

makedocs(
    sitename = "DiffMatic",
    format = Documenter.HTML(),
    modules = [DiffMatic],
    pages = ["Introduction" => "index.md", "API Reference" => "api.md"],
    checkdocs = :exports,
)

deploydocs(repo = "https://github.com/asterycs/DiffMatic.jl")
