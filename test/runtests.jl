# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using Documenter
using Test

include("TestUtils.jl")

@testset "Aqua" begin
    include("Aqua.jl")
end

@testset "RicciTest" begin
    include("RicciTest.jl")
end

@testset "ForwardTest" begin
    include("ForwardTest.jl")
end

@testset "SimplifyTest" begin
    include("SimplifyTest.jl")
end

@testset "StdTest" begin
    include("StdTest.jl")
end

@testset "StdStrTest" begin
    include("StdStrTest.jl")
end

@testset "JuliaTest" begin
    include("JuliaTest.jl")
end

DocMeta.setdocmeta!(DiffMatic, :DocTestSetup, :(using DiffMatic); recursive = true)
doctest(DiffMatic)
