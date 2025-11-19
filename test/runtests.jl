# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using TestItems
using TestItemRunner

@testitem "Aqua" begin
    include("TestUtils.jl")
    include("Aqua.jl")
end

@testitem "RicciTest" begin
    include("TestUtils.jl")
    include("RicciTest.jl")
end

@testitem "ForwardTest" begin
    include("TestUtils.jl")
    include("ForwardTest.jl")
end

@testitem "SimplifyTest" begin
    include("TestUtils.jl")
    include("SimplifyTest.jl")
end

@testitem "StdTest" begin
    include("TestUtils.jl")
    include("StdTest.jl")
end

@testitem "StdStrTest" begin
    include("TestUtils.jl")
    include("StdStrTest.jl")
end

@testitem "JuliaTest" begin
    include("TestUtils.jl")
    include("JuliaTest.jl")
end

@testitem "DocTest" begin
    using Documenter
    DocMeta.setdocmeta!(DiffMatic, :DocTestSetup, :(using DiffMatic); recursive = true)
    doctest(DiffMatic)
end

@run_package_tests
