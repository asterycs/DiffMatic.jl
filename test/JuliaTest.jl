# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using ForwardDiff

using LinearAlgebra: tr, diagm, I

@testset "test interface: valid user supplied function arguments" begin
    @matrix A B
    @vector x

    fmt = JuliaFunc([x, B, A])

    expr = A * B * x

    fun = to_std(expr; format = fmt)
    fun = eval(fun)

    A = Matrix(I, 2, 2)
    B = Matrix(I, 2, 2)
    x = [1; 1]

    @test fun(x, B, A) == x
end

@testset "test interface: too many user supplied function arguments yields warning" begin
    @matrix A B C
    @vector x

    fmt = JuliaFunc([x, B, C, A])

    expr = A * B * x

    @test_logs (:warn, "Ignoring unused variables: [\"C\"]") fun =
        to_std(expr; format = fmt)
    fun = eval(fun)

    A = Matrix(I, 2, 2)
    B = Matrix(I, 2, 2)
    x = [1; 1]

    @test fun(x, B, A) == x
end

@testset "test interface: more than one occurrence per argument throws" begin
    @matrix A B C
    @vector x

    fmt = JuliaFunc([A, x, B, A])

    expr = A * B * x

    @test_throws DomainError to_std(expr; format = fmt)
end

@testset "test interface: throws on missing arguments" begin
    @matrix A B C
    @vector x

    fmt = JuliaFunc([A, x])

    expr = A * B * x

    @test_throws DomainError to_std(expr; format = fmt)
end

@testset "test Julia function" begin
    @matrix A B C
    @vector x y z

    x̂ = [
        0.055
        0.395
        0.821
    ]

    ŷ = [
        0.442
        0.630
        0.176
    ]

    ẑ = [
        0.578
        0.845
        0.711
    ]

    Â = [
        0.023 0.136 0.181
        0.443 0.132 0.576
        0.786 0.198 0.583
    ]

    B̂ = [
        0.570 0.987 0.124
        0.855 0.400 0.196
        0.111 0.469 0.406
    ]

    Ĉ = [
        0.884 0.947 0.401
        0.999 0.623 0.473
        0.415 0.483 0.969
    ]

    @testset "function tr(x*x')" begin
        f(x) = tr(x * x')

        jfun = eval(to_std(f(x); format = dc.JuliaFunc()))

        @test jfun(x̂) ≈ f(x̂)
    end

    @testset "function tr(A*B'*C)" begin
        f(A, B, C) = tr(A * B' * C)

        jfun = eval(to_std(f(A, B, C); format = dc.JuliaFunc()))

        @test jfun(Â, B̂, Ĉ) ≈ f(Â, B̂, Ĉ)
    end

    @testset "function log(A' * x)" begin
        f(A, x) = log.(A' * x)

        jfun = eval(to_std(f(A, x); format = dc.JuliaFunc()))

        @test jfun(Â, x̂) ≈ f(Â, x̂)
    end

    @testset "gradient of x'*x" begin
        f(x) = x' * x

        jgrad = eval(to_std(gradient(f(x), x); format = dc.JuliaFunc()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(f, x̂)
    end

    @testset "gradient of sum(x.^2)^2" begin
        f(x) = sum(x .^ 2)^2

        jgrad = eval(to_std(gradient(f(x), x); format = dc.JuliaFunc()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(f, x̂)
    end

    @testset "gradient of cos(tr(x * x'))" begin
        f(x) = cos(tr(x * x'))

        jgrad = eval(to_std(gradient(f(x), x); format = dc.JuliaFunc()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(f, x̂)
    end

    @testset "gradient of log.(A*x)' * x" begin
        f(A, x) = log.(A * x)' * x

        jgrad = eval(to_std(gradient(f(A, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, x̂) ≈ ForwardDiff.gradient(x -> f(Â, x), x̂)
    end

    @testset "gradient of log.(A*x)' * (A .* B)' * x" begin
        f(A, B, x) = log.(A * x)' * (A .* B)' * x

        jgrad = eval(to_std(gradient(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, B̂, x̂) ≈ ForwardDiff.gradient(x -> f(Â, B̂, x), x̂)
    end

    @testset "gradient of x' * sin.(A * x)" begin
        f(A, x) = x' * sin.(A * x)

        jgrad = eval(to_std(gradient(f(A, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, x̂) ≈ ForwardDiff.gradient(x -> f(Â, x), x̂)
    end

    @testset "jacobian of sin(A * x + y - z)" begin
        f(A, x, y, z) = sin.(A * x + y - z)

        jjac = eval(to_std(jacobian(f(A, x, y, z), x); format = dc.JuliaFunc()))

        @test jjac(Â, x̂, ŷ, ẑ) ≈ ForwardDiff.jacobian(x -> f(Â, x, ŷ, ẑ), x̂)
    end

    @testset "jacobian of (A .* B) * C * x)' * x * x" begin
        f(A, B, C, x) = ((A .* B) * C * x)' * x * x

        jjac = eval(to_std(jacobian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "jacobian of (A .* B) * log.(A * x)" begin
        f(A, B, x) = (A .* B) * log.(A * x)

        jjac = eval(to_std(jacobian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, x), x̂)
    end

    @testset "hessian of sum(log.(x)) * x' * x" begin
        f(x) = sum(log.(x)) * x' * x

        jhess = eval(to_std(hessian(f(x), x); format = dc.JuliaFunc()))

        @test jhess(x̂) ≈ ForwardDiff.hessian(x -> f(x), x̂)
    end

    @testset "hessian of x' * sin.(A * x)" begin
        f(A, x) = x' * sin.(A * x)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end
end
