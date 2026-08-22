# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using ForwardDiff

using LinearAlgebra: tr, diag, diagm, norm, I

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

    @testset "function tr(diagm(x) * A)" begin
        f(A, x) = tr(diagm(x) * A)

        jfun = eval(to_std(f(A, x); format = dc.JuliaFunc()))

        @test only(jfun(Â, x̂)) ≈ f(Â, x̂)
    end

    @testset "function log(A' * x)" begin
        f(A, x) = log.(A' * x)

        jfun = eval(to_std(f(A, x); format = dc.JuliaFunc()))

        @test jfun(Â, x̂) ≈ f(Â, x̂)
    end

    @testset "function sum(x .* y .* z)" begin
        f(x, y, z) = sum(x .* y .* z)

        jfun = eval(to_std(f(x, y, z); format = dc.JuliaFunc()))

        @test only(jfun(x̂, ŷ, ẑ)) ≈ f(x̂, ŷ, ẑ)
    end

    @testset "function sum(sin.(A * x) .* cos.(B * x) .* x)" begin
        f(A, B, x) = sum(sin.(A * x) .* cos.(B * x) .* x)

        jfun = eval(to_std(f(A, B, x); format = dc.JuliaFunc()))

        @test only(jfun(Â, B̂, x̂)) ≈ f(Â, B̂, x̂)
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

    @testset "function exp.(A * x)" begin
        f(A, x) = exp.(A * x)

        jfun = eval(to_std(f(A, x); format = dc.JuliaFunc()))

        @test jfun(Â, x̂) ≈ f(Â, x̂)
    end

    @testset "function exp(x' * x)" begin
        f(x) = exp(x' * x)

        jfun = eval(to_std(f(x); format = dc.JuliaFunc()))

        @test only(jfun(x̂)) ≈ f(x̂)
    end

    @testset "gradient of sum(exp.(A * x))" begin
        f(A, x) = sum(exp.(A * x))

        jgrad = eval(to_std(gradient(f(A, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, x̂) ≈ ForwardDiff.gradient(x -> f(Â, x), x̂)
    end

    @testset "gradient of exp(x' * x)" begin
        f(x) = exp(x' * x)

        jgrad = eval(to_std(gradient(f(x), x); format = dc.JuliaFunc()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(f, x̂)
    end

    @testset "gradient of y' * exp.(A * x)" begin
        f(A, x, y) = y' * exp.(A * x)

        jgrad = eval(to_std(gradient(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jgrad(Â, x̂, ŷ) ≈ ForwardDiff.gradient(x -> f(Â, x, ŷ), x̂)
    end

    @testset "jacobian of exp.(A * x)" begin
        f(A, x) = exp.(A * x)

        jjac = eval(to_std(jacobian(f(A, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, x), x̂)
    end

    @testset "graph: gradient of x' * sin.(A * x)" begin
        f(A, x) = x' * sin.(A * x)

        jgrad = eval(to_std(gradient(f(A, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, x̂) ≈ ForwardDiff.gradient(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of sin.(A * x)' * cos.(B * x)" begin
        f(A, B, x) = sin.(A * x)' * cos.(B * x)

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sin.(A * x)' * B * cos.(C * x)" begin
        f(A, B, C, x) = sin.(A * x)' * B * cos.(C * x)

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: hessian of sin.(A * x)' * B * C * cos.(C * x)" begin
        f(A, B, C, x) = sin.(A * x)' * B * C * cos.(C * x)

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: hessian of sin.(A * x)' * B * C * B * cos.(C * x)" begin
        f(A, B, C, x) = sin.(A * x)' * B * C * B * cos.(C * x)

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: hessian of (A * x)' * sin.(B * x)" begin
        f(A, B, x) = (A * x)' * sin.(B * x)

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of (A * x)' * sin.(A * x)" begin
        f(A, x) = (A * x)' * sin.(A * x)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of y' * sin.(A * x)" begin
        f(A, x, y) = y' * sin.(A * x)

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: hessian of sin.(A * x)' * y" begin
        f(A, x, y) = sin.(A * x)' * y

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: hessian of (x .* y)' * sin.(A * x)" begin
        f(A, x, y) = (x .* y)' * sin.(A * x)

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: hessian of (x .* y .* z)' * sin.(A * x)" begin
        f(A, x, y, z) = (x .* y .* z)' * sin.(A * x)

        jhess = eval(to_std(hessian(f(A, x, y, z), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ, ẑ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ, ẑ), x̂)
    end

    @testset "graph: hessian of (x .* sin.(A * x))' * y" begin
        f(A, x, y) = (x .* sin.(A * x))' * y

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: gradient of sum(sin.(A * x) .* y)" begin
        f(A, x, y) = sum(sin.(A * x) .* y)

        jgrad = eval(to_std(gradient(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jgrad(Â, x̂, ŷ) ≈ ForwardDiff.gradient(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: gradient of sum(sin.(A * x) .* cos.(B * x))" begin
        f(A, B, x) = sum(sin.(A * x) .* cos.(B * x))

        jgrad = eval(to_std(gradient(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, B̂, x̂) ≈ ForwardDiff.gradient(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x) .* cos.(B * x))" begin
        f(A, B, x) = sum(sin.(A * x) .* cos.(B * x))

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x) .* cos.(B * x) .* z)" begin
        f(A, B, x, z) = sum(sin.(A * x) .* cos.(B * x) .* z)

        jhess = eval(to_std(hessian(f(A, B, x, z), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂, ẑ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x, ẑ), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x) .* x .* y)" begin
        f(A, x, y) = sum(sin.(A * x) .* x .* y)

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: hessian of sum(x .* sin.(A * x))" begin
        f(A, x) = sum(x .* sin.(A * x))

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of sum((A * x) .* (B * x))" begin
        f(A, B, x) = sum((A * x) .* (B * x))

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x) .* (B * y))" begin
        f(A, B, x, y) = sum(sin.(A * x) .* (B * y))

        jhess = eval(to_std(hessian(f(A, B, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x, ŷ), x̂)
    end

    @testset "graph: hessian of sum((A * sin.(B * x)) .* z)" begin
        f(A, B, x, z) = sum((A * sin.(B * x)) .* z)

        jhess = eval(to_std(hessian(f(A, B, x, z), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂, ẑ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x, ẑ), x̂)
    end

    @testset "graph: hessian of sum(exp.(A * x))" begin
        f(A, x) = sum(exp.(A * x))

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of sum(log.(A * x))" begin
        f(A, x) = sum(log.(A * x))

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of exp(x' * x)" begin
        f(x) = exp(x' * x)

        jhess = eval(to_std(hessian(f(x), x); format = dc.JuliaFunc()))

        @test jhess(x̂) ≈ ForwardDiff.hessian(f, x̂)
    end

    @testset "graph: hessian of sin(x' * A * x)" begin
        f(A, x) = sin(x' * A * x)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of log(x' * A * x)" begin
        f(A, x) = log(x' * A * x)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of exp(sin(x' * A * x))" begin
        f(A, x) = exp(sin(x' * A * x))

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of (x' * A * x) ^ 2" begin
        f(A, x) = (x' * A * x)^2

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of x' * A * B * x" begin
        f(A, B, x) = x' * A * B * x

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of x' * A * B * C * x" begin
        f(A, B, C, x) = x' * A * B * C * x

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: hessian of (A * x)' * B * (C * x)" begin
        f(A, B, C, x) = (A * x)' * B * (C * x)

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: hessian of sin.(A * x)' * B * sin.(B * x)" begin
        f(A, B, x) = sin.(A * x)' * B * sin.(B * x)

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of (A * x)' * diagm(y) * (B * x)" begin
        f(A, B, x, y) = (A * x)' * diagm(y) * (B * x)

        jhess = eval(to_std(hessian(f(A, B, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x, ŷ), x̂)
    end

    @testset "graph: hessian of x' * diagm(sin.(A * x)) * x" begin
        f(A, x) = x' * diagm(sin.(A * x)) * x

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of sum(diagm(y) * sin.(A * x))" begin
        f(A, x, y) = sum(diagm(y) * sin.(A * x))

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: hessian of sum(diagm(sin.(A * x)) * y)" begin
        f(A, x, y) = sum(diagm(sin.(A * x)) * y)

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: hessian of (B * sin.(A * x))' * cos.(C * x)" begin
        f(A, B, C, x) = (B * sin.(A * x))' * cos.(C * x)

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: hessian of (B * sin.(A * x))' * (C * cos.(A * x))" begin
        f(A, B, C, x) = (B * sin.(A * x))' * (C * cos.(A * x))

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: hessian of norm(sin.(A * x), 2)" begin
        f(A, x) = norm(sin.(A * x), 2)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x)) ^ 2" begin
        f(A, x) = sum(sin.(A * x))^2

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x)) * sum(cos.(B * x))" begin
        f(A, B, x) = sum(sin.(A * x)) * sum(cos.(B * x))

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x)) * (x' * x)" begin
        f(A, x) = sum(sin.(A * x)) * (x' * x)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of ((A * x)' * sin.(B * x)) * (x' * x)" begin
        f(A, B, x) = ((A * x)' * sin.(B * x)) * (x' * x)

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of tr(A) * (x' * sin.(B * x))" begin
        f(A, B, x) = tr(A) * (x' * sin.(B * x))

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(A * sin.(B * x))" begin
        f(A, B, x) = sum(A * sin.(B * x))

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(A * B * sin.(C * x))" begin
        f(A, B, C, x) = sum(A * B * sin.(C * x))

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: gradient of sum(A * sin.(B * x))" begin
        f(A, B, x) = sum(A * sin.(B * x))

        jgrad = eval(to_std(gradient(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, B̂, x̂) ≈ ForwardDiff.gradient(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: function sum(A * x)" begin
        f(A, x) = sum(A * x)

        jfun = eval(to_std(f(A, x); format = dc.JuliaFunc()))

        @test only(jfun(Â, x̂)) ≈ f(Â, x̂)
    end

    @testset "graph: function sum(A * B * x)" begin
        f(A, B, x) = sum(A * B * x)

        jfun = eval(to_std(f(A, B, x); format = dc.JuliaFunc()))

        @test only(jfun(Â, B̂, x̂)) ≈ f(Â, B̂, x̂)
    end

    @testset "graph: gradient of sum(sin.(A * sin.(B * x)))" begin
        f(A, B, x) = sum(sin.(A * sin.(B * x)))

        jgrad = eval(to_std(gradient(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, B̂, x̂) ≈ ForwardDiff.gradient(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * sin.(B * x)))" begin
        f(A, B, x) = sum(sin.(A * sin.(B * x)))

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * cos.(B * x)))" begin
        f(A, B, x) = sum(sin.(A * cos.(B * x)))

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: hessian of y' * sin.(A * sin.(B * x))" begin
        f(A, B, x, y) = y' * sin.(A * sin.(B * x))

        jhess = eval(to_std(hessian(f(A, B, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x, ŷ), x̂)
    end

    @testset "graph: jacobian of sin.(A * x)" begin
        f(A, x) = sin.(A * x)

        jjac = eval(to_std(jacobian(f(A, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, x), x̂)
    end

    @testset "graph: jacobian of A * sin.(B * x)" begin
        f(A, B, x) = A * sin.(B * x)

        jjac = eval(to_std(jacobian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: jacobian of A * sin.(B * cos.(C * x))" begin
        f(A, B, C, x) = A * sin.(B * cos.(C * x))

        jjac = eval(to_std(jacobian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    @testset "graph: jacobian of sin.(A * sin.(B * x))" begin
        f(A, B, x) = sin.(A * sin.(B * x))

        jjac = eval(to_std(jacobian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: jacobian of sin.(A * x) .* cos.(B * x)" begin
        f(A, B, x) = sin.(A * x) .* cos.(B * x)

        jjac = eval(to_std(jacobian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: jacobian of diagm(y) * sin.(A * x)" begin
        f(A, x, y) = diagm(y) * sin.(A * x)

        jjac = eval(to_std(jacobian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jjac(Â, x̂, ŷ) ≈ ForwardDiff.jacobian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: jacobian of diagm(sin.(A * x)) * cos.(B * x)" begin
        f(A, B, x) = diagm(sin.(A * x)) * cos.(B * x)

        jjac = eval(to_std(jacobian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, x̂) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: jacobian of diagm(y) * A * diagm(z) * B * sin.(C * x)" begin
        f(A, B, C, x, y, z) = diagm(y) * A * diagm(z) * B * sin.(C * x)

        jjac = eval(to_std(jacobian(f(A, B, C, x, y, z), x); format = dc.JuliaFunc()))

        @test jjac(Â, B̂, Ĉ, x̂, ŷ, ẑ) ≈ ForwardDiff.jacobian(x -> f(Â, B̂, Ĉ, x, ŷ, ẑ), x̂)
    end

    @testset "graph: hessian of (sin.(A * x) .* y)' * B * cos.(C * x)" begin
        f(A, B, C, x, y) = (sin.(A * x) .* y)' * B * cos.(C * x)

        jhess = eval(to_std(hessian(f(A, B, C, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x, ŷ), x̂)
    end

    @testset "graph: hessian of (sin.(A * x) .* y)' * B * (cos.(C * x) .* z)" begin
        f(A, B, C, x, y, z) = (sin.(A * x) .* y)' * B * (cos.(C * x) .* z)

        jhess = eval(to_std(hessian(f(A, B, C, x, y, z), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂, ŷ, ẑ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x, ŷ, ẑ), x̂)
    end

    # Four pendants on a single node.
    @testset "graph: hessian of sum(x .* y .* z .* sin.(A * x))" begin
        f(A, x, y, z) = sum(x .* y .* z .* sin.(A * x))

        jhess = eval(to_std(hessian(f(A, x, y, z), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂, ŷ, ẑ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ, ẑ), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x) .* (B * C * y))" begin
        f(A, B, C, x, y) = sum(sin.(A * x) .* (B * C * y))

        jhess = eval(to_std(hessian(f(A, B, C, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x, ŷ), x̂)
    end

    @testset "graph: hessian of sum(sin.(A * x) .* (B * y) .* (C * z))" begin
        f(A, B, C, x, y, z) = sum(sin.(A * x) .* (B * y) .* (C * z))

        jhess = eval(to_std(hessian(f(A, B, C, x, y, z), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂, ŷ, ẑ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x, ŷ, ẑ), x̂)
    end

    @testset "graph: hessian of c * (x' * sin.(A * x))" begin
        @scalar c

        ĉ = 1.7

        f(A, c, x) = c * (x' * sin.(A * x))

        jhess = eval(to_std(hessian(f(A, c, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, ĉ, x), x̂)
    end

    @testset "graph: hessian of sin.(A' * x)' * cos.(B' * x)" begin
        f(A, B, x) = sin.(A' * x)' * cos.(B' * x)

        jhess = eval(to_std(hessian(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: gradient of x' * (A .* B) * x" begin
        f(A, B, x) = x' * (A .* B) * x

        jgrad = eval(to_std(gradient(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, B̂, x̂) ≈ ForwardDiff.gradient(x -> f(Â, B̂, x), x̂)
    end

    # Pendants that are powers rather than plain vectors.
    @testset "graph: hessian of ((A * x) .^ 2)' * y" begin
        f(A, x, y) = ((A * x) .^ 2)' * y

        jhess = eval(to_std(hessian(f(A, x, y), x); format = dc.JuliaFunc()))

        @test jhess(Â, ŷ) ≈ ForwardDiff.hessian(x -> f(Â, x, ŷ), x̂)
    end

    @testset "graph: hessian of sum((A * x) .^ 3)" begin
        f(A, x) = sum((A * x) .^ 3)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    # A long chain with no pendants at all.
    @testset "graph: hessian of x' * A * B * C * A * x" begin
        f(A, B, C, x) = x' * A * B * C * A * x

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    # A node sitting between two edges, both of which reach the boundary.
    @testset "graph: hessian of (A * x)' * diagm(sin.(B * x)) * (C * x)" begin
        f(A, B, C, x) = (A * x)' * diagm(sin.(B * x)) * (C * x)

        jhess = eval(to_std(hessian(f(A, B, C, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, B̂, Ĉ, x̂) ≈ ForwardDiff.hessian(x -> f(Â, B̂, Ĉ, x), x̂)
    end

    # Three pendants on a node with a single boundary end, so a vector result.
    @testset "graph: gradient of sum(sin.(A * x) .* cos.(B * x) .* x)" begin
        f(A, B, x) = sum(sin.(A * x) .* cos.(B * x) .* x)

        jgrad = eval(to_std(gradient(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, B̂, x̂) ≈ ForwardDiff.gradient(x -> f(Â, B̂, x), x̂)
    end

    # A node carrying only pendants, with no edge at all.
    @testset "graph: gradient of sum(x .* y .* z)" begin
        f(x, y, z) = sum(x .* y .* z)

        jgrad = eval(to_std(gradient(f(x, y, z), x); format = dc.JuliaFunc()))

        @test jgrad(ŷ, ẑ) ≈ ForwardDiff.gradient(x -> f(x, ŷ, ẑ), x̂)
    end

    # The same matrix on both pendants of a node.
    @testset "graph: hessian of sum(sin.(A * x) .* sin.(A * x))" begin
        f(A, x) = sum(sin.(A * x) .* sin.(A * x))

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "graph: gradient of tr(diagm(x) * A)" begin
        f(A, x) = tr(diagm(x) * A)

        jgrad = eval(to_std(gradient(f(A, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â) ≈ ForwardDiff.gradient(x -> f(Â, x), x̂)
    end

    @testset "graph: gradient of tr(diagm(sin.(A * x)) * B)" begin
        f(A, B, x) = tr(diagm(sin.(A * x)) * B)

        jgrad = eval(to_std(gradient(f(A, B, x), x); format = dc.JuliaFunc()))

        @test jgrad(Â, B̂, x̂) ≈ ForwardDiff.gradient(x -> f(Â, B̂, x), x̂)
    end

    @testset "graph: jacobian of diag(diagm(x) * A)" begin
        f(A, x) = diag(diagm(x) * A)

        jjac = eval(to_std(jacobian(f(A, x), x); format = dc.JuliaFunc()))

        @test_broken jjac(Â) ≈ ForwardDiff.jacobian(x -> f(Â, x), x̂)
    end

    @testset "graph: hessian of norm(A * x, 2)" begin
        f(A, x) = norm(A * x, 2)

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end

    @testset "hessian of sum(exp.(A * x))" begin
        f(A, x) = sum(exp.(A * x))

        jhess = eval(to_std(hessian(f(A, x), x); format = dc.JuliaFunc()))

        @test jhess(Â, x̂) ≈ ForwardDiff.hessian(x -> f(Â, x), x̂)
    end
end
