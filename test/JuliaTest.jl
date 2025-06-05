using ForwardDiff

using LinearAlgebra: tr, diagm, I

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
        jfun = eval(to_std(tr(x*x'); format = dc.JuliaFunc()))

        @test jfun(x̂) ≈ tr(x̂*x̂')
    end

    @testset "function tr(A*B'*C)" begin
        jfun = eval(to_std(tr(A*B'*C); format = dc.JuliaFunc()))

        @test jfun(B̂, Ĉ, Â) ≈ tr(Â * B̂' * Ĉ)
    end

    @testset "function log(A' * x)" begin
        jfun = eval(to_std(log.(A' * x); format = dc.JuliaFunc()))

        @test jfun(Â, x̂) ≈ log.(Â' * x̂)
    end

    @testset "gradient of x'*x" begin
        jgrad = eval(to_std(gradient(x' * x, x); format = dc.JuliaFunc()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(x -> x' * x, x̂)
    end

    @testset "gradient of sum(x.^2)^2" begin
        jgrad = eval(to_std(gradient(sum(x .^ 2)^2, x); format = dc.JuliaFunc()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(x -> sum(x .^ 2)^2, x̂)
    end

    @testset "gradient of cos(tr(x * x'))" begin
        jgrad = eval(to_std(gradient(cos(tr(x * x')), x); format = dc.JuliaFunc()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(x -> cos(tr(x * x')), x̂)
    end

    @testset "jacobian of sin(A * x + y - z)" begin
        jjac = eval(to_std(jacobian(sin.(A * x + y - z), x); format = dc.JuliaFunc()))

        @test jjac(Â, x̂, ŷ, ẑ) ≈ ForwardDiff.jacobian(x -> sin.(Â * x + ŷ - ẑ), x̂)
    end

    @testset "jacobian of (A .* B) * C * x)' * x * x" begin
        jjac =
            eval(to_std(jacobian(((A .* B) * C * x)' * x * x, x); format = dc.JuliaFunc()))

        @test jjac(x̂, Ĉ, Â, B̂) ≈
              ForwardDiff.jacobian(x -> ((Â .* B̂) * Ĉ * x)' * x * x, x̂)
    end
end
