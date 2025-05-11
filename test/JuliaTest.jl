using ForwardDiff

@testset "test Julia function" begin
    @vector x

    x̂ = [
        0.0552
        0.395
        0.821
        0.558
        0.048
        0.144
        0.519
        0.376
        0.264
        0.001
    ]

    @testset "gradient of x'*x" begin
        jgrad = eval(to_std(gradient(x' * x, x); format = dc.Julia()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(x -> x' * x, x̂)
    end

    @testset "gradient of sum(x.^2)^2" begin
        jgrad = eval(to_std(gradient(sum(x .^ 2)^2, x); format = dc.Julia()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(x -> sum(x .^ 2)^2, x̂)
    end

    @testset "gradient of cos(tr(x * x'))" begin
        jgrad = eval(to_std(gradient(cos(tr(x * x')), x); format = dc.Julia()))

        @test jgrad(x̂) ≈ ForwardDiff.gradient(x -> cos(tr(x * x')), x̂)
    end
end
