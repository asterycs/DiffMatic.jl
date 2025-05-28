@testset "test gradient in standard notation" begin
    @scalar a b
    @vector x y c
    @matrix A B C

    @test to_std(gradient(x' * x, x)) == "2x"
    @test to_std(gradient(tr(x * x'), x)) == "2x"
    @test to_std(gradient((y .* c)' * x, x)) == "y ⊙ c"
    @test to_std(gradient((x .* c)' * x, x)) == "2(x ⊙ c)"
    @test to_std(gradient((x + y)' * x, x)) == "2x + y"
    @test to_std(gradient((x - y)' * x, x)) == "2x - y"
    @test to_std(gradient(sin(tr(x * x')), x)) == "cos(xᵀx)2x"
    @test to_std(gradient(cos(tr(x * x')), x)) == "(-1)sin(xᵀx)2x"
    @test to_std(gradient(tr(A), x)) == "vec(0)"
    @test to_std(gradient(abs(x' * x), x)) == "sgn(xᵀx)2x"
    @test to_std(gradient(x' * B' * A * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std(gradient((A' * B * x)' * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std(gradient(a * sin.(y)' * x, x)) == "asin(y)"
    @test to_std(gradient(a * sin.(x)' * y, x)) == "a(cos(x) ⊙ y)"
    @test to_std(gradient(sin.(y)' * x * a, x)) == "asin(y)"
    @test to_std(gradient(sin.(x)' * y * a, x)) == "a(cos(x) ⊙ y)"
    @test to_std(gradient(x' * sin.(y) * a, x)) == "asin(y)"
    @test to_std(gradient(y' * sin.(x) * a, x)) == "a(cos(x) ⊙ y)"
    @test to_std(gradient(sin.(x .* y)' * x, x)) == "(cos(x ⊙ y) ⊙ y ⊙ x) + sin(x ⊙ y)"
    @test to_std(gradient(sum(x), x)) == "vec(1)"
    @test to_std(gradient(2 * sum(x), x)) == "2vec(1)"
    @test to_std(gradient(sum(2 * x), x)) == "2vec(1)"
    @test to_std(gradient(2 * sum(sin.(x)), x)) == "2cos(x)"
    @test to_std(gradient(sum(2 * sin.(x)), x)) == "2cos(x)"
    @test to_std(gradient(2 * sum(cos.(A * x + y)), x)) == "(-2)Aᵀsin(Ax + y)"
    @test to_std(gradient(sum(2 * cos.(A * x + y)), x)) == "(-2)Aᵀsin(Ax + y)"
    @test to_std(gradient(sum(x .^ 2), x)) == "2x"
    @test to_std(gradient(sum(x .^ 3), x)) == "3x^2"
    @test to_std(gradient(sum(x)^2, x)) == "2sum(xᵀ)vec(1)"
    @test to_std(gradient(sum(x .^ 2)^2, x)) == "2sum(xᵀ^2)2x"
    @test to_std(gradient(sum((x + y) .^ 2), x)) == "2(x + y)"
    @test to_std(gradient(sum((x .* y) .^ 2), x)) == "2(x ⊙ y ⊙ y)"
    @test to_std(gradient(sum((A * x - y) .^ 2), x)) == "2Aᵀ(Ax - y)"
    @test to_std(gradient((x' * A * x) ^ (-2), x)) == "(-2)(xᵀAᵀx)^(-3)(Aᵀx + Ax)"
    @test to_std(gradient((x' * A * x) ^ 2, x)) == "2xᵀAᵀx(Aᵀx + Ax)"
    @test to_std(gradient(((A .* B) * C * x)' * x, x)) == "(A ⊙ B)Cx + Cᵀ(Aᵀ ⊙ Bᵀ)x"
    @test to_std(gradient(((A .* (B .* C)) * C * x)' * x, x)) ==
          "(B ⊙ C ⊙ A)Cx + Cᵀ(Bᵀ ⊙ Cᵀ ⊙ Aᵀ)x"
    @test to_std(gradient(sum((A .* B) * C * x), x)) == "Cᵀ(Aᵀ ⊙ Bᵀ)vec(1)"
    to_std(gradient((x .^ 2 .* y)' * c, x)) == "2x ⊙ y ⊙ c"
end

@testset "test Jacobian in standard notation" begin
    @matrix A B C
    @vector x y

    @test to_std(jacobian(A * x, x)) == "A"
    @test to_std(jacobian(A' * x, x)) == "Aᵀ"
    @test to_std(jacobian(abs.(x), x)) == "diag(sgn(x))I"
    @test to_std(jacobian(sin.(A * x + y), x)) == "diag(cos(Ax + y))A"
    @test to_std(jacobian(((A .* B) * C * x)' * x * x, x)) ==
          "xᵀCᵀ(Aᵀ ⊙ Bᵀ)xI + x(xᵀCᵀ(Aᵀ ⊙ Bᵀ) + xᵀ(A ⊙ B)C)"
end

@testset "test derivative in standard notation" begin
    @matrix A B C X
    @vector x y z

    @test to_std(derivative(sum(-y .* (X*z)), X)) == "(-1)zyᵀ"
    @test to_std(derivative(sum((A .* B) * C * x), x)) == "vec(1)ᵀ(A ⊙ B)C"
end

@testset "test Hessian in standard notation" begin
    @matrix A
    @vector x

    @test to_std(hessian(x' * A * x, x)) == "Aᵀ + A"
    @test to_std(hessian(2 * x' * A * x, x)) == "2Aᵀ + 2A"
    @test to_std(hessian(2 * x' * x, x)) == "4I"
end
