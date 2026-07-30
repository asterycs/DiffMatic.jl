# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using LinearAlgebra: tr, diag, diagm, norm

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
    @test to_std(gradient(sin.(x .* y)' * x, x)) == "y ⊙ cos(x ⊙ y) ⊙ x + sin(x ⊙ y)"
    @test to_std(gradient(sum(x), x)) == "vec(1)"
    @test to_std(gradient(2 * sum(x), x)) == "2vec(1)"
    @test to_std(gradient(sum(2 * x), x)) == "2vec(1)"
    @test to_std(gradient(2 * sum(sin.(x)), x)) == "2cos(x)"
    @test to_std(gradient(sum(2 * sin.(x)), x)) == "2cos(x)"
    @test to_std(gradient(diag(A)'*x, x)) == "(A ⊙ I)vec(1)" # TODO: Output 'diag(A)'
    @test to_std(gradient(2 * sum(cos.(A * x + y)), x)) == "(-2)Aᵀsin(Ax + y)"
    @test to_std(gradient(sum(2 * cos.(A * x + y)), x)) == "(-2)Aᵀsin(Ax + y)"
    @test to_std(gradient(sum(x .^ 2), x)) == "2x"
    @test to_std(gradient(sum(x .^ 3), x)) == "3x²"
    @test to_std(gradient(sum(x)^2, x)) == "2sum(xᵀ)vec(1)"
    @test to_std(gradient(sum(x .^ 2)^2, x)) == "2sum(xᵀ²)2x"
    @test to_std(gradient(sum((x + y) .^ 2), x)) == "2(x + y)"
    @test to_std(gradient(sum((x .* y) .^ 2), x)) == "2(x ⊙ y ⊙ y)"
    @test to_std(gradient(sum((A * x - y) .^ 2), x)) == "2Aᵀ(Ax - y)"
    @test to_std(gradient(log.(x)'*x, x)) == "vec(1) + log(x)"
    @test to_std(gradient(log.(x)'*log.(x), x)) == "2(log(x) ⊘ x)"
    @test to_std(gradient((x' * A * x) ^ (-2), x)) == "(-2)(xᵀAᵀx)⁻³(Aᵀx + Ax)"
    @test to_std(gradient((x' * A * x) ^ 2, x)) == "2xᵀAᵀx(Aᵀx + Ax)"
    @test to_std(gradient(((A .* B) * C * x)' * x, x)) == "(A ⊙ B)Cx + Cᵀ(Aᵀ ⊙ Bᵀ)x"
    @test to_std(gradient(((A .* (B .* C)) * C * x)' * x, x)) ==
          "(B ⊙ C ⊙ A)Cx + Cᵀ(Bᵀ ⊙ Cᵀ ⊙ Aᵀ)x"
    @test to_std(gradient(sum((A .* B) * C * x), x)) == "Cᵀ(Aᵀ ⊙ Bᵀ)vec(1)"
    @test to_std(gradient((x .^ 2 .* y)' * c, x)) == "2(x ⊙ y ⊙ c)"
    @test to_std(gradient(norm(A * x, 2), x)) == "1/2sum((xᵀAᵀ)²)⁻¹⸍²2AᵀAx"
    @test to_std(gradient(log.(A*x)' * x, x)) == "Aᵀdiagm(vec(1) ⊘ (Ax))x + log(Ax)" # TODO: Simplify diagm(quotient) * vec
    @test to_std(gradient(x' * sin.(A * x), x)) == "Aᵀdiagm(cos(Ax))x + sin(Ax)"
end

@testset "test Jacobian in standard notation" begin
    @matrix A B C
    @vector x y

    @test to_std(jacobian(A * x, x)) == "A"
    @test to_std(jacobian(A' * x, x)) == "Aᵀ"
    @test to_std(jacobian((A .* B) * log.(A * x), x)) == "(A ⊙ B)diagm(vec(1) ⊘ (Ax))A"
    @test to_std(jacobian(abs.(x), x)) == "diagm(sgn(x))I"
    @test to_std(jacobian(log.(x), x)) == "diagm(vec(1) ⊘ x)I"
    @test to_std(jacobian(diagm(A*x)*x, x)) == "diagm(Ax)I + diagm(x)A"
    @test to_std(jacobian(sin.(A * x + y), x)) == "diagm(cos(Ax + y))A"
    @test to_std(jacobian(diag(diagm(x' * B' * A * A)), x)) == "AᵀAᵀB"
    @test to_std(jacobian(((A .* B) * C * x)' * x * x, x)) ==
          "xᵀCᵀ(Aᵀ ⊙ Bᵀ)xI + x(xᵀCᵀ(Aᵀ ⊙ Bᵀ) + xᵀ(A ⊙ B)C)"
end

@testset "test diagonal expressions in standard notation" begin
    @matrix A
    @vector x y z

    # The Kronecker delta that 'diagm' introduces is removed by
    # the vector of ones that 'diag' contracts against it, and that ones vector is then
    # tied element-wise to the remainder, where it contributes nothing.
    @test to_std(diag(diagm(x))) == "x"
    @test to_std(diag(diagm(A * x))) == "Ax"
    @test to_std(diag(diagm(diag(diagm(x))))) == "x"
    @test to_std(jacobian(diag(diagm(A * x)), x)) == "A"

    # Here KrD sits further inside the tree.
    @test to_std(diag(diagm(x)) * y') == "xyᵀ"
    @test to_std(y * diag(diagm(x))') == "yxᵀ"
    @test to_std(diag(diagm(x))' * y) == "xᵀy"
    @test to_std(A * diag(diagm(x))) == "Ax"
    @test to_std(diag(diagm(x)) .* y) == "x ⊙ y"
    @test to_std((diag(diagm(x)) .* y)' * z) == "(xᵀ ⊙ yᵀ)z"
    @test to_std(sum(diag(diagm(A * x)))) == "vec(1)ᵀAx"

    # Check that constants survive.
    @test to_std(diagm(x') * vector(1)) == "x"
    @test to_std(diagm(x') * vector(3)) == "3x"
end

@testset "test a contracted vector of ones is not dropped" begin
    @matrix A
    @vector x

    @test to_std(sum(x)) == "sum(x)"
    @test to_std(sum(A * x)) == "vec(1)ᵀAx"
    @test to_std(gradient(sum(x), x)) == "vec(1)"
    @test to_std(gradient(sum(A * x), x)) == "Aᵀvec(1)"
end

@testset "test elementwise multiplication in standard notation" begin
    @matrix A B
    @vector x

    # Matched variance
    @test to_std(A .* B) == "A ⊙ B"
    @test to_std(A .* A) == "A ⊙ A"
    @test to_std(A' .* B') == "Aᵀ ⊙ Bᵀ"

    # Mixed variance
    @test to_std(A .* B') == "A ⊙ Bᵀ"
    @test to_std(A' .* B) == "Aᵀ ⊙ B"
    @test to_std(A .* A') == "A ⊙ Aᵀ"
    @test to_std(A' .* A) == "Aᵀ ⊙ A"

    # Composed expressions
    @test to_std((A .* A') * x) == "(A ⊙ Aᵀ)x"
    @test to_std((A .* B') * x) == "(A ⊙ Bᵀ)x"
    @test to_std(x' * (A .* A') * x) == "xᵀ(A ⊙ Aᵀ)x"
    @test to_std(jacobian((A .* A') * x, x)) == "A ⊙ Aᵀ"
    @test to_std(jacobian((A .* B') * x, x)) == "A ⊙ Bᵀ"
    @test to_std(gradient(x' * (A .* A') * x, x)) == "(Aᵀ ⊙ A)x + (A ⊙ Aᵀ)x"
    @test to_std(gradient(x' * (A .* B') * x, x)) == "(Aᵀ ⊙ B)x + (A ⊙ Bᵀ)x"
    @test to_std(gradient(sum((A .* A') * x), x)) == "(Aᵀ ⊙ A)vec(1)"
end

@testset "test derivative in standard notation" begin
    @matrix A B C X
    @vector x y z

    @test to_std(derivative(diag(A)'*x, A)) == "Iᵀdiagm(x)"
    @test to_std(derivative(sum(-y .* (X*z)), X)) == "(-1)zyᵀ"
    @test to_std(derivative(sum((A .* B) * C * x), x)) == "vec(1)ᵀ(A ⊙ B)C"
end

@testset "test Hessian in standard notation" begin
    @matrix A B
    @vector x

    @test to_std(hessian(x' * A * x, x)) == "Aᵀ + A"
    @test to_std(hessian(2 * x' * A * x, x)) == "2Aᵀ + 2A"
    @test to_std(hessian(2 * x' * x, x)) == "4I"
    @test to_std(hessian(sin(cos(x' * A * B' * x)), x)) ==
          "cos(cos(xᵀBAᵀx))((-1)sin(xᵀBAᵀx)(BAᵀ + ABᵀ) + (BAᵀx + ABᵀx)(-1)cos(xᵀBAᵀx)(xᵀABᵀ + xᵀBAᵀ)) + (-1)(-1)sin(xᵀBAᵀx)(BAᵀx + ABᵀx)sin(cos(xᵀBAᵀx))(-1)sin(xᵀBAᵀx)(xᵀABᵀ + xᵀBAᵀ)"
    @test to_std(hessian(x' * sin.(A * x), x)) ==
          "diagm(cos(Ax))A + Aᵀdiagm(cos(Ax)) + (-1)Aᵀdiagm(x ⊙ sin(Ax))A"

    # These reach an order-3 intermediate.
    @test to_std(hessian((A * x)' * sin.(B * x), x)) ==
          "Aᵀdiagm(cos(Bx))B + (-1)Bᵀdiagm(Ax ⊙ sin(Bx))B + Bᵀdiagm(cos(Bx))A"
    @test to_std(hessian((A * x)' * sin.(A * x), x)) ==
          "Aᵀdiagm(cos(Ax))A + (-1)Aᵀdiagm(Ax ⊙ sin(Ax))A + Aᵀdiagm(cos(Ax))A"
end
