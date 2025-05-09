# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

using DiffMatic
using Test

using DiffMatic: Variable, KrD, Zero
using DiffMatic: evaluate
using DiffMatic: Upper, Lower

dc = DiffMatic

@testset "create column vector" begin
    @vector x
    @vector y A

    # TODO: Find a better way to keep track of the indices and remove all "equivalent"
    @test equivalent(x, Variable("x", Upper(1)))
    @test equivalent(y, Variable("y", Upper(2)))
    @test equivalent(A, Variable("A", Upper(3)))
end

@testset "create matrix" begin
    @matrix A
    @matrix B X

    @test equivalent(A, Variable("A", Upper(4), Lower(5)))
    @test equivalent(B, Variable("B", Upper(6), Lower(7)))
    @test equivalent(X, Variable("X", Upper(8), Lower(9)))
end

@testset "create scalar" begin
    @scalar a
    @scalar b c

    @test equivalent(a, Variable("a"))
    @test equivalent(b, Variable("b"))
    @test equivalent(c, Variable("c"))
end

@testset "to_std output is correct with standard form KrD" begin
    @test to_std(KrD(Upper(1), Lower(2))) == "I"
    @test to_std(KrD(Upper(2), Lower(1))) == "I"
    @test to_std(KrD(Lower(1), Upper(2))) == "Iᵀ"
    @test to_std(KrD(Lower(2), Upper(1))) == "Iᵀ"
end

@testset "to_std output is correct with standard form Zero" begin
    @test to_std(Zero(Upper(1))) == "vec(0)"
    @test to_std(Zero(Lower(1))) == "vec(0)ᵀ"

    @test to_std(Zero(Upper(1), Lower(2))) == "mat(0)"
    @test to_std(Zero(Upper(2), Lower(1))) == "mat(0)"
    @test to_std(Zero(Lower(1), Upper(2))) == "mat(0)ᵀ"
    @test to_std(Zero(Lower(2), Upper(1))) == "mat(0)ᵀ"
end

@testset "to_std output is correct with scalar-Variable multiplication" begin
    A = Variable("A", Upper(1), Lower(2))
    At = Variable("A", Lower(1), Upper(2))
    x = Variable("x", Upper(2))
    xt = Variable("x", Lower(2))
    a = Variable("a")
    b = Variable("b")

    function mult(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std(mult(A, a)) == "aA"
    @test to_std(mult(a, A)) == "aA"
    @test to_std(mult(At, a)) == "aAᵀ"
    @test to_std(mult(a, At)) == "aAᵀ"
    @test to_std(mult(x, a)) == "ax"
    @test to_std(mult(a, x)) == "ax"
    @test to_std(mult(xt, a)) == "axᵀ"
    @test to_std(mult(a, xt)) == "axᵀ"
    @test_broken to_std(mult(b, a)) == "ab" # TODO: Implement lexicographical ordering
end

@testset "to_std output is correct with matrix-vector contraction" begin
    A = Variable("A", Upper(1), Lower(2))
    At = Variable("A", Lower(1), Upper(2))
    x = Variable("x", Upper(2))
    xt = Variable("x", Lower(2))
    y = Variable("y", Upper(1))
    yt = Variable("y", Lower(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std(contract(A, x)) == "Ax"
    @test to_std(contract(x, A)) == "Ax"
    @test to_std(contract(A, yt)) == "yᵀA"
    @test to_std(contract(yt, A)) == "yᵀA"
    @test to_std(contract(At, xt)) == "xᵀAᵀ"
    @test to_std(contract(xt, At)) == "xᵀAᵀ"
    @test to_std(contract(At, y)) == "Aᵀy"
    @test to_std(contract(y, At)) == "Aᵀy"
end

@testset "to_std output is correct with trace-matrix/vector multiplication" begin
    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    function add(l, r)
        return dc.BinaryOperation{dc.Add}(l, r)
    end

    trA = Variable("A", Upper(2), Lower(2))

    A = Variable("A", Upper(3), Lower(4))
    B = Variable("B", Upper(4), Lower(5))
    x = Variable("x", Upper(4))

    @test to_std(trA) == "tr(A)"
    @test to_std(mul(trA, A)) == "tr(A)A"
    @test to_std(mul(A, trA)) == "tr(A)A"
    @test to_std(mul(mul(trA, A), x)) == "tr(A)Ax"

    trAB = mul(Variable("A", Upper(1), Lower(2)), Variable("B", Upper(2), Lower(1)))

    @test to_std(trAB) == "tr(AB)"
    @test to_std(mul(trAB, A)) == "tr(AB)A"
    @test to_std(mul(trAB, B)) == "tr(AB)B"
    @test to_std(mul(mul(trAB, A), x)) == "tr(AB)Ax"

    trApB = add(Variable("A", Upper(2), Lower(2)), Variable("B", Upper(2), Lower(2)))

    @test to_std(trApB) == "tr(A) + tr(B)"
    @test to_std(mul(trApB, A)) == "(tr(A) + tr(B))A"
    @test to_std(mul(trApB, B)) == "(tr(A) + tr(B))B"
    @test to_std(mul(mul(trApB, A), x)) == "(tr(A) + tr(B))Ax"
end

@testset "to_std output is correct with all covariant bilinar form-vector contraction" begin
    A = Variable("A", Lower(1), Lower(2))
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std(contract(A, x)) == "xᵀAᵀ"
    @test to_std(contract(x, A)) == "xᵀAᵀ"
    @test to_std(contract(A, y)) == "yᵀA"
    @test to_std(contract(y, A)) == "yᵀA"
end

@testset "to_std output is correct with all contravariant bilinear form-vector contraction" begin
    A = Variable("A", Upper(1), Upper(2))
    x = Variable("x", Lower(2))
    y = Variable("y", Lower(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std(contract(A, x)) == "Ax"
    @test to_std(contract(x, A)) == "Ax"
    @test to_std(contract(A, y)) == "Aᵀy"
    @test to_std(contract(y, A)) == "Aᵀy"
end

@testset "to_std output is correct with matrix-matrix contraction" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(2), Lower(3))
    C = Variable("C", Lower(1), Upper(3))
    D = Variable("D", Lower(3), Upper(2))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std(contract(A, B)) == "AB"
    @test to_std(contract(B, A)) == "AB"
    @test to_std(contract(A, C)) == "CᵀA"
    @test to_std(contract(C, A)) == "CᵀA"
    @test to_std(contract(A, D)) == "ADᵀ"
    @test to_std(contract(D, A)) == "ADᵀ"
    @test to_std(contract(C, D)) == "DᵀCᵀ"
    @test to_std(contract(D, C)) == "DᵀCᵀ"
end

@testset "to_std output is correct with matrix-matrix element wise multiplication" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(1), Lower(2))

    @test to_std(A .* B) == "A ⊙ B"
    @test to_std(A .* A) == "A ⊙ A"
    @test to_std(dc.evaluate(A * (A .* B))) == "A(A ⊙ B)"
    @test to_std(dc.evaluate((A .* B) * A)) == "(A ⊙ B)A"
    @test to_std(dc.evaluate(A .* (A * B))) == "AB ⊙ A"
    @test to_std(dc.evaluate((A * B) .* A)) == "AB ⊙ A"
end

@testset "to_std output is correct with matrix-matrix sum" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(1), Lower(2))
    C = Variable("C", Lower(2), Upper(1))

    function sum(l, r)
        return evaluate(dc.BinaryOperation{dc.Add}(l, r))
    end

    @test to_std(sum(A, B)) == "A + B"
    @test to_std(sum(B, A)) == "B + A"
    @test to_std(sum(A, C)) == "A + Cᵀ"
    @test to_std(sum(C, A)) == "Cᵀ + A"
end

@testset "to_std output is correct with vector-matrix element wise multiplication" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(1))
    y = Variable("y", Lower(2))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std(mul(A, x)) == "diag(x)A"
    @test to_std(mul(x, A)) == "diag(x)A"
    @test to_std(mul(A, y)) == "Adiag(yᵀ)"
    @test to_std(mul(y, A)) == "Adiag(yᵀ)"
end

@testset "to_std output is correct with vector-vector element wise multiplication" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(1))
    z = Variable("z", Upper(1))
    v = Variable("v", Upper(1))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std(mul(x, x)) == "x ⊙ x"
    @test to_std(mul(x, y)) == "x ⊙ y"
    @test to_std(mul(mul(x, y), z)) == "x ⊙ y ⊙ z"
    @test to_std(mul(z, mul(x, y))) == "x ⊙ y ⊙ z"
    @test to_std(mul(mul(mul(x, y), z), v)) == "x ⊙ y ⊙ z ⊙ v"
    @test to_std(mul(v, mul(mul(x, y), z))) == "x ⊙ y ⊙ z ⊙ v"
    @test to_std(mul(v, mul(z, mul(x, y)))) == "x ⊙ y ⊙ z ⊙ v"
    @test to_std(mul(mul(z, v), mul(x, y))) == "z ⊙ v ⊙ x ⊙ y"
end

@testset "to_std output is correct with vector sum" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Lower(2))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std(mul(x, KrD(Upper(1), Lower(1)))) == "sum(x)"
    @test to_std(mul(KrD(Upper(1), Lower(1)), x)) == "sum(x)"
    @test to_std(mul(y, KrD(Upper(2), Lower(2)))) == "sum(yᵀ)"
    @test to_std(mul(KrD(Upper(2), Lower(2)), y)) == "sum(yᵀ)"
    @test to_std(sum(x)) == "sum(x)"
    @test to_std(sum(y)) == "sum(yᵀ)"
end

@testset "to_std output is correct with KrD-KrD and one free index" begin
    l = KrD(Upper(1), Lower(2))
    u = KrD(Upper(2), Lower(1))
    t = KrD(Upper(1), Lower(1))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std(mul(u, t)) == "vec(1)"
    @test to_std(mul(t, u)) == "vec(1)"
    @test to_std(mul(l, t)) == "vec(1)ᵀ"
    @test to_std(mul(t, l)) == "vec(1)ᵀ"
end

@testset "to_std output is correct with complex expression" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(2))
    a = Variable("a")

    @test to_std(evaluate(a * sin(x)' * y)) == "asin(xᵀ)y"
    @test to_std(evaluate(sin(x)' * a * y)) == "asin(xᵀ)y"
    @test to_std(evaluate(sin(x)' * y * a)) == "asin(xᵀ)y"
end

@testset "derivative interface checks" begin
    @matrix A
    @vector x

    @test equivalent(derivative(x' * A * x, A), evaluate(x * x')) # scalar input works
    @test equivalent(derivative(A * x, x), A) # vector input works
end

@testset "gradient interface checks" begin
    @matrix A
    @vector x

    @test_throws DomainError gradient(A * x, x) # input not a scalar
    @test_throws DomainError gradient(x' * A * x, A) # A is a matrix
end

@testset "jacobian interface checks" begin
    @matrix A
    @vector x

    @test_throws DomainError jacobian(x' * A * x, x) # input not a vector
    @test_throws DomainError jacobian(A * x, A) # A is a matrix
end

@testset "hessian interface checks" begin
    @matrix A
    @vector x

    @test_throws DomainError hessian(A * x, x) # input not a scalar
    @test_throws DomainError hessian(x' * A * x, A) # A is a matrix
end

@testset "to_std of gradient" begin
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
    @test to_std(gradient(x' * B' * A * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std(gradient((A' * B * x)' * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std(gradient(a * sin(y)' * x, x)) == "asin(y)"
    @test to_std(gradient(a * sin(x)' * y, x)) == "a(cos(x) ⊙ y)"
    @test to_std(gradient(sin(y)' * x * a, x)) == "asin(y)"
    @test to_std(gradient(sin(x)' * y * a, x)) == "a(cos(x) ⊙ y)"
    @test to_std(gradient(x' * sin(y) * a, x)) == "asin(y)"
    @test to_std(gradient(y' * sin(x) * a, x)) == "a(cos(x) ⊙ y)"
    @test to_std(gradient(sum(x), x)) == "vec(1)"
    @test to_std(gradient(2 * sum(x), x)) == "2vec(1)"
    @test to_std(gradient(sum(2 * x), x)) == "2vec(1)"
    @test to_std(gradient(2 * sum(sin(x)), x)) == "2cos(x)"
    @test to_std(gradient(sum(2 * sin(x)), x)) == "2cos(x)"
    @test to_std(gradient(2 * sum(cos(A * x + y)), x)) == "(-2)Aᵀsin(Ax + y)"
    @test to_std(gradient(sum(2 * cos(A * x + y)), x)) == "(-2)Aᵀsin(Ax + y)"
    @test to_std(gradient(sum(x .^ 2), x)) == "2x"
    @test to_std(gradient(sum(x .^ 3), x)) == "3x^2"
    @test to_std(gradient(sum(x)^2, x)) == "2sum(xᵀ)vec(1)"
    @test to_std(gradient(sum(x .^ 2)^2, x)) == "2sum(xᵀ^2)2x"
    @test to_std(gradient(sum((x + y) .^ 2), x)) == "2(x + y)"
    @test to_std(gradient(sum((x .* y) .^ 2), x)) == "2(x ⊙ y ⊙ y)"
    @test to_std(gradient(sum((A * x - y) .^ 2), x)) == "2Aᵀ(Ax - y)"
    @test to_std(gradient((x' * A * x) ^ (-2), x)) == "(-2)(xᵀAᵀx)^(-3)(Aᵀx + Ax)"
    @test to_std(gradient(((A .* B) * C * x)' * x, x)) == "(A ⊙ B)Cx + Cᵀ(Aᵀ ⊙ Bᵀ)x"
    @test to_std(gradient(((A .* (B .* C)) * C * x)' * x, x)) ==
          "(B ⊙ C ⊙ A)Cx + Cᵀ(Bᵀ ⊙ Cᵀ ⊙ Aᵀ)x"
    @test to_std(gradient(sum((A .* B) * C * x), x)) == "Cᵀ(Aᵀ ⊙ Bᵀ)vec(1)"
    to_std(gradient((x .^ 2 .* y)' * c, x)) == "2x ⊙ y ⊙ c"
end

@testset "to_std of jacobian" begin
    @matrix A
    @vector x y

    @test to_std(jacobian(A * x, x)) == "A"
    @test to_std(jacobian(A' * x, x)) == "Aᵀ"
    @test to_std(jacobian(sin(A * x + y), x)) == "diag(cos(Ax + y))A"
    @test to_std(jacobian(((A .* B) * C * x)' * x * x, x)) ==
          "xᵀCᵀ(Aᵀ ⊙ Bᵀ)xI + x(xᵀCᵀ(Aᵀ ⊙ Bᵀ) + xᵀ(A ⊙ B)C)"
end

@testset "to_std of derivative {A, A'} * x" begin
    @matrix A B C X
    @vector x y z

    @test to_std(derivative(sum(-y .* (X*z)), X)) == "(-1)zyᵀ"
    @test to_std(derivative(sum((A .* B) * C * x), x)) == "vec(1)ᵀ(A ⊙ B)C"
end

@testset "to_std of hessian" begin
    @matrix A
    @vector x

    @test to_std(hessian(x' * A * x, x)) == "Aᵀ + A"
    @test to_std(hessian(2 * x' * A * x, x)) == "2Aᵀ + 2A"
    @test to_std(hessian(2 * x' * x, x)) == "4I"
end
