# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

using DiffMatic
using Test

using DiffMatic: Monomial, KrD, Zero
using DiffMatic: evaluate
using DiffMatic: Upper, Lower

dc = DiffMatic

@testset "create column vector" begin
    @vector x
    @vector y A

    # TODO: Find a better way to keep track of the indices and remove all "equivalent"
    @test equivalent(x, Monomial("x", Upper(1)))
    @test equivalent(y, Monomial("y", Upper(2)))
    @test equivalent(A, Monomial("A", Upper(3)))
end

@testset "create matrix" begin
    @matrix A
    @matrix B X

    @test equivalent(A, Monomial("A", Upper(4), Lower(5)))
    @test equivalent(B, Monomial("B", Upper(6), Lower(7)))
    @test equivalent(X, Monomial("X", Upper(8), Lower(9)))
end

@testset "create scalar" begin
    @scalar a
    @scalar b c

    @test equivalent(a, Monomial("a"))
    @test equivalent(b, Monomial("b"))
    @test equivalent(c, Monomial("c"))
end

@testset "to_std_string output is correct with standard form KrD" begin
    @test to_std_string(KrD(Upper(1), Lower(2))) == "I"
    @test to_std_string(KrD(Upper(2), Lower(1))) == "I"
    @test to_std_string(KrD(Lower(1), Upper(2))) == "Iᵀ"
    @test to_std_string(KrD(Lower(2), Upper(1))) == "Iᵀ"
end

@testset "to_std_string output is correct with scalar-Monomial multiplication" begin
    A = Monomial("A", Upper(1), Lower(2))
    At = Monomial("A", Lower(1), Upper(2))
    x = Monomial("x", Upper(2))
    xt = Monomial("x", Lower(2))
    a = Monomial("a")
    b = Monomial("b")

    function mult(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std_string(mult(A, a)) == "aA"
    @test to_std_string(mult(a, A)) == "aA"
    @test to_std_string(mult(At, a)) == "aAᵀ"
    @test to_std_string(mult(a, At)) == "aAᵀ"
    @test to_std_string(mult(x, a)) == "ax"
    @test to_std_string(mult(a, x)) == "ax"
    @test to_std_string(mult(xt, a)) == "axᵀ"
    @test to_std_string(mult(a, xt)) == "axᵀ"
    @test_broken to_std_string(mult(b, a)) == "ab" # TODO: Implement lexicographical ordering
end

@testset "to_std_string output is correct with matrix-vector contraction" begin
    A = Monomial("A", Upper(1), Lower(2))
    At = Monomial("A", Lower(1), Upper(2))
    x = Monomial("x", Upper(2))
    xt = Monomial("x", Lower(2))
    y = Monomial("y", Upper(1))
    yt = Monomial("y", Lower(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std_string(contract(A, x)) == "Ax"
    @test to_std_string(contract(x, A)) == "Ax"
    @test to_std_string(contract(A, yt)) == "yᵀA"
    @test to_std_string(contract(yt, A)) == "yᵀA"
    @test to_std_string(contract(At, xt)) == "xᵀAᵀ"
    @test to_std_string(contract(xt, At)) == "xᵀAᵀ"
    @test to_std_string(contract(At, y)) == "Aᵀy"
    @test to_std_string(contract(y, At)) == "Aᵀy"
end

@testset "to_std_string output is correct with trace-matrix/vector multiplication" begin
    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    function add(l, r)
        return dc.BinaryOperation{dc.Add}(l, r)
    end

    trA = mul(Monomial("A", Upper(1), Lower(2)), KrD(Upper(2), Lower(1)))
    A = Monomial("A", Upper(3), Lower(4))
    B = Monomial("B", Upper(4), Lower(5))
    x = Monomial("x", Upper(4))

    @test to_std_string(trA) == "tr(A)"
    @test to_std_string(mul(trA, A)) == "tr(A)A"
    @test to_std_string(mul(A, trA)) == "tr(A)A"
    @test to_std_string(mul(mul(trA, A), x)) == "tr(A)Ax"

    trAB = mul(
        mul(Monomial("A", Upper(1), Lower(2)), Monomial("B", Upper(2), Lower(3))),
        KrD(Upper(3), Lower(1)),
    )
    @test to_std_string(trAB) == "tr(AB)"
    @test to_std_string(mul(trAB, A)) == "tr(AB)A"
    @test to_std_string(mul(trAB, B)) == "tr(AB)B"
    @test to_std_string(mul(mul(trAB, A), x)) == "tr(AB)Ax"

    trApB = mul(
        add(Monomial("A", Upper(1), Lower(2)), Monomial("B", Upper(1), Lower(2))),
        KrD(Upper(2), Lower(1)),
    )
    @test to_std_string(trApB) == "tr(A + B)"
    @test to_std_string(mul(trApB, A)) == "tr(A + B)A"
    @test to_std_string(mul(trApB, B)) == "tr(A + B)B"
    @test to_std_string(mul(mul(trApB, A), x)) == "tr(A + B)Ax"
end

@testset "to_std_string output is correct with all covariant bilinar form-vector contraction" begin
    A = Monomial("A", Lower(1), Lower(2))
    x = Monomial("x", Upper(2))
    y = Monomial("y", Upper(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std_string(contract(A, x)) == "xᵀAᵀ"
    @test to_std_string(contract(x, A)) == "xᵀAᵀ"
    @test to_std_string(contract(A, y)) == "yᵀA"
    @test to_std_string(contract(y, A)) == "yᵀA"
end

@testset "to_std_string output is correct with all contravariant bilinear form-vector contraction" begin
    A = Monomial("A", Upper(1), Upper(2))
    x = Monomial("x", Lower(2))
    y = Monomial("y", Lower(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std_string(contract(A, x)) == "Ax"
    @test to_std_string(contract(x, A)) == "Ax"
    @test to_std_string(contract(A, y)) == "Aᵀy"
    @test to_std_string(contract(y, A)) == "Aᵀy"
end

@testset "to_std_string output is correct with matrix-matrix contraction" begin
    A = Monomial("A", Upper(1), Lower(2))
    B = Monomial("B", Upper(2), Lower(3))
    C = Monomial("C", Lower(1), Upper(3))
    D = Monomial("D", Lower(3), Upper(2))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std_string(contract(A, B)) == "AB"
    @test to_std_string(contract(B, A)) == "AB"
    @test to_std_string(contract(A, C)) == "CᵀA"
    @test to_std_string(contract(C, A)) == "CᵀA"
    @test to_std_string(contract(A, D)) == "ADᵀ"
    @test to_std_string(contract(D, A)) == "ADᵀ"
    @test to_std_string(contract(C, D)) == "DᵀCᵀ"
    @test to_std_string(contract(D, C)) == "DᵀCᵀ"
end

@testset "to_std_string output is correct with matrix-matrix element wise multiplication" begin
    A = Monomial("A", Upper(1), Lower(2))
    B = Monomial("B", Upper(1), Lower(2))

    @test to_std_string(A .* B) == "A ⊙ B"
    @test to_std_string(A .* A) == "A ⊙ A"
    @test to_std_string(dc.evaluate(A * (A .* B))) == "A(A ⊙ B)"
    @test to_std_string(dc.evaluate((A .* B) * A)) == "(A ⊙ B)A"
    @test to_std_string(dc.evaluate(A .* (A * B))) == "AB ⊙ A"
    @test to_std_string(dc.evaluate((A * B) .* A)) == "AB ⊙ A"
end

@testset "to_std_string output is correct with matrix-matrix sum" begin
    A = Monomial("A", Upper(1), Lower(2))
    B = Monomial("B", Upper(1), Lower(2))
    C = Monomial("C", Lower(2), Upper(1))

    function sum(l, r)
        return evaluate(dc.BinaryOperation{dc.Add}(l, r))
    end

    @test to_std_string(sum(A, B)) == "A + B"
    @test to_std_string(sum(B, A)) == "B + A"
    @test to_std_string(sum(A, C)) == "A + Cᵀ"
    @test to_std_string(sum(C, A)) == "Cᵀ + A"
end

@testset "to_std_string output is correct with vector-matrix element wise multiplication" begin
    A = Monomial("A", Upper(1), Lower(2))
    x = Monomial("x", Upper(1))
    y = Monomial("y", Lower(2))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std_string(mul(A, x)) == "diag(x)A"
    @test to_std_string(mul(x, A)) == "diag(x)A"
    @test to_std_string(mul(A, y)) == "A diag(yᵀ)"
    @test to_std_string(mul(y, A)) == "A diag(yᵀ)"
end

@testset "to_std_string output is correct with vector-vector element wise multiplication" begin
    x = Monomial("x", Upper(1))
    y = Monomial("y", Upper(1))
    z = Monomial("z", Upper(1))
    v = Monomial("v", Upper(1))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std_string(mul(x, x)) == "x ⊙ x"
    @test to_std_string(mul(x, y)) == "x ⊙ y"
    @test to_std_string(mul(mul(x, y), z)) == "x ⊙ y ⊙ z"
    @test to_std_string(mul(z, mul(x, y))) == "z ⊙ x ⊙ y"
    @test to_std_string(mul(mul(mul(x, y), z), v)) == "x ⊙ y ⊙ z ⊙ v"
    @test to_std_string(mul(v, mul(mul(x, y), z))) == "v ⊙ x ⊙ y ⊙ z"
    @test to_std_string(mul(v, mul(z, mul(x, y)))) == "v ⊙ z ⊙ x ⊙ y"
    @test to_std_string(mul(mul(z, v), mul(x, y))) == "z ⊙ v ⊙ x ⊙ y"
end

@testset "to_std_string output is correct with vector sum" begin
    x = Monomial("x", Upper(1))
    y = Monomial("y", Lower(2))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std_string(mul(x, KrD(Upper(1), Lower(1)))) == "sum(x)"
    @test to_std_string(mul(KrD(Upper(1), Lower(1)), x)) == "sum(x)"
    @test to_std_string(mul(y, KrD(Upper(2), Lower(2)))) == "sum(yᵀ)"
    @test to_std_string(mul(KrD(Upper(2), Lower(2)), y)) == "sum(yᵀ)"
    @test to_std_string(sum(x)) == "sum(x)"
    @test to_std_string(sum(y)) == "sum(yᵀ)"
end

@testset "to_std_string output is correct with KrD-KrD and one free index" begin
    l = KrD(Upper(1), Lower(2))
    u = KrD(Upper(2), Lower(1))
    t = KrD(Upper(1), Lower(1))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std_string(mul(u, t)) == "vec(1)"
    @test to_std_string(mul(t, u)) == "vec(1)"
    @test to_std_string(mul(l, t)) == "vec(1)ᵀ"
    @test to_std_string(mul(t, l)) == "vec(1)ᵀ"
end

@testset "to_std_string output is correct with complex expression" begin
    x = Monomial("x", Upper(1))
    y = Monomial("y", Upper(2))
    a = Monomial("a")

    @test to_std_string(evaluate(a * sin(x)' * y)) == "asin(xᵀ)y"
    @test to_std_string(evaluate(sin(x)' * a * y)) == "asin(xᵀ)y"
    @test to_std_string(evaluate(sin(x)' * y * a)) == "asin(xᵀ)y"
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

@testset "to_std_string of gradient" begin
    @scalar a b
    @vector x y c
    @matrix A B

    @test to_std_string(gradient(x' * x, x)) == "2x"
    @test to_std_string(gradient(tr(x * x'), x)) == "2x"
    @test to_std_string(gradient((y .* c)' * x, x)) == "y ⊙ c"
    @test to_std_string(gradient((x .* c)' * x, x)) == "2(x ⊙ c)"
    @test to_std_string(gradient((x + y)' * x, x)) == "2x + y"
    @test to_std_string(gradient((x - y)' * x, x)) == "2x - y"
    @test to_std_string(gradient(sin(tr(x * x')), x)) == "cos(xᵀx)2x"
    @test to_std_string(gradient(cos(tr(x * x')), x)) == "(-1)sin(xᵀx)2x"
    @test to_std_string(gradient(tr(A), x)) == "vec(0)"
    @test to_std_string(gradient(x' * B' * A * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std_string(gradient((A' * B * x)' * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std_string(gradient(a * sin(y)' * x, x)) == "asin(y)"
    @test to_std_string(gradient(a * sin(x)' * y, x)) == "a(cos(x) ⊙ y)"
    @test to_std_string(gradient(sin(y)' * x * a, x)) == "asin(y)"
    @test to_std_string(gradient(sin(x)' * y * a, x)) == "a(cos(x) ⊙ y)"
    @test to_std_string(gradient(x' * sin(y) * a, x)) == "asin(y)"
    @test to_std_string(gradient(y' * sin(x) * a, x)) == "a(cos(x) ⊙ y)"
    @test to_std_string(gradient(sum(x), x)) == "vec(1)"
    @test_broken to_std_string(gradient(2 * sum(x), x)) == "2(vec(1))"
    @test_broken to_std_string(gradient(sum(2 * x), x)) == "2(vec(1))"
    @test to_std_string(gradient(sum(x .^ 2), x)) == "2x"
    @test to_std_string(gradient(sum(x .^ 3), x)) == "3x²"
    @test to_std_string(gradient(sum((x + y) .^ 2), x)) == "2(x + y)"
    @test_broken to_std_string(gradient(sum((x .* y) .^ 2), x)) == "2(x ⊙ y ⊙ y)"
    @test to_std_string(gradient(sum((A * x - y) .^ 2), x)) == "2Aᵀ(Ax - y)"
    @test to_std_string(gradient((x' * A * x) .^ (-2), x)) == "(-2)(xᵀAᵀx)⁻³(Aᵀx + Ax)"
end

@testset "to_std_string of jacobian" begin
    @matrix A
    @vector x y

    @test to_std_string(jacobian(A * x, x)) == "A"
    @test to_std_string(jacobian(A' * x, x)) == "Aᵀ"
    @test to_std_string(jacobian(sin(A * x + y), x)) == "diag(cos(Ax + y))A"
end

@testset "to_std_string of derivative {A, A'} * x" begin
    @matrix X
    @vector x y z

    @test to_std_string(derivative(sum(-y .* (X*z)), X)) == "z(-1)yᵀ"
end

@testset "to_std_string of hessian" begin
    @matrix A
    @vector x

    @test to_std_string(hessian(x' * A * x, x)) == "Aᵀ + A"
    @test to_std_string(hessian(2 * x' * A * x, x)) == "2Aᵀ + 2A"
    @test to_std_string(hessian(2 * x' * x, x)) == "4I"
end
