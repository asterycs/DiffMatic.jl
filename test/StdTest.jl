# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

using DiffMatic
using Test

using DiffMatic: Tensor, KrD, Zero
using DiffMatic: evaluate
using DiffMatic: Upper, Lower

dc = DiffMatic

@testset "create column vector" begin
    @vector x
    @vector y A

    # TODO: Find a better way to keep track of the indices and remove all "equivalent"
    @test equivalent(x, Tensor("x", Upper(1)))
    @test equivalent(y, Tensor("y", Upper(2)))
    @test equivalent(A, Tensor("A", Upper(3)))
end

@testset "create matrix" begin
    @matrix A
    @matrix B X

    @test equivalent(A, Tensor("A", Upper(4), Lower(5)))
    @test equivalent(B, Tensor("B", Upper(6), Lower(7)))
    @test equivalent(X, Tensor("X", Upper(8), Lower(9)))
end

@testset "create scalar" begin
    @scalar a
    @scalar b c

    @test equivalent(a, Tensor("a"))
    @test equivalent(b, Tensor("b"))
    @test equivalent(c, Tensor("c"))
end

@testset "to_std_string output is correct with scalar-tensor multiplication" begin
    A = Tensor("A", Upper(1), Lower(2))
    At = Tensor("A", Lower(1), Upper(2))
    x = Tensor("x", Upper(2))
    xt = Tensor("x", Lower(2))
    a = Tensor("a")
    b = Tensor("b")

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
    @test to_std_string(mult(b, a)) == "ab"
end

@testset "to_std_string output is correct with matrix-vector contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    At = Tensor("A", Lower(1), Upper(2))
    x = Tensor("x", Upper(2))
    xt = Tensor("x", Lower(2))
    y = Tensor("y", Upper(1))
    yt = Tensor("y", Lower(1))

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

@testset "to_std_string output is correct with all covariant bilinar form-vector contraction" begin
    A = Tensor("A", Lower(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std_string(contract(A, x)) == "xᵀAᵀ"
    @test to_std_string(contract(x, A)) == "xᵀAᵀ"
    @test to_std_string(contract(A, y)) == "yᵀA"
    @test to_std_string(contract(y, A)) == "yᵀA"
end

@testset "to_std_string output is correct with all contravariant bilinear form-vector contraction" begin
    A = Tensor("A", Upper(1), Upper(2))
    x = Tensor("x", Lower(2))
    y = Tensor("y", Lower(1))

    function contract(l, r)
        return evaluate(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test to_std_string(contract(A, x)) == "Ax"
    @test to_std_string(contract(x, A)) == "Ax"
    @test to_std_string(contract(A, y)) == "Aᵀy"
    @test to_std_string(contract(y, A)) == "Aᵀy"
end

@testset "to_std_string output is correct with matrix-matrix contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))
    C = Tensor("C", Lower(1), Upper(3))
    D = Tensor("D", Lower(3), Upper(2))

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

@testset "to_std_string output is correct with matrix-matrix sum" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(1), Lower(2))
    C = Tensor("C", Lower(2), Upper(1))

    function sum(l, r)
        return evaluate(dc.BinaryOperation{dc.Add}(l, r))
    end

    @test to_std_string(sum(A, B)) == "A + B"
    @test to_std_string(sum(B, A)) == "B + A"
    @test to_std_string(sum(A, C)) == "A + Cᵀ"
    @test to_std_string(sum(C, A)) == "Cᵀ + A"
end

@testset "to_std_string output is correct with vector-matrix element wise multiplication" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(2))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std_string(mul(A, x)) == "diag(x)A"
    @test to_std_string(mul(x, A)) == "diag(x)A"
    @test to_std_string(mul(A, y)) == "A diag(yᵀ)"
    @test to_std_string(mul(y, A)) == "A diag(yᵀ)"
end

@testset "to_std_string output is correct with vector-vector element wise multiplication" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(1))
    z = Tensor("z", Upper(1))
    v = Tensor("v", Upper(1))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std_string(mul(x, x)) == "x ⊙ x"
    @test to_std_string(mul(x, y)) == "x ⊙ y"
    @test to_std_string(mul(mul(x, y), z)) == "(x ⊙ y) ⊙ z"
    @test to_std_string(mul(z, mul(x, y))) == "z ⊙ (x ⊙ y)"
    @test to_std_string(mul(mul(mul(x, y), z), v)) == "((x ⊙ y) ⊙ z) ⊙ v"
    @test to_std_string(mul(v, mul(mul(x, y), z))) == "v ⊙ ((x ⊙ y) ⊙ z)"
    @test to_std_string(mul(v, mul(z, mul(x, y)))) == "v ⊙ (z ⊙ (x ⊙ y))"
    @test to_std_string(mul(mul(z, v), mul(x, y))) == "(z ⊙ v) ⊙ (x ⊙ y)"
end

@testset "to_std_string output is correct with vector sum" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(2))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test to_std_string(mul(x, KrD(Upper(1), Lower(1)))) == "sum(x)"
    @test to_std_string(mul(KrD(Upper(1), Lower(1)), x)) == "sum(x)"
    @test to_std_string(mul(y, KrD(Upper(2), Lower(2)))) == "sum(yᵀ)"
    @test to_std_string(mul(KrD(Upper(2), Lower(2)), y)) == "sum(yᵀ)"
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
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))
    a = Tensor("a")

    @test to_std_string(evaluate(a * sin(x)' * y)) == "yᵀsin(x)a"
    @test to_std_string(evaluate(sin(x)' * a * y)) == "ayᵀsin(x)"
    @test to_std_string(evaluate(sin(x)' * y * a)) == "ayᵀsin(x)"
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
    @test to_std_string(gradient(cos(tr(x * x')), x)) == "-sin(xᵀx)2x"
    @test to_std_string(gradient(tr(A), x)) == "vec(0)"
    @test to_std_string(gradient(x' * B' * A * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std_string(gradient((A' * B * x)' * A * x, x)) == "AᵀAᵀBx + BᵀAAx"
    @test to_std_string(gradient(a * sin(y)' * x, x)) == "asin(y)"
    @test to_std_string(gradient(a * sin(x)' * y, x)) == "a(y ⊙ cos(x))"
    @test to_std_string(gradient(sin(y)' * x * a, x)) == "asin(y)"
    @test to_std_string(gradient(sin(x)' * y * a, x)) == "a(y ⊙ cos(x))"
    @test to_std_string(gradient(x' * sin(y) * a, x)) == "asin(y)"
    @test to_std_string(gradient(y' * sin(x) * a, x)) == "a(y ⊙ cos(x))"
    @test to_std_string(gradient(sum(x), x)) == "vec(1)"
    @test to_std_string(gradient(2 * sum(x), x)) == "2(vec(1))"
    @test to_std_string(gradient(sum(2 * x), x)) == "2(vec(1))"
end

@testset "to_std_string of jacobian {A, A'} * x" begin
    @matrix A
    @vector x

    @test to_std_string(jacobian(A * x, x)) == "A"
    @test to_std_string(jacobian(A' * x, x)) == "Aᵀ"
end

@testset "to_std_string of hessian" begin
    @matrix A
    @vector x

    @test to_std_string(hessian(x' * A * x, x)) == "Aᵀ + A"
    @test to_std_string(hessian(2 * x' * A * x, x)) == "2Aᵀ + 2A"
    @test to_std_string(hessian(2 * x' * x, x)) == "4I"
end
