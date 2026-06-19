# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic
using Test

using DiffMatic: Variable, Literal, KrD, Zero
using DiffMatic: Upper, Lower

using LinearAlgebra: norm, tr, I, diagm, diag

dc = DiffMatic

@testset "Variable constructor throws on invalid input" begin
    @test_throws DomainError Variable("A", Lower(2), Lower(2))
    @test_throws DomainError Variable("A", Lower(2), Upper(2), Lower(1), Lower(2))
end

@testset "Variable constructor succeeds on valid input" begin
    @test !isnothing(Variable("A", Upper(1), Lower(2)))
    @test !isnothing(Variable("B", Lower(1), Lower(2)))
    @test !isnothing(Variable("x", Upper(1)))
    @test !isnothing(Variable("y", Lower(1)))
    @test !isnothing(Variable("z"))
end

@testset "Literal constructor throws on invalid input" begin
    @test_throws DomainError Literal(2, Lower(2), Lower(2))
end

@testset "Literal constructor succeeds on valid input" begin
    @test Literal(0.6, Upper(1), Lower(2)) isa Literal
    @test Literal(5//1, Lower(1), Lower(2)) isa Literal
    @test Literal(4.0, Upper(1)) isa Literal
    @test Literal(3, Lower(1)) isa Literal
    @test Literal(2) isa Literal
end

@testset "Literal vector constructor" begin
    @test vector(2) == Literal(2, Upper(1))
    @test vector(4.2) == Literal(4.2, Upper(1))
end

@testset "Literal matrix constructor" begin
    @test matrix(2) == Literal(2, Upper(1), Lower(2))
    @test matrix(4.2) == Literal(4.2, Upper(1), Lower(2))
end

@testset "Diagonal matrix constructor" begin
    x = Variable("x", Upper(1))
    xt = Variable("x", Lower(1))

    @test diagm(x) == dc.BinaryOperation{dc.Mult}(KrD(Upper(1), Lower(2)), x)
    @test diagm(xt) == dc.BinaryOperation{dc.Mult}(KrD(Lower(1), Upper(2)), xt)
end

@testset "index equality operator" begin
    left = Lower(3)
    right = Lower(3)
    @test left == right
    @test left == Lower(3)
    @test left != Upper(3)
    @test left != Lower(1)
end

@testset "KrD constructor throws on invalid input" begin
    @test_throws DomainError KrD(Upper(2), Upper(2))
    @test_throws DomainError KrD(Lower(2), Lower(2))
    @test_throws DomainError KrD(Lower(1))
end

@testset "KrD equality operator" begin
    left = KrD(Upper(1), Lower(2))
    @test KrD(Upper(1), Lower(2)) == KrD(Upper(1), Lower(2))
    @test !(KrD(Upper(1), Lower(2)) === KrD(Upper(1), Lower(2)))
    @test left == KrD(Upper(1), Lower(2))
    @test left != KrD(Upper(1), Upper(2))
    @test left != KrD(Lower(1), Lower(2))
    @test left != KrD(Upper(3), Lower(2))
    @test left != KrD(Upper(1), Lower(3))
end

@testset "Zero constructor throws on invalid input" begin
    @test_throws DomainError Zero(Upper(2), Upper(2))
    @test_throws DomainError Zero(Lower(2), Lower(2))
    @test_throws DomainError Zero(Lower(1), Upper(2), Lower(1))
end

@testset "Zero equality operator" begin
    left = Zero(Upper(1), Lower(2))
    @test Zero(Upper(1), Lower(2)) == Zero(Upper(1), Lower(2))
    @test !(Zero(Upper(1), Lower(2)) === Zero(Upper(1), Lower(2)))
    @test left == Zero(Upper(1), Lower(2))
    @test left != Zero(Upper(1), Upper(2))
    @test left != Zero(Lower(1), Lower(2))
    @test left != Zero(Upper(3), Lower(2))
    @test left != Zero(Upper(1), Lower(3))
    @test left != Zero(Upper(1))
    @test left != Zero(Upper(1), Lower(2), Lower(3))
    @test left != Zero()
end

@testset "Variable equality operator" begin
    left = Variable("x", Upper(1), Lower(2))
    @test Variable("x", Upper(1), Lower(2)) == Variable("x", Upper(1), Lower(2))
    @test !(Variable("x", Upper(1), Lower(2)) === Variable("x", Upper(1), Lower(2)))
    @test left == Variable("x", Upper(1), Lower(2))
    @test left != Variable("x", Upper(1), Upper(2))
    @test left != Variable("x", Lower(1), Lower(2))
    @test left != Variable("x", Upper(3), Lower(2))
    @test left != Variable("x", Upper(1), Lower(3))
    @test left != Variable("x", Upper(1))
    @test left != Variable("x", Upper(1), Lower(2), Lower(3))
    @test left != Variable("x2", Upper(1), Lower(2))
end

@testset "Literal equality operator" begin
    left = Literal(2, Upper(1), Lower(2))
    @test Literal(2, Upper(1), Lower(2)) == Literal(2, Upper(1), Lower(2))
    @test !(Literal(2, Upper(1), Lower(2)) === Literal(2, Upper(1), Lower(2)))
    @test left == Literal(2, Upper(1), Lower(2))
    @test left != Literal(2, Upper(1), Upper(2))
    @test left != Literal(2, Lower(1), Lower(2))
    @test left != Literal(2, Upper(3), Lower(2))
    @test left != Literal(2, Upper(1), Lower(3))
    @test left != Literal(2, Upper(1))
    @test left != Literal(2, Upper(1), Lower(2), Lower(3))
    @test left != Literal(3, Upper(1), Lower(2))
end

@testset "Construct unary operations" begin
    a = KrD(Upper(1), Lower(2))
    b = Variable("b", Upper(2))
    c = Variable("c")

    funs = (abs, sign, sin, cos)
    ops = (dc.Abs, dc.Sgn, dc.Sin, dc.Cos)

    for (fun, op) ∈ zip(funs, ops)
        @test_throws DomainError fun(a * b)
        @test typeof(fun(c)) == dc.UnaryOperation{op}

        v = fun.(a * b)

        @test typeof(v) == dc.UnaryOperation{op}
        @test typeof(v.arg) == dc.BinaryOperation{dc.Mult}
    end
end

@testset "norm(⋅, 2) throws for inputs other than vectors" begin
    c = Variable("c")
    A = Variable("A", Upper(1), Lower(2))

    @test_throws DomainError norm(c, 2)
    @test_throws DomainError norm(A, 2)
end

@testset "norm(⋅, 1) throws for inputs other than vectors" begin
    c = Variable("c")
    A = Variable("A", Upper(1), Lower(2))

    @test_throws DomainError norm(c, 1)
    @test_throws DomainError norm(A, 1)
end

@testset "norm(⋅, 2) output" begin
    x = Variable("x", Upper(2))

    op = norm(x, 2)

    @test typeof(op) == dc.Power
    @test op.base == sum(x .^ 2)
    @test op.exponent == 1//2
end

@testset "norm(⋅, 1) output" begin
    x = Variable("x", Upper(2))

    op = norm(x, 1)

    @test op == sum(abs.(x))
end

@testset "UnaryOperation equality operator" begin
    a = KrD(Upper(1), Lower(2))
    b = Variable("b", Upper(2))

    inner = a * b

    ops = (dc.Sin, dc.Cos, dc.Abs, dc.Log, dc.Sgn)

    for (op, other_op) ∈ zip(ops, (ops[2], ops[1:(end-1)]...))
        left = dc.UnaryOperation{op}(inner)

        @test left == dc.UnaryOperation{op}(inner)
        @test left != dc.UnaryOperation{other_op}(inner)
        @test left != -dc.UnaryOperation{op}(inner)
    end
end

@testset "is_permutation true positive" begin
    l = [Lower(9); Upper(2); Lower(2); Lower(2)]
    r = collect(reverse(l))

    @test dc.is_permutation(l, l)
    @test dc.is_permutation(l, r)
    @test dc.is_permutation(r, l)
    @test dc.is_permutation(r, r)
end

@testset "is_permutation true negative" begin
    l = [Lower(9); Upper(2); Lower(2); Lower(2)]
    r1 = [Lower(9); Upper(2); Lower(2)]
    r2 = [Lower(9); Upper(2); Lower(2); Lower(2); Lower(2)]
    r3 = []
    r4 = [Lower(9); Upper(2); Upper(2); Lower(2)]

    @test !dc.is_permutation(l, r1)
    @test !dc.is_permutation(l, r2)
    @test !dc.is_permutation(l, r3)
    @test !dc.is_permutation(l, r4)
end

@testset "BinaryOperation{Mul,Add} equality operator" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Lower(1))

    function create(Op, l, r)
        return BinaryOperation{Op}(l, r)
    end

    for op ∈ (dc.Add, dc.Mult)
        left = create(op, a, b)

        @test create(op, a, b) == create(op, a, b)
        @test left == create(op, a, b)
        @test left == create(op, b, a)
        @test left != BinaryOperation{dc.Sub}(a, b)
    end
end

@testset "BinaryOperation{Sub} equality operator" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Lower(1))

    function create(Op, l, r)
        return BinaryOperation{Op}(l, r)
    end

    for op ∈ (dc.Sub,)
        left = create(op, a, b)

        @test create(op, a, b) == create(op, a, b)
        @test left == create(op, a, b)
        @test left != create(op, b, a)
        @test left != BinaryOperation{dc.Add}(a, b)
    end
end

@testset "Power equality operator" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    e = 1//3

    function create(l, r, e)
        return dc.Power(BinaryOperation{dc.Mult}(l, r), e)
    end

    left = create(a, b, e)

    @test create(a, b, e) == create(a, b, e)
    @test left == create(a, b, e)
    @test left == create(b, a, e)
    @test dc.Power(1//2, 3) != dc.Power(3, 1//2)
    @test left != BinaryOperation{dc.Add}(a, b)
end

@testset "BinaryOperation equivalent" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Lower(1))

    left = dc.BinaryOperation{dc.Mult}(a, b)

    @test equivalent(dc.BinaryOperation{dc.Mult}(a, b), dc.BinaryOperation{dc.Mult}(a, b))
    @test equivalent(left, dc.BinaryOperation{dc.Mult}(a, b))
    @test equivalent(left, dc.BinaryOperation{dc.Mult}(b, a))
    @test !equivalent(left, dc.BinaryOperation{dc.Add}(a, b))
    @test !equivalent(left, dc.BinaryOperation{dc.Mult}(a, Variable("x", Upper(1))))
end

@testset "index hash function" begin
    @test hash(Lower(3)) == hash(Lower(3))
    @test hash(Lower(3)) != hash(Lower(1))
    @test hash(Lower(1)) != hash(Lower(3))
    @test hash(Upper(3)) != hash(Lower(3))
end

@testset "flip" begin
    @test dc.flip(Lower(3)) == Upper(3)
    @test dc.flip(Upper(3)) == Lower(3)
end

@testset "get_free_indices with Variable * Variable and one matching pair" begin
    xt = Variable("x", Lower(1)) # row vector
    A = Variable("A", Upper(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(xt, A)
    op2 = dc.BinaryOperation{dc.Mult}(A, xt)

    @test dc.get_free_indices(op1) == [Lower(2)]
    @test dc.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Variable {+-} Variable" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Lower(2), Upper(1))

    ops = (dc.BinaryOperation{dc.Add}, dc.BinaryOperation{dc.Sub})

    for op ∈ ops
        op1 = op(A, A)
        op2 = op(A, B)
        op3 = op(B, A)

        @test dc.get_free_indices(op1) == [Upper(1); Lower(2)]
        @test dc.get_free_indices(op2) == [Upper(1); Lower(2)]
        @test dc.get_free_indices(op3) == [Lower(2); Upper(1)]
    end
end

@testset "get_free_indices with Variable * KrD and one matching pair" begin
    x = Variable("x", Upper(1))
    δ = KrD(Lower(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(x, δ)
    op2 = dc.BinaryOperation{dc.Mult}(δ, x)

    @test dc.get_free_indices(op1) == [Lower(2)]
    @test dc.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with scalar Variable * KrD" begin
    x = Variable("x")
    δ = KrD(Lower(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(x, δ)
    op2 = dc.BinaryOperation{dc.Mult}(δ, x)

    @test dc.get_free_indices(op1) == [Lower(1); Lower(2)]
    @test dc.get_free_indices(op2) == [Lower(1); Lower(2)]
end

@testset "get_free_indices with Variable * Variable and no matching pairs" begin
    x = Variable("x", Upper(1))
    A = Variable("A", Upper(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(x, A)
    op2 = dc.BinaryOperation{dc.Mult}(A, x)

    @test dc.get_free_indices(op1) == [Upper(1); Lower(2)]
    @test dc.get_free_indices(op2) == [Upper(1); Lower(2)]
end

@testset "multiplication of matrices" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(1), Lower(2))

    p1 = A * B
    p2 = A' * B
    p3 = A * B'
    p4 = A' * B'

    @test length(dc.get_free_indices(p1)) == 2
    @test p1.arg1.indices[2].letter == p1.arg2.indices[1].letter
    @test typeof(p1.arg1.indices[1]) == Upper
    @test typeof(p1.arg1.indices[2]) == Lower
    @test typeof(p1.arg2.indices[1]) == Upper
    @test typeof(p1.arg2.indices[2]) == Lower

    @test length(dc.get_free_indices(p2)) == 2
    @test p2.arg1.indices[1].letter == p2.arg2.indices[1].letter
    @test typeof(p2.arg1.indices[1]) == Lower
    @test typeof(p2.arg1.indices[2]) == Upper
    @test typeof(p2.arg2.indices[1]) == Upper
    @test typeof(p2.arg2.indices[2]) == Lower

    @test length(dc.get_free_indices(p3)) == 2
    @test p3.arg1.indices[2].letter == p3.arg2.indices[2].letter
    @test typeof(p3.arg1.indices[1]) == Upper
    @test typeof(p3.arg1.indices[2]) == Lower
    @test typeof(p3.arg2.indices[1]) == Lower
    @test typeof(p3.arg2.indices[2]) == Upper

    @test length(dc.get_free_indices(p4)) == 2
    @test p4.arg1.indices[1].letter == p4.arg2.indices[2].letter
    @test typeof(p4.arg1.indices[1]) == Lower
    @test typeof(p4.arg1.indices[2]) == Upper
    @test typeof(p4.arg2.indices[1]) == Lower
    @test typeof(p4.arg2.indices[2]) == Upper
end

@testset "addition of matrix and matrix (identity)" begin
    A = Variable("A", Upper(1), Lower(2))

    @test A + I == BinaryOperation{dc.Add}(
        A,
        BinaryOperation{dc.Mult}(true, KrD(Upper(1), Lower(2))),
    )
    @test I + A == BinaryOperation{dc.Add}(
        BinaryOperation{dc.Mult}(true, KrD(Upper(1), Lower(2))),
        A,
    )
end

@testset "subtraction of matrix and matrix (identity)" begin
    A = Variable("A", Upper(1), Lower(2))

    @test A - I == BinaryOperation{dc.Sub}(
        A,
        BinaryOperation{dc.Mult}(true, KrD(Upper(1), Lower(2))),
    )
    @test I - A == BinaryOperation{dc.Sub}(
        BinaryOperation{dc.Mult}(true, KrD(Upper(1), Lower(2))),
        A,
    )
end

@testset "multiplication with matching indices" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Lower(1))
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(2), Lower(3))

    @test dc.get_free_indices(A * x) ==
          dc.get_free_indices(dc.BinaryOperation{dc.Mult}(A, x))
    @test dc.get_free_indices(y * A) ==
          dc.get_free_indices(dc.BinaryOperation{dc.Mult}(y, A))
    @test dc.get_free_indices(A * B) ==
          dc.get_free_indices(dc.BinaryOperation{dc.Mult}(A, B))
end

@testset "multiplication with ambigous input fails" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Lower(1))
    z = Variable("z", Upper(1))
    A = Variable("A", Upper(1), Lower(2), Lower(3))
    B = Variable("B", Upper(1), Lower(2))

    @test_throws DomainError A * x
    @test_throws DomainError y * A
    @test_throws DomainError A * A
end

@testset "multiplication with scalars" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Lower(1))
    A = Variable("A", Upper(1), Lower(2), Lower(3))
    z = Variable("z")
    r = 42

    for n ∈ (z, r)
        for t ∈ (x, y, A, z)
            @test n * t == dc.BinaryOperation{dc.Mult}(n, t)
            @test t * n == dc.BinaryOperation{dc.Mult}(t, n)
        end
    end
end

@testset "elementwise multiplication matrix-matrix" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(3), Lower(4))

    op1 = A .* A

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test equivalent(op1.arg1, Variable("A", Upper(1), Lower(2)))
    @test equivalent(op1.arg2, Variable("A", Upper(1), Lower(2)))

    op2 = A .* B

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test equivalent(op2.arg1, Variable("A", Upper(3), Lower(4)))
    @test equivalent(op2.arg2, Variable("B", Upper(3), Lower(4)))

    op3 = A' .* B'

    @test typeof(op3) == dc.BinaryOperation{dc.Mult}
    @test equivalent(op3.arg1, Variable("A", Lower(1), Upper(2)))
    @test equivalent(op3.arg2, Variable("B", Lower(3), Upper(4)))
end

@testset "elementwise multiplication vector-vector" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(2))

    op1 = x .* x

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test equivalent(op1.arg1, Variable("x", Upper(1)))
    @test equivalent(op1.arg2, Variable("x", Upper(1)))

    op2 = x .* y

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test equivalent(op2.arg1, Variable("x", Upper(2)))
    @test equivalent(op2.arg2, Variable("y", Upper(2)))

    op3 = x' .* y'

    @test typeof(op3) == dc.BinaryOperation{dc.Mult}
    @test equivalent(op3.arg1, Variable("x", Lower(2)))
    @test equivalent(op3.arg2, Variable("y", Lower(2)))
end

@testset "elementwise multiplication with ambiguous input fails" begin
    x = Variable("x", Upper(1))
    A = Variable("A", Upper(3), Lower(4))
    B = Variable("B", Upper(5), Upper(6))
    T = Variable("T", Upper(7), Lower(8), Lower(9))

    @test_throws DomainError x .* x'
    @test_throws DomainError x' .* x
    @test_throws DomainError x .* A
    @test_throws DomainError A .* x
    @test_throws DomainError A .* x'
    @test_throws DomainError x' .* A
    @test_throws DomainError A .* B
    @test_throws DomainError B .* A
    @test_throws DomainError A .* T
    @test_throws DomainError T .* A
end

@testset "multiplication with UniformScaling 1" begin
    a = Variable("a")
    X = Variable("X", Upper(1), Lower(2))

    e = a * I * X

    @test e.arg1 == BinaryOperation{dc.Mult}(
        BinaryOperation{dc.Mult}(true, a),
        KrD(Upper(1), Lower(5)),
    )
    @test e.arg2 == Variable("X", Upper(5), Lower(4))
end

@testset "multiplication with UniformScaling 2" begin
    a = Variable("a")
    X = Variable("X", Upper(1), Lower(2))

    e = a * (I * X)

    @test e.arg1 == a
    @test e.arg2 == BinaryOperation{dc.Mult}(
        BinaryOperation{dc.Mult}(true, KrD(Upper(1), Lower(5))),
        Variable("X", Upper(5), Lower(4)),
    )
end

@testset "multiplication with UniformScaling 3" begin
    a = Variable("a")
    X = Variable("X", Upper(1), Lower(2))

    e = a * I .* X

    @test isempty(dc.get_indices(e.arg1.arg1))
    @test e.arg1.arg2 == KrD(Upper(3), Lower(4))
    @test e.arg2 == Variable("X", Upper(3), Lower(4))
end

@testset "multiplication with UniformScaling 4" begin
    a = Variable("a")
    X = Variable("X", Upper(1), Lower(2))

    e = a * (I .* X)

    @test e.arg1 == a
    @test dc.get_free_indices(e.arg2) == [Upper(3); Lower(4)]
end

@testset "elementwise multiplication with intersecting indices" begin
    # Upper(2) is a default index of KrD.
    # This checks that intersecting indices are updated correctly.
    X = Variable("X", Upper(2), Lower(3))

    e = I .* X

    @test dc.get_free_indices(e) == [Upper(4); Lower(3)]
end

@testset "update_index column vector" begin
    x = Variable("x", Upper(3))

    @test dc.update_index(x, Upper(3), Upper(3)) == x
    @test dc.update_index(x, Upper(3), Upper(1)) == Variable("x", Upper(1))
    @test dc.update_index(x, Upper(3), Upper(2)) == Variable("x", Upper(2))
end

@testset "update_index row vector" begin
    x = Variable("x", Lower(3))

    @test dc.update_index(x, Lower(3), Lower(3)) == x
    @test dc.update_index(x, Lower(3), Lower(1)) == Variable("x", Lower(1))
    @test dc.update_index(x, Lower(3), Lower(2)) == Variable("x", Lower(2))
end

@testset "update_index matrix" begin
    A = Variable("A", Upper(1), Lower(2))

    @test dc.update_index(A, Lower(2), Lower(2)) == A
    @test dc.update_index(A, Lower(2), Lower(3)) == Variable("A", Upper(1), Lower(3))
end

@testset "transpose vector" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Lower(1))

    @test equivalent(x', Variable("x", Lower(1)))
    @test equivalent(y', Variable("y", Upper(1)))
end

@testset "transpose KrD" begin
    d = KrD(Upper(1), Lower(2))

    @test equivalent(d', KrD(Lower(1), Upper(2)))
end

@testset "combined update_index and transpose vector" begin
    x = Variable("x", Upper(2))

    xt = x'
    x_indices = dc.get_free_indices(xt)
    updated_transpose = dc.update_index(xt, x_indices[1], Lower(1))

    @test equivalent(updated_transpose, Variable("x", Lower(1)))
end

@testset "transpose unary operations" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Lower(1))

    ops = (sin, cos)
    types = (dc.Sin, dc.Cos)

    for (op, type) ∈ zip(ops, types)
        op1 = op(y * x)'
        op2 = op.(x)'

        @test typeof(op1) == dc.UnaryOperation{type}
        @test dc.get_free_indices(op1.arg) == dc.get_free_indices(y * x)
        @test equivalent(op2.arg, Variable("x", Lower(1)))
    end
end

@testset "negate any operation" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))

    ops = (A, x, A * x, A + A, sin.(x), cos.(x), abs.(x), sign.(x), tr(A))

    for op ∈ ops
        @test typeof(-op) == dc.BinaryOperation{dc.Mult}
        @test (-op).arg2 == op
    end
end

@testset "transpose matrix" begin
    A = Variable("A", Upper(1), Lower(2))

    @test equivalent(A', Variable("A", Lower(1), Upper(2)))
end

@testset "transpose BinaryOperation{dc.Mult}" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(2))

    op_t = (A * x)'
    @test equivalent(
        op_t,
        dc.BinaryOperation{dc.Mult}(
            Variable("A", Lower(1), Lower(2)),
            Variable("x", Upper(2)),
        ),
    )
end

@testset "dc.Add/dc.Subtract Variables with different order fails" begin
    a = Variable("a")
    x = Variable("x", Upper(1))
    A = Variable("A", Upper(1), Lower(2))
    T = Variable("T", Upper(1), Lower(2), Lower(3))

    Variables = (a, x, A, T)

    for l ∈ Variables
        for r ∈ Variables
            if l == r
                continue
            end

            @test_throws DomainError l + r
            @test_throws DomainError l - r
        end
    end
end

@testset "dc.Add/dc.Subtract Variables with ambiguous indices succeeds" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(2), Lower(3))

    @test equivalent(
        A + B,
        dc.BinaryOperation{dc.Add}(
            Variable("A", Upper(1), Lower(2)),
            Variable("B", Upper(1), Lower(2)),
        ),
    )
end

@testset "dc.Add/dc.Subtract Variables with different indices" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(2))

    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(3), Lower(4))


    for op ∈ (+, -)
        for args ∈ ((x, y), (A, B))
            e = op(args[1], args[2])
            @test e.arg1.indices == e.arg2.indices
        end
    end
end

@testset "transpose BinaryOperation{+-}" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(2))
    z = Variable("z", Upper(3))

    for op ∈ (+, -)
        for ags ∈ ((x, y), (x, z))
            op_t = op(x, y)'
            @test op_t.arg1.indices == op_t.arg2.indices
        end
    end
end

@testset "adjoint of BinaryOperation and BinaryOpeartion of adjoints is consistent" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(3), Lower(4))
    x = Variable("x", Upper(5))
    y = Variable("y", Upper(6))

    @test equivalent(x' * A', (A * x)')
    @test equivalent(x' * A, (A' * x)')
    @test equivalent(x' * A * x, (A' * x)' * x)
end

@testset "trace with matrix input works" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(2), Lower(3))

    @test isempty(dc.get_free_indices(tr(A)))
    @test isempty(dc.get_free_indices(tr(A * B)))
end

@testset "trace with non-matrix input fails" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(2), Lower(3))
    x = Variable("x", Upper(3))

    @test_throws DomainError tr(x)
    @test_throws DomainError tr(A * x)
    @test_throws DomainError tr(x + x)
    @test_throws DomainError tr(A * B * x)
end

@testset "can_contract" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(3))
    z = Variable("z", Lower(1))
    d = KrD(Lower(1), Upper(3))

    @test dc.can_contract(A, x)
    @test dc.can_contract(x, A)
    @test !dc.can_contract(A, y)
    @test !dc.can_contract(y, A)
    @test dc.can_contract(A, d)
    @test dc.can_contract(d, A)
    @test dc.can_contract(A, z)
    @test dc.can_contract(z, A)
end

@testset "multiplication with non-matching indices matrix-vector" begin
    x = Variable("x", Upper(3))
    A = Variable("A", Upper(1), Lower(2))

    op1 = A * x

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test dc.can_contract(op1.arg1, op1.arg2)
    @test dc.get_free_indices(op1) == dc.LowerOrUpperIndex[Upper(1)]

    op2 = x' * A

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test dc.can_contract(op2.arg1, op2.arg2)
    @test dc.get_free_indices(op2) == dc.LowerOrUpperIndex[Lower(2)]
end

@testset "multiplication with non-compatible matrix-vector fails" begin
    x = Variable("x", Upper(3))
    A = Variable("A", Upper(1), Lower(2))

    @test_throws DomainError x * A
end

@testset "multiplication with matrix'-matrix has correct indices" begin
    A = Variable("A", Upper(1), Lower(2))
    C = Variable("C", Upper(3), Lower(4))

    op = A' * C

    @assert typeof(op) == dc.BinaryOperation{dc.Mult}
    op_indices = dc.get_free_indices(op)
    @test length(op_indices) == 2

    @test typeof(dc.get_free_indices(op.arg1)[1]) == Lower
    @test dc.flip(dc.get_free_indices(op.arg1)[1]) == dc.get_free_indices(op.arg2)[1]
    @test typeof(dc.get_free_indices(op.arg2)[end]) == Lower
end

@testset "multiplication with matrix'-matrix' has correct indices" begin
    A = Variable("A", Upper(1), Lower(2))
    C = Variable("C", Upper(3), Lower(4))

    op = A' * C'

    @assert typeof(op) == dc.BinaryOperation{dc.Mult}
    op_indices = dc.get_free_indices(op)
    @test length(op_indices) == 2

    @test typeof(dc.get_free_indices(op.arg1)[1]) == Lower
    @test dc.flip(dc.get_free_indices(op.arg1)[1]) == dc.get_free_indices(op.arg2)[2]
    @test typeof(dc.get_free_indices(op.arg2)[end]) == Upper
end

@testset "vector inner product with mismatching indices" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(1))

    op1 = x' * y

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test dc.can_contract(op1.arg1, op1.arg2)
    @test isempty(dc.get_free_indices(op1))

    op2 = y' * x

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test dc.can_contract(op2.arg1, op2.arg2)
    @test isempty(dc.get_free_indices(op2))
end

@testset "multiplication with non-matching indices scalar-matrix" begin
    A = Variable("A", Upper(1), Lower(2))
    z = Variable("z")

    op1 = A * z
    op2 = z * A

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == z
    @test op1.arg2 == A

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == z
    @test op2.arg2 == A
end

@testset "multiplication with non-matching indices scalar-vector" begin
    x = Variable("x", Upper(3))
    z = Variable("z")

    op1 = z * x
    op2 = x * z

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == z
    @test op1.arg2 == x

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == z
    @test op2.arg2 == x
end

@testset "multiplication with adjoint and adjoint of multiplication is equal" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))

    @test dc.get_free_indices(x' * A') == dc.get_free_indices((A * x)')
    @test dc.get_free_indices(x' * A) == dc.get_free_indices((A' * x)')
end

@testset "to_string output is correct for primitive types" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(1), Upper(2), Upper(3), Lower(4), Upper(5), Lower(6), Lower(7))
    x = Variable("x", Upper(2))
    y = Variable("y", Lower(1))
    z = Variable("z")
    d1 = KrD(Upper(1), Upper(2))
    d2 = KrD(Upper(3), Lower(4))
    zero = Zero(Upper(1), Lower(3), Lower(4))

    @test dc.to_string(A) == "A¹₂"
    @test dc.to_string(B) == "B¹²³₄⁵₆₇"
    @test dc.to_string(x) == "x²"
    @test dc.to_string(y) == "y₁"
    @test dc.to_string(z) == "z"
    @test dc.to_string(d1) == "δ¹²"
    @test dc.to_string(d2) == "δ³₄"
    @test dc.to_string(zero) == "0¹₃₄"
end

@testset "to_string output is correct for BinaryOperation" begin
    a = Variable("a")
    b = Variable("b")

    mul = dc.BinaryOperation{dc.Mult}(a, b)
    add = dc.BinaryOperation{dc.Add}(a, b)
    sub = dc.BinaryOperation{dc.Sub}(a, b)

    @test dc.to_string(mul) == "ab"
    @test dc.to_string(add) == "a + b"
    @test dc.to_string(sub) == "a - b"
    @test dc.to_string(dc.BinaryOperation{dc.Add}(mul, b)) == "ab + b"
    @test dc.to_string(dc.BinaryOperation{dc.Add}(mul, mul)) == "ab + ab"
    @test dc.to_string(dc.BinaryOperation{dc.Sub}(mul, mul)) == "ab - ab"
    @test dc.to_string(dc.BinaryOperation{dc.Mult}(mul, mul)) == "abab"
    @test dc.to_string(dc.BinaryOperation{dc.Mult}(add, add)) == "(a + b)(a + b)"
    @test dc.to_string(dc.BinaryOperation{dc.Mult}(sub, add)) == "(a - b)(a + b)"
end

@testset "to_string output is correct for negated values" begin
    x = Variable("x", Upper(1))
    a = Variable("a")
    c = 2

    @test dc.to_string(-x) == "-x¹"
    @test dc.to_string(-a) == "-a"
    @test dc.to_string(-c) == "-2"
end
