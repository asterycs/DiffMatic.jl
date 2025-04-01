# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic
using Test

using DiffMatic: Tensor, KrD, Zero
using DiffMatic: evaluate
using DiffMatic: Upper, Lower

dc = DiffMatic

@testset "Tensor constructor throws on invalid input" begin
    @test_throws DomainError Tensor("A", Lower(2), Lower(2))
    @test_throws DomainError Tensor("A", Lower(2), Upper(2), Lower(1), Lower(2))
end

@testset "Tensor constructor succeeds on valid input" begin
    @test !isnothing(Tensor("A", Upper(1), Lower(2)))
    @test !isnothing(Tensor("B", Lower(1), Lower(2)))
    @test !isnothing(Tensor("x", Upper(1)))
    @test !isnothing(Tensor("y", Lower(1)))
    @test !isnothing(Tensor("z"))
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

@testset "Sin constructor" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    op = sin(a * b)

    @test typeof(op) == dc.Sin
    @test typeof(op.arg) == dc.BinaryOperation{dc.Mult}
end

@testset "Cos constructor" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    op = cos(a * b)

    @test typeof(op) == dc.Cos
    @test typeof(op.arg) == dc.BinaryOperation{dc.Mult}
end

@testset "UnaryOperation equality operator" begin
    a = KrD(Upper(1), Lower(2))
    b = Tensor("b", Upper(2))

    inner = a * b

    left = sin(inner)

    @test left == dc.Sin(inner)
    @test left != dc.Cos(inner)
    @test left != -dc.Sin(inner)
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

@testset "BinaryOperation equality operator" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Lower(1))

    left = dc.BinaryOperation{dc.Mult}(a, b)

    @test dc.BinaryOperation{dc.Mult}(a, b) == dc.BinaryOperation{dc.Mult}(a, b)
    @test left == dc.BinaryOperation{dc.Mult}(a, b)
    @test left == dc.BinaryOperation{dc.Mult}(b, a)
    @test left != dc.BinaryOperation{dc.Add}(a, b)
end

@testset "BinaryOperation equivalent" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Lower(1))

    left = dc.BinaryOperation{dc.Mult}(a, b)

    @test equivalent(dc.BinaryOperation{dc.Mult}(a, b), dc.BinaryOperation{dc.Mult}(a, b))
    @test equivalent(left, dc.BinaryOperation{dc.Mult}(a, b))
    @test equivalent(left, dc.BinaryOperation{dc.Mult}(b, a))
    @test !equivalent(left, dc.BinaryOperation{dc.Add}(a, b))
    @test !equivalent(left, dc.BinaryOperation{dc.Mult}(a, Tensor("x", Upper(1))))
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

@testset "get_free_indices with Tensor * Tensor and one matching pair" begin
    xt = Tensor("x", Lower(1)) # row vector
    A = Tensor("A", Upper(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(xt, A)
    op2 = dc.BinaryOperation{dc.Mult}(A, xt)

    @test dc.get_free_indices(op1) == [Lower(2)]
    @test dc.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with Tensor {+-} Tensor" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Lower(2), Upper(1))

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

@testset "get_free_indices with Tensor * KrD and one matching pair" begin
    x = Tensor("x", Upper(1))
    δ = KrD(Lower(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(x, δ)
    op2 = dc.BinaryOperation{dc.Mult}(δ, x)

    @test dc.get_free_indices(op1) == [Lower(2)]
    @test dc.get_free_indices(op2) == [Lower(2)]
end

@testset "get_free_indices with scalar Tensor * KrD" begin
    x = Tensor("x")
    δ = KrD(Lower(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(x, δ)
    op2 = dc.BinaryOperation{dc.Mult}(δ, x)

    @test dc.get_free_indices(op1) == [Lower(1); Lower(2)]
    @test dc.get_free_indices(op2) == [Lower(1); Lower(2)]
end

@testset "get_free_indices with Tensor * Tensor and no matching pairs" begin
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(1), Lower(2))

    op1 = dc.BinaryOperation{dc.Mult}(x, A)
    op2 = dc.BinaryOperation{dc.Mult}(A, x)

    @test dc.get_free_indices(op1) == [Upper(1); Lower(2)]
    @test dc.get_free_indices(op2) == [Upper(1); Lower(2)]
end

@testset "multiplication of matrices" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(1), Lower(2))

    p1 = dc.evaluate(A * B)
    p2 = dc.evaluate(A' * B)
    p3 = dc.evaluate(A * B')
    p4 = dc.evaluate(A' * B')

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

@testset "multiplication with matching indices" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test dc.get_free_indices(A * x) ==
          dc.get_free_indices(dc.BinaryOperation{dc.Mult}(A, x))
    @test dc.get_free_indices(y * A) ==
          dc.get_free_indices(dc.BinaryOperation{dc.Mult}(y, A))
    @test dc.get_free_indices(A * B) ==
          dc.get_free_indices(dc.BinaryOperation{dc.Mult}(A, B))
end

@testset "multiplication with ambigous input fails" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z", Upper(1))
    A = Tensor("A", Upper(1), Lower(2), Lower(3))
    B = Tensor("B", Upper(1), Lower(2))

    @test_throws DomainError A * x
    @test_throws DomainError y * A
    @test_throws DomainError A * A
end

@testset "multiplication with scalars" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    A = Tensor("A", Upper(1), Lower(2), Lower(3))
    z = Tensor("z")
    r = 42

    for n ∈ (z, r)
        for t ∈ (x, y, A, z)
            @test n * t == dc.BinaryOperation{dc.Mult}(n, t)
            @test t * n == dc.BinaryOperation{dc.Mult}(t, n)
        end
    end
end

@testset "elementwise multiplication matrix-matrix" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(3), Lower(4))

    op1 = A .* A

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test equivalent(evaluate(op1.arg1), Tensor("A", Upper(1), Lower(2)))
    @test equivalent(evaluate(op1.arg2), Tensor("A", Upper(1), Lower(2)))

    op2 = A .* B

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test equivalent(evaluate(op2.arg1), Tensor("A", Upper(3), Lower(4)))
    @test equivalent(evaluate(op2.arg2), Tensor("B", Upper(3), Lower(4)))

    op3 = A' .* B'

    @test typeof(op3) == dc.BinaryOperation{dc.Mult}
    @test equivalent(evaluate(op3.arg1), Tensor("A", Lower(1), Upper(2)))
    @test equivalent(evaluate(op3.arg2), Tensor("B", Lower(3), Upper(4)))
end

@testset "elementwise multiplication vector-vector" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))

    op1 = x .* x

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test equivalent(evaluate(op1.arg1), Tensor("x", Upper(1)))
    @test equivalent(evaluate(op1.arg2), Tensor("x", Upper(1)))

    op2 = x .* y

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test equivalent(evaluate(op2.arg1), Tensor("x", Upper(2)))
    @test equivalent(evaluate(op2.arg2), Tensor("y", Upper(2)))

    op3 = x' .* y'

    @test typeof(op3) == dc.BinaryOperation{dc.Mult}
    @test equivalent(evaluate(op3.arg1), Tensor("x", Lower(2)))
    @test equivalent(evaluate(op3.arg2), Tensor("y", Lower(2)))
end

@testset "elementwise multiplication with ambiguous input fails" begin
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(3), Lower(4))
    B = Tensor("B", Upper(5), Upper(6))
    T = Tensor("T", Upper(7), Lower(8), Lower(9))

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

@testset "update_index column vector" begin
    x = Tensor("x", Upper(3))

    @test dc.update_index(x, Upper(3), Upper(3)) == x
    @test dc.update_index(x, Upper(3), Upper(1)) == Tensor("x", Upper(1))
    @test dc.update_index(x, Upper(3), Upper(2)) == Tensor("x", Upper(2))
end

@testset "update_index row vector" begin
    x = Tensor("x", Lower(3))

    @test dc.update_index(x, Lower(3), Lower(3)) == x
    @test dc.update_index(x, Lower(3), Lower(1)) == Tensor("x", Lower(1))
    @test dc.update_index(x, Lower(3), Lower(2)) == Tensor("x", Lower(2))
end

@testset "update_index matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    @test dc.update_index(A, Lower(2), Lower(2)) == A
    @test dc.update_index(A, Lower(2), Lower(3)) == Tensor("A", Upper(1), Lower(3))
    @test dc.update_index(A, Upper(1), Upper(3)) == Tensor("A", Upper(3), Lower(2))
end

@testset "transpose vector" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    @test equivalent(evaluate(x'), Tensor("x", Lower(1)))
    @test equivalent(evaluate(y'), Tensor("y", Upper(1)))
end

@testset "transpose KrD" begin
    d = KrD(Upper(1), Lower(2))

    @test equivalent(evaluate(d'), KrD(Lower(1), Upper(2)))
end

@testset "combined update_index and transpose vector" begin
    x = Tensor("x", Upper(2))

    xt = x'
    x_indices = dc.get_free_indices(xt)
    updated_transpose = evaluate(dc.update_index(xt, x_indices[1], Lower(1)))

    @test equivalent(updated_transpose, Tensor("x", Lower(1)))
end

@testset "transpose unary operations" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Lower(1))

    ops = (sin, cos)
    types = (dc.Sin, dc.Cos)

    for (op, type) ∈ zip(ops, types)
        op1 = evaluate(op(y * x)')
        op2 = evaluate(op(x)')

        @test typeof(op1) == type
        @test dc.get_free_indices(op1.arg) == dc.get_free_indices(y * x)
        @test equivalent(evaluate(op2).arg, Tensor("x", Lower(1)))
    end
end

@testset "negate any operation" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    ops = (A, x, A * x, A + A, sin(x), cos(x), tr(A))

    for op ∈ ops
        @test typeof(-op) == dc.Negate
        @test (-op).arg == op
    end
end

@testset "transpose matrix" begin
    A = Tensor("A", Upper(1), Lower(2))

    At = evaluate(A')
    @test equivalent(At, Tensor("A", Lower(1), Upper(2)))
end

@testset "transpose BinaryOperation{dc.Mult}" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))

    op_t = evaluate((A * x)')
    @test equivalent(
        evaluate(op_t),
        dc.BinaryOperation{dc.Mult}(Tensor("A", Lower(1), Lower(2)), Tensor("x", Upper(2))),
    )
end

@testset "dc.Add/dc.Subtract tensors with different order fails" begin
    a = Tensor("a")
    x = Tensor("x", Upper(1))
    A = Tensor("A", Upper(1), Lower(2))
    T = Tensor("T", Upper(1), Lower(2), Lower(3))

    tensors = (a, x, A, T)

    for l ∈ tensors
        for r ∈ tensors
            if l == r
                continue
            end

            @test_throws DomainError l + r
            @test_throws DomainError l - r
        end
    end
end

@testset "dc.Add/dc.Subtract tensors with ambiguous indices succeeds" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test equivalent(
        evaluate(A + B),
        dc.BinaryOperation{dc.Add}(
            Tensor("A", Upper(1), Lower(2)),
            Tensor("B", Upper(1), Lower(2)),
        ),
    )
end

@testset "dc.Add/dc.Subtract tensors with different indices" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(2))

    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(3), Lower(4))


    for op ∈ (+, -)
        for args ∈ ((x, y), (A, B))
            e = evaluate((op(args[1], args[2])))
            @test e.arg1.indices == e.arg2.indices
        end
    end
end

@testset "transpose BinaryOperation{+-}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(2))
    z = Tensor("z", Upper(3))

    for op ∈ (+, -)
        for ags ∈ ((x, y), (x, z))
            op_t = evaluate((op(x, y))')
            @test op_t.arg1.indices == op_t.arg2.indices
        end
    end
end

@testset "trace with matrix input works" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))

    @test isempty(dc.get_free_indices(tr(A)))
    @test isempty(dc.get_free_indices(tr(A * B)))
end

@testset "trace with non-matrix input fails" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))
    x = Tensor("x", Upper(3))

    @test_throws DomainError tr(x)
    @test_throws DomainError tr(A * x)
    @test_throws DomainError tr(x + x)
    @test_throws DomainError tr(A * B * x)
end

@testset "can_contract" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))
    z = Tensor("z", Lower(1))
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
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

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
    x = Tensor("x", Upper(3))
    A = Tensor("A", Upper(1), Lower(2))

    @test_throws DomainError x * A
end

@testset "multiplication with matrix'-matrix has correct indices" begin
    A = Tensor("A", Upper(1), Lower(2))
    C = Tensor("C", Upper(3), Lower(4))

    op = A' * C

    @assert typeof(op) == dc.BinaryOperation{dc.Mult}
    op_indices = dc.get_free_indices(op)
    @test length(op_indices) == 2

    @test typeof(dc.get_free_indices(op.arg1)[1]) == Lower
    @test dc.flip(dc.get_free_indices(op.arg1)[1]) == dc.get_free_indices(op.arg2)[1]
    @test typeof(dc.get_free_indices(op.arg2)[end]) == Lower
end

@testset "multiplication with matrix'-matrix' has correct indices" begin
    A = Tensor("A", Upper(1), Lower(2))
    C = Tensor("C", Upper(3), Lower(4))

    op = A' * C'

    @assert typeof(op) == dc.BinaryOperation{dc.Mult}
    op_indices = dc.get_free_indices(op)
    @test length(op_indices) == 2

    @test typeof(dc.get_free_indices(op.arg1)[1]) == Lower
    @test dc.flip(dc.get_free_indices(op.arg1)[1]) == dc.get_free_indices(op.arg2)[2]
    @test typeof(dc.get_free_indices(op.arg2)[end]) == Upper
end

@testset "vector inner product with mismatching indices" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(1))

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
    A = Tensor("A", Upper(1), Lower(2))
    z = Tensor("z")

    op1 = A * z
    op2 = z * A

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == A
    @test op1.arg2 == z

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == z
    @test op2.arg2 == A
end

@testset "multiplication with non-matching indices scalar-vector" begin
    x = Tensor("x", Upper(3))
    z = Tensor("z")

    op1 = z * x
    op2 = x * z

    @test typeof(op1) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op1.arg1, op1.arg2)
    @test op1.arg1 == z
    @test op1.arg2 == x

    @test typeof(op2) == dc.BinaryOperation{dc.Mult}
    @test !dc.can_contract(op2.arg1, op2.arg2)
    @test op2.arg1 == x
    @test op2.arg2 == z
end

@testset "multiplication with adjoint and adjoint of multiplication is equal" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test dc.get_free_indices(x' * A') == dc.get_free_indices((A * x)')
    @test dc.get_free_indices(x' * A) == dc.get_free_indices((A' * x)')
end

@testset "to_string output is correct for primitive types" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(1), Upper(2), Upper(3), Lower(4), Upper(5), Lower(6), Lower(7))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(1))
    z = Tensor("z")
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
    a = Tensor("a")
    b = Tensor("b")

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
    x = Tensor("x", Upper(1))
    a = Tensor("a")
    c = 2

    @test dc.to_string(dc.Negate(x)) == "-x¹"
    @test dc.to_string(dc.Negate(a)) == "-a"
    @test_broken dc.to_string(dc.Negate(c)) == "-2"
end
