# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic
using Test

using DiffMatic: Tensor, KrD, Zero
using DiffMatic: evaluate
using DiffMatic: Upper, Lower

dc = DiffMatic

@testset "evaluate Tensor" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))
    z = Tensor("z")

    @test evaluate(A) == A
    @test evaluate(x) == x
    @test evaluate(z) == z
end

@testset "simplify with Tensor and KrD" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    z = Tensor("z")

    d1 = KrD(Lower(1), Upper(3))
    d2 = KrD(Upper(2), Lower(3))

    @test dc.simplify(dc.BinaryOperation{dc.Mult}(A, d1)) == Tensor("A", Upper(3), Lower(2))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(x, d1)) == Tensor("x", Upper(3))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(z, d1)) ==
          dc.BinaryOperation{dc.Mult}(z, d1)
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(A, d2)) == Tensor("A", Upper(1), Lower(3))
end

@testset "simplify transpose" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(1))
    z = Tensor("z")

    @test dc.simplify(A') == Tensor("A", Lower(1), Upper(2))
    @test dc.simplify(x') == Tensor("x", Lower(1))
    @test dc.simplify(z') == Tensor("z")
end

@testset "evaluate BinaryOperation{AdditiveOperation} Matrix and KrD" begin
    X = Tensor("X", Upper(2), Lower(3))
    d = KrD(Upper(2), Lower(3))

    for op ∈ (dc.Add, dc.Sub)
        op1 = dc.BinaryOperation{op}(d, X)
        op2 = dc.BinaryOperation{op}(X, d)
        @test evaluate(op1) == op1
        @test evaluate(op2) == op2
    end
end

@testset "evaluate Matrix + Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = dc.BinaryOperation{dc.Add}(Z, X)
    op2 = dc.BinaryOperation{dc.Add}(X, Z)
    @test evaluate(op1) == X
    @test evaluate(op2) == X
end

@testset "evaluate Matrix - Zero" begin
    X = Tensor("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = dc.BinaryOperation{dc.Sub}(Z, X)
    op2 = dc.BinaryOperation{dc.Sub}(X, Z)
    @test evaluate(op1) == -X
    @test evaluate(op2) == X
end

@testset "evaluate Negate - Negate product" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(1))

    op = dc.BinaryOperation{dc.Mult}(-x, -y)
    @test evaluate(op) == dc.BinaryOperation{dc.Mult}(x, y)
end

@testset "evaluate Zero - Zero product" begin
    zl = Zero(Upper(1), Upper(2))
    zr = Zero(Lower(1), Upper(3))

    op = dc.BinaryOperation{dc.Mult}(zl, zr)
    @test evaluate(op) == Zero(Upper(2), Upper(3))
end

@testset "evaluate Real - Zero product" begin
    a = 1
    z = Zero(Upper(1), Lower(2))

    @test evaluate(dc.BinaryOperation{dc.Mult}(a, z)) == z
    @test evaluate(dc.BinaryOperation{dc.Mult}(z, a)) == z
end

@testset "evaluate sum of Zero and difference" begin
    x = Tensor("x", Upper(1))
    y = Tensor("y", Upper(1))
    z = Zero(Upper(1))

    d = dc.BinaryOperation{dc.Sub}(x, y)

    @test evaluate(dc.BinaryOperation{dc.Add}(z, d)) == d
end

@testset "evaluate sum of addition and addition" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Upper(1))
    c = Tensor("c", Upper(1))
    d = Tensor("d", Upper(1))

    l = dc.BinaryOperation{dc.Add}(a, b)
    r = dc.BinaryOperation{dc.Add}(a, c)
    s = dc.BinaryOperation{dc.Add}(l, r)

    @test dc.evaluate(s) == dc.evaluate(2 * a + (b + c))

    l = dc.BinaryOperation{dc.Add}(a, b)
    r = dc.BinaryOperation{dc.Add}(c, a)
    s = dc.BinaryOperation{dc.Add}(l, r)

    @test dc.evaluate(s) == dc.evaluate(2 * a + (b + c))

    l = dc.BinaryOperation{dc.Add}(b, a)
    r = dc.BinaryOperation{dc.Add}(a, c)
    s = dc.BinaryOperation{dc.Add}(l, r)

    @test dc.evaluate(s) == dc.evaluate(2 * a + (b + c))

    l = dc.BinaryOperation{dc.Add}(b, a)
    r = dc.BinaryOperation{dc.Add}(c, a)
    s = dc.BinaryOperation{dc.Add}(l, r)

    @test dc.evaluate(s) == dc.evaluate(2 * a + (b + c))

    l = dc.BinaryOperation{dc.Add}(a, b)
    r = dc.BinaryOperation{dc.Add}(c, d)
    s = dc.BinaryOperation{dc.Add}(l, r)

    @test dc.evaluate(s) == dc.evaluate((a + b) + (c + d))
end


@testset "evaluate sum of subtraction and addition" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Upper(1))
    c = Tensor("c", Upper(1))
    d = Tensor("d", Upper(1))

    # a - b + a + b
    add_inner = dc.BinaryOperation{dc.Add}(a, b)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(sub, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Mult}(2, a)

    # a - b + b + a
    add_inner = dc.BinaryOperation{dc.Add}(b, a)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(sub, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Mult}(2, a)

    # a - b + c + b
    add_inner = dc.BinaryOperation{dc.Add}(c, b)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(sub, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(a, c)

    # a - b + b + d
    add_inner = dc.BinaryOperation{dc.Add}(b, d)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(sub, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(a, d)

    # a - b + a + d
    add_inner = dc.BinaryOperation{dc.Add}(a, d)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(sub, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(
        dc.BinaryOperation{dc.Mult}(2, a),
        dc.BinaryOperation{dc.Sub}(d, b),
    )

    # a - b + c + d
    add_inner = dc.BinaryOperation{dc.Add}(c, d)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(sub, add_inner)
    @test evaluate(add) == add
end

@testset "evaluate sum of subtraction and subtraction" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Upper(1))
    c = Tensor("c", Upper(1))
    d = Tensor("d", Upper(1))

    # a - b + a - b
    l = dc.BinaryOperation{dc.Sub}(a, b)
    r = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(l, r)
    @test evaluate(add) == dc.BinaryOperation{dc.Sub}(
        dc.BinaryOperation{dc.Mult}(2, a),
        dc.BinaryOperation{dc.Mult}(2, b),
    )

    # a - b + a - c
    l = dc.BinaryOperation{dc.Sub}(a, b)
    r = dc.BinaryOperation{dc.Sub}(a, c)
    add = dc.BinaryOperation{dc.Add}(l, r)
    @test evaluate(add) == dc.BinaryOperation{dc.Sub}(
        dc.BinaryOperation{dc.Mult}(2, a),
        dc.BinaryOperation{dc.Add}(b, c),
    )

    # a - b + c - a
    l = dc.BinaryOperation{dc.Sub}(a, b)
    r = dc.BinaryOperation{dc.Sub}(c, a)
    add = dc.BinaryOperation{dc.Add}(l, r)
    @test evaluate(add) == dc.BinaryOperation{dc.Sub}(c, b)

    # b - a + a - c
    l = dc.BinaryOperation{dc.Sub}(b, a)
    r = dc.BinaryOperation{dc.Sub}(a, c)
    add = dc.BinaryOperation{dc.Add}(l, r)
    @test evaluate(add) == dc.BinaryOperation{dc.Sub}(b, c)

    # b - a + c - a
    l = dc.BinaryOperation{dc.Sub}(b, a)
    r = dc.BinaryOperation{dc.Sub}(c, a)
    add = dc.BinaryOperation{dc.Add}(l, r)
    @test evaluate(add) == dc.BinaryOperation{dc.Sub}(
        dc.BinaryOperation{dc.Add}(b, c),
        dc.BinaryOperation{dc.Mult}(2, a),
    )

    # a - b + c - d
    l = dc.BinaryOperation{dc.Sub}(a, b)
    r = dc.BinaryOperation{dc.Sub}(c, d)
    add = dc.BinaryOperation{dc.Add}(l, r)
    @test evaluate(add) == add
end

@testset "evaluate sum of product and addition" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Upper(1))

    # 2 * a + (a + b)
    add_inner = dc.BinaryOperation{dc.Add}(a, b)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add = dc.BinaryOperation{dc.Add}(prod, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(dc.BinaryOperation{dc.Mult}(3, a), b)

    # (a + b) + 2 * a
    add_inner = dc.BinaryOperation{dc.Add}(a, b)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add = dc.BinaryOperation{dc.Add}(add_inner, prod)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(dc.BinaryOperation{dc.Mult}(3, a), b)

    # 2 * a + (b + a)
    add_inner = dc.BinaryOperation{dc.Add}(b, a)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add = dc.BinaryOperation{dc.Add}(prod, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(dc.BinaryOperation{dc.Mult}(3, a), b)

    # 2 * a + (2 * b + 2 * a)
    add_inner = dc.BinaryOperation{dc.Add}(
        dc.BinaryOperation{dc.Mult}(2, b),
        dc.BinaryOperation{dc.Mult}(2, a),
    )
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add = dc.BinaryOperation{dc.Add}(prod, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(
        dc.BinaryOperation{dc.Mult}(4, a),
        dc.BinaryOperation{dc.Mult}(2, b),
    )

    # 2 * a + (2 * a + b)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add_inner = dc.BinaryOperation{dc.Add}(prod, b)
    add = dc.BinaryOperation{dc.Add}(prod, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(dc.BinaryOperation{dc.Mult}(4, a), b)

    # 2 * a + (b + 2 * a)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add_inner = dc.BinaryOperation{dc.Add}(b, prod)
    add = dc.BinaryOperation{dc.Add}(prod, add_inner)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(dc.BinaryOperation{dc.Mult}(4, a), b)
end

@testset "evaluate sum of product and subtraction" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Upper(1))

    # # 2 * a + (a - b)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add = dc.BinaryOperation{dc.Add}(prod, sub)
    @test evaluate(add) == dc.BinaryOperation{dc.Sub}(dc.BinaryOperation{dc.Mult}(3, a), b)

    # # 2 * a + (b - a)
    sub = dc.BinaryOperation{dc.Sub}(b, a)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    add = dc.BinaryOperation{dc.Add}(prod, sub)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(a, b)

    # 2 * a + (2 * a - b)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    sub = dc.BinaryOperation{dc.Sub}(prod, b)
    add = dc.BinaryOperation{dc.Add}(prod, sub)
    @test evaluate(add) == dc.BinaryOperation{dc.Sub}(dc.BinaryOperation{dc.Mult}(4, a), b)

    # 2 * a + (b - 2 * a)
    prod = dc.BinaryOperation{dc.Mult}(2, a)
    sub = dc.BinaryOperation{dc.Sub}(b, prod)
    add = dc.BinaryOperation{dc.Add}(prod, sub)
    @test evaluate(add) == b
end

@testset "evaluate sum of product and unary value 1" begin
    A = Tensor("A", Upper(1), Lower(2))

    prods = (dc.BinaryOperation{dc.Mult}(2, A), dc.BinaryOperation{dc.Mult}(A, 2))

    for prod ∈ prods
        @test evaluate(dc.BinaryOperation{dc.Add}(prod, A)) ==
              dc.BinaryOperation{dc.Mult}(3, A)
        @test evaluate(dc.BinaryOperation{dc.Add}(A, prod)) ==
              dc.BinaryOperation{dc.Mult}(3, A)
    end
end

@testset "evaluate sum of product and unary value 2" begin
    A = Tensor("A", Upper(1), Lower(2))

    prods = (dc.BinaryOperation{dc.Mult}(1, A), dc.BinaryOperation{dc.Mult}(A, 1))


    for prod ∈ prods
        @test evaluate(dc.BinaryOperation{dc.Add}(prod, A)) ==
              dc.BinaryOperation{dc.Mult}(2, A)
        @test evaluate(dc.BinaryOperation{dc.Add}(A, prod)) ==
              dc.BinaryOperation{dc.Mult}(2, A)
    end
end

@testset "evaluate product of real and real - Tensor product" begin
    a = Tensor("a", Upper(1))
    b = Tensor("b", Upper(1))

    op1 = 2 * dc.BinaryOperation{dc.Mult}(a, 2)
    op2 = 2 * dc.BinaryOperation{dc.Mult}(2, a)
    op3 = 2 * dc.BinaryOperation{dc.Mult}(a, b)

    @test evaluate(op1) == dc.BinaryOperation{dc.Mult}(4, a)
    @test evaluate(op2) == dc.BinaryOperation{dc.Mult}(4, a)
    @test evaluate(op3) == op3

    op4 = dc.BinaryOperation{dc.Mult}(a, 2) * 2
    op5 = dc.BinaryOperation{dc.Mult}(2, a) * 2
    op6 = dc.BinaryOperation{dc.Mult}(a, b) * 2

    @test evaluate(op4) == dc.BinaryOperation{dc.Mult}(4, a)
    @test evaluate(op5) == dc.BinaryOperation{dc.Mult}(4, a)
    @test evaluate(op6) == op6
end

# TODO: evaluate is a no-op here, remove evaluate and move to RicciTest
@testset "evaluate adjoint is consistent" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(3), Lower(4))
    x = Tensor("x", Upper(5))
    y = Tensor("y", Upper(6))

    @test equivalent(evaluate(x' * A'), evaluate((A * x)'))
    @test equivalent(evaluate(x' * A), evaluate((A' * x)'))
    @test equivalent(evaluate(x' * A * x), evaluate((A' * x)' * x))
end

@testset "simplify BinaryOperation vector * KrD" begin
    x = Tensor("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Upper(3), Lower(2))

    @test dc.simplify(dc.BinaryOperation{dc.Mult}(d1, x)) == Tensor("x", Upper(3))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(x, d1)) == Tensor("x", Upper(3))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(d2, x)) == Tensor("x", Upper(3))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(x, d2)) == Tensor("x", Upper(3))
end

@testset "simplify BinaryOperation matrix * KrD" begin
    A = Tensor("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(3))
    d3 = KrD(Upper(4), Lower(1))

    @test dc.simplify(dc.BinaryOperation{dc.Mult}(d1, A)) == Tensor("A", Upper(3), Lower(4))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(A, d1)) == Tensor("A", Upper(3), Lower(4))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(d2, A)) == Tensor("A", Lower(3), Lower(4))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(A, d2)) == Tensor("A", Lower(3), Lower(4))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(d3, A)) == Tensor("A", Upper(2), Lower(1))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(A, d3)) == Tensor("A", Upper(2), Lower(1))
end

@testset "evaluate BinaryOperation matrix * Zero" begin
    A = Tensor("A", Upper(2), Lower(4))
    Z = Zero(Upper(4), Lower(3), Lower(5))

    @test evaluate(dc.BinaryOperation{dc.Mult}(Z, A)) == Zero(Upper(2), Lower(3), Lower(5))
    @test evaluate(dc.BinaryOperation{dc.Mult}(A, Z)) == Zero(Upper(2), Lower(3), Lower(5))
end

@testset "evaluate BinaryOperation Negate * Zero" begin
    A = Tensor("A", Upper(1), Lower(2))
    Z = Zero(Upper(2), Lower(3))

    @test evaluate(dc.BinaryOperation{dc.Mult}(Z, -A)) == dc.Zero(Upper(1), Lower(3))
    @test evaluate(dc.BinaryOperation{dc.Mult}(-A, Z)) == dc.Zero(Upper(1), Lower(3))
end

@testset "simplify BinaryOperation KrD * KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test dc.simplify(dc.BinaryOperation{dc.Mult}(d1, d2)) == KrD(Upper(1), Lower(3))
    @test dc.simplify(dc.BinaryOperation{dc.Mult}(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate fully collapsible Mult * Mult" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))
    d3 = KrD(Upper(3), Lower(4))
    A = Tensor("A", Upper(4), Lower(5))

    op = dc.BinaryOperation{dc.Mult}(
        dc.BinaryOperation{dc.Mult}(d1, d3),
        dc.BinaryOperation{dc.Mult}(A, d2),
    )

    @test dc.simplify(op) == Tensor("A", Upper(1), Lower(5))
end

@testset "evaluate BinaryOperation with outer product" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    @test evaluate(dc.BinaryOperation{dc.Mult}(A, x)) == dc.BinaryOperation{dc.Mult}(A, x)
    @test evaluate(dc.BinaryOperation{dc.Mult}(A, y)) == dc.BinaryOperation{dc.Mult}(A, y)
end

@testset "evaluate subtraction with * and +" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    op1 = A * x - (x + y)
    op2 = (x + y) - A * x

    @test length(dc.get_free_indices(evaluate(op1))) == 1
    @test length(dc.get_free_indices(evaluate(op2))) == 1
end

@testset "evaluate subtraction with * and + and simplify" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))

    op1 = (x + y) - (x + y)
    op2 = 2 * x - 2 * x

    @test equivalent(evaluate(op1), Zero(Upper(2)))
    @test equivalent(evaluate(op2), Zero(Upper(2)))
end

@testset "evaluate subtraction with product with real" begin
    A = Tensor("A", Upper(1), Lower(2))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    function sub(l, r)
        return dc.BinaryOperation{dc.Sub}(l, r)
    end

    @test dc.evaluate(sub(mul(2, A), A)) == A
    @test dc.evaluate(sub(mul(A, 2), A)) == A
end

@testset "evaluate subtraction with product and zero" begin
    A = Tensor("A", Upper(1), Lower(2))
    B = Tensor("B", Upper(2), Lower(3))
    Z = Zero(Upper(1), Lower(3))

    function mul(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    function sub(l, r)
        return dc.BinaryOperation{dc.Sub}(l, r)
    end

    prod = mul(A, B)

    @test dc.evaluate(sub(prod, Z)) == prod
    @test dc.evaluate(sub(Z, prod)) == -prod
end

@testset "evaluate unary operations" begin
    A = Tensor("A", Upper(1), Lower(2))

    ops = (sin, cos)
    types = (dc.Sin, dc.Cos)

    for (op, type) ∈ zip(ops, types)
        @test typeof(op(A)) == UnaryOperation{type}
        @test op(A).arg == A
    end
end

# TODO: Store the original degree in Tensor
# @testset "simplify trace" begin
#     A = Tensor("A", Upper(1), Lower(2))
#     B = Tensor("B", Upper(2), Lower(3))

#     @test dc.simplify(tr(A)) == Tensor("A", Upper(2), Lower(2))
#     @test equivalent(
#         dc.simplify(tr(A * B)),
#         dc.BinaryOperation{dc.Mult}(A, Tensor("B", Upper(2), Lower(1))),
#     )
# end

@testset "simplify outer product - contraction" begin
    A = Tensor("A", Upper(1), Lower(2))
    d = KrD(Upper(3), Lower(4))
    x = Tensor("x", Lower(3))

    mul = dc.BinaryOperation{dc.Mult}

    @test dc.simplify(mul(mul(A, d), x)) == mul(A, Tensor("x", Lower(4)))
    @test dc.simplify(mul(mul(d, A), x)) == mul(A, Tensor("x", Lower(4)))
    @test dc.simplify(mul(x, mul(A, d))) == mul(Tensor("x", Lower(4)), A)
    @test dc.simplify(mul(x, mul(d, A))) == mul(Tensor("x", Lower(4)), A)

    @test dc.simplify(mul(mul(d, A), x)) == mul(Tensor("x", Lower(4)), A)
    @test dc.simplify(mul(mul(A, d), x)) == mul(Tensor("x", Lower(4)), A)
    @test dc.simplify(mul(x, mul(d, A))) == mul(A, Tensor("x", Lower(4)))
    @test dc.simplify(mul(x, mul(A, d))) == mul(A, Tensor("x", Lower(4)))
end

@testset "diff Tensor" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(3))
    A = Tensor("A", Upper(4), Lower(5))

    # TODO: Making this work would require passing a list of all indices down the
    # tree in diff() since we need to have a safe (as in unused) temporary index
    # when splitting the trace.
    # @test dc.diff(x, x) ==
    #   dc.BinaryOperation{dc.Mult}(KrD(Upper(2), Lower(3)), KrD(Upper(3), Lower(2)))
    @test dc.diff(y, x) == Zero(Upper(3), Lower(2))
    @test dc.diff(A, x) == Zero(Upper(4), Lower(5), Lower(2))

    @test dc.diff(x, Tensor("x", Upper(1))) == KrD(Upper(2), Lower(1))
    @test dc.diff(y, Tensor("y", Upper(4))) == KrD(Upper(3), Lower(4))
    @test dc.diff(A, Tensor("A", Upper(6), Lower(7))) ==
          dc.BinaryOperation{dc.Mult}(KrD(Upper(4), Lower(6)), KrD(Lower(5), Upper(7)))
end

@testset "diff KrD" begin
    x = Tensor("x", Upper(3))
    y = Tensor("y", Lower(4))
    A = Tensor("A", Upper(5), Lower(6))
    d = KrD(Upper(1), Lower(2))

    @test dc.diff(d, x) == Zero(Upper(1), Lower(2), Lower(3))
    @test dc.diff(d, y) == Zero(Upper(1), Lower(2), Upper(4))
    @test dc.diff(d, A) == Zero(Upper(1), Lower(2), Lower(5), Upper(6))
end

@testset "diff BinaryOperation{dc.Mult}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Lower(2))

    op = dc.BinaryOperation{dc.Mult}(x, y)

    D = dc.diff(op, Tensor("x", Upper(3)))

    @test typeof(D) == dc.BinaryOperation{dc.Add}
    @test D.arg1 == dc.BinaryOperation{dc.Mult}(x, Zero(Lower(2), Lower(3)))
    @test D.arg2 == dc.BinaryOperation{dc.Mult}(KrD(Upper(2), Lower(3)), y)
end

@testset "diff BinaryOperation{AdditiveOperation}" begin
    x = Tensor("x", Upper(2))
    y = Tensor("y", Upper(2))

    for op ∈ (dc.Add, dc.Sub)
        v = dc.BinaryOperation{op}(x, y)

        D = dc.diff(v, Tensor("x", Upper(3)))

        @test D == dc.BinaryOperation{op}(KrD(Upper(2), Lower(3)), Zero(Upper(2), Lower(3)))
    end
end

@testset "diff trace" begin
    A = Tensor("A", Upper(1), Lower(2))

    op = tr(A)

    D = dc.diff(op, Tensor("A", Upper(3), Lower(4)))

    @test equivalent(evaluate(D), KrD(Upper(1), Lower(2)))
end

@testset "diff sin" begin
    x = Tensor("x", Upper(2))

    op = sin(x)

    D = dc.diff(op, Tensor("x", Upper(3)))

    @test equivalent(
        D,
        dc.BinaryOperation{dc.Mult}(dc.UnaryOperation{dc.Cos}(x), KrD(Upper(2), Lower(3))),
    )
end

@testset "diff cos" begin
    x = Tensor("x", Upper(2))

    op = cos(x)

    D = dc.diff(op, Tensor("x", Upper(3)))

    @test equivalent(
        D,
        dc.BinaryOperation{dc.Mult}(-dc.UnaryOperation{dc.Sin}(x), KrD(Upper(2), Lower(3))),
    )
end

@testset "diff negated vector" begin
    x = Tensor("x", Upper(2))

    op = -x

    D = dc.diff(op, Tensor("x", Upper(3)))

    expected = dc.BinaryOperation{dc.Add}(
        -KrD(Upper(2), Lower(3)),
        dc.BinaryOperation{dc.Mult}(dc.Zero(Lower(3)), x),
    )

    @test equivalent(D, expected)
end

@testset "free indices constant after evaluate" begin
    x = Tensor("x", Upper(2))
    c = Tensor("c", Upper(3))
    y = Tensor("y", Upper(4))

    op1 = (y .* c)' * x

    @test isempty(dc.get_free_indices(op1))
    @test isempty(dc.get_free_indices(evaluate(op1)))

    op2 = tr(x * x')

    @test isempty(dc.get_free_indices(op2))
    @test isempty(dc.get_free_indices(evaluate(op2)))
end

@testset "Differentiate Ax" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test equivalent(dc.diff(A * x, Tensor("x", Upper(5))), A)
end

@testset "Differentiate xᵀA " begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    @test equivalent(
        dc.diff(x' * A, Tensor("x", Upper(6))),
        Tensor("A", Lower(1), Lower(2)),
    )
end

@testset "Differentiate xᵀAx" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = dc.diff(x' * A * x, Tensor("x", Upper(7)))

    @test equivalent(dc.simplify(D.arg1), dc.simplify(x' * A))
    @test equivalent(
        dc.simplify(dc.evaluate(D.arg2)),
        evaluate(dc.BinaryOperation{dc.Mult}(Tensor("A", Lower(1), Lower(3)), x)),
    )
end

@testset "Differentiate xx'x" begin
    x = Tensor("x", Upper(1))

    lr = dc.BinaryOperation{dc.Mult}(Tensor("x", Lower(101)), Tensor("x", Upper(100)))
    l = dc.BinaryOperation{dc.Mult}(2, lr)
    rr = dc.BinaryOperation{dc.Mult}(
        dc.BinaryOperation{dc.Mult}(Tensor("x", Lower(1)), x),
        KrD(Upper(100), Lower(101)),
    )
    expected = dc.BinaryOperation{dc.Add}(l, rr)

    D = dc.diff(x * x' * x, Tensor("x", Upper(6)))

    # TODO: Make evaluate/simplify work until the rersult doesn't change anymore
    @test equivalent(dc.evaluate(dc.simplify(dc.evaluate(D))), expected)
end

@testset "Differentiate A(x + 2x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = dc.diff(A * (x + 2 * x), Tensor("x", Upper(5)))

    # TODO: Make evaluate/simplify work until the rersult doesn't change anymore
    @test equivalent(dc.evaluate(dc.simplify(dc.evaluate(D))), 3 * A)
end

@testset "Differentiate A(x + 2x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    # wrt should have the same index as x has in expr
    expr = A * (x + 2 * x)
    wrt = Tensor("x", Upper(4))

    D = dc.diff(expr, wrt)

    expected = dc.BinaryOperation{dc.Mult}(3, KrD(Upper(3), Lower(3)))
    expected = dc.BinaryOperation{dc.Mult}(expected, Tensor("A", Upper(1), Lower(3)))

    @test equivalent(evaluate(D), expected)
end

@testset "Differentiate A(2x + x)" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))

    D = dc.diff(A * (x + 2 * x), Tensor("x", Upper(5)))

    # TODO: Make evaluate/simplify work until the rersult doesn't change anymore
    @test equivalent(dc.evaluate(dc.simplify(dc.evaluate(D))), 3 * A)
end

# TODO: The order of the terms can be different.
# Need to either:
#  - Sort them e.g. lexigraphically or
#  - Overload the equality operator
@testset "evaluated derivative is equal to derivative of evaluated expression" begin
    A = Tensor("A", Upper(1), Lower(2))
    x = Tensor("x", Upper(3))
    y = Tensor("y", Upper(4))
    c = Tensor("c", Upper(5))

    exprs = ( #
        A * x, #
        x' * A, #
        x' * A * x, #
        A * (x + 2 * x), #
        A * (2 * x + x), #
        (x + 2 * x)' * A, #
        (2 * x + x)' * A, #
        x' * x, #
        tr(x * x'), #
        (y .* c)' * x, #
        (x .* c)' * x, #
        # (x + y)' * x, #
        # (x - y)' * x, #
        x' * (y .* c), #
        x' * (x .* c), #
        # x' * (x + y), #
        # x' * (x - y), #
        sin(tr(x * x')), #
        cos(tr(x * x')), #
        tr(sin(x * x')), #
        # tr(A), #
        sum((x + y) .^ 2), #
    )

    for expr ∈ exprs
        @testset "$(dc.to_string(expr))" begin
            var = Tensor("A", Upper(10), Lower(11))
            @test evaluate(dc.diff(evaluate(expr), var)) == evaluate(dc.diff(expr, var))
            var = Tensor("x", Upper(10))
            @test evaluate(dc.diff(evaluate(expr), var)) == evaluate(dc.diff(expr, var))
            var = Tensor("y", Upper(10))
            @test evaluate(dc.diff(evaluate(expr), var)) == evaluate(dc.diff(expr, var))
            var = Tensor("c", Upper(10))
            @test evaluate(dc.diff(evaluate(expr), var)) == evaluate(dc.diff(expr, var))
        end
    end
end
