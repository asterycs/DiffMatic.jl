# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic
using Test

using DiffMatic: Variable, Literal, KrD, Zero
using DiffMatic: evaluate
using DiffMatic: Upper, Lower

using LinearAlgebra: tr

dc = DiffMatic

@testset "evaluate Variable" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))
    z = Variable("z")

    @test evaluate(A) == A
    @test evaluate(x) == x
    @test evaluate(z) == z
end

@testset "evaluate with Variable and KrD" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(1))
    z = Variable("z")

    d1 = KrD(Lower(1), Upper(3))
    d2 = KrD(Upper(2), Lower(3))

    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(A, d1)) ==
          Variable("A", Upper(3), Lower(2))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(x, d1)) == Variable("x", Upper(3))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(z, d1)) ==
          dc.BinaryOperation{dc.Mult}(z, d1)
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(A, d2)) ==
          Variable("A", Upper(1), Lower(3))
end

@testset "evaluate transpose" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(1))
    z = Variable("z")

    @test dc.evaluate(A') == Variable("A", Lower(1), Upper(2))
    @test dc.evaluate(x') == Variable("x", Lower(1))
    @test dc.evaluate(z') == Variable("z")
end

@testset "evaluate log" begin
    x = Variable("x", Upper(1))
    d = KrD(Upper(2), Lower(1))

    function lgp(l, r)
        return dc.UnaryOperation{dc.Sin}(dc.BinaryOperation{dc.Mult}(l, r))
    end

    @test dc.evaluate(lgp(x, d)) == dc.UnaryOperation{dc.Sin}(Variable("x", Upper(2)))
    @test dc.evaluate(lgp(d, x)) == dc.UnaryOperation{dc.Sin}(Variable("x", Upper(2)))
end

@testset "evaluate product of log and KrD" begin
    x = Variable("x", Upper(1))
    d = KrD(Upper(2), Lower(1))
    d_unrelated = KrD(Upper(5), Lower(6))

    function prodlogl(l, r)
        return dc.BinaryOperation{dc.Mult}(dc.UnaryOperation{dc.Sin}(l), r)
    end

    function prodlogr(l, r)
        return dc.BinaryOperation{dc.Mult}(l, dc.UnaryOperation{dc.Sin}(r))
    end

    expected = dc.UnaryOperation{dc.Sin}(Variable("x", Upper(2)))

    @test dc.evaluate(prodlogl(x, d)) == expected
    @test dc.evaluate(prodlogr(d, x)) == expected

    expected2 = prodlogl(x, d_unrelated)

    @test dc.evaluate(prodlogl(x, d_unrelated)) == expected2
    @test dc.evaluate(prodlogr(d_unrelated, x)) == expected2
end

@testset "evaluate product of additions" begin
    a = Variable("a", Lower(1))
    b = Variable("b", Lower(1))
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(1))

    l = dc.BinaryOperation{dc.Add}(a, b)
    r = dc.BinaryOperation{dc.Add}(x, y)

    e = dc.BinaryOperation{dc.Mult}(l, r)

    @test dc.evaluate(e) == e
end

@testset "evaluate trivially simplifiable quotient" begin
    x = Variable("x", Upper(1))
    d = KrD(Upper(2), Lower(1))

    function prod(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    function reciprocal(t)
        return dc.Power(t, -1)
    end

    @test dc.evaluate(prod(x, reciprocal(x))) == Literal(1, Upper(1))
    @test dc.evaluate(prod(prod(x, d), reciprocal(prod(d, x)))) == Literal(1, Upper(2))
    @test dc.evaluate(prod(prod(d, x), reciprocal(prod(x, d)))) == Literal(1, Upper(2))
end

@testset "evaluate BinaryOperation{AdditiveOperation} Matrix and KrD" begin
    X = Variable("X", Upper(2), Lower(3))
    d = KrD(Upper(2), Lower(3))

    for op ∈ (dc.Add, dc.Sub)
        op1 = dc.BinaryOperation{op}(d, X)
        op2 = dc.BinaryOperation{op}(X, d)
        @test evaluate(op1) == op1
        @test evaluate(op2) == op2
    end
end

@testset "evaluate contracting and element-wise product of matrix and matrix" begin
    d = KrD(Upper(2), Upper(3))
    A = Variable("A", Upper(2), Lower(3))

    op1 = dc.BinaryOperation{dc.Mult}(A, d)
    op2 = dc.BinaryOperation{dc.Mult}(d, A)
    @test evaluate(op1) == op1
    @test evaluate(op2) == op2
end

@testset "evaluate Matrix + Zero" begin
    X = Variable("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = dc.BinaryOperation{dc.Add}(Z, X)
    op2 = dc.BinaryOperation{dc.Add}(X, Z)
    @test evaluate(op1) == X
    @test evaluate(op2) == X
end

@testset "evaluate Matrix - Zero" begin
    X = Variable("X", Upper(2), Lower(3))
    Z = Zero(Upper(2), Lower(3))

    op1 = dc.BinaryOperation{dc.Sub}(Z, X)
    op2 = dc.BinaryOperation{dc.Sub}(X, Z)
    @test evaluate(op1) == -X
    @test evaluate(op2) == X
end

@testset "evaluate Negate - Negate product" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(1))

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
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(1))
    z = Zero(Upper(1))

    d = dc.BinaryOperation{dc.Sub}(x, y)

    @test evaluate(dc.BinaryOperation{dc.Add}(z, d)) == d
end

@testset "evaluate sum of Zero and sum" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(1))
    z = Zero(Upper(1))

    d = dc.BinaryOperation{dc.Add}(x, y)

    @test evaluate(dc.BinaryOperation{dc.Add}(z, d)) == d
    @test evaluate(dc.BinaryOperation{dc.Add}(d, z)) == d
end

@testset "evaluate sum of real and real" begin
    function add(l, r)
        return dc.BinaryOperation{dc.Add}(l, r)
    end

    @test evaluate(add(2, 2)) == 4
end

@testset "evaluate sum of addition and Value" begin
    a = Variable("a", Upper(1))
    b = dc.UnaryOperation{dc.Log}(Variable("b", Upper(1)))

    l = dc.BinaryOperation{dc.Add}(a, b)
    s = dc.BinaryOperation{dc.Add}(l, b)
    n = dc.BinaryOperation{dc.Add}(l, 2)

    @test dc.evaluate(s) == dc.evaluate(2 * b + a)
    @test dc.evaluate(n) == n
end

@testset "evaluate sum of difference and Value" begin
    a = Variable("a", Upper(1))
    b = dc.UnaryOperation{dc.Log}(Variable("b", Upper(1)))

    l = dc.BinaryOperation{dc.Sub}(a, b)
    s = dc.BinaryOperation{dc.Add}(l, b)
    n = dc.BinaryOperation{dc.Add}(l, 2)

    @test dc.evaluate(s) == a
    @test dc.evaluate(n) == n
end

@testset "evaluate sum of difference and Zero" begin
    a = Variable("a", Upper(1))
    b = dc.UnaryOperation{dc.Sin}(Variable("b", Upper(1)))
    z = Zero(Upper(1))

    l = dc.BinaryOperation{dc.Sub}(a, b)
    s = dc.BinaryOperation{dc.Add}(l, z)
    s2 = dc.BinaryOperation{dc.Add}(z, l)

    @test dc.evaluate(s) == l
    @test dc.evaluate(s2) == l
end

@testset "evaluate sum of difference and product" begin
    a = Variable("a", Upper(1))
    b = dc.UnaryOperation{dc.Sin}(Variable("b", Lower(1)))
    c = Variable("c", Upper(1))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    p = mult(a, b)
    d = dc.BinaryOperation{dc.Sub}(p, c)
    sum = dc.BinaryOperation{dc.Add}(p, d)
    sum2 = dc.BinaryOperation{dc.Add}(d, p)

    expected = dc.BinaryOperation{dc.Sub}(mult(2, mult(a, b)), c)

    @test dc.evaluate(sum) == expected
    @test dc.evaluate(sum2) == expected
end

@testset "evaluate sum of addition and addition" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    c = Variable("c", Upper(1))
    d = Variable("d", Upper(1))

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
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    c = Variable("c", Upper(1))
    d = Variable("d", Upper(1))

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

    # c + d + a - b
    add_inner = dc.BinaryOperation{dc.Add}(c, d)
    sub = dc.BinaryOperation{dc.Sub}(a, b)
    add = dc.BinaryOperation{dc.Add}(add_inner, sub)
    @test evaluate(add) == add
end

@testset "evaluate sum of subtraction and sum of subtraction and variable" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    c = Variable("c", Upper(1))

    e = ((a - b) - c) + ((b - a) + c)

    @test evaluate(e) == Zero(Upper(1))
end

@testset "evaluate sum of subtraction and zero" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    c = Variable("c", Upper(1))

    # a + b - (c - c)
    add = dc.BinaryOperation{dc.Add}(a, b)
    sub = dc.BinaryOperation{dc.Sub}(c, c)
    add = dc.BinaryOperation{dc.Add}(add, sub)
    @test evaluate(add) == dc.BinaryOperation{dc.Add}(a, b)
end

@testset "evaluate sum of subtraction and subtraction" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    c = Variable("c", Upper(1))
    d = Variable("d", Upper(1))

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
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))

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
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))

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
    A = Variable("A", Upper(1), Lower(2))

    prods = (dc.BinaryOperation{dc.Mult}(2, A), dc.BinaryOperation{dc.Mult}(A, 2))

    for prod ∈ prods
        @test evaluate(dc.BinaryOperation{dc.Add}(prod, A)) ==
              dc.BinaryOperation{dc.Mult}(3, A)
        @test evaluate(dc.BinaryOperation{dc.Add}(A, prod)) ==
              dc.BinaryOperation{dc.Mult}(3, A)
    end
end

@testset "evaluate sum of product and unary value 2" begin
    A = Variable("A", Upper(1), Lower(2))

    prods = (dc.BinaryOperation{dc.Mult}(1, A), dc.BinaryOperation{dc.Mult}(A, 1))


    for prod ∈ prods
        @test evaluate(dc.BinaryOperation{dc.Add}(prod, A)) ==
              dc.BinaryOperation{dc.Mult}(2, A)
        @test evaluate(dc.BinaryOperation{dc.Add}(A, prod)) ==
              dc.BinaryOperation{dc.Mult}(2, A)
    end
end

@testset "evaluate product of real and real - Variable product" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))

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

@testset "evaluate product of real and product" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))

    op1 = 1 * dc.BinaryOperation{dc.Mult}(a, b)
    op2 = 2 * dc.BinaryOperation{dc.Mult}(2, b)
    op3 = 2 * dc.BinaryOperation{dc.Mult}(a, 2)

    @test evaluate(op1) == dc.BinaryOperation{dc.Mult}(a, b)
    @test evaluate(op2) == dc.BinaryOperation{dc.Mult}(4, b)
    @test evaluate(op3) == dc.BinaryOperation{dc.Mult}(4, a)
end

@testset "evaluate product of Zero and KrD" begin
    z = Zero(Upper(1), Lower(2))
    d = KrD(Upper(1), Lower(3))
    d2 = KrD(Upper(2), Lower(3))

    op1 = dc.BinaryOperation{dc.Mult}(z, d)
    op2 = dc.BinaryOperation{dc.Mult}(d, z)

    op3 = dc.BinaryOperation{dc.Mult}(z, d2)
    op4 = dc.BinaryOperation{dc.Mult}(d2, z)

    @test evaluate(op1) == Zero(Upper(1), Lower(2), Lower(3))
    @test evaluate(op2) == Zero(Upper(1), Lower(2), Lower(3))
    @test evaluate(op3) == Zero(Upper(1), Lower(3))
    @test evaluate(op4) == Zero(Upper(1), Lower(3))
end

@testset "evaluate product of zero (Real) and product" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Lower(2))

    p = dc.BinaryOperation{dc.Mult}(a, b)

    op1 = dc.BinaryOperation{dc.Mult}(p, 0)
    op2 = dc.BinaryOperation{dc.Mult}(0, p)

    @test evaluate(op1) == Zero(Upper(1), Lower(2))
    @test evaluate(op2) == Zero(Upper(1), Lower(2))
end

@testset "evaluate product of powers with same base" begin
    l = dc.Power(Variable("a", Upper(1)), 2)
    r1 = dc.Power(Variable("a", Upper(1)), 3)
    r2 = dc.Power(Variable("a", Upper(1)), -2)
    r3 = dc.Power(Variable("a", Upper(1)), -2)

    for r ∈ (r1, r2, r3)
        op1 = dc.BinaryOperation{dc.Mult}(l, r)
        op2 = dc.BinaryOperation{dc.Mult}(r, l)

        @test evaluate(op1) == dc.Power(Variable("a", Upper(1)), l.exponent + r.exponent)
        @test evaluate(op2) == dc.Power(Variable("a", Upper(1)), l.exponent + r.exponent)
    end
end

@testset "evaluate product of powers with differing base" begin
    l = dc.Power(Variable("a", Upper(1)), 2)
    r1 = dc.Power(Variable("b", Upper(1)), 3)
    r2 = dc.Power(Variable("a", Upper(2)), -2)
    r3 = dc.Power(Variable("a", Lower(1)), -2)

    for r ∈ (r1, r2, r3)
        op1 = dc.BinaryOperation{dc.Mult}(l, r)
        op2 = dc.BinaryOperation{dc.Mult}(r, l)

        @test evaluate(op1) == op1
        @test evaluate(op2) == op2
    end
end

@testset "evaluate product of Zero and power" begin
    p = dc.Power(Variable("a", Upper(1)), 2)
    z = Zero(Upper(1))

    op1 = dc.BinaryOperation{dc.Mult}(p, z)
    op2 = dc.BinaryOperation{dc.Mult}(z, p)

    @test evaluate(op1) == Zero(Upper(1))
    @test evaluate(op2) == Zero(Upper(1))
end

@testset "evaluate product of zero and power" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Lower(2))

    p = dc.BinaryOperation{dc.Mult}(a, b)

    op1 = dc.BinaryOperation{dc.Mult}(p, 0)
    op2 = dc.BinaryOperation{dc.Mult}(0, p)

    @test evaluate(op1) == Zero(Upper(1), Lower(2))
    @test evaluate(op2) == Zero(Upper(1), Lower(2))
end

@testset "evaluate product of product and real" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    op1 = mult(mult(a, b), 3)
    op2 = mult(3, mult(a, b))

    @test evaluate(op1) == op2
    @test evaluate(op2) == op2
end

@testset "evaluate product of products" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    c = Variable("c", Upper(1))
    d = Variable("d", Upper(1))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    op1 = mult(mult(-1, a), mult(-1, b))
    op2 = mult(mult(-1, a), mult(b, c))
    op3 = mult(mult(a, b), mult(-1, c))

    @test evaluate(op1) == mult(a, b)
    @test evaluate(op2) == mult(-1, mult(a, mult(b, c)))
    @test evaluate(op3) == mult(-1, mult(c, mult(a, b)))
end

@testset "evaluate product of products 2" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(2))
    x = Variable("x", Lower(1))
    y = Variable("y", Lower(2))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    op1 = mult(mult(a, b), mult(x, y))
    op2 = mult(mult(a, y), mult(b, x))

    @test evaluate(op1) == mult(mult(a, x), mult(b, y))
    @test evaluate(op2) == mult(mult(a, x), mult(b, y))
end

@testset "evaluate product of products 3" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(1))
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(2))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    op1 = mult(mult(a, b), mult(x, y))
    op2 = mult(mult(a, y), mult(b, x))
    op3 = mult(mult(x, y), mult(a, b))

    @test evaluate(op1) == op1
    @test evaluate(op2) == op1
    @test evaluate(op3) == op3
end

@testset "evaluate product of products 4" begin
    a = Variable("a", Upper(1))
    b = Variable("b", Upper(2))
    x = Variable("x", Lower(1))
    y = Variable("y", Upper(2))
    z = Variable("z", Upper(3))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    op1 = mult(mult(a, b), mult(x, y))
    op2 = mult(mult(a, y), mult(b, z))

    @test evaluate(op1) == mult(mult(a, x), mult(b, y))
    @test evaluate(op2) == mult(mult(mult(y, b), a), z)
end

@testset "evaluate product of product and KrD 1" begin
    X = Variable("X", Upper(1), Lower(2))
    A = Variable("A", Upper(2), Lower(3))
    d = KrD(Lower(1), Upper(4))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test evaluate(mult(mult(X, A), d)) == mult(Variable("X", Upper(4), Lower(2)), A)
    @test evaluate(mult(d, mult(X, A))) == mult(Variable("X", Upper(4), Lower(2)), A)


    @test evaluate(mult(mult(A, X), d)) == mult(A, Variable("X", Upper(4), Lower(2)))
    @test evaluate(mult(d, mult(A, X))) == mult(A, Variable("X", Upper(4), Lower(2)))
end

@testset "evaluate product of product and KrD 2" begin
    X = Variable("X", Upper(1), Lower(2))
    Y = Variable("Y", Upper(1), Lower(2))
    d = KrD(Lower(1), Upper(4))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test evaluate(mult(mult(X, Y), d)) ==
          mult(Variable("X", Upper(4), Lower(2)), Variable("Y", Upper(4), Lower(2)))
    @test evaluate(mult(d, mult(X, Y))) ==
          mult(Variable("X", Upper(4), Lower(2)), Variable("Y", Upper(4), Lower(2)))

    @test evaluate(mult(mult(Y, X), d)) ==
          mult(Variable("Y", Upper(4), Lower(2)), Variable("X", Upper(4), Lower(2)))
    @test evaluate(mult(d, mult(Y, X))) ==
          mult(Variable("Y", Upper(4), Lower(2)), Variable("X", Upper(4), Lower(2)))
end

@testset "evaluate product of product and KrD 3" begin
    X = Variable("X", Upper(1), Lower(2))
    Y = Variable("Y", Upper(1), Lower(3))
    d = KrD(Upper(2), Lower(4))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test evaluate(mult(mult(X, Y), d)) ==
          mult(Variable("X", Upper(1), Lower(4)), Variable("Y", Upper(1), Lower(3)))
    @test evaluate(mult(d, mult(X, Y))) ==
          mult(Variable("X", Upper(1), Lower(4)), Variable("Y", Upper(1), Lower(3)))

    @test evaluate(mult(mult(Y, X), d)) ==
          mult(Variable("Y", Upper(1), Lower(3)), Variable("X", Upper(1), Lower(4)))
    @test evaluate(mult(d, mult(Y, X))) ==
          mult(Variable("Y", Upper(1), Lower(3)), Variable("X", Upper(1), Lower(4)))
end

@testset "evaluate product of product and KrD 4" begin
    X = Variable("X", Upper(1), Lower(2))
    Y = Variable("Y", Upper(1), Lower(3))
    d = KrD(Upper(4), Lower(1))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test evaluate(mult(mult(X, Y), d)) ==
          mult(Variable("X", Upper(4), Lower(2)), Variable("Y", Upper(4), Lower(3)))
    @test evaluate(mult(d, mult(X, Y))) ==
          mult(Variable("X", Upper(4), Lower(2)), Variable("Y", Upper(4), Lower(3)))

    @test evaluate(mult(mult(Y, X), d)) ==
          mult(Variable("Y", Upper(4), Lower(3)), Variable("X", Upper(4), Lower(2)))
    @test evaluate(mult(d, mult(Y, X))) ==
          mult(Variable("Y", Upper(4), Lower(3)), Variable("X", Upper(4), Lower(2)))
end

@testset "evaluate product of product and KrD 5" begin
    X = Variable("X", Upper(1), Lower(2))
    d = KrD(Upper(2), Lower(4))

    function mult(l, r)
        return dc.BinaryOperation{dc.Mult}(l, r)
    end

    @test evaluate(mult(mult(2, X), d)) == mult(2, Variable("X", Upper(1), Lower(4)))
    @test evaluate(mult(d, mult(2, X))) == mult(2, Variable("X", Upper(1), Lower(4)))
end

@testset "evaluate BinaryOperation vector * KrD" begin
    x = Variable("x", Upper(2))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Upper(3), Lower(2))

    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(d1, x)) == Variable("x", Upper(3))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(x, d1)) == Variable("x", Upper(3))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(d2, x)) == Variable("x", Upper(3))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(x, d2)) == Variable("x", Upper(3))
end

@testset "evaluate BinaryOperation matrix * KrD" begin
    A = Variable("A", Upper(2), Lower(4))
    d1 = KrD(Lower(2), Upper(3))
    d2 = KrD(Lower(2), Lower(3))
    d3 = KrD(Upper(4), Lower(1))

    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(d1, A)) ==
          Variable("A", Upper(3), Lower(4))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(A, d1)) ==
          Variable("A", Upper(3), Lower(4))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(d2, A)) ==
          Variable("A", Lower(3), Lower(4))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(A, d2)) ==
          Variable("A", Lower(3), Lower(4))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(d3, A)) ==
          Variable("A", Upper(2), Lower(1))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(A, d3)) ==
          Variable("A", Upper(2), Lower(1))
end

@testset "evaluate BinaryOperation matrix * Zero" begin
    A = Variable("A", Upper(2), Lower(4))
    Z = Zero(Upper(4), Lower(3), Lower(5))

    @test evaluate(dc.BinaryOperation{dc.Mult}(Z, A)) == Zero(Upper(2), Lower(3), Lower(5))
    @test evaluate(dc.BinaryOperation{dc.Mult}(A, Z)) == Zero(Upper(2), Lower(3), Lower(5))
end

@testset "evaluate BinaryOperation Negate * Zero" begin
    A = Variable("A", Upper(1), Lower(2))
    Z = Zero(Upper(2), Lower(3))

    @test evaluate(dc.BinaryOperation{dc.Mult}(Z, -A)) == dc.Zero(Upper(1), Lower(3))
    @test evaluate(dc.BinaryOperation{dc.Mult}(-A, Z)) == dc.Zero(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation KrD * KrD" begin
    d1 = KrD(Upper(1), Lower(2))
    d2 = KrD(Upper(2), Lower(3))

    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(d1, d2)) == KrD(Upper(1), Lower(3))
    @test dc.evaluate(dc.BinaryOperation{dc.Mult}(d2, d1)) == KrD(Upper(1), Lower(3))
end

@testset "evaluate BinaryOperation with outer product" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(3))

    @test evaluate(dc.BinaryOperation{dc.Mult}(A, x)) == dc.BinaryOperation{dc.Mult}(A, x)
    @test evaluate(dc.BinaryOperation{dc.Mult}(A, y)) == dc.BinaryOperation{dc.Mult}(A, y)
end

@testset "evaluate subtraction with * and +" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(3))

    op1 = A * x - (x + y)
    op2 = (x + y) - A * x

    @test length(dc.get_free_indices(evaluate(op1))) == 1
    @test length(dc.get_free_indices(evaluate(op2))) == 1
end

@testset "evaluate subtraction with * and + and evaluate" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(3))

    op1 = (x + y) - (x + y)
    op2 = 2 * x - 2 * x

    @test equivalent(evaluate(op1), Zero(Upper(2)))
    @test equivalent(evaluate(op2), Zero(Upper(2)))
end

@testset "evaluate subtraction with product with real" begin
    A = Variable("A", Upper(1), Lower(2))

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
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(2), Lower(3))
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

@testset "evaluate subtraction with real and real" begin
    function sub(l, r)
        return dc.BinaryOperation{dc.Sub}(l, r)
    end

    @test dc.evaluate(sub(3, 2)) == 1
    @test dc.evaluate(sub(3, 3)) == 0
    @test dc.evaluate(sub(2, 3)) == -1
end

@testset "evaluate unary operations" begin
    A = Variable("A", Upper(1), Lower(2))

    ops = (sin, cos, abs, sign)
    types = (dc.Sin, dc.Cos, dc.Abs, dc.Sgn)

    for (op, type) ∈ zip(ops, types)
        @test typeof(op.(A)) == UnaryOperation{type}
        @test op.(A).arg == A
    end
end

@testset "evaluate trace" begin
    A = Variable("A", Upper(1), Lower(2))
    B = Variable("B", Upper(2), Lower(3))

    @test dc.evaluate(tr(A)) == Variable("A", Upper(2), Lower(2))
    @test equivalent(
        dc.evaluate(tr(A * B)),
        dc.BinaryOperation{dc.Mult}(A, Variable("B", Upper(2), Lower(1))),
    )
end

@testset "evaluate outer product - contraction" begin
    A = Variable("A", Upper(1), Lower(2))
    d = KrD(Upper(3), Lower(4))
    x = Variable("x", Lower(3))

    mul = dc.BinaryOperation{dc.Mult}

    @test dc.evaluate(mul(mul(A, d), x)) == mul(A, Variable("x", Lower(4)))
    @test dc.evaluate(mul(mul(d, A), x)) == mul(A, Variable("x", Lower(4)))
    @test dc.evaluate(mul(x, mul(A, d))) == mul(Variable("x", Lower(4)), A)
    @test dc.evaluate(mul(x, mul(d, A))) == mul(Variable("x", Lower(4)), A)

    @test dc.evaluate(mul(mul(d, A), x)) == mul(Variable("x", Lower(4)), A)
    @test dc.evaluate(mul(mul(A, d), x)) == mul(Variable("x", Lower(4)), A)
    @test dc.evaluate(mul(x, mul(d, A))) == mul(A, Variable("x", Lower(4)))
    @test dc.evaluate(mul(x, mul(A, d))) == mul(A, Variable("x", Lower(4)))
end

@testset "diff Variable" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(3))
    a = Variable("a")
    A = Variable("A", Upper(4), Lower(5))

    @test dc.diff(x, x) == KrD(Upper(2), Lower(2))
    @test dc.diff(y, x) == Zero(Upper(3), Lower(2))
    @test dc.diff(a, x) == Zero(Lower(2))
    @test dc.diff(A, x) == Zero(Upper(4), Lower(5), Lower(2))

    @test dc.diff(x, Variable("x", Upper(1))) == KrD(Upper(2), Lower(1))
    @test dc.diff(y, Variable("y", Upper(4))) == KrD(Upper(3), Lower(4))
    @test dc.diff(A, Variable("A", Upper(6), Lower(7))) ==
          dc.BinaryOperation{dc.Mult}(KrD(Upper(4), Lower(6)), KrD(Lower(5), Upper(7)))
end

@testset "diff KrD" begin
    x = Variable("x", Upper(3))
    y = Variable("y", Lower(4))
    A = Variable("A", Upper(5), Lower(6))
    d = KrD(Upper(1), Lower(2))

    @test dc.diff(d, x) == Zero(Upper(1), Lower(2), Lower(3))
    @test dc.diff(d, y) == Zero(Upper(1), Lower(2), Upper(4))
    @test dc.diff(d, A) == Zero(Upper(1), Lower(2), Lower(5), Upper(6))
end

@testset "diff Real" begin
    a = 1
    b = 0.0
    c = 1//3

    x = Variable("x", Upper(1))

    @test dc.diff(a, x) == Zero(Lower(1))
    @test dc.diff(b, x) == Zero(Lower(1))
    @test dc.diff(c, x) == Zero(Lower(1))
end

@testset "diff BinaryOperation{dc.Mult}" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Lower(2))

    op = dc.BinaryOperation{dc.Mult}(x, y)

    D = dc.diff(op, Variable("x", Upper(3)))

    @test typeof(D) == dc.BinaryOperation{dc.Add}
    @test D.arg1 == dc.BinaryOperation{dc.Mult}(x, Zero(Lower(2), Lower(3)))
    @test D.arg2 == dc.BinaryOperation{dc.Mult}(KrD(Upper(2), Lower(3)), y)
end

@testset "diff BinaryOperation{AdditiveOperation}" begin
    x = Variable("x", Upper(2))
    y = Variable("y", Upper(2))

    for op ∈ (dc.Add, dc.Sub)
        v = dc.BinaryOperation{op}(x, y)

        D = dc.diff(v, Variable("x", Upper(3)))

        @test D == dc.BinaryOperation{op}(KrD(Upper(2), Lower(3)), Zero(Upper(2), Lower(3)))
    end
end

@testset "diff trace" begin
    A = Variable("A", Upper(1), Lower(2))

    op = tr(A)

    D = dc.diff(op, Variable("A", Upper(3), Lower(4)))

    @test equivalent(evaluate(D), KrD(Upper(1), Lower(2)))
end

@testset "diff abs" begin
    x = Variable("x", Upper(2))

    op = UnaryOperation{dc.Abs}(x)

    D = dc.diff(op, Variable("x", Upper(3)))

    @test equivalent(
        D,
        dc.BinaryOperation{dc.Mult}(dc.UnaryOperation{dc.Sgn}(x), KrD(Upper(2), Lower(3))),
    )
end

@testset "diff sin" begin
    x = Variable("x", Upper(2))

    op = sin.(x)

    D = dc.diff(op, Variable("x", Upper(3)))

    @test equivalent(
        D,
        dc.BinaryOperation{dc.Mult}(dc.UnaryOperation{dc.Cos}(x), KrD(Upper(2), Lower(3))),
    )
end

@testset "diff cos" begin
    x = Variable("x", Upper(2))

    op = cos.(x)

    D = dc.diff(op, Variable("x", Upper(3)))

    @test equivalent(
        D,
        dc.BinaryOperation{dc.Mult}(-dc.UnaryOperation{dc.Sin}(x), KrD(Upper(2), Lower(3))),
    )
end

@testset "diff negated vector" begin
    x = Variable("x", Upper(2))

    op = -x

    D = dc.diff(op, Variable("x", Upper(3)))

    expected = dc.BinaryOperation{dc.Add}(
        -KrD(Upper(2), Lower(3)),
        dc.BinaryOperation{dc.Mult}(dc.Zero(Lower(3)), x),
    )

    @test equivalent(D, expected)
end

@testset "free indices constant after evaluate" begin
    x = Variable("x", Upper(2))
    c = Variable("c", Upper(3))
    y = Variable("y", Upper(4))

    op1 = (y .* c)' * x

    @test isempty(dc.get_free_indices(op1))
    @test isempty(dc.get_free_indices(evaluate(op1)))

    op2 = tr(x * x')

    @test isempty(dc.get_free_indices(op2))
    @test isempty(dc.get_free_indices(evaluate(op2)))
end

@testset "Differentiate Ax" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))

    first_term =
        BinaryOperation{dc.Mult}(Variable("A", Upper(1), Lower(4)), KrD(Upper(4), Lower(5)))
    second_term = BinaryOperation{dc.Mult}(
        Zero(Upper(1), Lower(4), Lower(5)),
        Variable("x", Upper(4)),
    )
    expected = BinaryOperation{dc.Add}(first_term, second_term)

    @test dc.diff(A * x, Variable("x", Upper(5))) == expected
end

@testset "Differentiate xᵀA " begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))

    first_term = BinaryOperation{dc.Mult}(
        Variable("x", Lower(4)),
        Zero(Upper(4), Lower(2), Lower(6)),
    )
    second_term =
        BinaryOperation{dc.Mult}(KrD(Lower(4), Lower(6)), Variable("A", Upper(4), Lower(2)))
    expected = BinaryOperation{dc.Add}(first_term, second_term)

    @test dc.diff(x' * A, Variable("x", Upper(6))) == expected
end

@testset "Differentiate xᵀAx" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))

    D = dc.diff(x' * A * x, Variable("x", Upper(7)))

    function mult(l, r)
        return BinaryOperation{dc.Mult}(l, r)
    end

    first_term = mult(
        mult(Variable("x", Lower(4)), Variable("A", Upper(4), Lower(5))),
        KrD(Upper(5), Lower(7)),
    )
    second_term_ll = mult(Variable("x", Lower(4)), Zero(Upper(4), Lower(5), Lower(7)))
    second_term_lr = mult(KrD(Lower(4), Lower(7)), Variable("A", Upper(4), Lower(5)))
    second_term = mult(
        BinaryOperation{dc.Add}(second_term_ll, second_term_lr),
        Variable("x", Upper(5)),
    )

    @test D.arg1 == first_term
    @test D.arg2 == second_term
end

@testset "Differentiate xx'x" begin
    x = Variable("x", Upper(1))

    l = dc.BinaryOperation{dc.Mult}(Variable("x", Upper(100)), Variable("x", Lower(101)))
    l = dc.BinaryOperation{dc.Mult}(2, l)
    r = dc.BinaryOperation{dc.Mult}(Variable("x", Upper(99)), Variable("x", Lower(99)))
    r = dc.BinaryOperation{dc.Mult}(r, KrD(Upper(100), Lower(101)))
    expected = dc.BinaryOperation{dc.Add}(l, r)

    D = dc.diff(x * x' * x, Variable("x", Upper(6)))

    @test equivalent(dc.evaluate(D), expected)
end

@testset "Differentiate A(x + 2x)" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))

    D = dc.diff(A * (x + 2 * x), Variable("x", Upper(5)))

    @test equivalent(dc.evaluate(D), 3 * A)
end

@testset "Differentiate A(2x + x)" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))

    D = dc.diff(A * (2 * x + x), Variable("x", Upper(5)))

    @test equivalent(dc.evaluate(D), 3 * A)
end

@testset "evaluated derivative is equal to derivative of evaluated expression" begin
    A = Variable("A", Upper(1), Lower(2))
    x = Variable("x", Upper(3))
    y = Variable("y", Upper(4))
    c = Variable("c", Upper(5))

    exprs = ( #
        A * x, #
        x' * A, #
        x' * A * x, #
        # A * (x + 2 * x), # # TODO: Only valid if we don't collapse KrD's in 'evaluate'.
        # A * (2 * x + x), #
        # (x + 2 * x)' * A, #
        # (2 * x + x)' * A, #
        x' * x, #
        tr(x * x'), #
        (y .* c)' * x, #
        (x .* c)' * x, #
        (x + y)' * x, #
        (x - y)' * x, #
        x' * (y .* c), #
        x' * (x .* c), #
        x' * (x + y), #
        x' * (x - y), #
        sin(tr(x * x')), #
        cos(tr(x * x')), #
        tr(sin.(x * x')), #
        tr(A), #
        sum((x + y) .^ 2), #
    )

    # Check semantic equality. Differentiation before and after evaluate starts from different
    # expressions with different letters and thus yields expressions with different letters.
    for expr ∈ exprs
        @testset "$(dc.to_string(expr))" begin
            var = Variable("A", Upper(10), Lower(11))
            @test equivalent(
                evaluate(dc.diff(evaluate(expr), var)),
                evaluate(dc.diff(expr, var)),
            )
            var = Variable("x", Upper(10))
            @test equivalent(
                evaluate(dc.diff(evaluate(expr), var)),
                evaluate(dc.diff(expr, var)),
            )
            var = Variable("y", Upper(10))
            @test equivalent(
                evaluate(dc.diff(evaluate(expr), var)),
                evaluate(dc.diff(expr, var)),
            )
            var = Variable("c", Upper(10))
            @test equivalent(
                evaluate(dc.diff(evaluate(expr), var)),
                evaluate(dc.diff(expr, var)),
            )
        end
    end
end

@testset "a ones-vector times KrD is absorbed" begin
    mult(l, r) = dc.BinaryOperation{dc.Mult}(l, r)

    x = Variable("x", Upper(1))
    d = KrD(Upper(1), Lower(2))
    ones = Literal(1, Upper(2))

    @test evaluate(mult(mult(d, x), ones)) == x
    @test evaluate(mult(mult(x, d), ones)) == x

    @test evaluate(mult(mult(d, x), Literal(3, Upper(2)))) == mult(3, x)
    @test evaluate(mult(mult(x, d), Literal(3, Upper(2)))) == mult(3, x)
end

@testset "a tied ones-vector is dropped but a contracted one is kept" begin
    mult(l, r) = dc.BinaryOperation{dc.Mult}(l, r)

    A = Variable("A", Upper(1), Lower(8))
    x = Variable("x", Upper(1))

    @test evaluate(mult(A, Literal(1, Upper(1)))) == A

    # Is a sum - must be left alone.
    summed = mult(x, Literal(1, Lower(1)))
    @test evaluate(summed) == summed

    threes = mult(A, Literal(3, Upper(1)))
    @test evaluate(threes) == mult(3, A)
end
