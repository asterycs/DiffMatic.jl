# Copyright 2026, Jimmy Envall and contributors
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
