# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic
using Test

using DiffMatic: Variable, KrD, Zero, Literal
using DiffMatic: evaluate
using DiffMatic: Upper, Lower
using DiffMatic: BinaryOperation, Mult
using DiffMatic: get_free_indices

using LinearAlgebra: diag, diagm

dc = DiffMatic

# Such expression trees cannot be created from standard notation and are not treated
# @testset "simplify fully collapsible Mult * Mult" begin
#     d1 = KrD(Upper(1), Lower(2))
#     d2 = KrD(Upper(2), Lower(3))
#     d3 = KrD(Upper(3), Lower(4))
#     A = Variable("A", Upper(4), Lower(5))

#     op = dc.BinaryOperation{dc.Mult}(
#         dc.BinaryOperation{dc.Mult}(d1, d3),
#         dc.BinaryOperation{dc.Mult}(A, d2),
#     )

#     @test dc.simplify(op) == Variable("A", Upper(1), Lower(5))
# end

@testset "KrD collapsed correctly on element wise multiplications" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(2))
    z = Variable("z", Upper(3))

    e = (y .* z)' * x

    expected = dc.BinaryOperation{dc.Mult}(Variable("y", Lower(1)), Variable("x", Lower(1)))

    @test equivalent(dc.simplify(dc.diff(e, Variable("z", Upper(9)))), expected)

    expected = dc.BinaryOperation{dc.Mult}(Variable("y", Upper(1)), Variable("x", Upper(1)))
    @test equivalent(dc.simplify(dc.diff(e, Variable("z", Upper(9)))'), expected)
    @test equivalent(dc.simplify(dc.diff(e', Variable("z", Upper(9)))), expected)
    @test equivalent(dc.simplify(dc.diff(e, Variable("z", Upper(9)))'), expected)
end

@testset "KrD collapsed correctly on element wise multiplications (mirrored)" begin
    x = Variable("x", Upper(1))
    y = Variable("y", Upper(2))
    z = Variable("z", Upper(3))

    e = x' * (y .* z)

    expected = dc.BinaryOperation{dc.Mult}(Variable("y", Lower(1)), Variable("x", Lower(1)))

    @test equivalent(dc.simplify(dc.diff(e, Variable("z", Upper(9)))), expected)

    expected = dc.BinaryOperation{dc.Mult}(Variable("y", Upper(1)), Variable("x", Upper(1)))
    @test equivalent(dc.simplify(dc.diff(e, Variable("z", Upper(9)))'), expected)
    @test equivalent(dc.simplify(dc.diff(e', Variable("z", Upper(9)))), expected)
    @test equivalent(dc.simplify(dc.diff(e, Variable("z", Upper(9)))'), expected)
end

@testset "simplify preserves the free indices" begin
    @matrix A B
    @vector x y

    for e ∈ (
        diag(diagm(x)),
        diag(diagm(A * x)),
        diag(diagm(x' * B' * A * A)),
        diag(diagm(x)) * y',
        diagm(x) * y,
        sum(diag(diagm(A * x))),
        jacobian(diag(diagm(A * x)), x),
        jacobian(diag(diagm(x' * B' * A * A)), x),
        gradient(sum(x .* y), x),
    )
        @test issetequal(get_free_indices(dc.simplify(e)), get_free_indices(e))
    end
end

@testset "a constant vector times KrD is absorbed" begin
    mult(l, r) = BinaryOperation{Mult}(l, r)

    # Example:
    #
    #     A⁷₉ δ₉⁸ 1⁹  ->  A⁷⁸
    #
    # Reachable as diagm(x') * vector(3).
    A = Variable("A", Upper(7), Lower(9))
    d = KrD(Lower(9), Upper(8))

    expected = Variable("A", Upper(7), Upper(8))

    @test dc.simplify(mult(mult(A, d), Literal(1, Upper(9)))) == expected
    @test dc.simplify(mult(mult(d, A), Literal(1, Upper(9)))) == expected

    @test dc.simplify(mult(mult(A, d), Literal(3, Upper(9)))) == mult(3, expected)
    @test dc.simplify(mult(mult(d, A), Literal(3, Upper(9)))) == mult(3, expected)
end

@testset "simplify of two KrD agrees with evaluate" begin
    mult(l, r) = BinaryOperation{Mult}(l, r)

    for (l, r) ∈ (
        (KrD(Upper(1), Lower(2)), KrD(Upper(2), Lower(3))),
        (KrD(Upper(1), Lower(2)), KrD(Upper(2), Lower(1))), # the trace of the identity
        (KrD(Upper(1), Lower(2)), KrD(Upper(1), Lower(3))),
        (KrD(Upper(1), Lower(2)), KrD(Lower(1), Upper(3))),
        (KrD(Upper(1), Lower(2)), KrD(Upper(3), Lower(4))), # nothing to contract
    )
        @test dc.simplify(Mult(), l, r) == evaluate(mult(l, r))
    end
end
