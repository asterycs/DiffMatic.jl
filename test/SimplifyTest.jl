# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic
using Test

using DiffMatic: Variable, KrD, Zero
using DiffMatic: evaluate
using DiffMatic: Upper, Lower

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
