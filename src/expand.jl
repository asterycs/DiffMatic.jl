# Copyright 2026, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function expand(::Mult, arg1::BinaryOperation{Add}, arg2::Tensor)
    return expand(Mult(), arg2, arg1)
end

function expand(::Mult, arg1::BinaryOperation{Sub}, arg2::Tensor)
    return expand(Mult(), arg2, arg1)
end

# TODO: It is pessimistic to always expand the sum/difference, but it always works.
# Consider the following input:
#
# to_std(hessian((A * x)' * diagm(sin.(B * x)) * (C * x), x)
#
# arg1 = C₁₀₁₃
# arg2 = A⁸₄x⁴cos(B₈¹²x₁₂)B₈¹¹δ₈¹⁰ + sin(B₈⁷x₇)δ₈¹⁰A⁸¹¹
#
# In order to identify that we need to expand the sum then we must collect the free
# indices of every factor and check if the length is > 2 and whether it contracts
# with arg1. For now we always expand.
function expand(::Mult, arg1::Tensor, arg2::BinaryOperation{Add})
    return BinaryOperation{Add}(
        expand(Mult(), arg1, expand(arg2.arg1)),
        expand(Mult(), arg1, expand(arg2.arg2)),
    )
end

function expand(::Mult, arg1::Tensor, arg2::BinaryOperation{Sub})
    return BinaryOperation{Sub}(
        expand(Mult(), arg1, expand(arg2.arg1)),
        expand(Mult(), arg1, expand(arg2.arg2)),
    )
end

function expand(::Mult, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Add})
    return BinaryOperation{Add}(
        BinaryOperation{Add}(
            expand(Mult(), arg1.arg1, arg2.arg1),
            expand(Mult(), arg1.arg1, arg2.arg2),
        ),
        BinaryOperation{Add}(
            expand(Mult(), arg1.arg2, arg2.arg1),
            expand(Mult(), arg1.arg2, arg2.arg2),
        ),
    )
end

function expand(::Mult, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Add})
    return expand(Mult(), arg2, arg1)
end

function expand(::Mult, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Sub})
    return BinaryOperation{Add}(
        BinaryOperation{Sub}(
            expand(Mult(), arg1.arg1, arg2.arg1),
            expand(Mult(), arg1.arg1, arg2.arg2),
        ),
        BinaryOperation{Sub}(
            expand(Mult(), arg1.arg2, arg2.arg1),
            expand(Mult(), arg1.arg2, arg2.arg2),
        ),
    )
end

function expand(::Mult, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Sub})
    return BinaryOperation{Add}(
        BinaryOperation{Sub}(
            expand(Mult(), arg1.arg1, arg2.arg1),
            expand(Mult(), arg1.arg1, arg2.arg2),
        ),
        BinaryOperation{Sub}(
            expand(Mult(), arg1.arg2, arg2.arg2),
            expand(Mult(), arg1.arg2, arg2.arg1),
        ),
    )
end

# Catch all
function expand(::Mult, arg1::Value, arg2::Value)
    return BinaryOperation{Mult}(arg1, arg2)
end

function expand(arg::Union{Variable,Literal,KrD,Zero,Real})
    return arg
end

function expand(op::Power)
    return Power(expand(op.base), op.exponent)
end

function expand(op::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(expand(op.arg))
end

function expand(op::BinaryOperation{Mult})
    return expand(Mult(), expand(op.arg1), expand(op.arg2))
end

function expand(op::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(expand(op.arg1), expand(op.arg2))
end
