# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function trim(arg::Union{Variable,Literal,KrD,Zero,Real})
    return arg
end

function trim(arg::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(trim(arg.arg))
end

function trim(::Mult, arg1::BinaryOperation{Mult}, arg2::Real)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::T, arg2::BinaryOperation{Mult}) where {T<:Real}
    if arg1 == T(1)
        return arg2
    end

    if arg1 == T(0)
        return Zero(get_free_indices(arg2)...)
    end

    if arg2.arg1 isa Real
        return BinaryOperation{Mult}(arg1 * arg2.arg1, arg2.arg2)
    end

    return BinaryOperation{Mult}(arg1, trim(arg2))
end

function indices_in_common(arg1, arg2)
    arg1_indices = get_indices(arg1)
    arg2_indices = get_indices(arg2)

    return intersect(arg1_indices, arg2_indices)
end

function is_elementwise_multiplication(arg1, arg2)
    return !isempty(indices_in_common(arg1, arg2))
end

function is_all_elementwise(arg1, arg2)
    num_common_ids = length(indices_in_common(arg1, arg2))
    return num_common_ids == length(get_free_indices(arg1)) &&
           num_common_ids == length(get_free_indices(arg2))
end

"""
Detect an element-wise product with a ones-vector. Matches e.g.

    A¹₈1¹
    x¹y₆1¹

Does not match e.g.

    x¹1₁
"""
function is_tied_constant(arg::Value, other::Value)
    if !(arg isa Literal) || length(arg.indices) != 1
        return false
    end

    return only(arg.indices) ∈ get_indices(other)
end

function is_diagm(arg::BinaryOperation{Mult})
    return is_diagm(arg.arg1, arg.arg2)
end

function is_diagm(arg)
    return false
end

function is_diagm(arg1::KrD, arg2::Tensor)
    return is_diagm(arg2, arg1)
end

function is_diagm(arg1::KrD, arg2::KrD)
    return false
end

function is_diagm(arg::Union{Variable,Literal,KrD,Zero})
    return false
end

function is_diagm(arg1::Tensor, arg2::KrD)
    arg1_indices, arg2_indices = get_free_indices.((arg1, arg2))

    return length(arg1_indices) == 1 && !isempty(intersect(arg1_indices, arg2_indices))
end

function is_diagm(arg1::Value, arg2::Value)
    if isempty(get_free_indices(arg1))
        return is_diagm(arg2)
    elseif isempty(get_free_indices(arg2))
        return is_diagm(arg1)
    end

    return is_diagm(arg1) && is_diagm(arg2)
end

function trim(::Mult, arg1::Union{Variable,Literal}, arg2::BinaryOperation{Mult})
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::BinaryOperation{Mult}, arg2::Union{Variable,Literal})
    if is_tied_constant(arg2, arg1)
        return trim(BinaryOperation{Mult}(arg2.value, arg1))
    end

    if arg1.arg1 isa Real
        return BinaryOperation{Mult}(arg1.arg1, BinaryOperation{Mult}(arg1.arg2, arg2))
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function trim(::Mult, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    if arg1.arg1 == -1 && arg2.arg1 == -1
        return BinaryOperation{Mult}(arg1.arg2, arg2.arg2)
    elseif arg1.arg1 == -1
        return BinaryOperation{Mult}(
            arg1.arg1,
            trim(BinaryOperation{Mult}(arg1.arg2, arg2)),
        )
    elseif arg2.arg1 == -1
        return BinaryOperation{Mult}(
            arg2.arg1,
            trim(BinaryOperation{Mult}(arg2.arg2, arg1)),
        )
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function trim(::Mult, arg1::Zero, arg2::UnaryOperation)
    return trim(Mult(), arg2, arg1)
end

# Assumes one argument is of type 'Zero'
function _multiply_by_zero(arg1, arg2)
    free_indices = unique([get_indices(arg1); get_indices(arg2)])

    return Zero(free_indices...)
end

function trim(::Mult, arg1::UnaryOperation, arg2::Zero)
    return _multiply_by_zero(arg1, arg2)
end

function trim(::Mult, arg1::Zero, arg2::Tensor)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::Tensor, arg2::Zero)
    return _multiply_by_zero(arg1, arg2)
end

function trim(::Mult, arg1::KrD, arg2::Zero)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::Zero, arg2::KrD)
    return _multiply_by_zero(arg1, arg2)
end

function trim(::Mult, arg1::KrD, arg2::UnaryOperation)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::UnaryOperation{Op}, arg2::KrD) where {Op}
    attempt = UnaryOperation{Op}(trim(Mult(), trim(arg1.arg), trim(arg2)))

    # This ensures that arg1.base and arg2 can contract and that the contraction is simple
    if length(get_free_indices(attempt)) == length(get_free_indices(arg1))
        return attempt
    end

    return BinaryOperation{Mult}(trim(arg1), trim(arg2))
end

function trim(::Mult, arg1::Power, arg2::KrD)
    attempt = Power(trim(Mult(), trim(arg1.base), trim(arg2)), arg1.exponent)

    # This ensures that arg1.base and arg2 can contract and that the contraction is simple
    if length(get_free_indices(attempt)) == length(get_free_indices(arg1))
        return attempt
    end

    return BinaryOperation{Mult}(trim(arg1), trim(arg2))
end

function trim(::Mult, arg1::Tensor, arg2::Power)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::Power, arg2::Tensor)
    if arg1.exponent == -1 && arg1.base == arg2
        return Literal(1, get_free_indices(arg2)...)
    end

    return BinaryOperation{Mult}(trim(arg1), trim(arg2))
end

function trim(::Mult, arg1::Power, arg2::Power)
    if isequal(arg1.base, arg2.base)
        return Power(arg1.base, arg1.exponent + arg2.exponent)
    end

    return BinaryOperation{Mult}(trim(arg1), trim(arg2))
end

function trim(::Mult, arg1::Zero, arg2::Power)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::Power, arg2::Zero)
    return _multiply_by_zero(arg1, arg2)
end

function trim(::Mult, arg1::Zero, arg2::BinaryOperation{Add})
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::Zero, arg2::BinaryOperation{Sub})
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::BinaryOperation{Add}, arg2::Zero)
    return _multiply_by_zero(arg1, arg2)
end

function trim(::Mult, arg1::BinaryOperation{Sub}, arg2::Zero)
    return _multiply_by_zero(arg1, arg2)
end

function trim(::Mult, arg1::BinaryOperation{Add}, arg2::Power)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::BinaryOperation{Sub}, arg2::Power)
    return trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::Value, arg2::Value)
    if is_tied_constant(arg2, arg1)
        return trim(BinaryOperation{Mult}(arg2.value, arg1))
    end

    if is_tied_constant(arg1, arg2)
        return trim(BinaryOperation{Mult}(arg1.value, arg2))
    end

    return BinaryOperation{Mult}(trim(arg1), trim(arg2))
end

function trim(::Mult, arg1::Tensor, arg2::Real)
    trim(Mult(), arg2, arg1)
end

function trim(::Mult, arg1::T, arg2::Tensor) where {T<:Real}
    if arg1 == T(1)
        return arg2
    end

    if arg1 == T(0)
        return Zero(get_free_indices(arg2)...)
    end


    return BinaryOperation{Mult}(arg1, arg2)
end

function trim(::Mult, arg1::Zero, arg2::Zero)
    return _multiply_by_zero(arg1, arg2)
end

function trim(::Mult, arg1::Zero, arg2::Real)
    return trim(arg1)
end

function trim(::Mult, arg1::Real, arg2::Zero)
    return trim(arg2)
end

function trim(::Add, arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return arg1
end

function trim(::Add, arg1::Zero, arg2::Value)
    @assert is_permutation(arg1, arg2)

    return trim(arg2)
end

function trim(::Add, arg1::Value, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return trim(arg1)
end

function trim(::Add, arg1::Real, arg2::Real)
    return arg1 + arg2
end

function trim(::Add, arg1::Value, arg2::Value)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return BinaryOperation{Mult}(2, arg1)
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Mult}, arg2::Zero)
    return invoke(trim, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    return _add_to_product(arg1, arg2)
end

function trim(::Add, arg1::Value, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
end

function trim(::Add, arg1::BinaryOperation{Mult}, arg2::Value)
    return _add_to_product(arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Sub})
    return _add_to_product(arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::Value)
    if trim(arg1.arg1) isa Real && trim(arg1.arg2) == trim(arg2)
        return BinaryOperation{Mult}(trim(arg1.arg1) + 1, trim(arg2))
    end

    return BinaryOperation{Add}(trim(arg1), trim(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Add})
    if trim(arg1) == trim(arg2.arg1)
        return BinaryOperation{Add}(
            trim(BinaryOperation{Mult}(2, trim(arg1))),
            trim(arg2.arg2),
        )
    end

    if trim(arg1) == trim(arg2.arg2)
        return BinaryOperation{Add}(
            trim(BinaryOperation{Mult}(2, trim(arg1))),
            trim(arg2.arg1),
        )
    end

    if trim(arg1.arg1) isa Real && trim(arg1.arg2) == trim(arg2.arg1)
        return trim(
            BinaryOperation{Add}(
                trim(BinaryOperation{Mult}(trim(arg1.arg1) + 1, trim(arg1.arg2))),
                trim(arg2.arg2),
            ),
        )
    end

    if trim(arg1.arg1) isa Real && trim(arg1.arg2) == trim(arg2.arg2)
        return trim(
            BinaryOperation{Add}(
                trim(BinaryOperation{Mult}(trim(arg1.arg1) + 1, trim(arg1.arg2))),
                trim(arg2.arg1),
            ),
        )
    end

    return BinaryOperation{Add}(trim(arg1), trim(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Sub})
    if trim(arg1) == trim(arg2.arg1)
        return BinaryOperation{Sub}(
            trim(BinaryOperation{Mult}(2, trim(arg1))),
            trim(arg2.arg2),
        )
    end

    if trim(arg1.arg1) isa Real && trim(arg1.arg2) == trim(arg2.arg1)
        return trim(
            BinaryOperation{Sub}(
                trim(BinaryOperation{Mult}(trim(arg1.arg1) + 1, trim(arg1.arg2))),
                trim(arg2.arg2),
            ),
        )
    end

    if trim(arg1.arg1) isa Real && trim(arg1.arg2) == trim(arg2.arg2)
        return trim(
            BinaryOperation{Add}(
                trim(BinaryOperation{Mult}(trim(arg1.arg1) - 1, trim(arg1.arg2))),
                trim(arg2.arg1),
            ),
        )
    end

    if trim(arg1) == trim(arg2.arg2)
        return trim(arg2.arg1)
    end

    return BinaryOperation{Add}(trim(arg1), trim(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    if is_permutation(collect_factors(arg1), collect_factors(arg2))
        return BinaryOperation{Mult}(2, trim(arg1))
    end

    return BinaryOperation{Add}(trim(arg1), trim(arg2))
end

function trim(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Add})
    if arg1.arg1 == arg2.arg1
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg1),
            trim(BinaryOperation{Add}(arg1.arg2, arg2.arg2)),
        )
    end

    if arg1.arg1 == arg2.arg2
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg1),
            trim(BinaryOperation{Add}(arg1.arg2, arg2.arg1)),
        )
    end

    if arg1.arg2 == arg2.arg1
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg2),
            trim(BinaryOperation{Add}(arg1.arg1, arg2.arg2)),
        )
    end

    if arg1.arg2 == arg2.arg2
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg2),
            trim(BinaryOperation{Add}(arg1.arg1, arg2.arg1)),
        )
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Add}, arg2::Zero)
    return invoke(trim, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function trim(::Add, arg1::Zero, arg2::BinaryOperation{Add})
    return invoke(trim, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function trim(::Add, arg1::Zero, arg2::BinaryOperation{Mult})
    invoke(trim, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function trim(::Add, arg1::Value, arg2::BinaryOperation{Add})
    return trim(Add(), arg2, arg1)
end

function trim(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Add})
    return _add_to_product(arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Add}, arg2::Value)
    if trim(arg1.arg1) == trim(arg2)
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, trim(arg1.arg1)),
            trim(arg1.arg2),
        )
    end

    if trim(arg1.arg2) == trim(arg2)
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, trim(arg1.arg2)),
            trim(arg1.arg1),
        )
    end

    return BinaryOperation{Add}(trim(arg1), trim(arg2))
end

function trim(::Add, arg1::Value, arg2::BinaryOperation{Sub})
    return trim(Add(), arg2, arg1)
end

function trim(::Add, arg1::BinaryOperation{Sub}, arg2::Value)
    if trim(arg1.arg1) == trim(arg2)
        return BinaryOperation{Sub}(
            BinaryOperation{Mult}(2, trim(arg1.arg1)),
            trim(arg1.arg2),
        )
    end

    if trim(arg1.arg2) == trim(arg2)
        return trim(arg1.arg1)
    end

    return BinaryOperation{Add}(trim(arg1), trim(arg2))
end

function trim(::Add, arg1::BinaryOperation{Sub}, arg2::Zero)
    return invoke(trim, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
end

function trim(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Sub})
    return trim(Add(), arg2, arg1)
end

function trim(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Add})
    if arg1.arg1 == arg2.arg1
        return trim(
            BinaryOperation{Add}(
                trim(BinaryOperation{Mult}(2, arg1.arg1)),
                trim(BinaryOperation{Sub}(arg2.arg2, arg1.arg2)),
            ),
        )
    end

    if arg1.arg1 == arg2.arg2
        return trim(
            BinaryOperation{Add}(
                trim(BinaryOperation{Mult}(2, arg1.arg1)),
                trim(BinaryOperation{Sub}(arg2.arg1, arg1.arg2)),
            ),
        )
    end

    if arg1.arg2 == arg2.arg1
        return trim(Add(), arg1.arg1, arg2.arg2)
    end

    if arg1.arg2 == arg2.arg2
        return trim(Add(), arg1.arg1, arg2.arg1)
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function trim(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Sub})
    if arg1.arg1 == arg2.arg1
        return trim(
            BinaryOperation{Sub}(
                trim(BinaryOperation{Mult}(2, arg1.arg1)),
                trim(BinaryOperation{Add}(arg1.arg2, arg2.arg2)),
            ),
        )
    end

    if arg1.arg1 == arg2.arg2
        return trim(BinaryOperation{Sub}(arg2.arg1, arg1.arg2))
    end

    if arg1.arg2 == arg2.arg1
        return trim(BinaryOperation{Sub}(arg1.arg1, arg2.arg2))
    end

    if arg1.arg2 == arg2.arg2
        return trim(
            BinaryOperation{Sub}(
                trim(BinaryOperation{Add}(arg1.arg1, arg2.arg1)),
                trim(BinaryOperation{Mult}(2, arg1.arg2)),
            ),
        )
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function trim(::Add, arg1::Zero, arg2::BinaryOperation{Sub})
    return invoke(trim, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function trim(::Sub, arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return arg1
end

function trim(::Sub, arg1::Zero, arg2::Value)
    @assert is_permutation(arg1, arg2)

    return -trim(arg2)
end

function trim(::Sub, arg1::Value, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return trim(arg1)
end

function trim(::Sub, arg1::Real, arg2::Real)
    return arg1 - arg2
end

function trim(::Sub, arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return Zero(arg1_indices...)
    end

    return BinaryOperation{Sub}(arg1, arg2)
end

function trim(::Sub, arg1::Zero, arg2::BinaryOperation{Mult})
    return invoke(trim, Tuple{Sub,Zero,Value}, Sub(), arg1, arg2)
end

function trim(::Sub, arg1::BinaryOperation{Mult}, arg2::Zero)
    return invoke(trim, Tuple{Sub,Value,Zero}, Sub(), arg1, arg2)
end

function trim(::Sub, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    return _sub_from_product(arg1, arg2)
end

function trim(::Sub, arg1::BinaryOperation{Mult}, arg2::Value)
    return _sub_from_product(arg1, arg2)
end

function _sub_from_product(arg1::BinaryOperation{Mult}, arg2::Value)
    if trim(arg1) == trim(arg2)
        arg1_indices = get_free_indices(arg1)
        return Zero(arg1_indices...)
    end

    if trim(arg1.arg1) isa Real && trim(arg1.arg2) == trim(arg2)
        return trim(BinaryOperation{Mult}(trim(arg1.arg1) - 1, trim(arg2)))
    end

    return BinaryOperation{Sub}(trim(arg1), trim(arg2))
end

function trim(op::Power)
    if op.exponent == 1
        return trim(op.base)
    end

    return Power(trim(op.base), op.exponent)
end

function trim(op::BinaryOperation{Mult})
    return trim(Mult(), trim(op.arg1), trim(op.arg2))
end

function trim(op::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return trim(Op(), trim(op.arg1), trim(op.arg2))
end
