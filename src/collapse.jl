# Copyright 2026, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function collapse(::Op, arg1, arg2) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(collapse(arg1), collapse(arg2))
end

function collapse(::Mult, arg1::KrD, arg2::KrD)
    return try_swap(arg1, first(arg2.indices), last(arg2.indices))
end

function collapse(::Mult, arg1::Value, arg2::KrD)
    return try_swap(arg1, first(arg2.indices), last(arg2.indices))
end

function collapse(::Mult, arg1::KrD, arg2::Value)
    return collapse(Mult(), arg2, arg1)
end

function collapse(::Mult, arg1::BinaryOperation, arg2::KrD)
    return try_swap(arg1, first(arg2.indices), last(arg2.indices))
end

function collapse(::Mult, arg1::KrD, arg2::BinaryOperation)
    return collapse(Mult(), arg2, arg1)
end

function collapse(::Mult, arg1::UnaryOperation{Op}, arg2::KrD) where {Op}
    ids = get_indices(arg1)

    f = first(arg2.indices)
    l = last(arg2.indices)

    if flip(f) in ids || flip(l) in ids
        return UnaryOperation{Op}(
            try_swap(arg1.arg, first(arg2.indices), last(arg2.indices)),
        )
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function collapse(::Mult, arg1::KrD, arg2::UnaryOperation{Op}) where {Op}
    return collapse(Mult(), arg2, arg1)
end

function collapse(::Mult, arg1::Value, arg2::Value)
    return BinaryOperation{Mult}(collapse(arg1), collapse(arg2))
end

function collapse(::Mult, arg1::Union{Variable,Literal}, arg2::BinaryOperation{Mult})
    return collapse(Mult(), arg2, arg1)
end

function collapse(::Mult, arg1::BinaryOperation{Mult}, arg2::Union{Variable,Literal})
    if arg1.arg1 isa KrD && can_contract(arg1.arg1, arg2) && !can_contract(arg1.arg2, arg2)
        new_arg1 = collapse(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, arg1.arg2)
    end

    if arg1.arg2 isa KrD && can_contract(arg1.arg2, arg2) && !can_contract(arg1.arg1, arg2)
        new_arg2 = collapse(Mult(), arg1.arg2, arg2)
        return trim(Mult(), arg1.arg1, new_arg2)
    end

    is_arg1_elementwise = is_elementwise_multiplication(arg1.arg1, arg1.arg2)
    are_both_elwise =
        is_elementwise_multiplication(arg1.arg1, arg2) &&
        is_elementwise_multiplication(arg1.arg2, arg2)

    if is_arg1_elementwise || are_both_elwise
        return BinaryOperation{Mult}(trim(arg1), arg2)
    end

    if can_contract(arg1.arg2, arg2)
        new_arg2 = collapse(Mult(), arg1.arg2, arg2)
        return BinaryOperation{Mult}(arg1.arg1, new_arg2)
    elseif can_contract(arg1.arg1, arg2)
        new_arg1 = collapse(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, arg1.arg2)
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function try_swap(arg::Value, l::LowerOrUpperIndex, r::LowerOrUpperIndex)
    if flip(l) == r
        return BinaryOperation{Mult}(arg, KrD(l, r))
    end

    eliminated = eliminated_indices(get_indices(arg))

    if l ∈ eliminated || r ∈ eliminated
        return BinaryOperation{Mult}(arg, KrD(l, r))
    end

    ids = get_indices(arg)

    if flip(r) in ids
        return _try_swap(arg, flip(r), l)
    end

    if flip(l) in ids
        return _try_swap(arg, flip(l), r)
    end

    return BinaryOperation{Mult}(arg, KrD(l, r))
end

function _try_swap(
    arg::BinaryOperation{Op},
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex,
) where {Op}
    return BinaryOperation{Op}(_try_swap(arg.arg1, from, to), _try_swap(arg.arg2, from, to))
end

function _try_swap(
    arg::UnaryOperation{Op},
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex,
) where {Op}
    return UnaryOperation{Op}(_try_swap(arg.arg, from, to))
end

function _try_swap(arg::Power, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    return Power(_try_swap(arg.base, from, to), arg.exponent)
end

function _try_swap(
    arg::Union{Variable,Literal,Zero,KrD},
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex,
)
    ids = get_indices(arg)

    i = findfirst(i -> i == from, ids)

    if isnothing(i)
        return arg
    end

    newarg = deepcopy(arg)

    newarg.indices[i] = to

    return newarg
end

function _try_swap(arg::Union{Real,Int}, from::LowerOrUpperIndex, to::LowerOrUpperIndex)
    return arg
end

function collapse(arg::Union{Variable,Literal,KrD,Zero,Real})
    return arg
end

function collapse(op::Power)
    return Power(collapse(op.base), op.exponent)
end

function collapse(op::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(collapse(op.arg))
end

function collapse(op::BinaryOperation{Mult})
    return collapse(Mult(), collapse(op.arg1), collapse(op.arg2))
end

function collapse(op::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(collapse(op.arg1), collapse(op.arg2))
end
