# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic

using DiffMatic: Monomial, KrD, Zero
using DiffMatic: BinaryOperation, UnaryOperation
using DiffMatic: IndexList

function can_remap(left::IndexList, right::IndexList)
    lu = unique(left)
    ru = unique(right)

    if length(lu) != length(ru)
        return false
    end

    let
        lu_letters = [i.letter for i ∈ left]
        ru_letters = [i.letter for i ∈ right]

        if length(unique(lu_letters)) != length(unique(ru_letters))
            return false
        end
    end

    index_map = Dict((l => r) for (l, r) ∈ zip(lu, ru))

    left_remap = [index_map[i] for i ∈ left]

    return left_remap == right
end

function equivalent(arg1::Real, arg2::Real)
    return arg1 == arg2
end

function equivalent(left, right)
    left_ids, right_ids = DiffMatic.get_free_indices.((left, right))

    return can_remap(left_ids, right_ids)
end

function equivalent(left::Monomial, right::Monomial)
    left_ids, right_ids = DiffMatic.get_free_indices.((left, right))

    return left.id == right.id && can_remap(left_ids, right_ids)
end

function equivalent(left::KrD, right::KrD)
    return can_remap(left.indices, right.indices)
end

function equivalent(left::Zero, right::Zero)
    return can_remap(DiffMatic.get_free_indices(left), DiffMatic.get_free_indices(right))
end

function equivalent(left::BinaryOperation, right::BinaryOperation)
    if typeof(left) != typeof(right)
        return false
    end

    same_types =
        (
            typeof(left.arg1) == typeof(right.arg1) &&
            typeof(left.arg2) == typeof(right.arg2)
        ) ||
        (typeof(left.arg1) == typeof(right.arg2) && typeof(left.arg2) == typeof(right.arg1))

    return same_types &&
           (equivalent(left.arg1, right.arg1) && equivalent(left.arg2, right.arg2)) ||
           (equivalent(left.arg1, right.arg2) && equivalent(left.arg2, right.arg1))
end

function equivalent(arg1::T, arg2::T) where {T<:UnaryOperation}
    return equivalent(arg1.arg, arg2.arg)
end
