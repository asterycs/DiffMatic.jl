# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

using DiffMatic

using DiffMatic: Variable, KrD, Zero
using DiffMatic: BinaryOperation, UnaryOperation
using DiffMatic: IndexList

dc = DiffMatic

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

function equivalent(left::Variable, right::Variable)
    left_ids, right_ids = dc.get_free_indices.((left, right))

    return left.id == right.id && can_remap(left_ids, right_ids)
end

function equivalent(left::KrD, right::KrD)
    return can_remap(left.indices, right.indices)
end

function equivalent(left::Zero, right::Zero)
    return can_remap(dc.get_free_indices(left), dc.get_free_indices(right))
end

function equivalent(left::BinaryOperation{T}, right::BinaryOperation{T}) where {T}
    return equivalent(left.arg1, right.arg1) && equivalent(left.arg2, right.arg2)
end

function equivalent(left::BinaryOperation{dc.Mult}, right::BinaryOperation{dc.Mult})
    return (equivalent(left.arg1, right.arg1) && equivalent(left.arg2, right.arg2)) ||
           (equivalent(left.arg1, right.arg2) && equivalent(left.arg2, right.arg1))
end

function equivalent(left::BinaryOperation{dc.Add}, right::BinaryOperation{dc.Add})
    return (equivalent(left.arg1, right.arg1) && equivalent(left.arg2, right.arg2)) ||
           (equivalent(left.arg1, right.arg2) && equivalent(left.arg2, right.arg1))
end

function equivalent(arg1::UnaryOperation{T}, arg2::UnaryOperation{T}) where {T}
    return equivalent(arg1.arg, arg2.arg)
end

function equivalent(arg1::T1, arg2::T2) where {T1,T2}
    return false
end
