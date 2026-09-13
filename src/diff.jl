# Copyright 2026, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function diff(arg::Variable, wrt::Variable)
    if arg.id == wrt.id
        @assert length(arg.indices) == length(wrt.indices)

        D = 1

        for (u, l) ∈ zip(arg.indices, wrt.indices)
            D = BinaryOperation{Mult}(D, KrD(u, flip(l)))
        end

        return evaluate(D) # evaluate to get rid of the constant factor
    end

    indices = LowerOrUpperIndex[arg.indices; [flip(i) for i ∈ wrt.indices]]

    return Zero(unique(indices)...)
end

function diff(arg::Literal, wrt::Variable)
    indices = union(arg.indices, [flip(i) for i ∈ wrt.indices])

    return Zero(unique(indices)...)
end

function diff(arg::KrD, wrt::Variable)
    indices = union(arg.indices, [flip(i) for i ∈ wrt.indices])

    return Zero(unique(indices)...)
end

function diff(arg::Real, wrt::Variable)
    return Zero([flip(i) for i ∈ wrt.indices]...)
end

function diff(arg::UnaryOperation{Abs}, wrt::Variable)
    outer = replace_bound_letters(arg.arg, wrt)

    return BinaryOperation{Mult}(UnaryOperation{Sgn}(outer), diff(arg.arg, wrt))
end

function diff(arg::UnaryOperation{Sin}, wrt::Variable)
    outer = replace_bound_letters(arg.arg, wrt)

    return BinaryOperation{Mult}(UnaryOperation{Cos}(outer), diff(arg.arg, wrt))
end

function diff(arg::UnaryOperation{Cos}, wrt::Variable)
    outer = replace_bound_letters(arg.arg, wrt)

    return BinaryOperation{Mult}(-UnaryOperation{Sin}(outer), diff(arg.arg, wrt))
end

function diff(arg::UnaryOperation{Log}, wrt::Variable)
    outer = replace_bound_letters(arg.arg, wrt)

    return BinaryOperation{Mult}(Power(outer, -1), diff(arg.arg, wrt))
end

function diff(arg::UnaryOperation{Exp}, wrt::Variable)
    outer = replace_bound_letters(arg.arg, wrt)

    return BinaryOperation{Mult}(UnaryOperation{Exp}(outer), diff(arg.arg, wrt))
end

function diff(arg::Power, wrt::Variable)
    outer = replace_bound_letters(arg.base, wrt)

    return BinaryOperation{Mult}(
        BinaryOperation{Mult}(arg.exponent, Power(outer, arg.exponent - 1)),
        diff(arg.base, wrt),
    )
end

function diff(arg::BinaryOperation{Mult}, wrt::Variable)
    d1 = diff(arg.arg1, wrt)
    d2 = diff(arg.arg2, wrt)

    return BinaryOperation{Add}(
        BinaryOperation{Mult}(replace_colliding_letters(arg.arg1, d2, wrt), d2),
        BinaryOperation{Mult}(d1, replace_colliding_letters(arg.arg2, d1, wrt)),
    )
end

function diff(arg::BinaryOperation{Op}, wrt::Variable) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(diff(arg.arg1, wrt), diff(arg.arg2, wrt))
end

function bound_letters(arg)
    return setdiff(get_letters(get_indices(arg)), get_letters(get_free_indices(arg)))
end

function replace_colliding_letters(arg, other, letters_to_skip...)
    colliding = intersect(bound_letters(arg), bound_letters(other))

    if isempty(colliding)
        return arg
    end

    next_letter = get_next_letter(arg, other, letters_to_skip...)
    letter_map = Dict(colliding[i] => next_letter + i - 1 for i ∈ eachindex(colliding))

    return replace_letters(arg, letter_map)
end

function replace_bound_letters(arg::Real, letters_to_skip...)
    return arg
end

function replace_bound_letters(arg::Tensor, letters_to_skip...)
    letters = unique(get_letters(get_indices(arg)))
    free_letters = unique(get_letters(get_free_indices(arg)))
    bound_letters = setdiff(letters, free_letters)
    next_letter = get_next_letter(arg, letters_to_skip...)

    letter_map =
        Dict(bound_letters[li] => next_letter + li - 1 for li ∈ eachindex(bound_letters))

    return replace_letters(arg, letter_map)
end
