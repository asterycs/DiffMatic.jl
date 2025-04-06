# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function simplify(arg::Union{Tensor,KrD})
    return arg
end

function simplify(::Mult, arg1::KrD, arg2::UnaryOp) where {UnaryOp<:UnaryOperation}
    if can_contract(arg1, arg2.arg)
        return UnaryOp(simplify(Mult(), arg1, arg2.arg))
    end

    return BinaryOperation{Mult}(arg2, arg1)
end

function simplify(::Mult, arg1::UnaryOp, arg2::KrD) where {UnaryOp<:UnaryOperation}
    if can_contract(arg1.arg, arg2)
        return UnaryOp(simplify(Mult(), arg1.arg, arg2))
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(
    ::Mult,
    arg1::BinaryOperation{Op},
    arg2::Union{Tensor,KrD},
) where {Op<:AdditiveOperation}
    return simplify(
        Op(),
        simplify(Mult(), simplify(arg1.arg1), simplify(arg2)),
        simplify(Mult(), simplify(arg1.arg2), simplify(arg2)),
    )
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::KrD)
    ci = indices_in_common(arg1.arg1, arg1.arg2)

    # TODO: Make this redundant
    if !isempty(ci)
        el = eliminated_indices([ci; arg2.indices[1]])
        er = eliminated_indices([ci; arg2.indices[2]])

        if !isempty(el)
            return simplify(
                Mult(),
                simplify(Mult(), arg1.arg1, arg2), # order of the indices in arg2 determines which one is contracted
                simplify(Mult(), arg1.arg2, arg2),
            )
        elseif !isempty(er)
            rd = KrD(reverse(arg2.indices)...)

            return simplify(
                Mult(),
                simplify(Mult(), arg1.arg1, rd),
                simplify(Mult(), arg1.arg2, rd),
            )
        end
    end

    @assert !(can_contract(arg1.arg1, arg2) && can_contract(arg1.arg2, arg2))

    if can_contract(arg1.arg2, arg2)
        new_arg2 = simplify(Mult(), arg1.arg2, arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg1), new_arg2)
    elseif can_contract(arg1.arg1, arg2)
        new_arg1 = simplify(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, evaluate(arg1.arg2))
    elseif arg1.arg1 isa Real
        return BinaryOperation{Mult}(arg1.arg1, BinaryOperation{Mult}(arg1.arg2, arg2))
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(::Mult, arg1::Tensor, arg2::BinaryOperation{Mult})
    return evaluate(Mult(), arg2, arg1)
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::Tensor)
    if can_contract(arg1.arg2, arg2)
        new_arg2 = evaluate(Mult(), arg1.arg2, arg2)
        return BinaryOperation{Mult}(arg1.arg1, new_arg2)
    elseif can_contract(arg1.arg1, arg2)
        new_arg1 = evaluate(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, arg1.arg2)
    else
        return BinaryOperation{Mult}(arg1, arg2)
    end
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    new_args = []

    available1 = Any[arg1.arg1; arg1.arg2]
    available2 = Any[arg2.arg1; arg2.arg2]

    for i ∈ eachindex(available1)
        if isnothing(available1[i])
            continue
        end
        for j ∈ eachindex(available2)
            if isnothing(available2[j]) || isnothing(available1[i])
                continue
            end

            if can_contract(available1[i], available2[j])
                push!(new_args, evaluate(Mult(), available1[i], available2[j]))
                available1[i] = nothing
                available2[j] = nothing
            end
        end
    end

    for i ∈ available1
        if !isnothing(i)
            push!(new_args, i)
        end
    end

    for i ∈ available2
        if !isnothing(i)
            push!(new_args, i)
        end
    end

    new_arg = nothing

    for args ∈ Iterators.partition(new_args, 2)
        if length(args) == 1
            if isnothing(new_arg)
                return args[1]
            else
                return simplify(BinaryOperation{Mult}(new_arg, args[1]))
            end
        end

        if isnothing(new_arg)
            new_arg = simplify(BinaryOperation{Mult}(args[1], args[2]))
        else
            new_arg = BinaryOperation{Mult}(
                new_arg,
                simplify(BinaryOperation{Mult}(args[1], args[2])),
            )
        end
    end

    return new_arg
end

function simplify(::Op, arg1, arg2) where {Op<:AdditiveOperation}
    @assert is_permutation(arg1, arg2)

    return BinaryOperation{Op}(simplify(arg1), simplify(arg2))
end

function simplify(::Mult, arg1::KrD, arg2::Tensor)
    return simplify(Mult(), arg2, arg1)
end

function simplify(::Mult, arg1::Tensor, arg2::Tensor)
    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(::Mult, arg1, arg2)
    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(::Mult, arg1::Union{Tensor,KrD}, arg2::KrD)
    arg1_indices = get_free_indices(arg1)
    contracting_index = eliminated_indices([arg1_indices; get_indices(arg2)])

    if isempty(contracting_index) # Is an outer product
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if is_elementwise_multiplication(arg1, arg2)
        return BinaryOperation{Mult}(arg1, arg2)
    end

    @assert can_contract(arg1, arg2)
    @assert length(arg2.indices) == 2

    newarg = deepcopy(arg1)
    empty!(newarg.indices)

    contracted = false

    for i ∈ arg1.indices
        if flip(i) == arg2.indices[1] && !contracted
            push!(newarg.indices, arg2.indices[2])
            contracted = true
        elseif flip(i) == arg2.indices[2] && !contracted
            push!(newarg.indices, arg2.indices[1])
            contracted = true
        else
            push!(newarg.indices, i)
        end
    end

    return newarg
end


function simplify(arg::Union{Tensor,KrD,Zero})
    return arg
end

function simplify(arg::Real)
    return arg
end

function simplify(arg::UnaryOp) where {UnaryOp<:UnaryOperation}
    return UnaryOp(simplify(arg.arg))
end

function simplify(arg::BinaryOperation{Add})
    return BinaryOperation{Add}(simplify(arg.arg1), simplify(arg.arg2))
end

function group_factors(factors::AbstractArray)
    indices = vcat([get_indices(f) for f ∈ factors]...)
    letters = unique([i.letter for i ∈ indices])

    chunked_factors = []
    remaining = Any[f for f ∈ factors]

    # Find groups where contractions happen over more than two indices
    for letter ∈ letters
        complex = []

        if all(isnothing.(remaining))
            break
        end

        for i ∈ eachindex(remaining)
            if isnothing(remaining[i])
                continue
            end
            if has_letter(remaining[i], letter)
                push!(complex, i)
            end
        end

        if isempty(complex)
            continue
        end

        complex_ids = LowerOrUpperIndex[]

        for ci ∈ complex
            append!(complex_ids, get_indices(remaining[ci]))
        end

        target_indices = unique(eliminate_indices(complex_ids))
        eliminated_ids = eliminated_indices(complex_ids)

        if any(typeof.(factors[complex]) .== Zero)
            free_indices = unique(eliminate_indices(complex_ids))
            push!(chunked_factors, Zero(free_indices...))
            remaining[complex] .= nothing
        elseif length(complex) == 2 &&
               is_regular_contraction(remaining[first(complex)], remaining[last(complex)])
            if typeof(remaining[first(complex)]) == KrD ||
               typeof(remaining[last(complex)]) == KrD
                push!(
                    chunked_factors,
                    simplify(Mult(), remaining[first(complex)], remaining[last(complex)]),
                )
                remaining[complex] .= nothing
            end
            continue
        elseif isempty(target_indices)
            push!(chunked_factors, remaining[complex])
            for ci ∈ complex
                remaining[ci] = nothing
            end
        elseif length(complex) == 1
            continue
        elseif isempty(eliminated_ids)
            push!(chunked_factors, remaining[complex])
            for ci ∈ complex
                remaining[ci] = nothing
            end
        elseif isempty(target_indices)
            push!(chunked_factors, remaining[complex])
            for ci ∈ complex
                remaining[ci] = nothing
            end
        elseif length(target_indices) == 1
            ordered_factors = []

            if all(typeof.(factors[complex]) .== KrD) # sum
                if length(complex) != 2
                    throw_not_std()
                end

                for di ∈ complex
                    push!(ordered_factors, to_standard(factors[di]))
                    remaining[di] = nothing
                end
            else
                for fi ∈ complex
                    factor = remaining[fi]

                    if typeof(factor) != KrD
                        @assert length(get_indices(factor)) == 1 # other orders not implemented

                        push!(ordered_factors, reshape(factor, target_indices...))
                        remaining[fi] = nothing
                    elseif isempty(get_indices(factor))
                        pushfirst!(ordered_factors, factor)
                        remaining[fi] = nothing
                    elseif factor isa Real
                        pushfirst!(ordered_factors, factor)
                        remaining[fi] = nothing
                    else
                        # drop unneeded KrD:s
                        remaining[fi] = nothing
                    end
                end
            end

            push!(chunked_factors, ordered_factors)
        else
            @show target_indices
            @assert false
        end
    end

    for i ∈ eachindex(remaining)
        if !isnothing(remaining[i])
            push!(chunked_factors, remaining[i])
            remaining[i] = nothing
        end
    end

    return chunked_factors
end

function simplify(arg::BinaryOperation{Mult})
    factors = collect_factors(arg)
    factors = map(simplify, factors) # recursion

    grouped_factors = group_factors(factors)

    for i ∈ eachindex(grouped_factors)
        if grouped_factors[i] isa AbstractArray
            grouped_factors[i] = to_binary_operation(Mult(), grouped_factors[i])
        end
    end

    if length(grouped_factors) == 1
        return first(grouped_factors)
    else
        op = to_binary_operation(Mult(), grouped_factors)
        return simplify(Mult(), op.arg1, op.arg2)
    end
end
