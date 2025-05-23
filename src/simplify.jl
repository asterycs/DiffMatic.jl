# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function simplify(arg::Value)
    return arg
end

function simplify(arg::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(simplify(arg.arg))
end

function simplify(arg::BinaryOperation{Op}) where {Op}
    return evaluate(simplify(Op(), simplify(arg.arg1), simplify(arg.arg2)))
end

function simplify(::Mult, arg1::Variable, arg2::Variable)
    return BinaryOperation{Mult}(arg1, arg2)
end

function elementwise_indices(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return intersect(arg1_indices, arg2_indices)
end

function get_diag_delta(arg::BinaryOperation{Mult})
    l = get_diag_delta(arg.arg1)
    r = get_diag_delta(arg.arg2)

    if !isnothing(l)
        return l
    elseif !isnothing(r)
        return r
    end

    return nothing
end

function get_diag_delta(arg::KrD)
    if first(arg.indices).letter != last(arg.indices).letter
        return arg
    end

    return nothing
end

function get_diag_delta(arg)
    return nothing
end

function get_last_letter(indices::IndexList)
    current_last = Upper(0)

    for i ∈ indices
        if i.letter > current_last.letter
            current_last = i
        end
    end

    return current_last
end


function to_binary_operation(op::Op, terms::AbstractArray) where {Op}
    if length(terms) == 1
        return first(terms)
    end

    return BinaryOperation{Op}(to_binary_operation(op, terms[1:(end-1)]), terms[end])
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::KrD)
    if is_diag(arg1)
        d = get_diag_delta(arg1)

        @assert !isnothing(d)

        factors = collect_factors(arg1)
        reshaped = []

        for f ∈ factors
            free_ids = get_free_indices(f)
            if isempty(free_ids)
                push!(reshaped, f)
            elseif length(free_ids) == 1
                target_indices =
                    eliminate_indices(vcat(get_free_indices(arg1), get_indices(arg2)))
                @assert length(target_indices) == 1

                current_idx = intersect(free_ids, get_free_indices(d))
                f = update_index(
                    f,
                    only(current_idx),
                    only(target_indices);
                    allow_shape_change = true,
                )
                push!(reshaped, f)
            end
        end

        return to_binary_operation(Mult(), reshaped)
    end

    if is_trace(arg2)
        s = first(arg2.indices)

        elwise_ids = elementwise_indices(arg1.arg1, arg1.arg2)
        last_index = get_last_letter(union(get_free_indices(arg1), get_free_indices(arg2)))

        if !isempty(elwise_ids)
            if s ∈ elwise_ids || flip(s) ∈ elwise_ids
                if last_index ∈ get_free_indices(arg1.arg1)
                    return evaluate(BinaryOperation{Mult}(arg1.arg1, adjoint(arg1.arg2)))
                elseif last_index ∈ get_free_indices(arg1.arg2)
                    return evaluate(BinaryOperation{Mult}(adjoint(arg1.arg1), arg1.arg2))
                end

                @assert false "Unreachable"
            end
        end
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::Variable)
    if is_diag(arg1) && !is_elementwise_multiplication(arg1, arg2)
        d = get_diag_delta(arg1)

        @assert !isnothing(d)

        target_indices = eliminate_indices(vcat(get_free_indices(arg1), get_indices(arg2)))
        factors = collect_factors(arg1)
        reshaped = []

        for f ∈ factors
            if f isa KrD
                continue
            end

            free_ids = get_free_indices(f)
            if isempty(free_ids)
                push!(reshaped, f)
            elseif length(free_ids) == 1
                @assert length(target_indices) == 1

                current_idx = intersect(free_ids, get_free_indices(d))
                f = update_index(
                    f,
                    only(current_idx),
                    only(target_indices);
                    allow_shape_change = true,
                )
                push!(reshaped, f)
            else
                @assert false "Not implemented, please open an issue with your input"
            end
        end

        arg2_ids = get_free_indices(arg2)

        if length(arg2_ids) == 1
            arg2 = update_index(
                arg2,
                only(arg2_ids),
                only(target_indices);
                allow_shape_change = true,
            )
            push!(reshaped, arg2)
        else
            @assert false "Not implemented, please open an issue with your input"
        end

        return to_binary_operation(Mult(), reshaped)
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(::Mult, arg1::Value, arg2::Value)
    return evaluate(BinaryOperation{Mult}(arg1, arg2))
end

function simplify(::Op, arg1::Value, arg2::Value) where {Op<:AdditiveOperation}
    return evaluate(BinaryOperation{Op}(arg1, arg2))
end
