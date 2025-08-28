# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function simplify(arg::Value)
    return arg
end

function simplify(arg::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(simplify(evaluate(arg.arg)))
end

function simplify(arg::BinaryOperation{Op}) where {Op}
    return evaluate(
        simplify(Op(), simplify(evaluate(arg.arg1)), simplify(evaluate(arg.arg2))),
    )
end

function simplify(::Mult, arg1::Variable, arg2::Variable)
    return evaluate(
        BinaryOperation{Mult}(simplify(evaluate(arg1)), simplify(evaluate(arg2))),
    )
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

function to_binary_operation(op::Op, terms::AbstractArray) where {Op}
    if length(terms) == 1
        return first(terms)
    end

    return BinaryOperation{Op}(to_binary_operation(op, terms[1:(end-1)]), terms[end])
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::Literal)
    arg2_free_ids = get_free_indices(arg2)
    eliminated = eliminated_indices([get_free_indices(arg1); get_free_indices(arg2)])

    if is_diag(arg1) && !isempty(eliminated)
        d = get_diag_delta(arg1)

        @assert !isnothing(d)

        reshaped = []

        push!(
            reshaped,
            update_index(
                arg1.arg1,
                first(eliminated),
                last(eliminated);
                allow_shape_change = true,
            ),
        )
        push!(reshaped, arg1.arg2)

        reshaped = to_binary_operation(Mult(), reshaped)

        if arg2.value != 1
            reshaped = BinaryOperation{Mult}(arg2.value, reshaped)
        end

        return simplify(reshaped)
    end

    if can_contract(arg1, arg2) && length(arg2_free_ids) == 1
        elwise_ids = elementwise_indices(arg1.arg1, arg1.arg2)

        target_idx = only(arg2_free_ids)

        if flip(target_idx) ∈ elwise_ids
            tmp_letter = get_next_letter(arg1)
            new_r = BinaryOperation{Mult}(
                BinaryOperation{Mult}(
                    arg1.arg2,
                    KrD(target_idx, same_to(target_idx, tmp_letter)),
                ),
                KrD(flip_to(target_idx, tmp_letter), target_idx),
            )
            reshaped = BinaryOperation{Mult}(arg1.arg1, evaluate(new_r))

            if arg2.value != 1
                reshaped = BinaryOperation{Mult}(arg2.value, reshaped)
            end

            return reshaped
        end
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(::Mult, arg1::Tensor, arg2::BinaryOperation{Mult})
    return simplify(Mult(), arg2, arg1)
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::Tensor)
    op = BinaryOperation{Mult}(arg1, arg2)

    if is_diag(arg1) &&
       !is_elementwise_multiplication(arg1, arg2) &&
       length(get_free_indices(op)) == 1
        d = get_diag_delta(arg1)

        @assert !isnothing(d)

        target_indices = eliminate_indices(vcat(get_free_indices(arg1), get_indices(arg2)))
        factors = collect_factors(arg1)
        vector_factors = filter(f -> f != d, factors)
        reshaped = []

        for f ∈ factors
            if isequal(f, d)
                continue
            end

            free_ids = get_free_indices(f)

            if isempty(free_ids)
                push!(reshaped, f)
            elseif length(free_ids) == 1 || length(free_ids) == 2
                @assert length(target_indices) == 1

                vector_index =
                    only(get_free_indices(to_binary_operation(Mult(), vector_factors)))

                current_idx = intersect(free_ids, [vector_index])

                if !isempty(current_idx)
                    f = update_index(
                        f,
                        vector_index,
                        only(target_indices);
                        allow_shape_change = true,
                    )
                end

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

    return op
end

function simplify(::Mult, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    if is_diag(arg1)
        return invoke(
            simplify,
            Tuple{Mult,BinaryOperation{Mult},Tensor},
            Mult(),
            arg1,
            arg2,
        )
    elseif is_diag(arg2)
        return invoke(
            simplify,
            Tuple{Mult,BinaryOperation{Mult},Tensor},
            Mult(),
            arg2,
            arg1,
        )
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function simplify(::Mult, arg1::Value, arg2::Value)
    return evaluate(
        BinaryOperation{Mult}(evaluate(simplify(arg1)), evaluate(simplify(arg2))),
    )
end

function simplify(::Op, arg1::Value, arg2::Value) where {Op<:AdditiveOperation}
    return evaluate(BinaryOperation{Op}(simplify(evaluate(arg1)), simplify(evaluate(arg2))))
end
