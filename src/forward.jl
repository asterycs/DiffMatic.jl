# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function diff(arg::Tensor, wrt::Tensor)
    if arg.id == wrt.id
        @assert length(arg.indices) == length(wrt.indices)

        D = 1

        for (u, l) ∈ zip(arg.indices, wrt.indices)
            D = BinaryOperation{Mult}(D, KrD(u, flip(l)))
        end

        return evaluate(D) # evaluate to get rid of the constant factor
    end

    return Zero(eliminate_indices(arg.indices)..., [flip(i) for i ∈ wrt.indices]...)
end

function diff(arg::KrD, wrt::Tensor)
    # This is arguably inconsistent with Tensor but the result will be Zero anyway
    # and this way the double indices are confined to the KrDs.
    return BinaryOperation{Mult}(arg, Zero([flip(i) for i ∈ wrt.indices]...))
end

function diff(arg::Real, wrt::Tensor)
    return Zero([flip(i) for i ∈ wrt.indices]...)
end

function diff(arg::Negate, wrt::Tensor)
    return Negate(diff(arg.arg, wrt))
end

function diff(arg::Sin, wrt::Tensor)
    return BinaryOperation{Mult}(Cos(arg.arg), diff(arg.arg, wrt))
end

function diff(arg::Cos, wrt::Tensor)
    return BinaryOperation{Mult}(Negate(Sin(arg.arg)), diff(arg.arg, wrt))
end

function diff(arg::BinaryOperation{Pow}, wrt::Tensor)
    return BinaryOperation{Mult}(
        BinaryOperation{Mult}(arg.arg2, BinaryOperation{Pow}(arg.arg1, arg.arg2 - 1)),
        diff(arg.arg1, wrt),
    )
end

function diff(arg::BinaryOperation{Mult}, wrt::Tensor)
    return BinaryOperation{Add}(
        BinaryOperation{Mult}(arg.arg1, diff(arg.arg2, wrt)),
        BinaryOperation{Mult}(diff(arg.arg1, wrt), arg.arg2),
    )
end

function diff(arg::BinaryOperation{Op}, wrt::Tensor) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(diff(arg.arg1, wrt), diff(arg.arg2, wrt))
end

function is_regular_contraction(arg1, arg2)
    arg1_free_ids, arg2_free_ids = get_free_indices.((arg1, arg2))

    eliminated = eliminated_indices([arg1_free_ids; arg2_free_ids])

    # TODO: Refactor
    return !isempty(intersect(arg1_free_ids, eliminated)) &&
           length(eliminated) == 2 &&
           length(intersect(get_indices(arg1), eliminated)) == 1 &&
           length(intersect(get_indices(arg2), eliminated)) == 1
end

function collect_factors(arg::BinaryOperation{Mult})
    return [collect_factors(arg.arg1); collect_factors(arg.arg2)]
end

function collect_factors(arg)
    return [arg]
end

function has_letter(tensor::Value, letter::Letter)
    ids = get_indices(tensor)
    letters = [i.letter for i ∈ ids]

    return letter ∈ letters
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

function simplify(arg::BinaryOperation{Mult})
    factors = collect_factors(arg)
    factors = map(simplify, factors) # recursion
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
                    exec(Mult(), remaining[first(complex)], remaining[last(complex)]),
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

    for i ∈ eachindex(chunked_factors)
        if chunked_factors[i] isa AbstractArray
            chunked_factors[i] = to_binary_operation(NonStdCon(), chunked_factors[i])
        end
    end

    return to_binary_operation(Mult(), chunked_factors)
end

# TODO: Rename evaluate to e.g. expand. Evaluate does not evaluate anymore in order to retain more context.
function evaluate(arg::Negate)
    return Negate(evaluate(arg.arg))
end

function evaluate(arg::Union{Tensor,KrD,Zero,Real})
    return arg
end

function evaluate(arg::Sin)
    return Sin(evaluate(arg.arg))
end

function evaluate(arg::Cos)
    return Cos(evaluate(arg.arg))
end

# TODO: There is some technical debt here, many evaluate methods could be removed now.
function evaluate(::Mult, arg1::BinaryOperation{Pow}, arg2::KrD)
    return BinaryOperation{Mult}(arg1, arg2)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::Real)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Real, arg2::BinaryOperation{Mult})
    if arg2.arg1 isa Real
        return BinaryOperation{Mult}(arg1 * arg2.arg1, arg2.arg2)
    else
        return BinaryOperation{Mult}(arg1, evaluate(arg2))
    end
end

function indices_in_common(arg1, arg2)
    arg1_indices = get_indices(arg1)
    arg2_indices = get_indices(arg2)

    return intersect(arg1_indices, arg2_indices)
end

function is_elementwise_multiplication(arg1, arg2)
    return !isempty(indices_in_common(arg1, arg2))
end

function evaluate(::Mult, arg1::Tensor, arg2::BinaryOperation{Mult})
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::Tensor)
    return BinaryOperation{Mult}(arg1, arg2)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    return BinaryOperation{Mult}(arg1, arg2)
end

function evaluate(::Mult, arg1::KrD, arg2::BinaryOperation{Mult})
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::KrD)
    return BinaryOperation{Mult}(arg1, arg2)
end

function evaluate(::Mult, arg1::Zero, arg2::UnaryOperation)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::UnaryOperation, arg2::Zero)
    free_indices = unique(eliminate_indices([get_indices(arg1); get_indices(arg2)]))

    return Zero(free_indices...)
end

function evaluate(::Mult, arg1::Zero, arg2::TensorExpr)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::TensorExpr, arg2::Zero)
    free_indices = unique(eliminate_indices([get_indices(arg1); get_indices(arg2)]))

    return Zero(free_indices...)
end

function evaluate(::Mult, arg1::KrD, arg2::Zero)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Zero, arg2::KrD)
    contracting_index = eliminated_indices([get_indices(arg1); get_free_indices(arg2)])

    if isempty(contracting_index)
        return Zero(union(arg1.indices, arg2.indices)...)
    end

    @assert can_contract(arg1, arg2)
    @assert length(arg2.indices) == 2

    free_indices = unique(eliminate_indices([get_indices(arg1); get_indices(arg2)]))

    return Zero(free_indices...)
end

function evaluate(::Mult, arg1::Negate, arg2::TensorExpr)
    return Negate(evaluate(Mult(), arg1.arg, arg2))
end

function evaluate(::Mult, arg1::TensorExpr, arg2::Negate)
    return Negate(evaluate(Mult(), arg1, arg2.arg))
end

function evaluate(::Mult, arg1::Negate, arg2::KrD)
    return invoke(evaluate, Tuple{Mult,UnaryOperation,KrD}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::KrD, arg2::Negate)
    return invoke(evaluate, Tuple{Mult,KrD,UnaryOperation}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::UnaryOperation, arg2::KrD)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::KrD, arg2::UnaryOp) where {UnaryOp<:UnaryOperation}
    return BinaryOperation{Mult}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Mult, arg1::Tensor, arg2::KrD)
    return _multiply_with_krd(arg1, arg2)
end

function evaluate(::Mult, arg1::KrD, arg2::Tensor)
    return _multiply_with_krd(arg2, arg1)
end

function evaluate(::Mult, arg1::KrD, arg2::KrD)
    return _multiply_with_krd(arg1, arg2)
end

# TODO: Delete
function _multiply_with_krd(arg1::Union{Tensor,KrD}, arg2::KrD)
    return BinaryOperation{Mult}(arg1, arg2)
end

function evaluate(
    ::Mult,
    arg1::BinaryOperation{Op},
    arg2::Union{Tensor,KrD},
) where {Op<:AdditiveOperation}
    return evaluate(
        Op(),
        evaluate(Mult(), evaluate(arg1.arg1), evaluate(arg2)),
        evaluate(Mult(), evaluate(arg1.arg2), evaluate(arg2)),
    )
end

function evaluate(
    ::Mult,
    arg1::Union{Tensor,KrD},
    arg2::BinaryOperation{Op},
) where {Op<:AdditiveOperation}
    return evaluate(
        Op(),
        evaluate(Mult(), arg1, evaluate(arg2.arg1)),
        evaluate(Mult(), arg1, evaluate(arg2.arg2)),
    )
end

function evaluate(::Mult, arg1::Value, arg2::Value)
    return BinaryOperation{Mult}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Mult, arg1::Negate, arg2::Negate)
    return evaluate(Mult(), arg1.arg, arg2.arg)
end

function evaluate(::Mult, arg1::Negate, arg2::Zero)
    return invoke(evaluate, Tuple{Mult,TensorExpr,Zero}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::Zero, arg2::Negate)
    return invoke(evaluate, Tuple{Mult,Zero,TensorExpr}, Mult(), arg1, arg2)
end

function evaluate(::Mult, arg1::TensorExpr, arg2::Real)
    evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Real, arg2::TensorExpr)
    if arg1 == 1
        return arg2
    else
        BinaryOperation{Mult}(arg1, arg2)
    end
end

function evaluate(::Mult, arg1::Zero, arg2::Zero)
    new_indices = eliminate_indices([get_free_indices(arg1); get_free_indices(arg2)])

    return Zero(new_indices...)
end

function evaluate(::Mult, arg1::Zero, arg2::Real)
    return evaluate(arg1)
end

function evaluate(::Mult, arg1::Real, arg2::Zero)
    return evaluate(arg2)
end

function evaluate(::Add, arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return arg1
end

function evaluate(::Add, arg1::Zero, arg2::Value)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg2)
end

function evaluate(::Add, arg1::Value, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg1)
end

function evaluate(::Add, arg1::Real, arg2::Real)
    return arg1 + arg2
end

function evaluate(::Add, arg1::Value, arg2::Value)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return BinaryOperation{Mult}(2, arg1)
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::Zero)
    return invoke(evaluate, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    return _add_to_product(arg1, arg2)
end

function evaluate(::Add, arg1::Value, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::Value)
    return _add_to_product(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Sub})
    return _add_to_product(arg1, arg2)
end

function mirror_add(arg::BinaryOperation{Add})
    return BinaryOperation{Add}(arg.arg2, arg.arg1)
end

function mirror_add(arg)
    return arg
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Mult})
    return mirror_add(_add_to_product(arg2, arg1))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::Value)
    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg1) + 1, evaluate(arg2))
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Add})
    if evaluate(arg1) == evaluate(arg2.arg1)
        return BinaryOperation{Add}(
            evaluate(BinaryOperation{Mult}(2, evaluate(arg1))),
            evaluate(arg2.arg2),
        )
    end

    if evaluate(arg1) == evaluate(arg2.arg2)
        return BinaryOperation{Add}(
            evaluate(BinaryOperation{Mult}(2, evaluate(arg1))),
            evaluate(arg2.arg1),
        )
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2.arg1)
        return evaluate(
            BinaryOperation{Add}(
                evaluate(
                    BinaryOperation{Mult}(evaluate(arg1.arg1) + 1, evaluate(arg1.arg2)),
                ),
                evaluate(arg2.arg2),
            ),
        )
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2.arg2)
        return evaluate(
            BinaryOperation{Add}(
                evaluate(
                    BinaryOperation{Mult}(evaluate(arg1.arg1) + 1, evaluate(arg1.arg2)),
                ),
                evaluate(arg2.arg1),
            ),
        )
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Sub})
    if evaluate(arg1) == evaluate(arg2.arg1)
        return BinaryOperation{Sub}(
            evaluate(BinaryOperation{Mult}(2, evaluate(arg1))),
            evaluate(arg2.arg2),
        )
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2.arg1)
        return evaluate(
            BinaryOperation{Sub}(
                evaluate(
                    BinaryOperation{Mult}(evaluate(arg1.arg1) + 1, evaluate(arg1.arg2)),
                ),
                evaluate(arg2.arg2),
            ),
        )
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2.arg2)
        return evaluate(
            BinaryOperation{Add}(
                evaluate(
                    BinaryOperation{Mult}(evaluate(arg1.arg1) - 1, evaluate(arg1.arg2)),
                ),
                evaluate(arg2.arg1),
            ),
        )
    end

    if evaluate(arg1) == evaluate(arg2.arg2)
        return evaluate(arg2.arg1)
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function _add_to_product(arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    if evaluate(arg1) == evaluate(arg2)
        return BinaryOperation{Mult}(2, evaluate(arg1))
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Add})
    if arg1.arg1 == arg2.arg1
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg1),
            evaluate(BinaryOperation{Add}(arg1.arg2, arg2.arg2)),
        )
    end

    if arg1.arg1 == arg2.arg2
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg1),
            evaluate(BinaryOperation{Add}(arg1.arg2, arg2.arg1)),
        )
    end

    if arg1.arg2 == arg2.arg1
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg2),
            evaluate(BinaryOperation{Add}(arg1.arg1, arg2.arg2)),
        )
    end

    if arg1.arg2 == arg2.arg2
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, arg1.arg2),
            evaluate(BinaryOperation{Add}(arg1.arg1, arg2.arg1)),
        )
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::Zero)
    return invoke(evaluate, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::Zero, arg2::BinaryOperation{Add})
    return invoke(evaluate, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::Zero, arg2::BinaryOperation{Mult})
    invoke(evaluate, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::Value, arg2::BinaryOperation{Add})
    return evaluate(Add(), arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Add})
    return _add_to_product(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::Value)
    if evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, evaluate(arg1.arg1)),
            evaluate(arg1.arg2),
        )
    end

    if evaluate(arg1.arg2) == evaluate(arg2)
        return BinaryOperation{Add}(
            BinaryOperation{Mult}(2, evaluate(arg1.arg2)),
            evaluate(arg1.arg1),
        )
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Add, arg1::Value, arg2::BinaryOperation{Sub})
    return evaluate(Add(), arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::Value)
    if evaluate(arg1.arg1) == evaluate(arg2)
        return BinaryOperation{Sub}(
            BinaryOperation{Mult}(2, evaluate(arg1.arg1)),
            evaluate(arg1.arg2),
        )
    end

    if evaluate(arg1.arg2) == evaluate(arg2)
        return evaluate(arg1.arg1)
    end

    return BinaryOperation{Add}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::Zero)
    return invoke(evaluate, Tuple{Add,Value,Zero}, Add(), arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
end

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Sub})
    return evaluate(Add, arg2, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Add})
    if arg1.arg1 == arg2.arg1
        return evaluate(
            BinaryOperation{Add}(
                evaluate(BinaryOperation{Mult}(2, arg1.arg1)),
                evaluate(BinaryOperation{Sub}(arg2.arg2, arg1.arg2)),
            ),
        )
    end

    if arg1.arg1 == arg2.arg2
        return evaluate(
            BinaryOperation{Add}(
                evaluate(BinaryOperation{Mult}(2, arg1.arg1)),
                evaluate(BinaryOperation{Sub}(arg2.arg1, arg1.arg2)),
            ),
        )
    end

    if arg1.arg2 == arg2.arg1
        return BinaryOperation{Add}(arg1.arg1, arg2.arg2)
    end

    if arg1.arg2 == arg2.arg2
        return BinaryOperation{Add}(arg1.arg1, arg2.arg1)
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function evaluate(::Add, arg1::BinaryOperation{Sub}, arg2::BinaryOperation{Sub})
    if arg1.arg1 == arg2.arg1
        return evaluate(
            BinaryOperation{Sub}(
                evaluate(BinaryOperation{Mult}(2, arg1.arg1)),
                evaluate(BinaryOperation{Add}(arg1.arg2, arg2.arg2)),
            ),
        )
    end

    if arg1.arg1 == arg2.arg2
        return evaluate(BinaryOperation{Sub}(arg2.arg1, arg1.arg2))
    end

    if arg1.arg2 == arg2.arg1
        return evaluate(BinaryOperation{Sub}(arg1.arg1, arg2.arg2))
    end

    if arg1.arg2 == arg2.arg2
        return evaluate(
            BinaryOperation{Sub}(
                evaluate(BinaryOperation{Add}(arg1.arg1, arg2.arg1)),
                evaluate(BinaryOperation{Mult}(2, arg1.arg2)),
            ),
        )
    end

    return BinaryOperation{Add}(arg1, arg2)
end

function evaluate(::Add, arg1::Zero, arg2::BinaryOperation{Sub})
    return invoke(evaluate, Tuple{Add,Zero,Value}, Add(), arg1, arg2)
end

function evaluate(::Sub, arg1::Zero, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return arg1
end

function evaluate(::Sub, arg1::Zero, arg2::Value)
    @assert is_permutation(arg1, arg2)

    return Negate(evaluate(arg2))
end

function evaluate(::Sub, arg1::Value, arg2::Zero)
    @assert is_permutation(arg1, arg2)

    return evaluate(arg1)
end

function evaluate(::Sub, arg1::Real, arg2::Real)
    return arg1 - arg2
end

function evaluate(::Sub, arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    @assert is_permutation(arg1_indices, arg2_indices)

    if arg1 == arg2
        return Zero(arg1_indices...)
    end

    return BinaryOperation{Sub}(arg1, arg2)
end

function evaluate(::Sub, arg1::Zero, arg2::BinaryOperation{Mult})
    return invoke(evaluate, Tuple{Sub,Zero,Value}, Sub(), arg1, arg2)
end

function evaluate(::Sub, arg1::BinaryOperation{Mult}, arg2::Zero)
    return invoke(evaluate, Tuple{Sub,Value,Zero}, Sub(), arg1, arg2)
end

function evaluate(::Sub, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    return _sub_from_product(arg1, arg2)
end

function evaluate(::Sub, arg1::BinaryOperation{Mult}, arg2::Value)
    return _sub_from_product(arg1, arg2)
end

function _sub_from_product(arg1::BinaryOperation{Mult}, arg2::Value)
    if evaluate(arg1) == evaluate(arg2)
        arg1_indices = get_free_indices(arg1)
        return Zero(arg1_indices...)
    end

    if evaluate(arg1.arg1) isa Real && evaluate(arg1.arg2) == evaluate(arg2)
        return evaluate(BinaryOperation{Mult}(evaluate(arg1.arg1) - 1, evaluate(arg2)))
    end

    return BinaryOperation{Sub}(evaluate(arg1), evaluate(arg2))
end

function evaluate(op::BinaryOperation{Pow})
    if op.arg2 == 1
        return evaluate(op.arg1)
    end

    return BinaryOperation{Pow}(evaluate(op.arg1), op.arg2)
end

function evaluate(op::BinaryOperation{Mult})
    evaluate(Mult(), evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(op::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    evaluate(Op(), evaluate(op.arg1), evaluate(op.arg2))
end
