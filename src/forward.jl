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
