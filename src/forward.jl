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
    indices = eliminate_indices(indices)

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
    return BinaryOperation{Mult}(UnaryOperation{Sgn}(arg.arg), diff(arg.arg, wrt))
end

function diff(arg::UnaryOperation{Sin}, wrt::Variable)
    return BinaryOperation{Mult}(UnaryOperation{Cos}(arg.arg), diff(arg.arg, wrt))
end

function diff(arg::UnaryOperation{Cos}, wrt::Variable)
    return BinaryOperation{Mult}(-UnaryOperation{Sin}(arg.arg), diff(arg.arg, wrt))
end

function diff(arg::Power, wrt::Variable)
    return BinaryOperation{Mult}(
        BinaryOperation{Mult}(arg.exponent, Power(arg.base, arg.exponent - 1)),
        diff(arg.base, wrt),
    )
end

function diff(arg::BinaryOperation{Mult}, wrt::Variable)
    return BinaryOperation{Add}(
        BinaryOperation{Mult}(arg.arg1, diff(arg.arg2, wrt)),
        BinaryOperation{Mult}(diff(arg.arg1, wrt), arg.arg2),
    )
end

function diff(arg::BinaryOperation{Op}, wrt::Variable) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(diff(arg.arg1, wrt), diff(arg.arg2, wrt))
end

function collect_factors(arg::BinaryOperation{Mult})
    return Value[collect_factors(arg.arg1); collect_factors(arg.arg2)]
end

function collect_factors(arg)
    return Value[arg]
end

function evaluate(arg::Union{Variable,Literal,KrD,Zero,Real})
    return arg
end

function evaluate(arg::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(evaluate(arg.arg))
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::Real)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Real, arg2::BinaryOperation{Mult})
    if arg1 == 1
        return arg2
    end

    if arg2.arg1 isa Real
        return BinaryOperation{Mult}(arg1 * arg2.arg1, arg2.arg2)
    elseif arg2.arg2 isa Real
        return BinaryOperation{Mult}(arg1 * arg2.arg2, arg2.arg1)
    end

    return BinaryOperation{Mult}(arg1, evaluate(arg2))
end

function indices_in_common(arg1, arg2)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return intersect(arg1_indices, arg2_indices)
end

function is_elementwise_multiplication(arg1, arg2)
    return !isempty(indices_in_common(arg1, arg2))
end


function is_diag(arg::BinaryOperation{Mult})
    return is_diag(arg.arg1, arg.arg2)
end

function is_diag(arg)
    return false
end

function is_diag(arg1::KrD, arg2::Tensor)
    return is_diag(arg2, arg1)
end

function is_diag(arg1::KrD, arg2::KrD)
    return false
end

function is_diag(arg::Union{Variable,Literal,KrD,Zero})
    return false
end

function is_diag(arg1::Tensor, arg2::KrD)
    arg1_indices, arg2_indices = get_free_indices.((arg1, arg2))

    return length(arg1_indices) == 1 && !isempty(intersect(arg1_indices, arg2_indices))
end

function is_diag(arg1::Value, arg2::Value)
    if isempty(get_free_indices(arg1))
        return is_diag(arg2)
    elseif isempty(get_free_indices(arg2))
        return is_diag(arg1)
    end

    return is_diag(arg1) || is_diag(arg2)
end

function evaluate(::Mult, arg1::Union{Variable,Literal}, arg2::BinaryOperation{Mult})
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::Union{Variable,Literal})
    if arg1.arg1 isa Real
        return BinaryOperation{Mult}(arg1.arg1, BinaryOperation{Mult}(arg1.arg2, arg2))
    end

    is_arg1_elementwise = is_elementwise_multiplication(arg1.arg1, arg1.arg2)
    is_all_elementwise =
        is_elementwise_multiplication(arg1.arg1, arg2) &&
        is_elementwise_multiplication(arg1.arg2, arg2)

    if is_arg1_elementwise || is_all_elementwise
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if can_contract(arg1.arg2, arg2) || is_elementwise_multiplication(arg1.arg2, arg2)
        new_arg2 = evaluate(Mult(), arg1.arg2, arg2)
        return BinaryOperation{Mult}(arg1.arg1, new_arg2)
    elseif can_contract(arg1.arg1, arg2) || is_elementwise_multiplication(arg1.arg1, arg2)
        new_arg1 = evaluate(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, arg1.arg2)
    end

    return BinaryOperation{Mult}(arg1, arg2)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::BinaryOperation{Mult})
    if arg1.arg1 == -1 && arg2.arg1 == -1
        return BinaryOperation{Mult}(arg1.arg2, arg2.arg2)
    elseif arg1.arg1 == -1
        return BinaryOperation{Mult}(
            arg1.arg1,
            evaluate(BinaryOperation{Mult}(arg1.arg2, arg2)),
        )
    elseif arg2.arg1 == -1
        return BinaryOperation{Mult}(
            arg2.arg1,
            evaluate(BinaryOperation{Mult}(arg2.arg2, arg1)),
        )
    end

    new_args = []

    available1 = Any[arg1.arg1; arg1.arg2]
    if is_elementwise_multiplication(arg1.arg1, arg1.arg2)
        available1 = Any[arg1]
    end

    available2 = Any[arg2.arg1; arg2.arg2]
    if is_elementwise_multiplication(arg2.arg1, arg2.arg2)
        available2 = Any[arg2]
    end


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
                return BinaryOperation{Mult}(new_arg, args[1])
            end
        end

        if isnothing(new_arg)
            new_arg = BinaryOperation{Mult}(args[1], args[2])
        else
            new_arg = BinaryOperation{Mult}(
                new_arg,
                evaluate(BinaryOperation{Mult}(args[1], args[2])),
            )
        end
    end

    return new_arg
end

function evaluate(::Mult, arg1::KrD, arg2::BinaryOperation{Mult})
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::BinaryOperation{Mult}, arg2::KrD)
    ci = indices_in_common(arg1.arg1, arg1.arg2)

    if !isempty(ci) && !is_trace(arg2)
        el = eliminated_indices([ci; arg2.indices[1]])
        er = eliminated_indices([ci; arg2.indices[2]])

        if !isempty(el)
            return evaluate(
                BinaryOperation{Mult}(
                    evaluate(Mult(), arg1.arg1, arg2), # order of the indices in arg2 determines which one is contracted
                    evaluate(Mult(), arg1.arg2, arg2),
                ),
            )
        elseif !isempty(er)
            rd = KrD(reverse(arg2.indices)...)

            return evaluate(
                BinaryOperation{Mult}(
                    evaluate(Mult(), arg1.arg1, rd),
                    evaluate(Mult(), arg1.arg2, rd),
                ),
            )
        end
    end

    if can_contract(arg1.arg2, arg2) && !is_trace(arg2)
        new_arg2 = evaluate(Mult(), arg1.arg2, arg2)
        return BinaryOperation{Mult}(evaluate(arg1.arg1), new_arg2)
    elseif can_contract(arg1.arg1, arg2) && !is_trace(arg2)
        new_arg1 = evaluate(Mult(), arg1.arg1, arg2)
        return BinaryOperation{Mult}(new_arg1, evaluate(arg1.arg2))
    elseif arg1.arg1 isa Real
        return BinaryOperation{Mult}(arg1.arg1, BinaryOperation{Mult}(arg1.arg2, arg2))
    else
        return BinaryOperation{Mult}(arg1, arg2)
    end
end

function evaluate(::Mult, arg1::Zero, arg2::UnaryOperation)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::UnaryOperation, arg2::Zero)
    free_indices = unique(eliminate_indices([get_indices(arg1); get_indices(arg2)]))

    return Zero(free_indices...)
end

function evaluate(::Mult, arg1::Zero, arg2::Tensor)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Tensor, arg2::Zero)
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

function evaluate(::Mult, arg1::UnaryOperation, arg2::KrD)
    return evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::KrD, arg2::UnaryOp) where {UnaryOp<:UnaryOperation}
    if can_contract(evaluate(arg1), evaluate(arg2.arg))
        return UnaryOp(evaluate(Mult(), evaluate(arg1), evaluate(arg2.arg)))
    end

    return BinaryOperation{Mult}(evaluate(arg2), evaluate(arg1))
end

function evaluate(::Mult, arg1::Power, arg2::KrD)
    attempt = Power(evaluate(Mult(), evaluate(arg1.base), evaluate(arg2)), arg1.exponent)

    # This ensures that arg1.base and arg2 can contract and that the contraction is simple
    if length(get_free_indices(attempt)) == length(get_free_indices(arg1))
        return attempt
    end

    return BinaryOperation{Mult}(evaluate(arg1), evaluate(arg2))
end

function evaluate(::Mult, arg1::Union{Variable,Literal}, arg2::KrD)
    return _multiply_with_krd(arg1, arg2)
end

function evaluate(::Mult, arg1::KrD, arg2::Union{Variable,Literal})
    return _multiply_with_krd(arg2, arg1)
end

function evaluate(::Mult, arg1::KrD, arg2::KrD)
    return _multiply_with_krd(arg1, arg2)
end

function _multiply_with_krd(arg1::Union{Variable,Literal,KrD}, arg2::KrD)
    arg1_indices = get_free_indices(arg1)
    contracting_index = eliminated_indices([arg1_indices; get_indices(arg2)])

    if isempty(contracting_index) # Is an outer product
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if is_elementwise_multiplication(arg1, arg2)
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if is_trace(arg1) || is_trace(arg2)
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

function evaluate(
    ::Mult,
    arg1::BinaryOperation{Op},
    arg2::Union{Literal,KrD},
) where {Op<:AdditiveOperation}
    return evaluate(
        Op(),
        evaluate(Mult(), evaluate(arg1.arg1), evaluate(arg2)),
        evaluate(Mult(), evaluate(arg1.arg2), evaluate(arg2)),
    )
end

function evaluate(
    ::Mult,
    arg1::Union{Literal,KrD},
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

function evaluate(::Mult, arg1::Tensor, arg2::Real)
    evaluate(Mult(), arg2, arg1)
end

function evaluate(::Mult, arg1::Real, arg2::Tensor)
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

function evaluate(::Add, arg1::BinaryOperation{Add}, arg2::BinaryOperation{Mult})
    return _add_to_product(arg2, arg1)
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
    if is_permutation(collect_factors(arg1), collect_factors(arg2))
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

    return -evaluate(arg2)
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

function evaluate(op::Power)
    if op.exponent == 1
        return evaluate(op.base)
    end

    return Power(evaluate(op.base), op.exponent)
end

function evaluate(op::BinaryOperation{Mult})
    evaluate(Mult(), evaluate(op.arg1), evaluate(op.arg2))
end

function evaluate(op::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    evaluate(Op(), evaluate(op.arg1), evaluate(op.arg2))
end
