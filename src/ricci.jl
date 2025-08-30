# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import LinearAlgebra

export vector
export matrix

abstract type Tensor end

Value = Union{Tensor,Real}

function get_indices(arg::Real)
    return LowerOrUpperIndex[]
end

struct Variable <: Tensor
    id::String
    indices::IndexList

    function Variable(id, indices::LowerOrUpperIndex...)
        # Convert type
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(unique(indices)) != length(indices)
            throw(DomainError(indices, "Indices of $id are invalid"))
        end

        new(id, indices)
    end
end

Base.hash(m::Variable, h::UInt) = hash(Variable, hash(m.id, hash(m.indices, h)))

function Base.:(==)(left::Variable, right::Variable)
    return left.id == right.id && left.indices == right.indices
end

struct Literal <: Tensor
    value::Real
    indices::IndexList

    function Literal(value::Real, indices::LowerOrUpperIndex...)
        # Convert type
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(unique(indices)) != length(indices)
            throw(
                DomainError(
                    indices,
                    "Indices of literal with value '$(string(value))' are invalid",
                ),
            )
        end

        new(value, indices)
    end
end

Base.hash(l::Literal, h::UInt) = hash(Literal, hash(l.value, hash(l.indices, h)))

function Base.:(==)(left::Literal, right::Literal)
    return left.value == right.value && left.indices == right.indices
end

function vector(arg::Real)
    return Literal(arg, Upper(1))
end

function matrix(arg::Real)
    return Literal(arg, Upper(1), Lower(2))
end

function are_unique(arg::AbstractArray)
    return length(unique(arg)) == length(arg)
end

struct KrD <: Tensor
    indices::IndexList

    function KrD(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if !are_unique(indices) || length(indices) != 2
            throw(DomainError(indices, "Indices of δ are invalid"))
        end

        new(indices)
    end
end

Base.hash(m::KrD, h::UInt) = hash(KrD, hash(m.indices, h))

function Base.:(==)(left::KrD, right::KrD)
    return left.indices == right.indices
end

struct Zero <: Tensor
    indices::IndexList

    function Zero(indices::LowerOrUpperIndex...)
        indices = LowerOrUpperIndex[i for i ∈ indices]

        if length(unique(indices)) != length(indices)
            throw(DomainError(indices, "Indices of 0 are invalid"))
        end

        new(indices)
    end
end

Base.hash(m::Zero, h::UInt) = hash(Zero, hash(m.indices, h))

function Base.:(==)(left::Zero, right::Zero)
    left_ids = get_free_indices(left)
    right_ids = get_free_indices(right)

    return issetequal(left_ids, right_ids)
end

struct BinaryOperation{Op} <: Tensor where {Op}
    arg1::Value
    arg2::Value
end

Base.hash(op::BinaryOperation{Op}, h::UInt) where {Op} = hash(op.arg1, hash(op.arg1, h))

abstract type AdditiveOperation end
struct Add <: AdditiveOperation end
struct Sub <: AdditiveOperation end
struct Mult end

function collect_factors(arg::BinaryOperation{Mult})
    return Value[collect_factors(arg.arg1); collect_factors(arg.arg2)]
end

function collect_factors(arg)
    return Value[arg]
end

function Base.:(==)(left::BinaryOperation{Add}, right::BinaryOperation{Add})
    return (left.arg1 == right.arg1 && left.arg2 == right.arg2) ||
           (left.arg1 == right.arg2 && left.arg2 == right.arg1)
end

function Base.:(==)(left::BinaryOperation{Sub}, right::BinaryOperation{Sub})
    return left.arg1 == right.arg1 && left.arg2 == right.arg2
end

function Base.:(==)(left::BinaryOperation{Mult}, right::BinaryOperation{Mult})
    left_factors = collect_factors(left)
    right_factors = collect_factors(right)

    return issetequal(left_factors, right_factors)
end

struct Power <: Tensor
    base::Value
    exponent::Union{Int,Rational{Int}}
end

Base.hash(op::Power, h::UInt) = hash(op.exponent, hash(op.base, hash(Power, h)))

function Base.:(==)(left::Power, right::Power)
    return left.base == right.base && left.exponent == right.exponent
end

struct UnaryOperation{Op} <: Tensor where {Op}
    arg::Value
end

Base.hash(op::UnaryOperation{Op}, h::UInt) where {Op} = hash(op.arg, hash(Op, h))

struct Abs end
struct Sgn end
struct Sin end
struct Cos end
struct Log end

function Base.:(==)(left::UnaryOperation{Op}, right::UnaryOperation{Op}) where {Op}
    return left.arg == right.arg
end

function Base.sin(arg::Tensor)
    if !isempty(get_free_indices(arg))
        throw(
            DomainError(arg, "Argument is not a scalar. Did you mean to use 'sin.(...)'?"),
        )
    end

    return UnaryOperation{Sin}(arg)
end

function Base.broadcasted(::typeof(sin), arg::Tensor)
    return UnaryOperation{Sin}(arg)
end

function Base.cos(arg::Tensor)
    if !isempty(get_free_indices(arg))
        throw(
            DomainError(arg, "Argument is not a scalar. Did you mean to use 'cos.(...)'?"),
        )
    end

    return UnaryOperation{Cos}(arg)
end

function Base.broadcasted(::typeof(cos), arg::Tensor)
    return UnaryOperation{Cos}(arg)
end

function _eliminate_indices(arg::IndexList)
    CanBeNothing = Union{Nothing,Lower,Upper}
    available = CanBeNothing[i for i ∈ arg]
    eliminated = LowerOrUpperIndex[]

    for i ∈ eachindex(arg)
        if flip(arg[i]) in arg
            push!(eliminated, arg[i])
            available[i] = nothing
        end
    end

    filtered = filter(i -> i ∈ available, arg)

    return filtered, eliminated
end

function eliminate_indices(arg::IndexList)
    return first(_eliminate_indices(arg))
end

function eliminated_indices(arg::IndexList)
    return last(_eliminate_indices(arg))
end

function count_values(input::AbstractArray{T}) where {T}
    return Dict((i => count(==(i), input)) for i ∈ unique(input))
end

function get_next_letter(exprs...)
    exprs = [e for e ∈ exprs]
    indices = get_indices.(exprs)

    letters = [index.letter for index ∈ Iterators.flatten(indices)]

    if isempty(letters)
        return 1
    end

    return maximum(letters) + 1
end

function is_permutation(l::AbstractArray{T}, r::AbstractArray{T}) where {T}
    if length(l) != length(r)
        return false
    end

    l_element_count = count_values(l)
    r_element_count = count_values(r)

    for index ∈ keys(l_element_count)
        if !haskey(r_element_count, index) ||
           l_element_count[index] != r_element_count[index]
            return false
        end
    end

    return true
end

function is_permutation(arg1::Tensor, arg2::Tensor)
    arg1_indices = get_free_indices(arg1)
    arg2_indices = get_free_indices(arg2)

    return is_permutation(unique(arg1_indices), unique(arg2_indices))
end

function get_indices(arg::Union{Variable,Literal,KrD,Zero})
    @assert length(unique(arg.indices)) == length(arg.indices)

    return arg.indices
end

function get_indices(arg::UnaryOperation)
    return get_indices(arg.arg)
end

function get_indices(arg::BinaryOperation{Mult})
    return [get_indices(arg.arg1); get_indices(arg.arg2)]
end

function get_indices(arg::Power)
    return get_indices(arg.base)
end

function get_indices(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    arg1_free_ids, arg2_free_ids = get_free_indices.((arg.arg1, arg.arg2))

    @assert is_permutation(arg1_free_ids, arg2_free_ids)

    arg1_ids, arg2_ids = get_indices.((arg.arg1, arg.arg2))

    return union(arg1_ids, arg2_ids)
end

function get_free_indices(arg)
    return unique(eliminate_indices(get_indices(arg)))
end

function can_contract(arg1::Value, arg2::Value)
    arg1_indices = get_indices(arg1)
    arg2_indices = get_indices(arg2)

    # If there is at least one matching index pair then a contraction is possible.
    pairs = Dict{Letter,Int}()

    for i ∈ arg1_indices
        for j ∈ arg2_indices
            if flip(i) == j
                if haskey(pairs, i.letter)
                    pairs[i.letter] += 1
                else
                    pairs[i.letter] = 1
                end
            end
        end
    end

    if !isempty(pairs)
        return true
    end

    return false
end

function LinearAlgebra.tr(arg::Tensor)
    free_ids = get_free_indices(arg)

    de = DomainError("Trace is defined only for matrices")

    if length(free_ids) != 2
        throw(de)
    end

    if all(typeof.(free_ids) .== Upper) || all(typeof.(free_ids) .== Lower)
        throw(de)
    end

    return evaluate(BinaryOperation{Mult}(arg, KrD(flip(free_ids[2]), flip(free_ids[1]))))
end

function LinearAlgebra.norm(arg::Tensor, p::Real)
    if p == 1
        return norm1(arg)
    elseif p == 2
        return norm2(arg)
    end

    throw(DomainError(p, "$p-norm not implemented for $arg"))
end

function norm2(arg::Tensor)
    free_ids = get_free_indices(arg)

    if length(free_ids) != 1
        throw(DomainError("Norms are currently implemented only for vectors."))
    end

    p = 2

    return sum(arg .^ p)^(1//p)
end

function norm1(arg::Tensor)
    free_ids = get_free_indices(arg)

    if length(free_ids) != 1
        throw(DomainError("Norms are currently implemented only for vectors."))
    end

    return sum(abs.(arg))
end

function Base.sum(arg::Tensor)
    free_ids = get_free_indices(arg)

    if length(free_ids) != 1
        throw(DomainError("Sum is defined only for vectors"))
    end

    return BinaryOperation{Mult}(arg, Literal(1, flip(only(free_ids))))
end

function Base.broadcasted(::typeof(abs), arg::Tensor)
    return UnaryOperation{Abs}(arg)
end

function Base.abs(arg::Tensor)
    if !isempty(get_free_indices(arg))
        throw(
            DomainError(arg, "Argument is not a scalar. Did you mean to use 'abs.(...)'?"),
        )
    end

    return UnaryOperation{Abs}(arg)
end

function Base.broadcasted(::typeof(sign), arg::Tensor)
    return UnaryOperation{Sgn}(arg)
end

function Base.sign(arg::Tensor)
    if !isempty(get_free_indices(arg))
        throw(
            DomainError(arg, "Argument is not a scalar. Did you mean to use 'sign.(...)'?"),
        )
    end

    return UnaryOperation{Sgn}(arg)
end

function Base.broadcasted(::typeof(*), arg1::Tensor, arg2::Tensor)
    arg1_free_indices = get_free_indices(arg1)
    arg2_free_indices = get_free_indices(arg2)

    if length(arg1_free_indices) != length(arg2_free_indices)
        return throw(
            DomainError(
                (arg1, arg2),
                "Cannot do elementwise multiplication with inputs of different order",
            ),
        )
    end

    if typeof.(arg1_free_indices) != typeof.(arg2_free_indices)
        return throw(
            DomainError(
                (arg1, arg2),
                "Elementwise multiplication is ambiguous for tensors with different co/contravariance",
            ),
        )
    end

    new_arg1 = arg1

    for (li, ri) ∈ zip(arg1_free_indices, arg2_free_indices)
        new_arg1 = update_index(new_arg1, li, ri)
    end

    return BinaryOperation{Mult}(new_arg1, arg2)
end

function Base.broadcasted(
    ::typeof(*),
    arg1::LinearAlgebra.UniformScaling{T},
    arg2::Tensor,
) where {T<:Real}
    return (arg1.λ * KrD(Upper(1), Lower(2))) .* arg2
end

function Base.broadcasted(
    ::typeof(*),
    arg1::Tensor,
    arg2::LinearAlgebra.UniformScaling{T},
) where {T<:Real}
    return arg1 .* (arg2.λ * KrD(Upper(1), Lower(2)))
end

function Base.broadcasted(
    ::typeof(Base.literal_pow),
    f::Function,
    base::Tensor,
    exponent::Val{E},
) where {E}
    return Base.broadcasted(f, base, E)
end

function Base.broadcasted(::typeof(^), base::Tensor, exponent::Union{Int,Rational{Int}})
    return Power(base, exponent)
end

function Base.:(^)(base::Tensor, exponent::Union{Int,Rational{Int}})
    if !isempty(get_free_indices(base))
        throw(DomainError(base, "Argument is not a scalar, use .^ for element-wise power."))
    end

    return Power(base, exponent)
end

function Base.literal_pow(f::typeof(^), base::Tensor, exponent::Val{E}) where {E}
    if !isempty(get_free_indices(base))
        throw(DomainError(base, "Argument is not a scalar, use .^ for element-wise power."))
    end

    return Power(base, E)
end

function Base.log(arg::Tensor)
    if !isempty(get_free_indices(arg))
        throw(DomainError(arg, "Argument is not a scalar, use log. for element-wise log."))
    end

    return UnaryOperation{Log}(arg)
end

function Base.broadcasted(::typeof(log), arg::Tensor)
    return UnaryOperation{Log}(arg)
end

function replace_letters(arg::BinaryOperation{Mult}, letter_map::Dict)
    return BinaryOperation{Mult}(
        replace_letters(arg.arg1, letter_map),
        replace_letters(arg.arg2, letter_map),
    )
end

function replace_letters(arg::Power, letter_map::Dict)
    return Power(replace_letters(arg.base, letter_map), arg.exponent)
end

function replace_letters(arg::BinaryOperation{Op}, letter_map::Dict) where {Op}
    return BinaryOperation{Op}(
        replace_letters(arg.arg1, letter_map),
        replace_letters(arg.arg2, letter_map),
    )
end

function replace_letters(arg::Union{Variable,Literal,Zero,KrD}, letter_map::Dict)
    new_indices = LowerOrUpperIndex[]

    for i ∈ arg.indices
        if haskey(letter_map, i.letter)
            push!(new_indices, same_to(i, letter_map[i.letter]))
        else
            push!(new_indices, i)
        end
    end

    newarg = deepcopy(arg)
    empty!(newarg.indices)

    for ni ∈ new_indices
        push!(newarg.indices, ni)
    end

    return newarg
end

function replace_letters(arg::UnaryOperation{Op}, letter_map::Dict) where {Op}
    return UnaryOperation{Op}(replace_letters(arg.arg, letter_map))
end

function replace_letters(arg::Real, letter_map::Dict)
    return arg
end

function Base.:(*)(arg1::LinearAlgebra.UniformScaling{T}, arg2::Tensor) where {T<:Real}
    return arg1.λ * KrD(Upper(1), Lower(2)) * arg2
end

function Base.:(*)(arg1::Tensor, arg2::LinearAlgebra.UniformScaling{T}) where {T<:Real}
    return arg1 * arg2.λ * KrD(Upper(1), Lower(2))
end

function Base.:(*)(arg1::Tensor, arg2::Real)
    return arg2 * arg1
end

function Base.:(*)(arg1::Value, arg2::Tensor)
    arg1_indices, arg2_indices = unique.(get_indices.((arg1, arg2)))
    intersecting_letters =
        unique(intersect(get_letters(arg1_indices), get_letters(arg2_indices)))

    new_letters = Dict()
    next_letter = get_next_letter(arg1, arg2)

    for l ∈ intersecting_letters
        new_letters[l] = next_letter
        next_letter += 1
    end

    arg2 = replace_letters(arg2, new_letters)

    arg1_free_indices = get_free_indices(arg1)
    arg2_free_indices = get_free_indices(arg2)

    if isempty(arg1_free_indices)
        return BinaryOperation{Mult}(arg1, arg2)
    end

    if isempty(arg2_free_indices) # This is to keep scalars first
        return BinaryOperation{Mult}(arg2, arg1)
    end

    if length(arg1_free_indices) > 2
        throw(DomainError(arg1, "Multiplication involving tensor \"$arg1\" is ambiguous"))
    end

    if length(arg2_free_indices) > 2
        throw(DomainError(arg2, "Multiplication involving tensor \"$arg2\" is ambiguous"))
    end

    arg1_first = first(arg1_free_indices)
    arg1_last = last(arg1_free_indices)
    arg2_first = first(arg2_free_indices)
    arg2_last = last(arg2_free_indices)

    if arg1_last isa Lower && arg2_first isa Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(arg1, arg1_last, same_to(arg1_last, new_letter)),
            update_index(arg2, arg2_first, same_to(arg2_first, new_letter)),
        )
    end

    if arg1_first isa Lower && arg2_first isa Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(arg1, arg1_first, same_to(arg1_first, new_letter)),
            update_index(arg2, arg2_first, same_to(arg2_first, new_letter)),
        )
    end

    if arg1_first isa Lower && arg2_last isa Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(arg1, arg1_first, same_to(arg1_first, new_letter)),
            update_index(arg2, arg2_last, same_to(arg2_last, new_letter)),
        )
    end

    if arg1_last isa Lower && arg2_last isa Upper
        new_letter = get_next_letter(arg1, arg2)

        return BinaryOperation{Mult}(
            update_index(arg1, arg1_last, same_to(arg1_last, new_letter)),
            update_index(arg2, arg2_last, same_to(arg2_last, new_letter)),
        )
    end

    if length(arg1_free_indices) == 1 &&
       length(arg2_free_indices) == 1 &&
       arg1_first isa Upper &&
       arg2_first isa Lower
        return BinaryOperation{Mult}(
            arg1,
            update_index(
                arg2,
                arg2_first,
                same_to(arg2_first, get_next_letter(arg1, arg2)),
            ),
        )
    end

    throw(DomainError((arg1, arg2), "Multiplication with $arg1 and $arg2 is ambiguous"))
end

function get_letters(indices::IndexList)
    return [i.letter for i ∈ indices]
end

function Base.:(+)(arg1::Tensor, arg2::Tensor)
    return create_additive_op(Add(), arg1, arg2)
end

function Base.:(-)(arg1::Tensor, arg2::Tensor)
    return create_additive_op(Sub(), arg1, arg2)
end

function get_index_type_count(indices::IndexList)
    return [(t, count(==(t), typeof.(indices))) for t ∈ (Upper, Lower)]
end

function create_additive_op(
    op::Op,
    arg1::Tensor,
    arg2::Tensor,
) where {Op<:AdditiveOperation}
    arg1_ids, arg2_ids = get_free_indices.((arg1, arg2))

    if length(unique(arg1_ids)) != length(unique(arg2_ids))
        op_text = if Op == Add
            "add"
        elseif Op == Sub
            "subtract"
        end

        throw(DomainError((arg1, arg2), "Cannot $op_text tensors of different order"))
    end

    if get_index_type_count(arg1_ids) != get_index_type_count(arg2_ids)
        throw(
            DomainError(
                (arg1, arg2),
                "Cannot add tensors with different components. Did you try to add e.g. a row and a column vector?",
            ),
        )
    end

    if arg1_ids == arg2_ids
        return BinaryOperation{Op}(arg1, arg2)
    end

    next_letter = get_next_letter(arg1, arg2)

    new_ids = [next_letter + i - 1 for i ∈ 1:length(unique(arg1_ids))]

    arg1_index_map = Dict((old => new for (old, new) ∈ zip(unique(arg1_ids), new_ids)))
    arg2_index_map = Dict((old => new for (old, new) ∈ zip(unique(arg2_ids), new_ids)))

    for index ∈ unique(arg1_ids)
        arg1 = update_index(arg1, index, same_to(index, arg1_index_map[index]))
    end

    for index ∈ union(arg2_ids)
        arg2 = update_index(arg2, index, same_to(index, arg2_index_map[index]))
    end

    return BinaryOperation{Op}(arg1, arg2)
end

function update_index(
    arg::BinaryOperation{Op},
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex;
    allow_shape_change = false,
) where {Op}
    arg1 = arg.arg1
    arg2 = arg.arg2

    arg1_free_ids = get_free_indices(arg1)
    arg2_free_ids = get_free_indices(arg2)

    if from ∈ arg1_free_ids
        arg1 = update_index(arg1, from, to; allow_shape_change)
    end

    if from ∈ arg2_free_ids
        arg2 = update_index(arg2, from, to; allow_shape_change)
    end

    return BinaryOperation{Op}(arg1, arg2)
end

function update_index(
    arg::UnaryOperation{Op},
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex;
    allow_shape_change = false,
) where {Op}
    return UnaryOperation{Op}(update_index(arg.arg, from, to; allow_shape_change))
end

function update_index(
    arg::Power,
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex;
    allow_shape_change = false,
)
    return Power(update_index(arg.base, from, to; allow_shape_change), arg.exponent)
end

function update_index(
    arg::Log,
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex;
    allow_shape_change = false,
)
    return Log(update_index(arg.arg, from, to; allow_shape_change))
end

function update_index(
    arg::Union{Variable,Literal,KrD,Zero},
    from::LowerOrUpperIndex,
    to::LowerOrUpperIndex;
    allow_shape_change = false,
)
    indices = get_indices(arg)

    if from == to
        return arg
    end

    @assert from ∈ indices

    if !allow_shape_change
        if typeof(from) != typeof(to)
            throw(DomainError("A shape change is not permitted"))
        end
    end

    e = deepcopy(arg)

    for i ∈ eachindex(e.indices)
        if e.indices[i] == from
            e.indices[i] = to
        end
    end

    return e
end

function Base.:(-)(arg::Tensor)
    return BinaryOperation{Mult}(-1, arg)
end

function Base.adjoint(arg::T) where {T<:UnaryOperation}
    return T(arg.arg')
end

function Base.adjoint(arg::Power)
    return Power(adjoint(arg.base), arg.exponent)
end

function Base.adjoint(arg::BinaryOperation{Op}) where {Op}
    return evaluate(BinaryOperation{Op}(adjoint(arg.arg1), adjoint(arg.arg2)))
end

function Base.adjoint(arg::Union{Variable,Literal,KrD,Zero})
    e = deepcopy(arg)
    e.indices[:] = flip.(arg.indices)

    return e
end

function LinearAlgebra.diagm(v::Tensor)
    indices = get_free_indices(v)

    if length(indices) != 1
        throw(
            DomainError(indices, "Input is not a vector, cannot create a diagonal matrix"),
        )
    end

    vector_index = only(indices)
    next_letter = get_next_letter(v)
    id = KrD(vector_index, flip_to(vector_index, next_letter))

    return BinaryOperation{Mult}(id, v)
end

function script(index::Lower)
    @assert index.letter >= 0
    text = []

    letter = abs(index.letter)

    for d ∈ reverse(digits(letter))
        push!(text, Char(0x2080 + d))
    end

    if index.letter < 0
        pushfirst!(text, "₋")
    end

    return join(text)
end

function script(index::Upper)
    text = []

    letter = abs(index.letter)

    for d ∈ reverse(digits(letter))
        if d == 0
            push!(text, Char(0x2070))
        end
        if d == 1
            push!(text, Char(0x00B9))
        end
        if d == 2
            push!(text, Char(0x00B2))
        end
        if d == 3
            push!(text, Char(0x00B3))
        end
        if d > 3
            push!(text, Char(0x2070 + d))
        end
    end

    if index.letter < 0
        pushfirst!(text, "⁻")
    end

    return join(text)
end

function to_string(arg::Variable)
    scripts = [script(i) for i ∈ arg.indices]

    return arg.id * join(scripts)
end

function to_string(arg::Literal)
    scripts = [script(i) for i ∈ arg.indices]

    return string(arg.value) * join(scripts)
end

function to_string(arg::KrD)
    scripts = [script(i) for i ∈ arg.indices]

    return "δ" * join(scripts)
end

function to_string(arg::Real)
    return string(arg)
end

function to_string(arg::Rational)
    return string(arg.num) * "/" * string(arg.den)
end

function to_string(arg::Zero)
    scripts = [script(i) for i ∈ arg.indices]

    return "0" * join(scripts)
end

function to_string(arg::UnaryOperation{Abs})
    return "|$(arg.arg)|"
end

function to_string(arg::UnaryOperation{Sgn})
    return "sgn($(arg.arg))"
end

function to_string(arg::UnaryOperation{Sin})
    return "sin($(arg.arg))"
end

function to_string(arg::UnaryOperation{Cos})
    return "cos($(arg.arg))"
end

function to_string(arg::UnaryOperation{Log})
    return "log($(arg.arg))"
end

function parenthesize(arg)
    return to_string(arg)
end

function parenthesize(arg::Rational)
    return "(" * to_string(arg) * ")"
end

function parenthesize(arg::BinaryOperation{Add})
    return "(" * to_string(arg) * ")"
end

function parenthesize(arg::BinaryOperation{Sub})
    return "(" * to_string(arg) * ")"
end

function to_string(arg::Power)
    b = to_string(arg.base)

    if arg.base isa BinaryOperation || arg.base isa UnaryOperation
        b = "(" * b * ")"
    end

    return b * ".^" * parenthesize(arg.exponent)
end

function to_string(arg::BinaryOperation{Mult})
    if arg.arg1 == -1
        return "-" * parenthesize(arg.arg2)
    end

    return parenthesize(arg.arg1) * parenthesize(arg.arg2)
end

function to_string(arg::BinaryOperation{Add})
    return to_string(arg.arg1) * " + " * to_string(arg.arg2)
end

function to_string(arg::BinaryOperation{Sub})
    return to_string(arg.arg1) * " - " * parenthesize(arg.arg2)
end

function Base.show(io::IO, expr::Tensor)
    return print(io, to_string(expr))
end
