# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

export @matrix
export @vector
export @scalar

export derivative
export gradient
export jacobian
export hessian

export to_std_string

function create_matrix(name::String)
    T = Monomial(name, Upper(1), Lower(2))

    return T
end

function create_vector(name::String)
    T = Monomial(name, Upper(1))

    return T
end

function create_scalar(name::String)
    T = Monomial(name)

    return T
end

"""
    @matrix(ids...)

Create one or more matrices. Example:
```jldoctest; output=false
@matrix A X

# output

X¹₂
```
"""
macro matrix(ids...)
    # From https://discourse.julialang.org/t/macro-question-create-variables/90160
    syms = (:($(esc(id)) = create_matrix($(string(id)))) for id ∈ ids)
    return Expr(:block, syms...)
end

"""
    @vector(ids...)

Create one or more vectors. Example:
```jldoctest; output=false
@vector x y

# output

y¹
```
"""
macro vector(ids...)
    syms = (:($(esc(id)) = create_vector($(string(id)))) for id ∈ ids)
    return Expr(:block, syms...)
end

"""
    @scalar(ids...)

Create one or more scalars. Example:
```jldoctest; output=false
@scalar a β

# output

β
```
"""
macro scalar(ids...)
    syms = (:($(esc(id)) = create_scalar($(string(id)))) for id ∈ ids)
    return Expr(:block, syms...)
end

"""
    derivative(expr, wrt::Tensor)

Compute the derivative of `expr` with respect to `wrt`. Example:
```jldoctest
@matrix A
@vector x

derivative(x' * x, x)

# output

2x₄
```
"""
function derivative(expr, wrt::Monomial)
    ∂ = Monomial(wrt.id)

    for index ∈ wrt.indices
        push!(∂.indices, same_to(index, get_next_letter(expr, ∂)))
    end

    D = diff(expr, ∂)

    return evaluate(evaluate(D))
end

"""
    gradient(expr, wrt::Tensor)

Compute the gradient of `expr` with respect to `wrt`. `expr` must be a scalar and `wrt` a vector. Example:
```jldoctest
@matrix A
@vector x

gradient(x' * A * x, x)

# output

x₃A³⁵ + A⁵₄x⁴
```
"""
function gradient(expr, wrt::Monomial)
    free_indices = get_free_indices(evaluate(expr))

    if !isempty(free_indices)
        throw(DomainError(evaluate(expr), "Input is not a scalar"))
    end

    if length(wrt.indices) != 1
        throw(DomainError(wrt, "\"$wrt\" is not a vector"))
    end

    D = derivative(expr, wrt)
    gradient = evaluate(D')

    return gradient
end

"""
    jacobian(expr, wrt::Tensor)

Compute the jacobian of `expr` with respect to `wrt`. `expr` must be a column vector and `wrt` a vector. Example:
```jldoctest
@matrix A
@vector x

jacobian(A * x, x)

# output

A³₅
```
"""
function jacobian(expr, wrt::Monomial)
    free_indices = get_free_indices(evaluate(expr))

    if length(free_indices) != 1 || typeof(free_indices[1]) != Upper
        throw(DomainError(evaluate(expr), "Input is not a column vector"))
    end

    if length(wrt.indices) != 1
        throw(DomainError(wrt, "\"$wrt\" is not a vector"))
    end

    D = derivative(expr, wrt)

    return D
end

"""
    hessian(expr, wrt::Tensor)

Compute the hessian of `expr` with respect to `wrt`. `expr` must be a scalar and `wrt` a vector. Example:
```jldoctest
@matrix A
@vector x

hessian(x' * A * x, x)

# output

A₆⁵ + A⁵₆
```
"""
function hessian(expr, wrt::Monomial)
    free_indices = get_free_indices(evaluate(expr))

    if !isempty(free_indices)
        throw(DomainError(evaluate(expr), "Input is not a scalar"))
    end

    if length(wrt.indices) != 1
        throw(DomainError(wrt, "\"$wrt\" is not a vector"))
    end

    D = derivative(expr, wrt)
    g = evaluate(D')
    H = derivative(g, wrt)

    return evaluate(H)
end

function throw_not_std()
    throw(DomainError("Cannot write expression in standard notation"))
end

# This type is for tagging contractions that need special treatment before converting to
# standard notation (if at all possible).
struct NonStdCon end

function get_indices(arg::BinaryOperation{NonStdCon})
    return [get_indices(arg.arg1); get_indices(arg.arg2)]
end

function to_string(arg::BinaryOperation{NonStdCon})
    return parenthesize(arg.arg1) * parenthesize(arg.arg2)
end

function evaluate(arg::BinaryOperation{NonStdCon})
    return arg
end

function _to_std_string(arg::Monomial)
    ids = get_indices(arg)

    if length(ids) == 2
        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return arg.id
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return arg.id * "ᵀ"
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return arg.id
        elseif typeof(ids[1]) == Lower
            return arg.id * "ᵀ"
        end
    elseif isempty(ids)
        return arg.id
    end

    throw_not_std()
end

function _to_std_string(arg::KrD)
    ids = get_indices(arg)

    if length(ids) == 2
        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return "I"
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return "Iᵀ"
        end
    end

    throw_not_std()
end

function _to_std_string(arg::Zero)
    ids = get_indices(arg)

    if length(ids) == 2
        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return "mat(0)"
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return "mat(0)ᵀ"
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return "vec(0)"
        elseif typeof(ids[1]) == Lower
            return "vec(0)ᵀ"
        end
    end

    throw_not_std()
end

function _to_std_string(arg::Real)
    return to_string(arg)
end

function _to_std_string(arg::UnaryOperation{Sin})
    return "sin(" * _to_std_string(arg.arg) * ")"
end

function _to_std_string(arg::UnaryOperation{Cos})
    return "cos(" * _to_std_string(arg.arg) * ")"
end

function _to_std_string(::Add)
    return "+"
end

function _to_std_string(::Sub)
    return "-"
end

function _to_std_string(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return _to_std_string(arg.arg1) *
           " " *
           _to_std_string(Op()) *
           " " *
           _to_std_string(arg.arg2)
end

function add_contracting_factor_ordered(factor, ordered_factors)
    for fixed ∈ (first(ordered_factors), last(ordered_factors))
        fixed_indices = get_free_indices(fixed)
        next_indices = get_free_indices(factor)

        ### Contractions
        ################
        if typeof(last(next_indices)) == Lower &&
           flip(last(next_indices)) == first(fixed_indices)
            pushfirst!(ordered_factors, factor)
            return true
        end

        if typeof(last(next_indices)) == Upper &&
           flip(last(next_indices)) == first(fixed_indices)
            push!(ordered_factors, factor)
            return true
        end

        if typeof(first(next_indices)) == Upper &&
           flip(first(next_indices)) == last(fixed_indices)
            push!(ordered_factors, factor)
            return true
        end

        if typeof(first(next_indices)) == Lower &&
           flip(first(next_indices)) == last(fixed_indices)
            pushfirst!(ordered_factors, factor)
            return true
        end

        if typeof(last(next_indices)) == Upper &&
           flip(last(next_indices)) == last(fixed_indices)
            push!(ordered_factors, factor)
            return true
        end

        if typeof(last(next_indices)) == Lower &&
           flip(last(next_indices)) == last(fixed_indices)
            pushfirst!(ordered_factors, factor)
            return true
        end

        if typeof(first(next_indices)) == Upper &&
           flip(first(next_indices)) == first(fixed_indices)
            push!(ordered_factors, factor)
            return true
        end

        if typeof(first(next_indices)) == Lower &&
           flip(first(next_indices)) == first(fixed_indices)
            pushfirst!(ordered_factors, factor)
            return true
        end
    end

    return false
end

function _to_std_string(arg::BinaryOperation{Mult})
    factors = group_non_std_factors(arg)
    factors = Any[f for f ∈ factors]

    ordered_factors = []
    reals = Real[]
    scalars = []

    while !all(isnothing.(factors))
        term_was_added = false

        for i ∈ eachindex(factors)
            if isnothing(factors[i])
                continue
            end

            factor = factors[i]
            free_ids = get_free_indices(factor)

            if factor isa Real
                push!(reals, factor)
                factors[i] = nothing
                term_was_added = true
            elseif isempty(free_ids)
                push!(scalars, factor)
                factors[i] = nothing
                term_was_added = true
            else
                if isempty(ordered_factors)
                    push!(ordered_factors, factor)
                    factors[i] = nothing
                    term_was_added = true
                    continue
                end

                if add_contracting_factor_ordered(factor, ordered_factors)
                    factors[i] = nothing
                    term_was_added = true
                    continue
                end
            end
        end

        if !term_was_added
            break
        end
    end

    for i ∈ eachindex(factors) # Remaining factors make outer products
        if !isnothing(factors[i])
            if typeof(last(get_free_indices(factors[i]))) == Upper
                pushfirst!(ordered_factors, factors[i])
                factors[i] = nothing
            else
                push!(ordered_factors, factors[i])
                factors[i] = nothing
            end
        end
    end

    @assert all(isnothing.(factors))

    prepend!(ordered_factors, scalars)
    if !isempty(reals)
        pushfirst!(ordered_factors, prod(reals))
    end

    out = ""

    for f ∈ ordered_factors
        out *= parenthesize_std(f)
    end

    return out
end

function _to_std_string(arg::BinaryOperation{Pow})
    return parenthesize_std(arg.arg1) * script(Upper(arg.arg2))
end

function _to_std_string(arg::BinaryOperation{NonStdCon})
    indices = get_indices(arg)
    target_indices = unique(eliminate_indices(indices))
    terms = collect_factors(arg)

    if length(target_indices) == 1
        if all(length.(get_indices.(terms)) .== 1)
            return reduce(
                (l, r) -> l * " ⊙ " * _to_std_string(r),
                terms[2:end];
                init = _to_std_string(terms[1]),
            )
        elseif all(typeof.(terms) .== KrD)
            if typeof(target_indices[1]) == Upper
                return "vec(1)"
            else
                return "vec(1)ᵀ"
            end
        else
            throw_not_std()
        end
    end

    if length(terms) == 2 && length(target_indices) == 2
        arg1_ids, arg2_ids = get_indices.(terms)

        if maximum(length.((arg1_ids, arg2_ids))) == 2 &&
           minimum(length.((arg1_ids, arg2_ids))) == 1
            matrix = terms[1]
            vector = terms[2]

            if length(arg2_ids) == 2
                matrix, vector = vector, matrix
            end

            m_ids = get_indices(matrix)
            v_ids = get_indices(vector)

            if m_ids[1] == v_ids[1]
                return "diag(" * _to_std_string(vector) * ")" * _to_std_string(matrix)
            elseif m_ids[2] == v_ids[1]
                return _to_std_string(matrix) * " diag(" * _to_std_string(vector) * ")"
            end
        end

        if length(arg1_ids) == length(arg2_ids)
            if all(arg1_ids .== arg2_ids)
                return _to_std_string(terms[1]) * " ⊙ " * _to_std_string(terms[2])
            else
                throw_not_std()
            end
        end

        throw_not_std()
    end

    if isempty(target_indices) && length(terms) == 2
        tensor = nothing

        if isempty(get_free_indices(terms[1])) && typeof(terms[1]) == KrD
            tensor = terms[2]
        elseif isempty(get_free_indices(terms[2])) && typeof(terms[2]) == KrD
            tensor = terms[1]
        else
            throw_not_std()
        end

        return "sum(" * _to_std_string(tensor) * ")"
    end

    throw_not_std()
end

function parenthesize_std(arg)
    return _to_std_string(arg)
end

function parenthesize_std(arg::Real)
    if arg < 0
        return "(" * _to_std_string(arg) * ")"
    end

    return _to_std_string(arg)
end

function parenthesize_std(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return "(" * _to_std_string(arg) * ")"
end

function parenthesize_std(arg::BinaryOperation{NonStdCon})
    return "(" * _to_std_string(arg) * ")"
end

function collect_factors(arg::BinaryOperation{NonStdCon})
    return [collect_factors(arg.arg1); collect_factors(arg.arg2)]
end

# TODO: Remove
function is_trace(arg)
    terms = collect_factors(arg)

    if length(terms) == 1
        ids = get_indices(arg)
        free_ids = get_free_indices(arg)

        if length(ids) == 2 && isempty(free_ids)
            return true
        else
            return false
        end
    end

    return all(length.(get_free_indices.(terms)) .== 2) && isempty(get_free_indices(arg))
end

function reshape(term::Monomial, indices::LowerOrUpperIndex...)
    return Monomial(term.id, indices...)
end

function reshape(term::Zero, indices::LowerOrUpperIndex...)
    return Zero(indices...)
end

function reshape(term::KrD, indices::LowerOrUpperIndex...)
    return KrD(indices...)
end

function reshape(term::UnaryOperation{Op}, indices::LowerOrUpperIndex...) where {Op}
    return UnaryOperation{Op}(reshape(term.arg, indices...))
end

function reshape(
    term::BinaryOperation{Op},
    indices::LowerOrUpperIndex...,
) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(
        reshape(term.arg1, indices...),
        reshape(term.arg2, indices...),
    )
end

function reshape(arg::BinaryOperation{Pow}, indices::LowerOrUpperIndex...)
    return BinaryOperation{Pow}(reshape(arg.arg1, indices...), arg.arg2)
end

function to_standard(
    term::UnaryOperation{Op};
    upper_letter = nothing,
    lower_letter = nothing,
) where {Op}
    return UnaryOperation{Op}(to_standard(term.arg; upper_letter, lower_letter))
end

function to_standard(
    term::BinaryOperation{Pow};
    upper_letter = nothing,
    lower_letter = nothing,
)
    return BinaryOperation{Pow}(to_standard(term.arg1), term.arg2)
end

function to_standard(term::Monomial; upper_letter = nothing, lower_letter = nothing)
    ids = term.indices

    if length(ids) == 2
        if typeof(last(term.indices)) == Lower
            return Monomial(term.id, Upper(ids[1].letter), Lower(ids[2].letter))
        else
            return Monomial(term.id, Lower(ids[1].letter), Upper(ids[2].letter))
        end
    elseif length(ids) == 1
        return term
    elseif isempty(ids)
        return Monomial(term.id)
    end

    throw_not_std()
end

function to_standard(term::Union{KrD,Zero}; upper_letter = nothing, lower_letter = nothing)
    ids = term.indices

    if length(ids) == 2
        if typeof(last(term.indices)) == Lower
            return typeof(term)(Upper(ids[1].letter), Lower(ids[2].letter))
        else
            return typeof(term)(Lower(ids[1].letter), Upper(ids[2].letter))
        end
    elseif length(ids) == 1
        return term
    elseif isempty(ids)
        return typeof(term)()
    end

    throw_not_std()
end

function to_standard(
    arg::BinaryOperation{Op};
    upper_letter = nothing,
    lower_letter = nothing,
) where {Op<:AdditiveOperation}
    target_indices = get_free_indices(arg)

    l = to_standard(arg.arg1)
    r = to_standard(arg.arg2)

    if isempty(setdiff(get_free_indices(l), get_free_indices(r))) &&
       isempty(setdiff(get_free_indices(l), target_indices))
        return BinaryOperation{Op}(l, r)
    elseif isempty(setdiff(get_free_indices(transpose(l)), get_free_indices(r))) &&
           isempty(setdiff(get_free_indices(transpose(l)), target_indices))
        return BinaryOperation{Op}(transpose(l), r)
    elseif isempty(setdiff(get_free_indices(l), get_free_indices(transpose(r)))) &&
           isempty(setdiff(get_free_indices(l), target_indices))
        return BinaryOperation{Op}(l, transpose(r))
    elseif isempty(
        setdiff(get_free_indices(transpose(l)), get_free_indices(transpose(r))),
    ) && isempty(setdiff(get_free_indices(transpose(l)), target_indices))
        return BinaryOperation{Op}(transpose(l), transpose(r))
    end

    throw_not_std()
end

function to_standard(arg::Real; upper_letter = nothing, lower_letter = nothing)
    return arg
end

function get_flipped(new_term, old_term)
    new_ids = get_free_indices(new_term)
    old_ids = get_free_indices(old_term)

    flipped = Dict()

    for (l, r) ∈ zip(new_ids, old_ids)
        @assert l.letter == r.letter

        if typeof(l) != typeof(r)
            flipped[r] = l
        end
    end

    return flipped
end

function was_flipped(index, flips)
    if flip(index) ∈ keys(flips)
        return true
    end

    return false
end

# TODO: Constrain to Mult and NonStdCon
function to_binary_operation(op::Op, terms::AbstractArray) where {Op}
    binop = nothing

    for t ∈ terms
        if isnothing(binop)
            binop = t
            continue
        end

        binop = BinaryOperation{Op}(binop, t)
    end

    return binop
end

function to_binary_operation(op::Op, term) where {Op}
    return term
end

function to_standard(
    arg::BinaryOperation{NonStdCon};
    upper_letter = nothing,
    lower_letter = nothing,
)
    ids = unique(get_indices(arg))
    terms = collect_factors(arg)

    reshaped = []

    # TODO: record reshaped indices and reshape all others

    if isempty(eliminate_indices(ids))
        return arg
    end

    for t ∈ terms
        ids = unique(get_indices(t))

        if length(ids) == 1
            if ids[1].letter == upper_letter || ids[1].letter == lower_letter
                push!(reshaped, t)
                continue
            else
                throw_not_std()
            end
        elseif length(ids) == 2
            if isnothing(upper_letter) && isnothing(lower_letter)
                return arg
            end

            if isnothing(upper_letter)
                if ids[2].letter == lower_letter
                    push!(reshaped, reshape(t, Upper(ids[1].letter), Lower(ids[2].letter)))
                    continue
                end

                if ids[1].letter == lower_letter
                    push!(reshaped, reshape(t, Lower(ids[1].letter), Upper(ids[2].letter)))
                    continue
                end
            end

            if isnothing(lower_letter)
                if ids[2].letter == upper_letter
                    push!(reshaped, reshape(t, Lower(ids[1].letter), Upper(ids[2].letter)))
                    continue
                end

                if ids[1].letter == upper_letter
                    push!(reshaped, reshape(t, Upper(ids[1].letter), Lower(ids[2].letter)))
                    continue
                end
            end

            if upper_letter == ids[1].letter && lower_letter == ids[2].letter
                push!(reshaped, reshape(t, Upper(ids[1].letter), Lower(ids[2].letter)))
                continue
            end

            if upper_letter == ids[2].letter && lower_letter == ids[1].letter
                push!(reshaped, reshape(t, Lower(ids[1].letter), Upper(ids[2].letter)))
                continue
            end

            # neither upper_letter nor lower_letter is in this term
            # happens e.g. for sums
            push!(reshaped, reshape(t, Upper(ids[1].letter), Lower(ids[2].letter)))
            continue
        else
            @assert false && "TODO"
        end
    end

    return to_binary_operation(NonStdCon(), reshaped)
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

            if length(ordered_factors) == 1
                ordered_factors = first(ordered_factors)
            end

            push!(chunked_factors, ordered_factors)
        else
            # TODO: Refactor
            @show factors[complex]
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

function group_non_std_factors(arg::BinaryOperation{Mult})
    factors = collect_factors(arg)
    factors = map(evaluate, factors) # recursion

    grouped_factors = group_factors(factors)

    for i ∈ eachindex(grouped_factors)
        if grouped_factors[i] isa AbstractArray
            grouped_factors[i] = to_binary_operation(NonStdCon(), grouped_factors[i])
        end
    end

    return grouped_factors
end


function transpose(arg::T) where {T<:UnaryOperation}
    return T(arg.arg')
end

function transpose(arg::BinaryOperation{Pow})
    return BinaryOperation{Pow}(transpose(arg.arg1), arg.arg2)
end

function transpose(arg::BinaryOperation{Mult})
    return BinaryOperation{Mult}(transpose(arg.arg1), transpose(arg.arg2))
end

function transpose(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    return BinaryOperation{Op}(transpose(arg.arg1), transpose(arg.arg2))
end

function transpose(arg::Monomial)
    indices = get_indices(arg)

    if length(indices) > 2
        throw(DomainError(arg.id, "Transpose is only defined for vectors and matrices"))
    end

    return Monomial(arg.id, flip.(indices)...)
end

function transpose(arg::Union{KrD,Zero})
    indices = get_indices(arg)

    if length(indices) > 2
        throw(DomainError(arg.id, "Transpose is only defined for vectors and matrices"))
    end

    return typeof(arg)(flip.(indices)...)
end

function to_standard(
    arg::BinaryOperation{Mult};
    upper_letter = nothing,
    lower_letter = nothing,
)
    target_indices = unique(get_free_indices(arg))
    target_len = length(target_indices)

    if length(target_indices) > 2
        throw_not_std()
    end

    l = to_standard(arg.arg1)
    r = to_standard(arg.arg2)

    term = nothing

    # TODO: Refactor
    if get_free_indices(BinaryOperation{Mult}(l, r)) == target_indices
        term = BinaryOperation{Mult}(l, r)
    elseif get_free_indices(BinaryOperation{Mult}(transpose(l), r)) == target_indices
        term = BinaryOperation{Mult}(transpose(l), r)
    elseif get_free_indices(BinaryOperation{Mult}(l, transpose(r))) == target_indices
        term = BinaryOperation{Mult}(l, transpose(r))
    elseif get_free_indices(BinaryOperation{Mult}(transpose(l), transpose(r))) ==
           target_indices
        term = BinaryOperation{Mult}(transpose(l), transpose(r))
    elseif length(get_free_indices(BinaryOperation{Mult}(l, r))) == target_len
        term = BinaryOperation{Mult}(l, r)
    elseif length(get_free_indices(BinaryOperation{Mult}(transpose(l), r))) == target_len
        term = BinaryOperation{Mult}(transpose(l), r)
    elseif length(get_free_indices(BinaryOperation{Mult}(l, transpose(r)))) == target_len
        term = BinaryOperation{Mult}(l, transpose(r))
    elseif length(get_free_indices(BinaryOperation{Mult}(transpose(l), transpose(r)))) ==
           target_len
        term = BinaryOperation{Mult}(transpose(l), transpose(r))
    else
        throw_not_std()
    end

    terms = group_non_std_factors(term)

    standardized_term = nothing

    for t ∈ terms
        if isnothing(standardized_term)
            standardized_term = t
            continue
        end

        standardized_term = BinaryOperation{Mult}(standardized_term, t)
    end

    ordered_expr_ids = get_free_indices(standardized_term)

    @assert length(unique(ordered_expr_ids)) == length(target_indices)

    return standardized_term
end

"""
    to_std_string(expr)

Convert the expression `expr` to standard matrix notation. `expr` must be a scalar and `wrt` a vector. Example:
```jldoctest
@matrix A
@vector x

to_std_string(gradient(x' * A * x, x))

# output

"Aᵀx + Ax"
```
"""
function to_std_string(arg)
    arg = simplify(arg)
    free_indices = unique(get_free_indices(arg))

    standardized = if length(free_indices) == 2
        if typeof(free_indices[1]) == Upper && typeof(free_indices[2]) == Lower
            to_standard(
                arg;
                upper_letter = free_indices[1].letter,
                lower_letter = free_indices[2].letter,
            )
        elseif typeof(free_indices[1]) == Lower && typeof(free_indices[2]) == Upper
            to_standard(
                arg;
                upper_letter = free_indices[2].letter,
                lower_letter = free_indices[1].letter,
            )
        else
            throw_not_std()
        end
    elseif length(free_indices) == 1
        if typeof(free_indices[1]) == Lower
            to_standard(arg; lower_letter = free_indices[1].letter)
        elseif typeof(free_indices[1]) == Upper
            to_standard(arg; upper_letter = free_indices[1].letter)
        else
            throw_not_std()
        end
    else
        to_standard(arg)
    end

    trace = is_trace(arg)

    argstr = _to_std_string(standardized)

    # TODO: Move inside _to_std_string
    if trace
        argstr = "tr(" * argstr * ")"
    end

    return argstr
end
