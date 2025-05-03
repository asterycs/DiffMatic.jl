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

export to_std

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

x⁴A₄⁶ + A⁶⁵x₅
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

A¹₅
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

A₇⁶ + A⁶₇
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

function throw_not_std(arg::Tensor)
    throw(DomainError(arg, "Cannot write expression in standard notation"))
end

"""
Returns an intermediate representation of the input expression. The input must be in standard form.
"""
function to_ir end

function is_standard_form(arg::Tensor)
    free_ids = unique(get_free_indices(arg))

    if length(free_ids) > 2
        return false
    end

    if length(free_ids) == 2
        if first(free_ids) isa Upper && last(free_ids) isa Lower ||
           first(free_ids) isa Lower && last(free_ids) isa Upper
            return true
        end

        return false
    end

    return true
end

function to_ir(arg::Monomial)
    @assert is_standard_form(arg)

    ids = get_indices(arg)

    if length(ids) == 2
        if flip(ids[1]) == ids[2]
            return ir.Trace(ir.Mat(arg.id))
        elseif typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return ir.Mat(arg.id)
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return ir.Transpose(ir.Mat(arg.id))
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return ir.Vec(arg.id)
        elseif typeof(ids[1]) == Lower
            return ir.Transpose(ir.Vec(arg.id))
        end
    end

    return ir.Scal(arg.id)
end

function to_ir(arg::KrD)
    @assert is_standard_form(arg)

    ids = get_indices(arg)

    if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
        return ir.Identity()
    elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
        return ir.Transpose(ir.Identity())
    end
end

function to_ir(arg::Zero)
    @assert is_standard_form(arg)

    ids = get_indices(arg)

    if length(ids) == 2
        if typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return ir.Mat(0)
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return ir.Transpose(ir.Mat(0))
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return ir.Vec(0)
        elseif typeof(ids[1]) == Lower
            return ir.Transpose(ir.Vec(0))
        end
    end
end

function to_ir(arg::Real)
    return ir.Scal(arg)
end

function to_ir(arg::UnaryOperation{Sin})
    return ir.Sin(to_ir(arg.arg))
end

function to_ir(arg::UnaryOperation{Cos})
    return ir.Cos(to_ir(arg.arg))
end

function to_ir(arg::BinaryOperation{Add})
    return ir.Add(to_ir(arg.arg1), to_ir(arg.arg2))
end

function to_ir(arg::BinaryOperation{Sub})
    return ir.Sub(to_ir(arg.arg1), to_ir(arg.arg2))
end

function get_contra_covariant_matrix(arg1::Tensor, arg2::Tensor)
    arg1_ids, arg2_ids = get_free_indices.((arg1, arg2))
    arg1_letters = [i.letter for i ∈ get_free_indices(arg1)]
    arg2_letters = [i.letter for i ∈ get_free_indices(arg2)]
    common_letter = intersect(arg1_letters, arg2_letters)

    if length(common_letter) == 1
        arg1_filt = filter(i->i.letter == first(common_letter), arg1_ids)
        arg2_filt = filter(i->i.letter == first(common_letter), arg2_ids)

        if typeof(first(arg1_filt)) == Lower && typeof(first(arg2_filt)) == Upper
            return (arg2, arg1)
        elseif typeof(first(arg1_filt)) == Upper && typeof(first(arg2_filt)) == Lower
            return (arg1, arg2)
        end
    elseif length(common_letter) == 2 # is a trace
        return (arg1, arg2)
    end

    throw_not_std(to_binary_operation(Mult(), (arg1, arg2)))
end

function to_ir(arg::BinaryOperation{Mult})
    @assert is_standard_form(arg)

    indices = get_indices(arg)
    target_indices = unique(eliminate_indices(indices))
    terms = (arg.arg1, arg.arg2)
    arg1_free_ids, arg2_free_ids = get_free_indices.(terms)

    if is_elementwise_multiplication(arg.arg1, arg.arg2)
        if length(terms) == 2 && length(target_indices) == 2

            if maximum(length.((arg1_free_ids, arg2_free_ids))) == 2 &&
               minimum(length.((arg1_free_ids, arg2_free_ids))) == 1
                matrix = terms[1]
                vector = terms[2]

                if length(arg2_free_ids) == 2
                    matrix, vector = vector, matrix
                end

                m_ids = get_indices(matrix)
                v_ids = get_indices(vector)

                if m_ids[1] == v_ids[1]
                    return ir.Product(ir.Diag(to_ir(vector)), to_ir(matrix))
                elseif m_ids[2] == v_ids[1]
                    return ir.Product(to_ir(matrix), ir.Diag(to_ir(vector)))
                end
            end

            if length(arg1_free_ids) == length(arg2_free_ids) &&
               all(arg1_free_ids .== arg2_free_ids)
                return ir.HadamardProduct(to_ir(terms[1]), to_ir(terms[2]))
            end

            throw_not_std(arg)
        end
    end

    if length(target_indices) == 1
        if all(length.(unique(get_free_indices.(terms))) .== 1)
            return reduce(
                (l, r) -> ir.HadamardProduct(l, to_ir(r)),
                terms[2:end];
                init = to_ir(terms[1]),
            )
        elseif all(typeof.(terms) .== KrD)
            if typeof(target_indices[1]) == Upper
                return ir.Vec(1)
            else
                return ir.Transpose(ir.Vec(1))
            end
        elseif (arg.arg1 isa KrD && is_trace(arg.arg1)) ||
               (arg.arg2 isa KrD && is_trace(arg.arg2))
            tensor = if arg.arg1 isa KrD
                arg.arg2
            else
                arg.arg1
            end

            if typeof(target_indices[1]) == Upper
                return ir.Product(to_ir(tensor), ir.Vec(1))
            else
                return ir.Product(ir.Transpose(ir.Vec(1)), to_ir(tensor))
            end
        end
    end

    if is_trace(arg)
        return ir.Trace(ir.Product(to_ir(arg.arg1), to_ir(arg.arg2)))
    end

    if isempty(target_indices) && (typeof(terms[1]) == KrD || typeof(terms[2]) == KrD)
        tensor = if typeof(first(terms)) == KrD
            last(terms)
        else
            first(terms)
        end

        tensor_free_ids = get_free_indices(tensor)

        if length(tensor_free_ids) == 1
            return ir.Sum(to_ir(tensor))
        end

        throw_not_std(arg)
    end

    if length(arg1_free_ids) == 2 && length(arg2_free_ids) == 2
        contra, covariant = get_contra_covariant_matrix(arg.arg1, arg.arg2)

        return ir.Product(to_ir(covariant), to_ir(contra))
    end

    if (length(arg1_free_ids) == 2 && length(arg2_free_ids) == 1) ||
       (length(arg2_free_ids) == 2 && length(arg1_free_ids) == 1)
        mat = if length(arg1_free_ids) == 2
            arg.arg1
        else
            arg.arg2
        end
        vec = if length(arg1_free_ids) == 1
            arg.arg1
        else
            arg.arg2
        end
        mat_ids = if length(arg1_free_ids) == 2
            arg1_free_ids
        else
            arg2_free_ids
        end
        vec_ids = if length(arg1_free_ids) == 1
            arg1_free_ids
        else
            arg2_free_ids
        end

        if typeof(last(mat_ids)) == Lower && flip(last(mat_ids)) == first(vec_ids)
            return ir.Product(to_ir(mat), to_ir(vec))
        elseif typeof(first(mat_ids)) == Lower && flip(first(mat_ids)) == first(vec_ids)
            return ir.Product(to_ir(mat), to_ir(vec))
        elseif typeof(first(mat_ids)) == Upper && flip(first(mat_ids)) == first(vec_ids)
            return ir.Product(to_ir(vec), to_ir(mat))
        elseif typeof(last(mat_ids)) == Upper && flip(last(mat_ids)) == first(vec_ids)
            return ir.Product(to_ir(vec), to_ir(mat))
        end
    end

    if length(arg1_free_ids) == 1 && length(arg2_free_ids) == 1
        if isempty(get_free_indices(arg))
            if typeof(first(arg1_free_ids)) == Lower &&
               typeof(first(arg2_free_ids)) == Upper
                return ir.Product(to_ir(arg.arg1), to_ir(arg.arg2))
            elseif typeof(first(arg1_free_ids)) == Upper &&
                   typeof(first(arg2_free_ids)) == Lower
                return ir.Product(to_ir(arg.arg2), to_ir(arg.arg1))
            end
        else
            if typeof(first(arg1_free_ids)) == Lower &&
               typeof(first(arg2_free_ids)) == Upper
                return ir.Product(to_ir(arg.arg2), to_ir(arg.arg1))
            elseif typeof(first(arg1_free_ids)) == Upper &&
                   typeof(first(arg2_free_ids)) == Lower
                return ir.Product(to_ir(arg.arg1), to_ir(arg.arg2))
            end
        end
    end

    if (isempty(arg1_free_ids) && !isempty(arg2_free_ids)) ||
       (isempty(arg2_free_ids) && !isempty(arg1_free_ids))
        scalar = if isempty(arg1_free_ids)
            arg.arg1
        else
            arg.arg2
        end
        tensor = if !isempty(arg1_free_ids)
            arg.arg1
        else
            arg.arg2
        end

        return ir.Product(to_ir(scalar), to_ir(tensor))
    end

    if (isempty(arg1_free_ids) && isempty(arg2_free_ids))
        if arg.arg1 isa Real
            return ir.Product(to_ir(arg.arg1), to_ir(arg.arg2))
        elseif arg.arg2 isa Real
            return ir.Product(to_ir(arg.arg2), to_ir(arg.arg1))
        elseif arg.arg1 isa Monomial
            return ir.Product(to_ir(arg.arg1), to_ir(arg.arg2))
        elseif arg.arg2 isa Monomial
            return ir.Product(to_ir(arg.arg2), to_ir(arg.arg1))
        end
    end

    throw_not_std(arg)
end

function to_ir(arg::Power)
    return ir.Power(to_ir(arg.base), arg.exponent)
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

function parenthesize_std(arg::BinaryOperation{Mult})
    if is_elementwise_multiplication(arg.arg1, arg.arg2)
        return "(" * _to_std_string(arg) * ")"
    end

    return _to_std_string(arg)
end

function is_trace(arg)
    terms = collect_factors(arg)

    if length(terms) == 1
        return length(get_indices(first(terms))) == 2 && isempty(get_free_indices(arg))
    end

    return all(length.(get_free_indices.(terms)) .== 2) && isempty(get_free_indices(arg))
end

function to_standard(term::UnaryOperation{Op}) where {Op}
    return UnaryOperation{Op}(to_standard(term.arg))
end

function to_standard(term::Power)
    return Power(to_standard(term.base), term.exponent)
end

function to_standard(term::Monomial)
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

    throw_not_std(term)
end

function to_standard(term::Union{KrD,Zero})
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

    throw_not_std(term)
end

function to_standard(arg::BinaryOperation{Op}) where {Op<:AdditiveOperation}
    target_indices = get_free_indices(arg)

    l = to_standard(arg.arg1)
    r = to_standard(arg.arg2)

    attempts = ((l, r), (adjoint(l), r), (l, adjoint(r)), (adjoint(l), adjoint(r)))

    for (l, r) ∈ attempts
        l_free_indices = get_free_indices(l)
        if isempty(setdiff(l_free_indices, get_free_indices(r))) &&
           isempty(setdiff(l_free_indices, target_indices))
            return BinaryOperation{Op}(l, r)
        end
    end

    throw_not_std(arg)
end

function to_standard(arg::Real)
    return arg
end

function to_standard(arg::BinaryOperation{Mult})
    target_indices = unique(get_free_indices(arg))
    target_len = length(target_indices)
    is_scalar = isempty(target_indices)

    if length(target_indices) > 2
        throw_not_std(arg)
    end

    l = to_standard(arg.arg1)
    r = to_standard(arg.arg2)

    attempts = (
        BinaryOperation{Mult}(l, r),
        BinaryOperation{Mult}(adjoint(l), r),
        BinaryOperation{Mult}(l, adjoint(r)),
        BinaryOperation{Mult}(adjoint(l), adjoint(r)),
    )

    for attempt ∈ attempts
        if length(get_free_indices(attempt)) == target_len &&
           (is_scalar || last(get_free_indices(attempt)) == last(target_indices))
            return attempt
        end
    end

    throw_not_std(arg)
end

struct Ir end
struct StdStr end

function to_std_str(arg::ir.Mat)
    if arg.id isa String
        return arg.id
    end

    return "mat(" * to_std_str(arg.id) * ")"
end

function to_std_str(arg::ir.Vec)
    if arg.id isa String
        return arg.id
    end

    return "vec(" * to_std_str(arg.id) * ")"
end

function to_std_str(arg::ir.Scal)
    return to_std_str(arg.id)
end

function to_std_str(arg::String)
    return arg
end

function to_std_str(arg::ir.Real)
    out = string(arg)

    if arg < 0
        out = "(" * out * ")"
    end

    return out
end

function to_std_str(arg::ir.Identity)
    return "I"
end

function to_std_str(arg::ir.Sin)
    return "sin(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Cos)
    return "cos(" * to_std_str(arg.arg) * ")"
end

function parenthesize(f, arg::ir.Add)
    return "(" * f(arg) * ")"
end

function parenthesize(f, arg::ir.Sub)
    return "(" * f(arg) * ")"
end

function parenthesize(f, arg::ir.HadamardProduct)
    return "(" * f(arg.l) * " ⊙ " * f(arg.r) * ")"
end

function parenthesize(f, arg)
    return f(arg)
end

function to_std_str(arg::ir.Add)
    return parenthesize(to_std_str, arg.l) * " + " * parenthesize(to_std_str, arg.r)
end

function to_std_str(arg::ir.Sub)
    return parenthesize(to_std_str, arg.l) * " - " * parenthesize(to_std_str, arg.r)
end

function to_std_str(arg::ir.Product)
    return parenthesize(to_std_str, arg.l) * parenthesize(to_std_str, arg.r)
end

function to_std_str(arg::ir.HadamardProduct)
    return to_std_str(arg.l) * " ⊙ " * to_std_str(arg.r)
end

function to_std_str(arg::ir.Power)
    out = to_std_str(arg.base)

    if arg.base isa ir.Product ||
       arg.base isa ir.HadamardProduct ||
       arg.base isa ir.Add ||
       arg.base isa ir.Sub
        out = "(" * out * ")"
    end

    return out * script(Upper(arg.exponent))
end

function to_std_str(arg::ir.Trace)
    return "tr(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Diag)
    return "diag(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Transpose)
    return parenthesize(to_std_str, arg.arg) * "ᵀ"
end

function to_std_str(arg::ir.Sum)
    return "sum(" * to_std_str(arg.arg) * ")"
end

function standardize(arg)
    arg = simplify(arg)
    free_indices = unique(get_free_indices(arg))

    standardized = if length(free_indices) == 2
        if typeof(free_indices[1]) == Upper && typeof(free_indices[2]) == Lower
            to_standard(arg)
        elseif typeof(free_indices[1]) == Lower && typeof(free_indices[2]) == Upper
            to_standard(arg)
        else
            throw_not_std(arg)
        end
    elseif length(free_indices) == 1
        to_standard(arg)
    else
        to_standard(arg)
    end

    return standardized
end

"""
    to_std(expr)

Convert the expression `expr` to standard matrix notation. Example:
```jldoctest
@matrix A
@vector x

to_std(gradient(x' * A * x, x))

# output

"Aᵀx + Ax"
```
"""
function to_std(arg; format = StdStr())
    return _to_std(format, arg)
end

function _to_std(format::StdStr, arg)
    standardized = standardize(arg)

    return to_std_str(to_ir(standardized))
end

function _to_std(format::Ir, arg)
    standardized = standardize(arg)

    return to_ir(standardized)
end
