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

function to_std_str(arg::Real)
    out = string(arg)

    if arg < 0
        out = "(" * out * ")"
    end

    return out
end

function to_std_str(arg::Rational)
    out = string(arg)

    return "(" * out * ")"
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

    return out * "^" * to_std_str(arg.exponent)
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
