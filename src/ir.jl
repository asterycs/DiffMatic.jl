# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

module ir

abstract type IR end

struct Var <: IR
    id::String
end

struct Literal <: IR
    value::Real
end

struct Mat <: IR
    id::Union{Var,Literal}
end

struct Vec <: IR
    id::Union{Var,Literal}
end

struct Scal <: IR
    id::Union{Var,Literal}
end

struct Identity <: IR end

struct Abs <: IR
    arg::IR
end

struct Sgn <: IR
    arg::IR
end

struct Sin <: IR
    arg::IR
end

struct Cos <: IR
    arg::IR
end

struct Add <: IR
    l::IR
    r::IR
end

struct Sub <: IR
    l::IR
    r::IR
end

struct Product <: IR
    l::IR
    r::IR
end

struct Quotient <: IR
    num::IR
    den::IR
end

struct HadamardProduct <: IR
    l::IR
    r::IR
end

struct Log <: IR
    arg::IR
end

struct Power <: IR
    base::IR
    exponent::Union{Int,Rational{Int}}
end

struct Trace <: IR
    arg::IR
end

struct Diag <: IR
    arg::IR
end

struct Transpose <: IR
    arg::IR
end

struct Sum <: IR
    arg::IR
end

struct PartialSum <: IR
    arg::IR
    dim::Int
end

function _get_variables(arg::Mat)
    return _get_variables(arg.id)
end

function _get_variables(arg::Vec)
    return _get_variables(arg.id)
end

function _get_variables(arg::Scal)
    return _get_variables(arg.id)
end

function _get_variables(arg::Var)
    return arg.id
end

function _get_variables(arg::Literal)
    return _get_variables(arg.value)
end

function _get_variables(arg::Real)
    return nothing
end

function _get_variables(arg::Identity)
    return nothing
end

function _get_variables(arg::Sin)
    return _get_variables(arg.arg)
end

function _get_variables(arg::Cos)
    return _get_variables(arg.arg)
end

function _get_variables(arg::Log)
    return _get_variables(arg.arg)
end

function _get_variables(arg::Add)
    return [_get_variables(arg.l); _get_variables(arg.r)]
end

function _get_variables(arg::Sub)
    return [_get_variables(arg.l); _get_variables(arg.r)]
end

function _get_variables(arg::Product)
    return [_get_variables(arg.l); _get_variables(arg.r)]
end

function _get_variables(arg::HadamardProduct)
    return [_get_variables(arg.l); _get_variables(arg.r)]
end

function _get_variables(arg::Power)
    return [_get_variables(arg.base); _get_variables(arg.exponent)]
end

function _get_variables(arg::Trace)
    return _get_variables(arg.arg)
end

function _get_variables(arg::Diag)
    return _get_variables(arg.arg)
end

function _get_variables(arg::Transpose)
    return _get_variables(arg.arg)
end

function _get_variables(arg::Sum)
    return _get_variables(arg.arg)
end

function get_variables(arg::IR)
    s = [_get_variables(arg);]

    return filter(x -> !isnothing(x), unique(s))
end

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

function to_ir(arg::Variable)
    @assert is_standard_form(arg)

    ids = get_indices(arg)

    if length(ids) == 2
        if flip(ids[1]) == ids[2]
            return ir.Trace(ir.Mat(ir.Var(arg.id)))
        elseif typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return ir.Mat(ir.Var(arg.id))
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return ir.Transpose(ir.Mat(ir.Var(arg.id)))
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return ir.Vec(ir.Var(arg.id))
        elseif typeof(ids[1]) == Lower
            return ir.Transpose(ir.Vec(ir.Var(arg.id)))
        end
    end

    return ir.Scal(ir.Var(arg.id))
end

function to_ir(arg::Literal)
    @assert is_standard_form(arg)

    ids = get_indices(arg)

    if length(ids) == 2
        if flip(ids[1]) == ids[2]
            return ir.Trace(ir.Mat(ir.Literal(arg.value)))
        elseif typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
            return ir.Mat(ir.Literal(arg.value))
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return ir.Transpose(ir.Mat(ir.Literal(arg.value)))
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return ir.Vec(ir.Literal(arg.value))
        elseif typeof(ids[1]) == Lower
            return ir.Transpose(ir.Vec(ir.Literal(arg.value)))
        end
    end

    return ir.Scal(ir.Literal(arg.value))
end

function to_ir(arg::KrD)
    @assert is_standard_form(arg)

    ids = get_indices(arg)

    if flip(ids[1]) == ids[2]
        return ir.Trace(ir.Identity())
    elseif typeof(ids[1]) == Upper && typeof(ids[2]) == Lower
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
            return ir.Mat(ir.Literal(0))
        elseif typeof(ids[1]) == Lower && typeof(ids[2]) == Upper
            return ir.Transpose(ir.Mat(ir.Literal(0)))
        end
    elseif length(ids) == 1
        if typeof(ids[1]) == Upper
            return ir.Vec(ir.Literal(0))
        elseif typeof(ids[1]) == Lower
            return ir.Transpose(ir.Vec(ir.Literal(0)))
        end
    end
end

function to_ir(arg::Real)
    return ir.Scal(ir.Literal(arg))
end

function to_ir(arg::UnaryOperation{Abs})
    return ir.Abs(to_ir(arg.arg))
end

function to_ir(arg::UnaryOperation{Sgn})
    return ir.Sgn(to_ir(arg.arg))
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
                return ir.Vec(ir.Literal(1))
            else
                return ir.Transpose(ir.Vec(ir.Literal(1)))
            end
        elseif arg.arg1 isa Literal || arg.arg2 isa Literal
            tensor = if arg.arg1 isa Literal
                arg.arg2
            else
                arg.arg1
            end

            if typeof(target_indices[1]) == Upper
                return ir.PartialSum(to_ir(tensor), 2)
            else
                return ir.PartialSum(to_ir(tensor), 1)
            end
        end
    end

    if is_trace(arg)
        return ir.Trace(ir.Product(to_ir(arg.arg1), to_ir(arg.arg2)))
    end

    if isempty(target_indices) && (first(terms) isa Literal || last(terms) isa Literal)
        tensor = if first(terms) == Literal
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
        elseif arg.arg1 isa Variable
            return ir.Product(to_ir(arg.arg1), to_ir(arg.arg2))
        elseif arg.arg2 isa Variable
            return ir.Product(to_ir(arg.arg2), to_ir(arg.arg1))
        end
    end

    throw_not_std(arg)
end

function to_ir(arg::BinaryOperation{Div})
    return ir.Quotient(to_ir(arg.arg1), to_ir(arg.arg2))
end

function to_ir(arg::UnaryOperation{Log})
    return ir.Log(to_ir(arg.arg))
end

function to_ir(arg::Power)
    return ir.Power(to_ir(arg.base), arg.exponent)
end

function is_trace(arg)
    terms = collect_factors(arg)

    if length(terms) == 1
        return length(get_indices(first(terms))) == 2 && isempty(get_free_indices(arg))
    end

    return all(length.(get_free_indices.(terms)) .== 2) && isempty(get_free_indices(arg))
end
