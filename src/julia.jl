# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function to_julia(arg::ir.Var)
    return Symbol(arg.id)
end

function to_julia(arg::ir.Literal)
    return :($(arg.value))
end

function to_julia(arg::ir.Mat)
    if arg.id isa ir.Var
        return to_julia(arg.id)
    end

    throw(
        ArgumentError("Unable to generate julia code for constant matrix of unknown size"),
    )
end

function to_julia(arg::ir.Vec)
    if arg.id isa ir.Var
        return to_julia(arg.id)
    end

    if arg.id isa ir.Literal
        return to_julia(arg.id)
    end

    throw(
        ArgumentError("Unable to generate julia code for constant vector of unknown size"),
    )
end

function to_julia(arg::ir.Scal)
    return to_julia(arg.id)
end

function to_julia(arg::ir.Real)
    return arg
end

function to_julia(arg::ir.Identity)
    return Symbol("I")
end

for (op, f) ∈
    ((:Sin, :sin), (:Cos, :cos), (:Log, :log), (:Exp, :exp), (:Abs, :abs), (:Sgn, :sign))
    @eval to_julia(arg::ir.$op) = Expr(:., $(QuoteNode(f)), Expr(:tuple, to_julia(arg.arg)))
end

function to_julia(arg::ir.Add)
    if get_size(arg.l) == Scalar::Size ||
       get_size(arg.l) == Inferred::Size ||
       get_size(arg.r) == Scalar::Size ||
       get_size(arg.r) == Inferred::Size
        return :($(to_julia(arg.l)) .+ $(to_julia(arg.r)))
    end

    return :($(to_julia(arg.l)) + $(to_julia(arg.r)))
end

function to_julia(arg::ir.Sub)
    if get_size(arg.l) == Scalar::Size ||
       get_size(arg.l) == Inferred::Size ||
       get_size(arg.r) == Scalar::Size ||
       get_size(arg.r) == Inferred::Size
        return :($(to_julia(arg.l)) .- $(to_julia(arg.r)))
    end

    return :($(to_julia(arg.l)) - $(to_julia(arg.r)))
end

function to_julia(arg::ir.Product)
    if get_size(arg) == NoSize::Size
        throw(
            DomainError(
                arg,
                "Unable to generate julia code; Couldn't infer size of one or more operands",
            ),
        )
    end

    if get_size(arg.l) == Array::Size &&
       get_size(arg.r) == Inferred::Size &&
       get_kind(arg.l) == Mat::Kind &&
       get_kind(arg.r) == ColVec::Kind
        return :($(to_julia(arg.r)) .* vec(sum($(to_julia(arg.l)), dims = 2)))
    end

    if get_size(arg.l) == Inferred::Size &&
       get_size(arg.r) == Array::Size &&
       get_kind(arg.l) == RowVec::Kind &&
       get_kind(arg.r) == Mat::Kind
        return :($(to_julia(arg.l)) .* (sum($(to_julia(arg.r)), dims = 1)))
    end

    return :($(to_julia(arg.l)) * $(to_julia(arg.r)))
end

function to_julia(arg::ir.HadamardProduct)
    return :($(to_julia(arg.l)) .* $(to_julia(arg.r)))
end

function to_julia(arg::ir.Power)
    return :($(to_julia(arg.base)) .^ $(to_julia(arg.exponent)))
end

for (op, f) ∈ (
    (:Trace, :tr),
    (:Diag, :diag),
    (:Diagm, :diagm),
    (:Transpose, :transpose),
    (:Sum, :sum),
)
    @eval to_julia(arg::ir.$op) = Expr(:call, $(QuoteNode(f)), to_julia(arg.arg))
end
