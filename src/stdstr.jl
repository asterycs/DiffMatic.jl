# Copyright 2025, Jimmy Envall and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

function parenthesize(::ir.HadamardProduct, f, arg::Union{ir.Add,ir.Sub})
    return "(" * f(arg) * ")"
end

function parenthesize(::ir.HadamardProduct, f, arg)
    return f(arg)
end

function parenthesize(::ir.Product, f, arg)
    return f(arg)
end

function parenthesize(::ir.Product, f, arg::Union{ir.HadamardProduct,ir.Add,ir.Sub})
    return "(" * f(arg) * ")"
end

function parenthesize(::ir.Add, f, arg)
    return f(arg)
end

function parenthesize(::ir.Sub, f, arg)
    return f(arg)
end

function parenthesize(::ir.Sub, f, arg::ir.Add)
    return "(" * f(arg) * ")"
end

function parenthesize(
    ::ir.Transpose,
    f,
    arg::Union{ir.Product,ir.HadamardProduct,ir.Power,ir.Add,ir.Sub},
)
    return "(" * f(arg) * ")"
end

function parenthesize(::ir.Transpose, f, arg)
    return f(arg)
end

function to_std_str(arg::ir.Mat)
    if arg.id isa ir.Var
        return to_std_str(arg.id)
    end

    return "mat(" * to_std_str(arg.id) * ")"
end

function to_std_str(arg::ir.Vec)
    if arg.id isa ir.Var
        return to_std_str(arg.id)
    end

    return "vec(" * to_std_str(arg.id) * ")"
end

function to_std_str(arg::ir.Scal)
    return to_std_str(arg.id)
end

function to_std_str(arg::ir.Var)
    return arg.id
end

function to_std_str(arg::ir.Literal)
    return to_std_str(arg.value)
end

function to_std_str(arg::Real)
    out = string(arg)

    if arg < 0
        out = "(" * out * ")"
    end

    return out
end

function to_std_str(arg::Rational)
    out = string(arg.num) * "/" * string(arg.den)

    return out
end

function to_std_str(arg::ir.Identity)
    return "I"
end

function to_std_str(arg::ir.Abs)
    return "abs(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Sgn)
    return "sgn(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Sin)
    return "sin(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Cos)
    return "cos(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Add)
    return parenthesize(arg, to_std_str, arg.l) *
           " + " *
           parenthesize(arg, to_std_str, arg.r)
end

function to_std_str(arg::ir.Sub)
    return to_std_str(arg.l) * " - " * parenthesize(arg, to_std_str, arg.r)
end

function to_std_str(arg::ir.Product)
    return parenthesize(arg, to_std_str, arg.l) * parenthesize(arg, to_std_str, arg.r)
end

function reciprocal(arg::ir.Power)
    return ir.Power(arg.base, -arg.exponent)
end

function to_std_str(arg::ir.HadamardProduct)
    t = arg
    if t isa ir.Transpose
        t = arg.arg
    end

    if t.l isa ir.Power && t.l.exponent < 0
        return parenthesize(arg, to_std_str, t.r) *
               " ⊘ " *
               parenthesize(arg, to_std_str, reciprocal(t.l))
    elseif arg.r isa ir.Power && arg.r.exponent < 0
        return parenthesize(arg, to_std_str, t.l) *
               " ⊘ " *
               parenthesize(arg, to_std_str, reciprocal(t.r))
    end

    return parenthesize(arg, to_std_str, arg.l) *
           " ⊙ " *
           parenthesize(arg, to_std_str, arg.r)
end

function to_std_str(arg::ir.Log)
    out = to_std_str(arg.arg)

    return "log(" * out * ")"
end

function superscript(value::Rational)
    absvalue = abs(value)

    num_c = script(Upper(absvalue.num))
    den_c = script(Upper(absvalue.den))

    if value < 0
        num_c = "⁻" * num_c
    end

    return join(num_c) * "⸍" * join(den_c)
end

function to_std_str(arg::ir.Power)
    base = to_std_str(arg.base)

    if arg.base isa ir.Product ||
       arg.base isa ir.HadamardProduct ||
       arg.base isa ir.Add ||
       arg.base isa ir.Sub
        base = "(" * base * ")"
    end

    if arg.exponent == -1
        if arg.base isa ir.Transpose
            return "vec(1)ᵀ ⊘ " * base
        else
            return "vec(1) ⊘ " * base
        end
    end

    if arg.exponent == 1
        return base
    end

    exponent = nothing

    if arg.exponent isa Rational
        exponent = superscript(arg.exponent)
    else
        exponent = script(Upper(arg.exponent))
    end

    return base * exponent
end

function to_std_str(arg::ir.Trace)
    return "tr(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Diag)
    return "diag(" * to_std_str(arg.arg) * ")"
end

function to_std_str(arg::ir.Transpose)
    return parenthesize(arg, to_std_str, arg.arg) * "ᵀ"
end

function to_std_str(arg::ir.Sum)
    return "sum(" * to_std_str(arg.arg) * ")"
end
