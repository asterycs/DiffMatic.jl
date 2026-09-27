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

function to_julia(arg::ir.Sin)
    return :(sin.($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Cos)
    return :(cos.($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Log)
    return :(log.($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Exp)
    return :(exp.($(to_julia(arg.arg))))
end

# Add and Sub operands render differently depending on the operands.
#
# - An operand of only literal values renders as a number that gets broadcast.
#
# - An Identity operand renders as a UniformScaling, which supports + and -
#   with arrays but no broadcasting.
#
# - Anything containing a variable renders as an array, which works with both.
#
# The operator is therefore chosen per node using 'is_uniform' and 'is_scaling'.
function is_uniform(arg::Union{ir.Literal,ir.Real})
    return true
end

function is_uniform(arg::Union{ir.Var,ir.Identity})
    return false
end

function is_uniform(arg::Union{ir.Mat,ir.Vec,ir.Scal})
    return is_uniform(arg.id)
end

function is_uniform(arg::Union{ir.Add,ir.Sub,ir.Product,ir.HadamardProduct})
    return is_uniform(arg.l) && is_uniform(arg.r)
end

function is_uniform(arg::ir.Power)
    return is_uniform(arg.base)
end

function is_uniform(arg::ir.IR)
    return is_uniform(arg.arg)
end

function is_scaling(arg::ir.Identity)
    return true
end

function is_scaling(arg::Union{ir.Literal,ir.Real,ir.Var})
    return false
end

function is_scaling(arg::Union{ir.Mat,ir.Vec,ir.Scal})
    return is_scaling(arg.id)
end

function is_scaling(arg::Union{ir.Add,ir.Sub,ir.HadamardProduct})
    return is_scaling(arg.l) || is_scaling(arg.r)
end

function is_scaling(arg::ir.Product)
    if is_scaling(arg.l)
        return is_scaling(arg.r) || is_uniform(arg.r)
    elseif is_scaling(arg.r)
        return is_uniform(arg.l)
    end

    return false
end

function is_scaling(arg::ir.Power)
    return is_scaling(arg.base)
end

function is_scaling(arg::ir.IR)
    return is_scaling(arg.arg)
end

function to_julia(arg::ir.Add)
    if (is_uniform(arg.l) || is_uniform(arg.r)) && !(is_scaling(arg.l) || is_scaling(arg.r))
        return :($(to_julia(arg.l)) .+ $(to_julia(arg.r)))
    end

    return :($(to_julia(arg.l)) + $(to_julia(arg.r)))
end

function to_julia(arg::ir.Sub)
    if (is_uniform(arg.l) || is_uniform(arg.r)) && !(is_scaling(arg.l) || is_scaling(arg.r))
        return :($(to_julia(arg.l)) .- $(to_julia(arg.r)))
    end

    return :($(to_julia(arg.l)) - $(to_julia(arg.r)))
end

function to_julia(arg::ir.Product)
    if arg.l isa ir.Transpose && arg.l.arg isa ir.Vec && arg.l.arg.id isa ir.Literal
        return :($(to_julia(arg.l.arg.id)) .* (sum($(to_julia(arg.r)), dims = 1)))
    elseif arg.r isa ir.Vec && arg.r.id isa ir.Literal
        return :($(to_julia(arg.r.id)) .* vec(sum($(to_julia(arg.l)), dims = 2)))
    end

    return :($(to_julia(arg.l)) * $(to_julia(arg.r)))
end

function to_julia(arg::ir.HadamardProduct)
    return :($(to_julia(arg.l)) .* $(to_julia(arg.r)))
end

function to_julia(arg::ir.Power)
    return :($(to_julia(arg.base)) .^ $(to_julia(arg.exponent)))
end

function to_julia(arg::ir.Trace)
    return :(tr($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Diag)
    return :(diag($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Diagm)
    return :(diagm($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Transpose)
    return :(transpose($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Sum)
    return :(sum($(to_julia(arg.arg))))
end
