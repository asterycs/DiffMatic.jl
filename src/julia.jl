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

    throw(RuntimeError("Unable to generate julia code for constant matrix of unknown size"))
end

function to_julia(arg::ir.Vec)
    if arg.id isa ir.Var
        return to_julia(arg.id)
    end

    throw(RuntimeError("Unable to generate julia code for constant vector of unknown size"))
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

function to_julia(arg::ir.Add)
    return :($(to_julia(arg.l)) + $(to_julia(arg.r)))
end

function to_julia(arg::ir.Sub)
    return :($(to_julia(arg.l)) - $(to_julia(arg.r)))
end

function to_julia(arg::ir.Product)
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
    return :(diagm($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Transpose)
    return :(transpose($(to_julia(arg.arg))))
end

function to_julia(arg::ir.Sum)
    return :(sum($(to_julia(arg.arg))))
end
