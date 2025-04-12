# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import LinearAlgebra.tr

export tr

struct Tensor{O}
    id::String

    function Tensor{Order}(id::String) where {Order}
        new{Order}(id)
    end
end

struct Adjoint{Ord}
    arg::Any
end

function Base.adjoint(arg::Tensor{Ord}) where {Ord}
    return Adjoint{Ord}(arg)
end

Letter = Int64
IndexSet = Vector{Letter}

struct Prod
    arg1::Any
    arg2::Any
    s1::IndexSet
    s2::IndexSet
    s3::IndexSet
end

struct Add
    arg1::Any
    arg2::Any
end

function Base.:(*)(arg1::Tensor{1}, arg2::Tensor{1})
    return Prod(arg1, arg2, [1], [1], [1])
end

function Base.:(*)(arg1::Tensor{2}, arg2::Tensor{1})
    return Prod(arg1, arg2, [1, 2], [2], [1])
end

function Base.:(*)(arg1::Adjoint{2}, arg2::Tensor{1})
    return Prod(arg1.arg, arg2, [1, 2], [1], [2])
end

function Base.:(*)(arg1::Adjoint{1}, arg2::Tensor{1})
    return Prod(arg1.arg, arg2, [1], [1], [])
end

function Base.:(*)(arg1::Adjoint{2}, arg2::Tensor{2})
    return Prod(arg1.arg, arg2, [1, 2], [1, 3], [2, 3])
end

function Base.:(*)(arg1::Adjoint{1}, arg2::Tensor{2})
    return Prod(arg1.arg, arg2, [1], [1, 2], [2])
end

function Base.:(*)(arg1::Tensor{2}, arg2::Adjoint{2})
    return Prod(arg1, arg2.arg, [1, 2], [3, 1], [2, 3])
end

function Base.:(*)(arg1::Tensor{2}, arg2::Adjoint{1})
    return Prod(arg1, arg2.arg, [1, 2], [1], [2])
end

function Base.:(*)(arg1::Adjoint{2}, arg2::Adjoint{2})
    return Prod(arg1.arg, arg2.arg, [1, 2], [3, 1], [2, 3])
end

function Base.:(*)(arg1::Adjoint{1}, arg2::Adjoint{2})
    return Prod(arg1.arg, arg2.arg, [1], [2, 1], [2])
end

struct KrD end
struct Zero{Ord} end

function diff(arg::Tensor{Ord}, wrt::Tensor{OrdD}) where {Ord,OrdD}
    op = nothing

    if arg.id == wrt.id
        for i ∈ 1:Ord
            if isnothing(op)
                op = KrD()
            else
                op = Prod(op, KrD(), [1, 2], [3, 4], [1, 2, 3, 4])
            end
        end
    else
        # TODO: Order of wrt somehow
        op = Zero{Ord + OrdD}()
    end

    return op
end

function diff(arg::Prod, wrt)
    wrt_index = maximum(vcat(arg.s1, arg.s2, arg.s3)) + 1
    l = Prod(
        arg.arg2,
        diff(arg.arg1, wrt),
        arg.s2,
        union(arg.s1, wrt_index),
        union(arg.s3, wrt_index),
    )
    r = Prod(
        arg.arg1,
        diff(arg.arg2, wrt),
        arg.s1,
        union(arg.s2, wrt_index),
        union(arg.s3, wrt_index),
    )

    return Add(l, r)
end

function script(index::Lower)
    @assert index.letter >= 0
    text = []

    for d ∈ reverse(digits(index.letter))
        push!(text, Char(0x2080 + d))
    end

    return join(text)
end

function to_string(arg::Tensor)
    return arg.id
end

function to_string(arg::KrD)
    return "δ"
end

function to_string(arg::Zero)
    return "0"
end

function parenthesize(arg)
    return to_string(arg)
end

function to_string(arg::Prod)
    index_str = ""

    for ids ∈ (arg.s1, arg.s2, arg.s3)
        if !isempty(index_str)
            index_str *= ","
        end

        for i ∈ ids
            index_str *= script(Lower(i))
        end
    end

    return parenthesize(arg.arg1) * "₍" * index_str * "₎" * parenthesize(arg.arg2)
end

function Base.show(io::IO, expr::Prod)
    return print(io, to_string(expr))
end
