module ir

abstract type IR end

struct Mat <: IR
    id::Union{String,Real}
end

struct Vec <: IR
    id::Union{String,Real}
end

struct Scal <: IR
    id::Union{String,Real}
end

struct Identity <: IR end

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

struct HadamardProduct <: IR
    l::IR
    r::IR
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

end
