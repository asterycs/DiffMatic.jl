# DiffMatic.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://asterycs.github.io/DiffMatic.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://asterycs.github.io/DiffMatic.jl/dev)
[![Run tests](https://github.com/asterycs/DiffMatic.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/asterycs/DiffMatic.jl/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/asterycs/DiffMatic.jl/graph/badge.svg?token=XIVXM5EPAC)](https://codecov.io/gh/asterycs/DiffMatic.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Symbolic differentiation of vector/matrix/tensor expressions in Julia

### Example

Create a matrix and two vectors:

```julia
julia> using DiffMatic

julia> @matrix A;

julia> @vector x y;
```
Create an expression and differentiate it:
```julia
julia> expr = x' * sin.(A * x);

julia> g = gradient(expr, x);

julia> H = hessian(expr, x);
```
Convert the gradient and the Hessian to standard notation using `to_std`:
```julia
julia> to_std(g)

# output

"Aᵀ(cos(Ax) ⊙ x) + sin(Ax)"

julia> to_std(H)

# output

"diagm(cos(Ax))A + Aᵀdiagm(cos(Ax)) + (-1)Aᵀdiagm(x ⊙ sin(Ax))A"
```

Jacobians can be computed with `jacobian`:

```julia
julia> to_std(jacobian(sin.(A * x + y), x))

# output

"diagm(cos(Ax + y))A"
```

The function `derivative` can be used to compute arbitrary derivatives.

```julia
julia> to_std(derivative(tr(A), A))

# output

"I"
```
Runnable Julia code can also be generated directly:

```julia
julia> to_std(H; format = JuliaFunc())

# output

quote
    #= ... =#
    function (A, x)
        #= ... =#
        return diagm(cos.(A * x)) * A + (transpose(A) * diagm(cos.(A * x)) + -1 * (transpose(A) * (diagm(x .* sin.(A * x)) * A)))
    end
end
```

### Supported functions and operators

- Basic operators `+`, `-`, `'`, `*`, `^`, `abs`, `sign`, `sin`, `cos`, `log` and `exp`
- Element-wise operators `sin.`, `cos.`, `abs.`, `sign.`, `.*`, `.^`, `log.` and `exp.`
- Diagonal matrix using `LinearAlgebra.diagm`
- Vector of a matrix diagonal using `LinearAlgebra.diag`
- Vector 1-norm and 2-norm using `LinearAlgebra.norm(..., 1)` and `LinearAlgebra.norm(..., 2)`
- Sums of vectors using `sum`
- Matrix traces using `LinearAlgebra.tr`
- `LinearAlgebra.I` for the identity matrix
- Standard notation output: `tr`, `diag`, `diagm`, `sum`, `vec(1)`, `⊙` (element-wise
  product) and `⊘` (element-wise division)

### Installation

Installation from the general registry:

```julia
using Pkg; Pkg.add("DiffMatic")
```

### Acknowledgements

The implementation is based on the ideas presented in

> S. Laue, M. Mitterreiter, and J. Giesen.
> Computing Higher Order Derivatives of Matrix and Tensor Expressions, NeurIPS 2018.
