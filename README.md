# DiffMatic.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://asterycs.github.io/DiffMatic.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://asterycs.github.io/DiffMatic.jl/dev)
[![Run tests](https://github.com/asterycs/DiffMatic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/asterycs/DiffMatic.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/asterycs/DiffMatic.jl/graph/badge.svg?token=XIVXM5EPAC)](https://codecov.io/gh/asterycs/DiffMatic.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Symbolic differentiation of vector/matrix/tensor expressions in Julia

### Example

Create a matrix and a vector:

```julia
using DiffMatic

@matrix A
@vector x
```
Create an expression:
```julia
expr = x' * A * x
```
The variable `expr` now contains an internal representation of the expression `x' * A * x`.

Compute the gradient and the Hessian with respect to the vector `x`.
```julia
g = gradient(expr, x)
H = hessian(expr, x)
```
Convert the gradient and the Hessian to standard notation using `to_std`:
```julia
to_std(g) # "Aᵀx + Ax"
to_std(H) # "Aᵀ + A"
```

Jacobians can be computed with `jacobian`:

```julia
to_std(jacobian(A * x, x)) # "A"
```

The function `derivative` can be used to compute arbitrary derivatives.

```julia
to_std(derivative(tr(A), A)) # "I"
```
The function `to_std` will throw an exception when given an expression that that cannot be converted to
standard notation.

### Supported functions and operators

`+`, `-`, `'`, `*`, `^`, `abs`, `sin`, `cos`

Element-wise operations `sin.`, `cos.`, `abs.`, `.*` and `.^` are supported.  
Vector 1-norm and 2-norm can be computed with `LinearAlgebra.norm(..., 1)` and `LinearAlgebra.norm(..., 2)`.  
Sums of vectors can be computed with `sum`.  
Matrix traces can be computed with `LinearAlgebra.tr`.

### Installation

Installation from the general registry:

```julia
using Pkg; Pkg.add("DiffMatic")
```

### Acknowledgements

The implementation is based on the ideas presented in

> S. Laue, M. Mitterreiter, and J. Giesen.
> Computing Higher Order Derivatives of Matrix and Tensor Expressions, NeurIPS 2018.
