# DiffMatic.jl

DiffMatic is a package for computing derivatives of matrix expressions, also known as matrix calculus. Provided are methods for computing gradients, Jacobians, Hessians and general matrix derivatives.

## Installation
The package is currently unregistered and can only be installed from GitHub:
```julia
julia> using Pkg; Pkg.add("https://github.com/asterycs/DiffMatic.jl.git")
```

## Usage
Expressions are constructed from Julia syntax.

Creating a matrix and a vector:

```jldoctest intro; output = false
using DiffMatic

@matrix A
@vector x

# output

x¹
```
Creating an expression:
```jldoctest intro
expr = x' * A * x

# output
x₄A⁴₅x⁵
```
The variable `expr` now contains an internal representation of the expression `x' * A * x`.

Compute the gradient and the Hessian with respect to the vector `x`.
```jldoctest intro; output = false
g = gradient(expr, x)
H = hessian(expr, x)

# output
A₇⁶ + A⁶₇
```
Convert the gradient into standard notation using `to_std`:
```jldoctest intro
to_std(g)

# output

"Aᵀx + Ax"
```

Convert the the Hessian into standard notation:
```jldoctest intro
to_std(H)

# output

"Aᵀ + A"
```

Jacobians can be computed with `jacobian`:

```jldoctest intro
to_std(jacobian(A * x, x))

# output

"A"
```

The method `derivative` can be used to compute arbitrary derivatives.

```jldoctest intro
to_std(derivative(tr(A), A))

# output

"I"
```
The method `to_std` will throw an exception when given an expression that that cannot be converted to
standard notation:
```jldoctest intro
to_std(derivative(A, A))

# output

ERROR: DomainError
```
