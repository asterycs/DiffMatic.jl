# DiffMatic.jl

Documentation for DiffMatic.jl

```@docs
@matrix(ids...)
@vector(ids...)
@scalar(ids...)
derivative(expr, wrt::DiffMatic.Monomial)
gradient(expr, wrt::DiffMatic.Monomial)
jacobian(expr, wrt::DiffMatic.Monomial)
hessian(expr, wrt::DiffMatic.Monomial)
to_std_string(expr)
```
