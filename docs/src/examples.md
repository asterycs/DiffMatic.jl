# Examples
## Creating Variables
```jldoctest usage; output = false
using LinearAlgebra: diag, diagm, I, norm, tr
using DiffMatic

@matrix A B C
@vector x y z
@scalar c

# output

c
```
## Creating Expressions
### Matrix Multiplication and Transpose
```jldoctest usage
expr = x' * A * x

# output

x₄A⁴₅x⁵
```
```jldoctest usage
expr = x' * A' * x

# output

x₄A₅⁴x⁵
```
```jldoctest usage
expr = A * B * x

# output

A¹₅B⁵₇x⁷
```
#### Hadamard/Element-wise Product
```jldoctest usage
expr = A .* B .* C

# output

A¹₂B¹₂C¹₂
```
```jldoctest usage
expr = x .* y .* z

# output

x¹y¹z¹
```
#### Powers and Element-wise Powers
```jldoctest usage
expr = (x' * A * x)^2

# output

(x₄A⁴₅x⁵).^2
```
```jldoctest usage
expr = (x .* y).^2

# output

(x²y²).^2
```
```jldoctest usage
expr = (A * x).^2

# output

(A¹₄x⁴).^2
```

#### Trigonometric Functions
```jldoctest usage
expr = sin.(A * x)

# output

sin(A¹₄x⁴)
```
```jldoctest usage
expr = cos(x' * x)

# output

cos(x₃x³)
```
#### Absolute Value
```jldoctest usage
expr = abs.(A * x)

# output

|A¹₄x⁴|
```
```jldoctest usage
expr = abs(x' * x)

# output

|x₃x³|
```
#### Sum of a vector
```jldoctest usage
expr = sum(x .* y)

# output

x²y²1₂
```
```jldoctest usage
expr = sum(A * x)

# output

A¹₄x⁴1₁
```
#### Log and element-wise log
```jldoctest usage
expr = log(x' * y)

# output

log(x₃y³)
```
```jldoctest usage
expr = log.(x)' * y

# output

log(x₃)y³
```
#### Vector Norms
```jldoctest usage
expr = norm(A * x, 2)

# output

((A¹₄x⁴).^21₁).^(1/2)
```
```jldoctest usage
expr = norm(A * x, 1)

# output

|A¹₄x⁴|1₁
```
#### Matrix Trace
```jldoctest usage
expr = tr(A)

# output

A²₂
```
```jldoctest usage
expr = tr(A*B*B'*C)

# output

A¹₅B⁵₇B₉⁷C⁹₁
```

## Derivatives in Standard Notation
### Gradient
```jldoctest usage
to_std(gradient(tr(x * x'), x))

# output

"2x"
```

```jldoctest usage
to_std(gradient(tr(x * x'), x))

# output

"2x"
```
```jldoctest usage
to_std(gradient((x .* y)' * x, x))

# output

"2(x ⊙ y)"
```
```jldoctest usage
to_std(gradient((x - y)' * x, x))

# output

"2x - y"
```
```jldoctest usage
to_std(gradient(sin(tr(x * x')), x))

# output

"cos(xᵀx)2x"
```
```jldoctest usage
to_std(gradient(abs(x' * x), x))

# output

"sgn(xᵀx)2x"
```
```jldoctest usage
to_std(gradient(log.(x)'*x, x))

# output

"vec(1) + log(x)"
```
```jldoctest usage
to_std(gradient(2 * sum(cos.(A * x + y)), x))

# output

"(-2)Aᵀsin(Ax + y)"
```
```jldoctest usage
to_std(gradient((x' * A * x) ^ (-2), x))

# output

"(-2)(xᵀAᵀx)⁻³(Aᵀx + Ax)"
```
```jldoctest usage
to_std(gradient(((A .* (B .* C)) * C * x)' * x, x))

# output

"(B ⊙ C ⊙ A)Cx + Cᵀ(Bᵀ ⊙ Cᵀ ⊙ Aᵀ)x"
```
```jldoctest usage
to_std(gradient(sum((A .* B) * C * x), x))

# output

"Cᵀ(Aᵀ ⊙ Bᵀ)vec(1)"
```
### Jacobian
```jldoctest usage
to_std(jacobian(A * x, x))

# output

"A"
```
```jldoctest usage
to_std(jacobian(A' * x, x))

# output

"Aᵀ"
```
```jldoctest usage
to_std(jacobian(sin.(A * x + y), x))

# output

"diagm(cos(Ax + y))A"
```
```jldoctest usage
to_std(jacobian(((A .* B) * C * x)' * x * x, x))

# output

"xᵀCᵀ(Aᵀ ⊙ Bᵀ)xI + x(xᵀCᵀ(Aᵀ ⊙ Bᵀ) + xᵀ(A ⊙ B)C)"
```
```jldoctest usage
to_std(jacobian(diag(diagm(x' * B' * A * A)), x))

# output

"diagm(vec(1))AᵀAᵀB"
```

### Derivative
```jldoctest usage
to_std(derivative(c * I, c))

# output

"I"
```
