abstract type Method end

@kwdef struct GAMP <: Method
    max_iter::Integer
    rtol::Float64
end

@kwdef struct ERM <: Method
end

### 

function fit(problem::Ridge, X::AbstractMatrix, y::AbstractVector, ::ERM)
    (; Δ̂, λ) = problem
    w = (X' * X + Δ̂ * λ * I) \ (X' * y)
    return w
end

function fit(problem::Lasso, X::AbstractMatrix, y::AbstractVector, ::ERM)
    (; λ) = problem
    model = MLJLinearModels.LassoRegression(
        λ; fit_intercept=false, scale_penalty_with_samples=false
    )
    w = MLJLinearModels.fit(model, X, y)
    return w

end

function fit(problem::Logistic, X::AbstractMatrix, y::AbstractVector, ::ERM)
    (; λ) = problem
    model = MLJLinearModels.LogisticRegression(
        λ; fit_intercept=false, scale_penalty_with_samples=false
    )
    w = MLJLinearModels.fit(model, X, y)
    return w
end


function fit(problem::Pinball, X::AbstractMatrix, y::AbstractVector, ::ERM)
    """
    https://juliaai.github.io/MLJLinearModels.jl/dev/api/#MLJLinearModels.QuantileRegression
    """
    model = MLJLinearModels.QuantileRegression(
        problem.q, problem.λ; fit_intercept = false, scale_penalty_with_samples = false
    )
    return  MLJLinearModels.fit(model, X, y)
end

## 

function fit(problem::Problem, X::AbstractMatrix, y::AbstractVector, method::GAMP)
    (; max_iter, rtol) = method
    return gamp(problem, X, y; max_iter=max_iter, rtol=rtol).x̂
end

## 

function fit_leave_one_out(problem::Problem, X::AbstractMatrix, y::AbstractVector, ::ERM)
    n, d = size(X)
    w = zeros(n, d)
    for i in 1:n
        w[i, :] = fit(problem, X[1:n .!= i, :], y[1:n .!= i], ERM())
    end
    return w
end

function fit_leave_one_out(problem::Problem, X::AbstractMatrix, y::AbstractVector, method::GAMP)
    (; x̂, v̂, ω) = gamp(problem, X, y; max_iter=method.max_iter, rtol=method.rtol)
    return get_cavity_means_from_gamp(problem, X, y, x̂, v̂, ω; rtol = method.rtol)
end

## TODO : Regrouper dans les memes fonctions  

function predict(::Union{Ridge, Lasso}, ŵ::AbstractVector, X::AbstractMatrix)
    return X * ŵ
end

function predict(::Logistic, ŵ::AbstractVector, X::AbstractMatrix)
    return sign.(X * ŵ)
end


function predict(::Union{Ridge, Lasso}, ŵ::AbstractMatrix, X::AbstractMatrix)
    # ŵ can be a matrix to accomodate the cavities
    return X * ŵ'
end

function predict(::Logistic, ŵ::AbstractMatrix, X::AbstractMatrix)
    return sign.(X * ŵ')
end

function predict(::Union{Ridge, Lasso}, ŵ::AbstractMatrix, x::AbstractVector)
    # ŵ can be a matrix to accomodate the cavities
    return  ŵ * x
end

function predict(::Logistic, ŵ::AbstractMatrix, x::AbstractVector)
    return sign.(ŵ * x)
end

function predict(::Union{Ridge, Lasso}, ŵ::AbstractVector, x::AbstractVector)
    # ŵ can be a matrix to accomodate the cavities
    return  ŵ'x
end

function predict(::Logistic, ŵ::AbstractVector, x::AbstractVector)
    return sign(ŵ'x)
end

#