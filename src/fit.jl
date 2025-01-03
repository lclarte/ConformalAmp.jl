abstract type Method end

@kwdef struct GAMP <: Method
    max_iter::Integer
    rtol::Float64
end

@kwdef struct GAMPTaylor <: Method
    # perturbation of last sample used to compute 1st derivative
    # different from thep δy such that ỹ[n] = y[n] + δy  
    max_iter::Integer
    rtol::Float64
end

@kwdef struct ERM <: Method
end

@kwdef struct VAMP <: Method 
    max_iter::Integer
    rtol::Float64
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
    with an intercept, the intercept is the LAST ELEMENT OF THE VECTOR RETURNED
    """
    n, d = size(X)
    model = MLJLinearModels.QuantileRegression(
        ; delta = problem.q, lambda = problem.λ / n, gamma = 0.0, fit_intercept = problem.use_bias, scale_penalty_with_samples = false, penalize_intercept = false
    )
    return MLJLinearModels.fit(model, X, y)
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

function fit_leave_one_out(problem::Problem, X::AbstractMatrix, y::AbstractVector, method::Union{GAMP, GAMPTaylor})
    result = gamp(problem, X, y; max_iter=method.max_iter, rtol=method.rtol)
    return get_cavity_means_from_gamp(X, result)
end

function predict(::RegressionProblem, ŵ::AbstractVector, X::AbstractMatrix; bias::Real = 0.0)
    # add the same bias since we have only one predictor
    return X * ŵ .+ bias
end

function predict(::Logistic, ŵ::AbstractVector, X::AbstractMatrix)
    return sign.(X * ŵ)
end

# attention au type de bias qui DOIT inclure "Nothing" pour ne pas buguer 
function predict(::RegressionProblem, ŵ::AbstractMatrix, X::AbstractMatrix; bias::Union{AbstractVector, Nothing} = nothing)
    """
    If we have a k x d matrix ŵ, we return a n x k matrix
    """
    # ŵ can be a matrix to accomodate the cavities
    if isnothing(bias)
        return X * ŵ'
    else
        @assert length(bias) == size(ŵ, 1)
        # stach the bias horzizontally n times
        # when bias is a k-dimensional vector
        return X * ŵ' + vcat([bias' for i in 1:size(X, 1)])
    end
end

function predict(::Logistic, ŵ::AbstractMatrix, X::AbstractMatrix)
    return sign.(X * ŵ')
end

function predict(::RegressionProblem, ŵ::AbstractMatrix, x::AbstractVector; bias::Real = 0.0)
    # ŵ can be a matrix to accomodate the cavities
    return  ŵ * x .+ bias
end

function predict(::Logistic, ŵ::AbstractMatrix, x::AbstractVector)
    return sign.(ŵ * x)
end

# function predict(::Union{Ridge, Lasso}, ŵ::AbstractVector, x::AbstractVector)
function predict(::RegressionProblem, ŵ::AbstractVector, x::AbstractVector)
    # ŵ can be a matrix to accomodate the cavities
    return  ŵ'x
end

function predict(::Logistic, ŵ::AbstractVector, x::AbstractVector)
    return sign(ŵ'x)
end

### for VAMP

function fit(problem::Union{Ridge, Lasso}, X::AbstractMatrix, y::AbstractVector, method::VAMP)
    x̂_1, α_1, γ_1, x̂_2, α_2, γ_2 = vamp(problem, X, y; n_iterations = method.max_iter, rtol = method.rtol)
    return x̂_1
end

function fit_leave_one_out(problem::Ridge, X::AbstractMatrix, y::AbstractVector, method::VAMP)
    (; Δ̂, λ) = problem
    x̂_1, α_1, γ_1, x̂_2, α_2, γ_2  = vamp(problem, X, y; n_iterations = method.max_iter, rtol = method.rtol)
    # return loo_vamp(X, y, Δ̂, x̂_1, α_1, γ_1, x̂_2, α_2, γ_2)
    return iterate_loo_vamp_ridge(X, y, problem.Δ̂, x̂_1, α_1, γ_1, x̂_2, α_2, γ_2)
end