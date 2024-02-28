function fit(problem::Ridge, X::AbstractMatrix, y::AbstractVector)
    (; Δ̂, λ) = problem
    w = (X' * X + Δ̂ * λ * I) \ (X' * y)
    return w
end

function fit(problem::Logistic, X::AbstractMatrix, y::AbstractVector)
    (; λ) = problem
    model = MLJLinearModels.LogisticRegression(
        λ; fit_intercept=false, scale_penalty_with_samples=false
    )
    w = MLJLinearModels.fit(model, X, y)
    return w
end

# 

function predict(::Ridge, ŵ::AbstractVector, X::AbstractMatrix)
    return X * ŵ
end

function predict(::Logistic, ŵ::AbstractVector, X::AbstractMatrix)
    return sign.(X * ŵ)
end


function predict(::Ridge, ŵ::AbstractMatrix, X::AbstractMatrix)
    # ŵ can be a matrix to accomodate the cavities
    return X * ŵ'
end

function predict(::Logistic, ŵ::AbstractMatrix, X::AbstractMatrix)
    return sign.(X * ŵ')
end

#

function fit_leave_one_out(problem::Problem, X::AbstractMatrix, y::AbstractVector)
    n, d = size(X)
    w = zeros(n, d)
    for i in 1:n
        w[i, :] = fit(problem, X[1:n .!= i, :], y[1:n .!= i])
    end
    return w
end