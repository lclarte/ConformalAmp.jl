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

function fit_leave_one_out(problem::Problem, X::AbstractMatrix, y::AbstractVector)
    n, d = size(X)
    w = zeros(n, d)
    for i in 1:n
        X_i = X[1:n .!= i, :]
        y_i = y[1:n .!= i]
        w[i, :] = fit(problem, X_i, y_i)
    end
    return w
end