
function split_conformal(problem::Problem, X::AbstractMatrix, y::AbstractVector, coverage::Real)
    κ = 1. - coverage

    n, d = size(X)
    ntest = floor(Int, n * 0.2)
    ntrain = n - ntest
    # train test split
    
    Xtrain = X[1:ntrain, :]
    ytrain = y[1:ntrain]
    Xtest = X[ntrain+1:end, :]
    ytest = y[ntrain+1:end]

    ŵ = fit(problem, Xtrain, ytrain, ERM())
    ŷ = Xtest * ŵ
    residuals = abs.(ytest - ŷ)

    # compute the quantile of the residuals
    q = Statistics.quantile(residuals, ceil( (1.0 - κ) * (ntest + 1))/ntest)

    return ŵ, q
end