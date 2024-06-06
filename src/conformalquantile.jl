"""
# we won't use ScikitLearn for now 
using ScikitLearn
@sk_import linear_model: QuantileRegressor

using ScikitLearn.CrossValidation: train_test_split
using ScikitLearn: fit!, predict
"""

"""
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split

def get_estimators_qcp(X, y, coverage, lambda_, train_val_split=0.8):
    alpha = 1.0 - coverage
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_val_split, random_state=0)

    n, d  = X_train.shape


    qr_upper = QuantileRegressor(quantile = 1.0 - alpha / 2.0, alpha = lambda_ / n, solver="highs")
    qr_lower = QuantileRegressor(quantile = alpha / 2.0, alpha = lambda_ / n, solver="highs")

    qr_upper.fit(X_train, y_train)
    qr_lower.fit(X_train, y_train)

    # compute the residuals 
    residuals_upper = y_val - qr_upper.predict(X_val)
    residuals_lower = qr_lower.predict(X_val) - y_val

    scores_val = [max( a, b ) for a,b in zip(residuals_upper, residuals_lower)]
    # get the coverage * (1.0 + 1 / n_val ) quantile of the scores
    threshold = np.quantile(scores_val, coverage * (1.0 + 1.0 / len(y_val)))

    return qr_upper, qr_lower, threshold

def get_intervals_qcp(X_train, y_train, X_test, coverage, lambda_):
    qr_upper, qr_lower, threshold = get_estimators_qcp(X_train, y_train, coverage, lambda_)
    upper = qr_upper.predict(X_test)
    lower = qr_lower.predict(X_test)

    return lower - threshold, upper + threshold
"""

using StableRNGs: AbstractRNG
using Random: shuffle
using Statistics

# convert the code above in Julia

function train_test_split(X, y, at, rng::AbstractRNG)
    n = size(X, 1)
    idx = shuffle(rng, 1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    X[train_idx,:], X[test_idx,:], y[train_idx], y[test_idx]
end

function get_estimators_qcp(X::AbstractMatrix, y::AbstractVector, coverage::Real, λ::Real, train_val_split::Real, rng::AbstractRNG)
    κ = 1.0 - coverage
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_val_split, rng)
    n, d = size(X_train)
    α = d / n

    problem_upper = ConformalAmp.Pinball(α = α, Δ = 1.0, q = 1.0 - κ / 2.0, λ = λ, use_bias = true)
    problem_lower = ConformalAmp.Pinball(α = α, Δ = 1.0, q = κ / 2.0, λ = λ, use_bias = true)

    model_upper = ConformalAmp.fit(
        problem_upper, X_train, y_train, ERM()
    )

    model_lower = ConformalAmp.fit(
        problem_lower, X_train, y_train, ERM()
    )

    residuals_upper = y_val - ConformalAmp.predict(problem_upper, model_upper[1:end - 1], X_val, bias = model_upper[end])
    residuals_lower = ConformalAmp.predict(problem_lower, model_lower[1:end - 1], X_val, bias = model_lower[end]) - y_val 

    scores_val = [max(a, b) for (a, b) in zip(residuals_upper, residuals_lower)]
    threshold = Statistics.quantile(scores_val, coverage * (1.0 + 1.0 / length(y_val)))
    
    return model_upper, model_lower, threshold
end




function get_intervals_qcp(X_train::AbstractMatrix, y_train::AbstractVector, X_test::AbstractMatrix, coverage::Real, λ::Real, rng::AbstractRNG)
    model_upper, model_lower, threshold = get_estimators_qcp(X_train, y_train, coverage, λ, 0.8, rng)

    n, d = size(X_train)
    problem = Ridge(α = d / n, Δ = 1.0, Δ̂ = 1.0, λ = λ)
    lower = ConformalAmp.predict(problem, model_lower[1:end - 1],  X_test, bias = model_lower[end]) .- threshold
    upper = ConformalAmp.predict(problem, model_upper[1:end - 1], X_test, bias = model_upper[end]) .+ threshold
    return lower, upper
end