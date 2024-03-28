"""
    Script to generate the data
"""

using StableRNGs: AbstractRNG
using LogExpFunctions

function get_n(α::Real, d::Integer)
    return ceil(Int, α * d)
end

function sample_data(rng::AbstractRNG, problem::Problem, d::Integer)
    n = get_n(problem.α, d)
    return sample_data_any_n(rng, d, n)
end

function sample_data_any_n(rng::AbstractRNG, d::Integer, n::Integer)

    X = randn(rng, n, d) ./ sqrt(d)
    return X
end

function sample_weights(rng::AbstractRNG, problem::Problem, d::Integer)
    w = randn(rng, d) 
    return w .* sqrt(d) / norm(w, 2)
end

function sample_labels(rng::AbstractRNG, ::Logistic, X::AbstractMatrix, w::AbstractVector;)
    n = size(X, 1)
    y = 2. .* (rand(rng, n) .< logistic.(X * w)) .- 1.
    return y
end

function sample_labels(rng::AbstractRNG, problem::RegressionProblem, X::AbstractMatrix, w::AbstractVector;)
    n = size(X, 1)
    y = X * w .+ sqrt.(problem.Δ) .* randn(rng, n)
    return y
end

function sample_all(rng::AbstractRNG, problem::Problem, d::Integer)
    X = sample_data(rng, problem, d)
    w = sample_weights(rng, problem, d)
    y = sample_labels(rng, problem, X, w)
    return (; X, w, y)
end
