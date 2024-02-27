"""
    Script to generate the data
"""

using StableRNGs: AbstractRNG
using LogExpFunctions

function sample_data(rng::AbstractRNG, problem::Problem, n::Integer)
    d = ceil(Int, n / problem.α)
    X = randn(rng, n, d) ./ sqrt(d)
    return X
end

function sample_weights(rng::AbstractRNG, problem::Problem, n::Integer)
    d = ceil(Int, n / problem.α)
    w = randn(rng, d) 
    return w .* sqrt(d) / norm(w, 2)
end

function sample_labels(rng::AbstractRNG, ::Logistic, X::AbstractMatrix, w::AbstractVector;)
    n = size(X, 1)
    y = 2. .* (rand(rng, n) .< logistic.(X * w)) .- 1.
    return y
end

function sample_labels(
    rng::AbstractRNG, problem::Ridge, X::AbstractMatrix, w::AbstractVector;
)
    n = size(X, 1)
    y = X * w .+ sqrt.(problem.Δ) .* randn(rng, n)
    return y
end

function sample_all(rng::AbstractRNG, problem::Problem, n::Integer)
    X = sample_data(rng, problem, n)
    w = sample_weights(rng, problem, n)
    y = sample_labels(rng, problem, X, w)
    return (; X, w, y)
end
