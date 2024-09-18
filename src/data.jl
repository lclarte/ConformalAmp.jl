"""
    Script to generate the data
"""

using Distributions
using StableRNGs: AbstractRNG
using LogExpFunctions

# ===== for gauss-bernoulli model =====

function sample_bernoulli_gauss(rng::AbstractRNG, p::Real)
    if rand(rng) < p
        return 0.0
    else
        return rand(rng, Normal(0, 1))
    end
end

# sampe a n x d matrix with a bernoulli-gaussian model
function sample_bernoulli_gauss_matrix(rng::AbstractRNG, n::Integer, d::Integer, p::Real)
    X = zeros(n, d)
    for i in 1:n
        for j in 1:d
            X[i, j] = sample_bernoulli_gauss(rng, p)
        end
    end
    return X
end

### 

function get_n(α::Real, d::Integer)
    return ceil(Int, α * d)
end

function sample_data(rng::AbstractRNG, problem::Problem, d::Integer; model::String = "gaussian")
    n = get_n(problem.α, d)
    return sample_data_any_n(rng, d, n; model = model)
end

function sample_data_any_n(rng::AbstractRNG, d::Integer, n::Integer; model = "gaussian")
    if model == "gaussian"
        X = randn(rng, n, d) ./ sqrt(d)
    elseif model == "laplace"
        X = rand(rng, Laplace(0.0, 1.0), n, d) ./ sqrt(d)
    elseif model == "bernoulli-gauss"
        X = sample_bernoulli_gauss_matrix(rng, n, d, 0.5)
    else
        throw("Model not recognized")
    end
    return X
end

function sample_weights(rng::AbstractRNG, problem::Problem, d::Integer)
    if problem isa BayesOptimalLasso
        w = rand(Distributions.Laplace(0., 1.0), d)
        # variance of the Laplace distribution is 2
        return w .* sqrt(2 * d) / norm(w, 2)
    else
        w = randn(rng, d) 
        return w .* sqrt(d) / norm(w, 2)
    end
end

function sample_labels(rng::AbstractRNG, ::Union{Logistic, BayesOptimalLogistic}, X::AbstractMatrix, w::AbstractVector;)
    n = size(X, 1)
    y = 2. .* (rand(rng, n) .< logistic.(X * w)) .- 1.
    return y
end

function sample_labels(rng::AbstractRNG, problem::RegressionProblem, X::AbstractMatrix, w::AbstractVector;)
    n = size(X, 1)
    y = X * w .+ sqrt.(problem.Δ) .* randn(rng, n)
    return y
end

function sample_all(rng::AbstractRNG, problem::Problem, d::Integer; model::String = "gaussian")
    X = sample_data(rng, problem, d; model = model)
    w = sample_weights(rng, problem, d)
    y = sample_labels(rng, problem, X, w)
    return (; X, w, y)
end