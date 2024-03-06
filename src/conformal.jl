
abstract type UncertaintyAlgorithm end

@kwdef struct FullConformal <: UncertaintyAlgorithm
    # we'll test [ŷ - δy_range, ŷ + δy_range]
    δy_range::AbstractRange
    coverage::Float64
end

@kwdef struct JacknifePlus <: UncertaintyAlgorithm
    coverage::Float64
end

# score function 

function score(::Logistic, y::Real, confidence::Real)
    # note : confidence is the proba. of 1st class
    if y == 1.0
        return 1.0 - confidence
    else 
        return confidence
    end
end

function score(pb::Logistic, y::AbstractVector, confidence::AbstractVector)
    return [score(pb, y[i], confidence[i]) for i in eachindex(y)]
end

function score(::Ridge, y::Real, ŷ::Real)
    # score must be higher for bad predictions
    return abs(y - ŷ)
end

function score(::Ridge, y::AbstractVector, ŷ::AbstractVector)
    return abs.(y - ŷ)
end

##

function get_confidence_interval(problem::Ridge, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algorithm::JacknifePlus, method::Method)
    # for jacknife, coverage = 1 - 2 * α
    n, d = size(X)
    α = (1.0  - algorithm.coverage) / 2.0

    what_cavity = fit_leave_one_out(problem, X, y, method)
    
    # not optimal, the matrix multiplication is O(n^3)
    training_residuals = diag(X * what_cavity') - y
    lower_bound        = predict(problem, what_cavity, xtest) - abs.(training_residuals)
    upper_bound        = predict(problem, what_cavity, xtest) + abs.(training_residuals)

    return Statistics.quantile(lower_bound, floor(Int, α * (n + 1)) / n), Statistics.quantile(upper_bound, floor(Int, (1 - α) * (n + 1)) / n)
end

# 

function get_confidence_interval(problem::Ridge, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algo::FullConformal, method::Method)
    """
    Use GAMP to compute the confidence interval
    """
    (; coverage, δy_range) = algo
    n, d = size(X)
    @assert size(xtest, 2) == 1
    
    α = 1.0 - coverage

    # augment the dataset by adding xtest to X
    X_augmented = vcat(X, xtest')
    y_augmented = vcat(y, 0.0)

    prediction_set = []

    ŵ = fit(problem, X, y, method)
    ŷ = predict(problem, ŵ, xtest)

    # LOWER BOUND 
    for δy in reverse(δy_range)
        # Candidate label
        y_augmented[n+1] = ŷ - δy
        # Compute the score for all n samples by 1) computing the leave-one-out and corresponding score
        weights          = fit_leave_one_out(problem, X_augmented, y_augmented, method)
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[n+1] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ - δy)
        end
    end

    # UPPER BOUND 
    for δy in δy_range
        # Candidate label
        y_augmented[n+1] = ŷ + δy
        # Compute the score for all n samples by 1) computing the leave-one-out and corresponding score
        weights          = fit_leave_one_out(problem, X_augmented, y_augmented, method)
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[n+1] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ + δy)
        end
    end

    return prediction_set
end

function get_confidence_interval(problem::Logistic, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algo::FullConformal, method::Method)
    (; coverage) = algo
    n, d = size(X)
    @assert size(xtest, 2) == 1
    
    α = 1.0 - coverage

    # augment the dataset by adding xtest to X
    X_augmented = vcat(X, xtest')
    y_augmented = vcat(y, 0.0)
    
    prediction_set = []

    for y in [-1, 1]
        y_augmented[n+1] = y
        weights = fit_leave_one_out(problem, X_augmented, y_augmented, method)
        scores  = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        if scores[n+1] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, y)
        end
    end

    return prediction_set
end

##

function get_order_one_amp_perturbation(problem::Ridge, X::AbstractMatrix, y::AbstractVector, gampresult::GampResult; rtol::Real = 1e-3)
    """
    return the vector Δŵ such that for changing the last label from y[n] to y[n] + δy, the estimator is changed to ŵ + δy * Δŵ
    """
    (; x̂, v̂, ω, V, A, b) = gampresult
    n, d = size(X)
    V = (X .* X) * v̂
    ∂yg = RidgeChannel.∂ygₒᵤₜ_and_∂ωgₒᵤₜ(y[n], ω[n], V[n]; Δ = problem.Δ, rtol = rtol)[1]
    return ∂yg .* X[n, :] ./ (1.0 .+ A)
end