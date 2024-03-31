
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

function score(::RegressionProblem, y::Real, ŷ::Real)
    # score must be higher for bad predictions
    return abs(y - ŷ)
end

function score(::RegressionProblem, y::AbstractVector, ŷ::AbstractVector)
    return abs.(y - ŷ)
end

## for jacknife

function get_confidence_interval(problem::RegressionProblem, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algorithm::JacknifePlus, method::Method)
    # for jacknife, coverage = 1 - 2 * α
    n, d = size(X)
    α = (1.0  - algorithm.coverage) / 2.0

    what_cavity = fit_leave_one_out(problem, X, y, method)
    
    # not optimal, the matrix multiplication is O(n^3)
    training_residuals = diag(predict(problem, what_cavity, X)) - y
    lower_bound        = predict(problem, what_cavity, xtest) - abs.(training_residuals)
    upper_bound        = predict(problem, what_cavity, xtest) + abs.(training_residuals)

    return Statistics.quantile(lower_bound, floor(Int, α * (n + 1)) / n), Statistics.quantile(upper_bound, floor(Int, (1 - α) * (n + 1)) / n)
end

# for full conformal prediction

function get_confidence_interval(problem::RegressionProblem, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algo::FullConformal, method::Method)
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

function get_confidence_interval(problem::RegressionProblem, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algo::FullConformal, method::GAMP)
    """
    With GAMP, we only need to fit twice gamp, and compute the derivative with finite difference
    we just have to to an matrix multilplication to get the new scores
    """
    (; coverage, δy_range) = algo
    n, d = size(X)
    @assert size(xtest, 2) == 1
    
    α = 1.0 - coverage

    # augment the dataset by adding xtest to X
    X_augmented = vcat(X, xtest')
    y_augmented = vcat(y, 0.0)

    prediction_set = []

    # the only goal of result₀ is to provide a 1st approximation ŷ of the output
    result₀ = gamp(problem, X, y; max_iter = method.max_iter, rtol = method.rtol)
    ŷ = predict(problem, result₀.x̂, xtest)
    
    # NOTE : technically our method treats the test sample not symmetrically w.r.t 
    # the other samples as we compute the deritvative w.r.t the last sample, but 
    # in practice it should not change much (from empirical observations)
    
    δy₀ = 0.01
    y_augmented[end] = ŷ
    result_1  = gamp(problem, X_augmented, y_augmented; max_iter = method.max_iter, rtol = method.rtol)
    y_augmented[end] = ŷ + δy₀
    result_2  = gamp(problem, X_augmented, y_augmented; max_iter = method.max_iter, rtol = method.rtol)

    Ŵ_1     = get_cavity_means_from_gamp(X_augmented, result_1)
    Ŵ_2     = get_cavity_means_from_gamp(X_augmented, result_2)
    # we divide by δy_perturbation so that we only have to multiply by δy after
    ΔŴ      = (1.0 / δy₀) * (Ŵ_2 - Ŵ_1)

    # LOWER BOUND 
    for δy in reverse(δy_range)
        # Candidate label
        y_augmented[end] = ŷ - δy
        weights = Ŵ_1 - δy * ΔŴ
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[end] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ - δy)
        end
    end

    # UPPER BOUND 
    for δy in δy_range
        # Candidate label
        y_augmented[end] = ŷ + δy
        weights = Ŵ_1 + δy * ΔŴ
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[end] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ + δy)
        end
    end

    return prediction_set
end 

function get_confidence_interval(problem::RegressionProblem, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algo::FullConformal, method::GAMPTaylor)
    """
    With GAMPTaylor, we only need to fit twice gamp, one for gamp and one for the derivative, then at each y 
    we just have to to an matrix multilplication to get the new scores
    """
    (; coverage, δy_range) = algo
    n, d = size(X)
    @assert size(xtest, 2) == 1
    
    α = 1.0 - coverage

    # augment the dataset by adding xtest to X
    X_augmented = vcat(X, xtest')
    y_augmented = vcat(y, 0.0)

    prediction_set = []

    # the only goal of result₀ is to provide a 1st approximation of the output
    result₀ = gamp(problem, X, y; max_iter = method.max_iter, rtol = method.rtol)
    ŷ = predict(problem, result₀.x̂, xtest)
    
    # NOTE : technically our method treats the test sample not symmetrically w.r.t 
    # the other samples as we compute the deritvative w.r.t the last sample, but 
    # in practice it should not change much (from empirical observations)
    
    y_augmented[end] = ŷ
    result  = gamp(problem, X_augmented, y_augmented; max_iter = method.max_iter, rtol = method.rtol)
    Δresult = compute_order_one_perturbation_gamp(problem, X_augmented, y_augmented, result; 
                                            max_iter = method.max_iter, rtol = method.rtol, δy = method.δy_perturbation)
    Ŵ_0     = get_cavity_means_from_gamp(X_augmented, result)
    # we divide by δy_perturbation so that we only have to multiply by δy after
    ΔŴ      = (1.0 / method.δy_perturbation) * get_derivative_cavity_means(X_augmented, result, Δresult)

    # LOWER BOUND 
    for δy in reverse(δy_range)
        # Candidate label
        y_augmented[end] = ŷ - δy
        weights = Ŵ_0 - δy * ΔŴ
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[end] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ - δy)
        end
    end

    # UPPER BOUND 
    for δy in δy_range
        # Candidate label
        y_augmented[end] = ŷ + δy
        weights = Ŵ_0 + δy * ΔŴ
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[end] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ + δy)
        end
    end

    return prediction_set
end

function get_confidence_interval(problem::RegressionProblem, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algo::FullConformal, method::ERMTaylor)
    (; coverage, δy_range) = algo
    n, d = size(X)
    @assert size(xtest, 2) == 1
    
    α = 1.0 - coverage

    # augment the dataset by adding xtest to X
    X_augmented = vcat(X, xtest')
    y_augmented = vcat(y, 0.0)

    prediction_set = []

    # the only goal of result₀ is to provide a 1st approximation of the output
    result₀ = fit(problem, X, y, ERM())
    ŷ = predict(problem, result₀, xtest)
    
    δy₀ = 0.01
    y_augmented[end] = ŷ
    Ŵ_1     = fit_leave_one_out(problem, X_augmented, y_augmented, ERM())
    y_augmented[end] = ŷ + δy₀
    Ŵ_2     = fit_leave_one_out(problem, X_augmented, y_augmented, ERM())
    # we divide by δy_perturbation so that we only have to multiply by δy after
    ΔŴ      = (1.0 / δy₀) * (Ŵ_2 - Ŵ_1)

    # LOWER BOUND 
    for δy in reverse(δy_range)
        # Candidate label
        y_augmented[end] = ŷ - δy
        weights = Ŵ_1 - δy * ΔŴ
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[end] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ - δy)
        end
    end

    # UPPER BOUND 
    for δy in δy_range
        # Candidate label
        y_augmented[end] = ŷ + δy
        weights = Ŵ_1 + δy * ΔŴ
        scores           = score(problem, diag(predict(problem, weights, X_augmented)), y_augmented)
        # Compute the quantiles and add y to the interval if it's in the quantile
        if scores[end] <= Statistics.quantile(scores[1:n], ceil(Int, coverage * (n+1)) / n)
            push!(prediction_set, ŷ + δy)
        end
    end

    return prediction_set
end

# for classification problem

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