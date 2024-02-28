
abstract type UncertaintyAlgorithm end

@kwdef struct FullConformal <: UncertaintyAlgorithm
    y_bound::Float64
    coverage::Float64
end

@kwdef struct JacknifePlus <: UncertaintyAlgorithm
    coverage::Float64
end

@kwdef struct Bound
    lower::Float64
    upper::Float64
end

function get_confidence_interval(problem::Problem, X::AbstractMatrix, y::AbstractVector, xtest::AbstractVector, algorithm::JacknifePlus)
    # for jacknife, coverage = 1 - 2 * α
    n, d = size(X)
    α = (1.0  - algorithm.coverage) / 2.0
    (; xhat, vhat, ω) = gamp(problem, X, y)
    # shape is (n, d)
    xhat_cavity = get_cavity_means_from_gamp(problem, X, y, xhat, vhat, ω)
    
    # not optimal, the matrix multiplication is O(n^3)
    training_residuals = diag(X * xhat_cavity') - y
    lower_bound        = what_cavity'xtest - abs(training_residuals)
    upper_bound        = what_cavity'xtest + abs(training_residuals)

    return Bound(
        lower = StatsBase.quantile(lower_bound, floor(Int, alpha * (n + 1)) / n),
        upper = StatsBase.quantile(upper_bound, floor(Int, 1 - alpha * (n + 1)) / n)
    )
end

function get_confidence_interval(problem::Problem, X::AbstractMatrix, y::AbstractVector, algo::FullConformal)
    # 
end