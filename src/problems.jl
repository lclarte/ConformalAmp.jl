abstract type Problem end
abstract type RegressionProblem <: Problem end
abstract type ClassificationProblem <: Problem end

@kwdef struct Logistic <: ClassificationProblem
    λ::Float64
    α::Float64
end

# 

@kwdef struct Ridge <: RegressionProblem
    Δ::Float64
    Δ̂::Float64
    λ::Float64
    α::Float64
end

@kwdef struct Pinball <: RegressionProblem
    Δ::Float64
    q::Float64 # to estimate quantile q
    λ::Float64
    α::Float64
end

@kwdef struct Lasso <: RegressionProblem
    Δ::Float64
    Δ̂::Float64
    λ::Float64
    α::Float64
end