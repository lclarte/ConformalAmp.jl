abstract type Problem end
abstract type RegressionProblem <: Problem end

@kwdef struct Logistic <: Problem
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

@kwdef struct Lasso <: RegressionProblem
    Δ::Float64
    Δ̂::Float64
    λ::Float64
    α::Float64
end

@kwdef struct Quantile <: RegressionProblem
    """
    See e.g https://scikit-learn.org/stable/modules/model_evaluation.html#pinball-loss
    """
    quantile::Float64
    Δ::Float64
    α::Float64
    λ::Float64
end