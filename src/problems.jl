using Base

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

# model corresponding the Gaussian prior and L2 loss
@kwdef struct BayesOptimalRidge <: RegressionProblem
    Δ::Float64
    Δ̂::Float64
    λ::Float64
    α::Float64
end

# model corresponding the Laplace prior and L2 loss 
@kwdef struct BayesOptimalLasso <: RegressionProblem
    Δ::Float64
    Δ̂::Float64
    λ::Float64
    α::Float64
end

function Base.show(io::IO, problem::Ridge) 
    print(io, "Ridge(Δ = $(problem.Δ), Δ̂ = $(problem.Δ̂), λ = $(problem.λ), α = $(problem.α))")
end

function Base.show(io::IO, problem::Lasso) 
    print(io, "Lasso(Δ = $(problem.Δ), Δ̂ = $(problem.Δ̂), λ = $(problem.λ), α = $(problem.α))")
end

function Base.show(io::IO, problem::Pinball) 
    print(io, "Pinball(Δ = $(problem.Δ), q = $(problem.q), λ = $(problem.λ), α = $(problem.α))")
end