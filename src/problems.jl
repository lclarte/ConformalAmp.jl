abstract type Problem end

@kwdef struct Logistic <: Problem
    λ::Float64
    α::Float64
end

# 

@kwdef struct Ridge <: Problem
    Δ::Float64
    Δ̂::Float64
    λ::Float64
    α::Float64
end

@kwdef struct Lasso <: Problem
    Δ::Float64
    Δ̂::Float64
    λ::Float64
    α::Float64
end
