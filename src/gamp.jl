"""
Contains the code to run the BayesOpt estimator for logistic regression
"""

import Base: -, +, *

@kwdef struct GampResult 
    x̂::AbstractVector
    v̂::AbstractVector
    ω::AbstractVector
    V::AbstractVector
    A::AbstractVector
    b::AbstractVector
    g::AbstractVector
    dg::AbstractVector
end


function Base.:+(res1::GampResult, res2::GampResult)
    return GampResult(
        v̂ = res1.v̂ + res2.v̂,
        x̂ = res1.x̂ + res2.x̂,
        ω = res1.ω + res2.ω,
        V = res1.V + res2.V,
        A = res1.A + res2.A,
        b = res1.b + res2.b,
        dg = res1.dg + res2.dg,
    )
end

function Base.:-(res1::GampResult, res2::GampResult)
    return GampResult(
        x̂ = res1.x̂ - res2.x̂,
        v̂ = res1.v̂ - res2.v̂,
        ω = res1.ω - res2.ω,
        V = res1.V - res2.V,
        A = res1.A - res2.A,
        b = res1.b - res2.b,
        g = res1.g - res2.g,
        dg = res1.dg - res2.dg,
    )
end

function Base.:*(c::Real, res::GampResult)
    return GampResult(
        x̂ = c * res.x̂,
        v̂ = c * res.v̂,
        ω = c * res.ω,
        V = c * res.V,
        A = c * res.A,
        b = c * res.b,
        g = c * res.g,
        dg= c * res.dg
    )
end

## 

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, ::Logistic; rtol = 1e-3)
    return LogisticChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol)
end

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, problem::RegressionProblem; rtol = 1e-3)
    # use Δ̂ as it's the factor used by the student
    return RidgeChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol, Δ = problem.Δ̂)
end

# 

function ∂ωchannel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, problem::RegressionProblem; rtol = 1e-3)
    # use Δ̂ as it's the factor used by the student
    return RidgeChannel.∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol, Δ = problem.Δ̂)
end

function ∂ychannel(y::Real, ω::Real, V::Real, problem::RegressionProblem; rtol = 1e-3)
    # use Δ̂ as it's the factor used by the student
    return RidgeChannel.∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol, Δ = problem.Δ̂)
end

function prior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Logistic})
    """
    Only works for Gaussian 
    """
    (; λ) = problem

    return (b ./ A) ./ (λ ./ A .+ 1.0), (1. ./ A) ./ (λ ./ A .+ 1.0)
end

function prior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    """
    Only works for Gaussian 
    """
    (; λ) = problem

    function fa(b_, A_) # sigma = 1 / A > 0, r = b / A
        if abs(b_) < λ
            return 0.0
        elseif b_ > λ
            return (b_ - λ) / A_
        else
            return (b_ + λ) / A_
        end
    end

    function fv(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return 1.0 / A_
       end
    end

    return fa.(b, A), fv.(b, A)
end

## derivatives of the prior for regression problem, useful for 

function ∂bprior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Logistic})
    (; λ) = problem

    return (1 ./ A) ./ (λ ./ A .+ 1.0), zeros(size(b))
end

function ∂Aprior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Logistic})
    (; λ) = problem

    return - b ./ (λ .+ A).^2., - 1.0 ./ (λ .+ A).^2.
end

function ∂bprior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    (; λ) = problem

    function ∂bfa(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return 1.0 / A_
        end
    end

    return ∂bfa.(b, A), zeros(size(b))
end

function ∂Aprior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    (; λ) = problem

    function ∂Afa(b_, A_) # sigma = 1 / A > 0, r = b / A
        if abs(b_) < λ
            return 0.0
        elseif b_ > λ
            return -(b_ - λ) / A_^2.
        else
            return -(b_ + λ) / A_^2.
        end
    end
    
    function ∂Afv(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return - 1.0 / A_^2.
       end
    end

    return ∂Afa.(b, A), ∂Afv.(b, A)
end



## 

function gamp(problem::Problem, X::AbstractMatrix, y::AbstractVector; max_iter::Integer = 100, rtol::Real = 1e-3)
    (; λ) = problem
    n, d = size(X)
    X_squared = X .* X

    xhat = zeros(d)
    vhat = ones(d)
    xhat_old = zeros(d)
    
    g  = zeros(n)
    dg = zeros(n)
    
    V = zeros(n)
    ω = zeros(n)

    A = zeros(d)
    b = zeros(d)

    for iteration in 1:max_iter
        V = X_squared * vhat

        ω = X * xhat - V .* g
        g, dg = channel(y, ω, V, problem, rtol=rtol)

        A = - X_squared' * dg
        b = A .* xhat + X' * g
        
        xhat_old = copy(xhat)

        xhat, vhat = prior(b, A, problem)

        if norm(xhat - xhat_old) / norm(xhat) < rtol
            break
        end
    end

    # return (; xhat, vhat, ω)
    return GampResult(x̂ = xhat, v̂ = vhat, ω = ω, V = V, A = A, b = b, g = g, dg = dg)
end

function compute_order_one_perturbation_gamp(problem::RegressionProblem, X::AbstractMatrix, y::AbstractVector, result::GampResult; max_iter::Integer = 10, rtol::Real = 1e-3, δy::Real = 1.0)
    """
    The idea is to use the convergence of GAMP algo and iterate over Δx̂, Δv̂, ΔV and Δω until convergence. the Δw returned is for δy = 1 
    By convention, we'll assume the last sample sees its label changed
    """
    n, d = size(X)
    (; x̂, v̂, ω, V, A, b) = result
    g, ∂g = channel(y, ω, V, problem; rtol = rtol)

    
    
    X_squared = X .* X
    Δx̂, Δv̂ = zeros(d), zeros(d)
    ΔA, Δb = zeros(d), zeros(d)

    Δg, Δ∂g= zeros(n), zeros(n)
    Δω, ΔV = zeros(n), zeros(n)
    
    ∂ychannel_ = ∂ychannel(y[n], ω[n], V[n], problem; rtol = rtol)
    ∂ωchannel_ = ∂ωchannel(y, ω, V, problem; rtol = rtol)
    
    ∂bprior_ = ∂bprior(b, A, problem)
    ∂Aprior_ = ∂Aprior(b, A, problem)

    for iteration in 1:max_iter
        ΔV = X_squared * Δv̂
        Δω = X * Δx̂ - ΔV .* g - V .* Δg

        Δg, Δ∂g = ∂ωchannel_[1] .* Δω, ∂ωchannel_[2] .* Δω

        Δg[n] += δy * ∂ychannel_[1]
        Δ∂g[n]+= δy * ∂ychannel_[2]

        ΔA = - (X_squared)' * Δ∂g
        Δb = X' * Δg + A .* Δx̂ + ΔA .* x̂

        Δx̂_old = copy(Δx̂)

        Δx̂ = ∂bprior_[1] .* Δb .+ ∂Aprior_[1] .* ΔA
        Δv̂ = ∂bprior_[2] .* Δb .+ ∂Aprior_[2] .* ΔA

        if norm(Δx̂ - Δx̂_old) / norm(Δx̂) < rtol
            break
        end
    end

    return GampResult(x̂ = Δx̂, v̂ = Δv̂, ω = Δω, V = ΔV, A = ΔA, b = Δb, g = Δg, dg = Δ∂g)
end

function get_cavity_means_from_gamp(problem::Problem, X::AbstractMatrix, y::AbstractVector, xhat::AbstractVector, vhat::AbstractVector, ω::AbstractVector; rtol = 1e-3)
    """
    return a matrix n x d such that the i-th row is the estimator where the i-th sample has been removed 
    """
    Xsquared = X .* X
    n, d     = size(X)

    xhat_tiled  = repeat(xhat', n, 1)
    V = Xsquared * vhat
    gout, dgout = channel(y, ω, V, problem, rtol = rtol)
    return xhat_tiled - X .* (vhat * gout')'
end