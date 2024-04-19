"""
Contains the code to run the BayesOpt estimator for logistic regression
"""

using SpecialFunctions

import Base: -, +, *

# TODO : Write GampResult with a template for the problem type
# so that we can have the bias or not 

@kwdef struct GampResult 
    x̂::AbstractVector
    v̂::AbstractVector
    ω::AbstractVector
    V::AbstractVector
    A::AbstractVector
    b::AbstractVector
    g::AbstractVector
    dg::AbstractVector
    bias::Union{Real, Nothing} = nothing
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

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, ::Logistic; rtol = 1e-3)::Tuple{AbstractVector, AbstractVector}
    return LogisticChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol)
end

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, problem::Union{Ridge, Lasso, BayesOptimalRidge, BayesOptimalLasso}; rtol = 1e-3)::Tuple{AbstractVector, AbstractVector}
    # use Δ̂ as it's the factor used by the student
    return RidgeChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol, Δ = problem.Δ̂)
end

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, problem::Pinball; rtol::Real = 1e-3, b::Real = 0.0)::Tuple{AbstractVector, AbstractVector}
    # use Δ̂ as it's the factor used by the student
    return PinballChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; q = problem.q, bias = b)
end

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, ::BayesOptimalLogistic; rtol = 1e-3)::Tuple{AbstractVector, AbstractVector}
    return BayesianLogisticChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol)
end

# 

function ∂ωchannel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, problem::Union{Lasso, Ridge, BayesOptimalRidge, BayesOptimalLasso}; rtol = 1e-3)
    # use Δ̂ as it's the factor used by the student
    return RidgeChannel.∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol, Δ = problem.Δ̂)
end

function ∂ωchannel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, problem::Pinball; rtol = 1e-3)
    # use Δ̂ as it's the factor used by the student
    return PinballChannel.∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y, ω, V ; q = problem.q)
end

function ∂ychannel(y::Real, ω::Real, V::Real, problem::Union{Lasso, Ridge}; rtol = 1e-3)
    # use Δ̂ as it's the factor used by the student
    return RidgeChannel.∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol, Δ = problem.Δ̂)
end

function ∂ychannel(y::Real, ω::Real, V::Real, problem::Pinball; rtol = 1e-3)
    # use Δ̂ as it's the factor used by the student
    return PinballChannel.∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y, ω, V; q = problem.q)
end

function prior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Logistic, Pinball, BayesOptimalLogistic, BayesOptimalRidge})
    """
    For L2 penalty
    """
    (; λ) = problem
    # We don't need to assert that λ = 1.0, if it's not we will just not have the Bayes optimal estimator
    # if problem isa BayesOptimalLogistic || problem isa BayesOptimalRidge
    #     @assert λ == 1.0
    # end

    return b ./ (λ .+ A), 1. ./ (λ .+ A)
end

function prior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    """
    for l1 penalty
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

function prior(b::AbstractVector, A::AbstractVector, problem::BayesOptimalLasso)
    """
    for l1 penalty
    """
    function ∂RlogZ(Σ::AbstractVector, R::AbstractVector, λ::Real)
        # convert the Python code above in Julia
        tmp = sqrt.(2.0 * Σ)
        Rm, Rp = (R .- λ * Σ), (R .+ λ * Σ)
        return - λ * (1.0 .+ erf.(Rm ./ tmp) - exp.(2 * R * λ) .* erfc.(Rp ./ tmp)) ./ (1.0 .+ erf.(Rm ./ tmp) .+ exp.(2.0 * λ * R) .* erfc.(Rp ./ tmp))
    end

    function ∂∂RlogZ(Σ::AbstractVector, R::AbstractVector, λ::Real)
        """tmp = np.sqrt(2.0 * Sigma)
        tmp_exp = np.exp(2 * R * lambda_)
        tmp_pi = np.sqrt(2.0 / np.pi)
        Rm, Rp = R - lambda_ * Sigma, R + lambda_ * Sigma
        return 2 * lambda_ * tmp_exp * ( - np.exp(-Rp**2 / tmp**2) * tmp_pi + ( 2 * lambda_ * np.sqrt(Sigma) - tmp_pi * np.exp(-Rm**2 / tmp**2)) * erfc(Rp / tmp) + \
                erf(Rm / tmp) * (2 * lambda_ * np.sqrt(Sigma) * erfc(Rp / tmp) - np.exp(-Rp**2 / tmp**2) * tmp_pi)) / \
                (np.sqrt(Sigma) * (1.0 + erf(Rm / tmp) + tmp_exp * erfc(Rp / tmp) )**2)"""
        tmp = sqrt.(2.0 * Σ)
        tmp_exp = exp.(2 * R * λ)
        tmp_pi = sqrt.(2.0 / π)
        Rm, Rp = R .- λ * Σ, R .+ λ * Σ
        return 2 * λ * tmp_exp .* ( - exp.(- Rp.^2 ./ tmp.^2) .* tmp_pi + ( 2 * λ * sqrt.(Σ) .- tmp_pi .* exp.(- Rm.^2 ./ tmp.^2)) .* erfc.(Rp ./ tmp) .+ 
                erf.(Rm ./ tmp) .* (2 * λ * sqrt.(Σ) .* erfc.(Rp ./ tmp) .- exp.(- Rp.^2 ./ tmp.^2) .* tmp_pi)) ./ 
                (sqrt.(Σ) .* (1.0 .+ erf.(Rm ./ tmp) .+ tmp_exp .* erfc.(Rp ./ tmp) ).^2)
    end

    Σ = 1.0 ./ A
    R = b ./ A
    fa = Σ .* ∂RlogZ(Σ, R, problem.λ) + R
    fv = Σ.^2 .* ∂∂RlogZ(Σ, R, problem.λ) + Σ
    return fa, fv
end

## derivatives of the prior for regression problem, useful for 

function ∂bprior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Logistic, Pinball})
    (; λ) = problem

    return (1 ./ A) ./ (λ ./ A .+ 1.0), zeros(size(b))
end

function ∂Aprior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Logistic, Pinball})
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

# SPECIAL IMPLEMENTATION OF GAMP FOR QUANTILE REGRESSION, BECAUSE HERE WE ADD THE BIAS.
# For now we alternate the minimization of the bias inside the loop of GAMP (given the weights w⃗) 

function gamp(problem::Pinball, X::AbstractMatrix, y::AbstractVector; max_iter::Integer = 100, rtol::Real = 1e-3)
    """
    Modification of GAMP to integrate a bias term in the problem
    QUESTION : Can we derive state evolution equations from this ? Maybe not 
    """

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

    bias = 0.0

    for iteration in 1:max_iter
        xhat_old_outer = copy(xhat)
        for iteration in 1:max_iter
            V = X_squared * vhat

            ω = X * xhat - V .* g
            g, dg = channel(y, ω, V, problem, rtol=rtol, b = bias)

            A = - X_squared' * dg
            b = A .* xhat + X' * g
            
            xhat_old = copy(xhat)
            xhat, vhat = prior(b, A, problem)

            if norm(xhat - xhat_old) / norm(xhat) < rtol
                break
            end
        end

        # update the bias AFTER convergence of the AMP iterations
        if problem.use_bias
            bias_old = bias
            # update the bias here, it's equal to the q-th quantile of the training residuals
            bias = Statistics.quantile(y - X * xhat, problem.q)
            if norm(xhat - xhat_old_outer) / norm(xhat) < rtol && abs(bias - bias_old) / abs(bias) < rtol
                break
            end
        else
            bias = 0.0
            if norm(xhat - xhat_old_outer) / norm(xhat) < rtol
                break
            end
        end

        
    end

    # return (; xhat, vhat, ω)
    return GampResult(x̂ = xhat, v̂ = vhat, ω = ω, V = V, A = A, b = b, g = g, dg = dg, bias = bias)
end

##

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

function get_cavity_means_from_gamp(X::AbstractMatrix, result::GampResult)
    n, _ = size(X)
    x̂_tiled = repeat(result.x̂', n, 1)
    return x̂_tiled - X .* (result.v̂ * result.g')'
end

function get_derivative_cavity_means(X::AbstractMatrix, result::GampResult, Δresult::GampResult)
    repeat(Δresult.x̂', size(X)[1], 1) - X .* (result.v̂ * Δresult.g' + Δresult.v̂ * result.g')' 
end

function get_cavity_means_order_one(problem::Problem, X::AbstractMatrix, y::AbstractVector, result::GampResult, Δresult::GampResult; rtol = 1e-3)
    """
    arguments : 
        - Δresult : variation of GAMP result w.r.t δy > 0 (that implicitly multilplies Δresult so not included as argument)
    """
    n, _ = size(X)
    Ŵ_0 = get_cavity_means_from_gamp(problem, X, y, result.x̂, result.v̂, result.ω; rtol = rtol)
    ΔŴ  = get_derivative_cavity_means(X, result, Δresult)
    return  Ŵ_0 + ΔŴ
end