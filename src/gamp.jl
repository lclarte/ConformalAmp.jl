"""
Contains the code to run the BayesOpt estimator for logistic regression
"""

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, ::Logistic)
    return LogisticChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = 1e-3)
end

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, problem::Ridge)
    # use Δ̂ as it's the factor used by the student
    return RidgeChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = 1e-3, Δ = problem.Δ̂)
end

function prior(b::AbstractVector, A::AbstractVector, λ::Real)
    """
    Only works for Gaussian 
    """
    Σ = 1. ./ A
    R = b ./ A

    return R ./ (λ .* Σ .+ 1.0), Σ ./ (λ .* Σ .+ 1.0)
end

function gamp(problem::Problem, X::AbstractMatrix, y::AbstractVector; max_iter::Integer = 100, rtol::Real = 1e-3)
    (; λ) = problem
    n, d = size(X)
    X_squared = X .* X

    vhat = ones(d)
    xhat = zeros(d)
    
    g    = zeros(n)
    ω = zeros(n)

    for iteration in 1:max_iter
        V = X_squared * vhat

        ω = X * xhat - V .* g
        g, dg = channel(y, ω, V, problem)

        A = - X_squared' * dg
        b = A .* xhat + X' * g
        
        xhat_old = copy(xhat)

        xhat, vhat = prior(b, A, λ)

        if norm(xhat - xhat_old) / norm(xhat) < rtol
            break
        end
    end

    return (; xhat, vhat, ω)
end

function get_cavity_means_from_gamp(problem::Problem, X::AbstractMatrix, y::AbstractVector, xhat::AbstractVector, vhat::AbstractVector, ω::AbstractVector)
    """
    return a matrix n x d such that the i-th row is the estimator where the i-th sample has been removed 
    """
    Xsquared = X .* X
    n, d     = size(X)

    xhat_tiled = repeat(xhat', n, 1)
    V = Xsquared * vhat
    gout, dgout = channel(y, ω, V, problem)
    return xhat_tiled - X .* (vhat * gout')'
end