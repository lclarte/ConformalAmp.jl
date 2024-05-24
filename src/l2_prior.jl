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

## derivatives of the prior for regression problem for Taylor-gamp
# on pourrait inclure Logistic dedans mais ca servirait a rien

function ∂bprior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Pinball})
    (; λ) = problem

    return (1 ./ A) ./ (λ ./ A .+ 1.0), zeros(size(b))
end

function ∂Aprior(b::AbstractVector, A::AbstractVector, problem::Union{Ridge, Pinball})
    (; λ) = problem

    return - b ./ (λ .+ A).^2., - 1.0 ./ (λ .+ A).^2.
end