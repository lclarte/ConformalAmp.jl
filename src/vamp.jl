"""
    Implement here Vector-Approximate message passing (VAMP) algorithm for the Ridge case 
"""

using LinearAlgebra
using Plots

function vamp(problem::Union{Ridge, Lasso}, x::AbstractMatrix, y::AbstractVector; n_iterations::Integer = 100, rtol::Real = 1e-4)
    """
    We use the same notation as in the Vector-Approximate message passing (VAMP) paper : r and γ represent the mean and variance 
    of the messages, x₁ is the estimator linked to the prior, x₂ is the estimator linked to the likelihood 
    """
    n, d = size(x)
    delta_student = problem.Δ̂
    prior_variance = 1.0 / problem.λ

    r_1 = zeros(d)

    α_1 = 1.0
    α_2 = 1.0

    γ_1 = 1.0
    γ_2 = 1.0

    x̂_1 = zeros(d)
    x̂_2 = zeros(d)

    ε = 1e-20

    for iteration in 1:n_iterations
        x̂_1_old = copy(x̂_1)
        # step 1 : Denoising
        if problem isa Ridge
            x̂_1, α_1 = denoiser(r_1, γ_1, prior_variance)
        elseif problem isa Lasso
            x̂_1, α_1 = denoiser_lasso(r_1, γ_1, prior_variance)
        else
            error("Unknown problem")
        end
        η_1 = max(γ_1 / (ε + α_1), ε)
        γ_2 = max(η_1 - γ_1, ε)
        r_2 = (η_1 .* x̂_1 - γ_1 * r_1) / γ_2 

        # step 2 : LMMSE estimation 
        x̂_2, α_2 = likelihood(x, y, r_2, γ_2, delta_student)
        # remove the 1st dimension of x̂_2
        η_2 = max(γ_2 / (ε + α_2), ε)
        γ_1 = max(η_2 - γ_2, ε)

        r_1 = (η_2 .* x̂_2 - γ_2 * r_2) / γ_1

        if norm(x̂_1 - x̂_1_old) / norm(x̂_1) < rtol
            break
        end
    end

    return x̂_1, α_1, γ_1, x̂_2, α_2, γ_2
end

function denoiser(r::AbstractVector, γ::Real, prior_variance::Real)
    """
    Implement the prox. function related to the Gaussian prior
    return a vector and a scalar (that correspodns to the variance) since the variance
    seems to be encoded here in a scalar
    The prior has 0 mean and prior_variance variance
    """
    # return r ./ ( λ * γ + 1.0), γ / (λ * γ + 1.)    
    λ = 1.0 / prior_variance
    α =  γ / (λ + γ)
    return α * r, α
end

function denoiser_lasso(r::AbstractVector, γ::Real, prior_variance::Real)
    """
    Implement the prox. function related to the Lasso prior
    return a vector and a scalar (that correspodns to the variance) since the variance
    seems to be encoded here in a scalar
    The prior has 0 mean and prior_variance variance
    """
    λ = 1.0 / prior_variance
    x̂ = (r .> 0) .* max.(0, r .- (λ / γ)) .+ (r .< 0) .* min.(0, r .+ (λ / γ))
    α = sum(x̂ .!= 0) / length(x̂)
    return x̂, α
end

function likelihood(x::AbstractMatrix,y::AbstractVector, r::AbstractVector, γ::Real, delta_student::Real)
    """
    delta_student = 1-0 / γ_w in the paper
    """
    M = inv(x' * x ./ delta_student + γ .* I) 
    g_2 = M * (x' * y ./ delta_student + γ .* r)
    return g_2, γ * tr(M) / size(M, 1)
end

### Experimental 

function loo_vamp(x::AbstractMatrix, y::AbstractVector, delta_student::Real, x̂_1::AbstractVector, α_1::Real, γ_1::Real, x̂_2::AbstractVector, α_2::Real, γ_2::Real)
    """
    # version that gives a result but does not coincide with GAMP as it should
    n, d = size(x)
    M = inv(x' * x ./ delta_student + γ_2 .* I)
    prod = M * (y .* x)'
    # repeat x̂_2 n times to do a d x n matrix
    return repeat(x̂_1, outer=[1, n])' - γ_2 .* prod'
    """
    # ΔX̂_2 stores the differences in x̂_2 when each sample is removed, so it's a n x d matric 
    M = inv(x' * x ./ delta_student + γ_2 .* I)
    ΔX̂_2 = (M * (y .* x))' ./ delta_student
    η_2  = γ_2 / α_2
    
    n, d = size(x)
    return repeat(x̂_1, outer=[1, n])' - (α_1 * η_2 / γ_1) .* ΔX̂_2
end

### Experimental 2 : iteration to compute leave-one-out 

function iterate_loo_vamp_ridge(x::AbstractMatrix, y::AbstractVector, delta_student::Real, x̂_1::AbstractVector, α_1::Real, γ_1::Real, x̂_2::AbstractVector, α_2::Real, γ_2::Real)
    # start from the converged values of the VAMP algorithm
    # initialize ΔX̂_1, ΔX̂_2, Δr_1, Δr_2 as n × d matrices
    n, d = size(x)
    ΔX̂_1 = zeros(n, d)
    ΔX̂_2 = zeros(n, d)
    Δr_1 = zeros(n, d)
    Δr_2 = zeros(n, d)

    ε   = 1e-20
    η_1 = max(γ_1 / (ε + α_1), ε)
    η_2 = max(γ_2 / (ε + α_2), ε)

    M = inv(x' * x ./ delta_student + γ_2 .* I)
    ΔX= (-y .* x)

    n_iterations = 20
    for i in 1:n_iterations
        # for now only do for the denoiser of ridge
        # because α_1 is the derivative of the denomiser function (and it self averages)
        ΔX̂_1 = α_1 .* Δr_1
        ΔX̂_2 = α_2 .* Δr_2 + (M * ΔX')' ./ delta_student
        Δr_1 = (η_2 .* ΔX̂_2 - γ_2 .* Δr_2) / γ_1
        Δr_2 = (η_1 .* ΔX̂_1 - γ_1 .* Δr_1) / γ_2
    end

    # stack x̂_1 n times to get a n x d matrix
    return (repeat(x̂_1, outer=[1, n]) + ΔX̂_1')'
end
