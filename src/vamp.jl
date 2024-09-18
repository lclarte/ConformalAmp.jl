"""
    Implement here Vector-Approximate message passing (VAMP) algorithm for the Ridge case 
"""

using LinearAlgebra
using Plots

function vamp(x::AbstractMatrix, y::AbstractVector, delta_student::Real, prior_variance::Real; n_iterations = 100)
    """
    We use the same notation as in the Vector-Approximate message passing (VAMP) paper : r and γ represent the mean and variance 
    of the messages, x₁ is the estimator linked to the prior, x₂ is the estimator linked to the likelihood 
    """
    n, d = size(x)


    r_1 = zeros(d)
    γ_1 = 1.0
    γ_2 = 1.0

    x̂_1 = zeros(d)

    for iteration in 1:n_iterations
        # step 1 : Denoising
        x̂_1, α_1 = denoiser(r_1, γ_1, prior_variance)
        η_1 = γ_1 / α_1
        γ_2 = η_1 - γ_1
        r_2 = (η_1 .* x̂_1 - γ_1 * r_1) / γ_2 

        # step 2 : LMMSE estimation 
        x̂_2, α_2 = likelihood(x, y, r_2, γ_2, delta_student)
        # remove the 1st dimension of x̂_2
        η_2 = γ_2 / α_2
        γ_1 = η_2 - γ_2

        r_1 = (η_2 .* x̂_2 - γ_2 * r_2) / γ_1
    end

    print("$γ_1, $γ_2")

    return x̂_1
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

function likelihood(x::AbstractMatrix,y::AbstractVector, r::AbstractVector, γ::Real, delta_student::Real)
    """
    delta_student = 1-0 / γ_w in the paper
    """
    M = inv(x' * x ./ delta_student + γ .* I) 
    g_2 = M * (x' * y ./ delta_student + γ .* r)
    return g_2, γ * tr(M) / size(M, 1)
end



# test on random data and random teacher
d = 100
n = 1000

x = randn(n, d) / sqrt(d)
w = randn(d)
y = x * w

ŵ_erm = (x' * x + 1.0 * I) \ (x' * y)

ŵ = vamp(x, y, 1.0, 1.0; n_iterations = 10)
stephist(ŵ, bins=100)
stephist!(ŵ_erm, bins=100)