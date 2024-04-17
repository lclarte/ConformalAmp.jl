module BayesianLogisticChannel

using NLSolvers
using LinearAlgebra
using LogExpFunctions: log1pexp
using QuadGK
using StaticArrays
using StatsFuns: normpdf, poispdf

logistic(x::Real) = 1 / (1 + exp(-x))

function Z₀_and_∂μZ₀_and_∂∂μZ₀(
    y::Real, μ::Real, v::Real; rtol::Real
)
    """
    Is both the partition funtion for the teacher and is used in gₒᵤₜ_and_∂ωgₒᵤₜ
    """
    function Z₀_and_∂μZ₀_and_∂∂μZ₀_integrand(u::Real)
        z = u * sqrt(v) + μ
        σ      = BayesianLogisticChannel.logistic(y * z)
        σ_der  = σ * (1 - σ)
        σ_der_2= σ_der * (1 - 2σ) 
        res = SVector(σ, y[1] * σ_der, σ_der_2) * normpdf(u)
        return res
    end

    bound = 10.0
    double_integral, err = quadgk(
        Z₀_and_∂μZ₀_and_∂∂μZ₀_integrand, -bound, bound; rtol
    )

    Z₀ = double_integral[1]
    ∂μZ₀ = double_integral[2]
    ∂∂μZ₀ = double_integral[3]
    return Z₀, ∂μZ₀, ∂∂μZ₀
end

function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; rtol::Real)
    """
    Channel function for the logistic regression
    Reminder : gₒᵤₜ = ∂ωlogZ₀ =  and ∂ωgₒᵤₜ = ∂²ωlogZ₀
    """
    Z, ∂Z, ∂∂Z = Z₀_and_∂μZ₀_and_∂∂μZ₀(y, ω, V; rtol)
    gₒᵤₜ   = ∂Z / Z
    ∂ωgₒᵤₜ = ∂∂Z / Z - (∂Z / Z)^2
    
    return gₒᵤₜ, ∂ωgₒᵤₜ
end

function gₒᵤₜ_and_∂ωgₒᵤₜ(
    y::AbstractVector{<:Real},
    ω::AbstractVector{<:Real},
    V::AbstractVector{<:Real};
    rtol::Real
)
    """
    Vectorized version of the channel function
    """
    g, dg = similar(y), similar(y)
    for i in eachindex(y)
        g[i], dg[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; rtol)
    end

    return g, dg
end

end # module