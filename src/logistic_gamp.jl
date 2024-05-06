"""
logistic_gamp.jl
Functions used by gamp for logistic regression
"""

function channel(y::AbstractVector, ω::AbstractVector, V::AbstractVector, ::Logistic; rtol = 1e-3)::Tuple{AbstractVector, AbstractVector}
    return LogisticChannel.gₒᵤₜ_and_∂ωgₒᵤₜ(y, ω, V, ; rtol = rtol)
end