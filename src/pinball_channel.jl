# TODO : Lire le papier de Song mei pour le channel correspondant a la pinball loss 

module PinballChannel
    # corresponds to the loss function 
    # l(y, ŷ) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)

    using NLSolvers
    using LinearAlgebra
    using LogExpFunctions: log1pexp


    function loss(y::Real, ŷ::Real, q::Real)
        return q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)
    end

    function loss_der(y::Real, ŷ::Real, q::Real)
        return y > ŷ ? -q : (1 - q)
    end

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; q::Real, rtol::Real = 1e-3)
        prox = y
        if ω > y - (q - 1) * V
            prox = ω + (q - 1) * V
        elseif ω < y - q * V
            prox = ω + q * V
        end
        
        dprox = 0.0
        if ω > y - (q - 1) * V
            dprox = 1.0
        elseif ω < y - q * V
            dprox = 1.0
        end      

        gₒᵤₜ = (prox - ω) / V
        ∂gₒᵤₜ = (dprox - 1.0) / V
        return gₒᵤₜ, ∂gₒᵤₜ
    end

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; q::Real, rtol::Real = 1e-3)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        gₒᵤₜ, ∂ωgₒᵤₜ = zeros(length(y)), zeros(length(y))
        for i in eachindex(y)
            gₒᵤₜ[i], ∂ωgₒᵤₜ[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; q = q, rtol = rtol)
        end
        return gₒᵤₜ, ∂ωgₒᵤₜ
    end

    function ∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; rtol::Real)
        """
        Vectorized version of the channel function
        NOTE : Weird that the 1st and 2nd derivatives are 0 ... 
        """
        @assert length(y) == length(ω) == length(V)
        ∂ωgₒᵤₜ, ∂ω∂ωgₒᵤₜ = zeros(length(y)), zeros(length(y))
        for i in eachindex(y)
            _, ∂ωgₒᵤₜ[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; q = q, rtol = rtol)
        end
        return ∂ωgₒᵤₜ, ∂ω∂ωgₒᵤₜ
    end

    ## 

    function ∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; rtol::Real)
        """
        Needed for full cp
        """
        @assert length(y) == length(ω) == length(V)
        error("not implemented")
        ∂yg   = nothing
        ∂y∂ωg = nothing
        return ∂yg, ∂y∂ωg
    end

    # this function is normally useless 
    function Z₀_and_∂μZ₀(y::Real, ω::Real, v::Real; rtol::Real)
        """
        Corresponds to the partition function of the Gaussian teacher 
        Here Δ is the variance of the teacher
        """
        error("not implemented")
        Z₀   = nothing
        ∂μZ₀ = nothing
        return Z₀, ∂μZ₀
    end
end