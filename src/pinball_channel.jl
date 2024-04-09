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

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; q::Real)
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

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; q::Real)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        gₒᵤₜ, ∂ωgₒᵤₜ = zeros(length(y)), zeros(length(y))
        for i in eachindex(y)
            gₒᵤₜ[i], ∂ωgₒᵤₜ[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; q = q)
        end
        return gₒᵤₜ, ∂ωgₒᵤₜ
    end

    function ∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; q::Real)
        """
        Vectorized version of the channel function
        NOTE : Weird that the 1st and 2nd derivatives are 0 ... 
        """
        @assert length(y) == length(ω) == length(V)
        ∂ωgₒᵤₜ, ∂ω∂ωgₒᵤₜ = zeros(length(y)), zeros(length(y))
        for i in eachindex(y)
            _, ∂ωgₒᵤₜ[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; q = q)
        end
        return ∂ωgₒᵤₜ, ∂ω∂ωgₒᵤₜ
    end

    ## 

    function ∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; q::Real)
        """
        Needed for full cp
        """
        ∂yg   = 0.0
        # I think it's always 0 
        ∂y∂ωg = 0.0

        if y > ω + (q - 1) * V && y < ω + q * V
            ∂yg = 1.0 / V
        end

        return ∂yg, ∂y∂ωg
    end
end