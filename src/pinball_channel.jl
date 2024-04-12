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

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; q::Real, bias::Real = 0.0)
        """
        Here the loss is defined as 
        l_b(y, w ⋅ x) = q * max(y - (w ⋅ x + b), 0) + (1 - q) * max((w ⋅ x + b) - y, 0)
        """
        if bias != 0.0
            return gₒᵤₜ_and_∂ωgₒᵤₜ(y - bias, ω, V; q=q, bias = 0.0)
        else
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
    end

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; q::Real, bias::Real = 0.0)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        gₒᵤₜ, ∂ωgₒᵤₜ = zeros(length(y)), zeros(length(y))
        for i in eachindex(y)
            gₒᵤₜ[i], ∂ωgₒᵤₜ[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; q = q, bias = bias)
        end
        return gₒᵤₜ, ∂ωgₒᵤₜ
    end

    function ∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; q::Real, bias::Real = 0.0)
        """
        Vectorized version of the channel function
        NOTE : Weird that the 1st and 2nd derivatives are 0 ... 
        """
        @assert length(y) == length(ω) == length(V)
        ∂ωgₒᵤₜ, ∂ω∂ωgₒᵤₜ = zeros(length(y)), zeros(length(y))
        for i in eachindex(y)
            _, ∂ωgₒᵤₜ[i] = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; q = q, bias = bias)
        end
        return ∂ωgₒᵤₜ, ∂ω∂ωgₒᵤₜ
    end

    ## 

    function ∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; q::Real, bias::Real = 0.0)
        """
        Needed for full cp
        """
        if bias != 0.0
            return ∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y - bias, ω, V; q = q, bias = 0.0)
        else 
            ∂yg   = 0.0
            # I think it's always 0 
            ∂y∂ωg = 0.0

            if y > ω + (q - 1) * V && y < ω + q * V
                ∂yg = 1.0 / V
            end

            return ∂yg, ∂y∂ωg
        end
    
    end

end # module