module RidgeChannel

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; rtol::Real, Δ::Real = 1.0)
        """
            Δ is the denominator in the square loss 
        """
        gₒᵤₜ = (y - ω) / (Δ + V)
        ∂ωgₒᵤₜ = - 1.0 / (Δ + V)
        return gₒᵤₜ, ∂ωgₒᵤₜ 
    end

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; rtol::Real, Δ::Real = 1.0)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        g = (y - ω) ./ (Δ .+ V)
        ∂ωg = - 1.0 ./ (Δ .+ V)
        return g, ∂ωg
    end

    function ∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; rtol::Real, Δ::Real = 1.0)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        ∂ωg = - 1.0 ./ (Δ .+ V)
        ∂ω∂ωg = - 1.0 ./ (Δ .+ V)
        return ∂ωg, ∂ω∂ωg
    end

    function ∂Vgₒᵤₜ_and_∂V∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; rtol::Real, Δ::Real = 1.0)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        ∂Vg = - (y - ω) ./ (Δ .+ V).^2
        ∂V∂ωg = 1.0 ./ (Δ .+ V).^2
        return ∂Vg, ∂V∂ωg
    end

    ## 

    function ∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; rtol::Real, Δ::Real = 1.0)
        """
        Needed for full cp
        """
        @assert length(y) == length(ω) == length(V)
        ∂yg = 1.0 / (Δ + V)
        ∂y∂ωg = 0.0
        return ∂yg, ∂y∂ωg
    end

    function Z₀_and_∂μZ₀(y::Real, ω::Real, v::Real; rtol::Real, Δ::Real = 1.0)
        """
        Corresponds to the partition function of the Gaussian teacher 
        Here Δ is the variance of the teacher
        """
        Z₀   = (- 0.5 * (y - ω)^2 / (Δ + v)).exp() / sqrt(2.0 * π * (Δ + v))
        ∂μZ₀ = (y - ω) / (Δ + v) * exp(- 0.5 * (y - ω)^2 / (Δ + v)) / sqrt(2.0 * π * (Δ + v))
        return Z₀, ∂μZ₀
    end

end # module RidgeChannel