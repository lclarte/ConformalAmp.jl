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

    """
    fn z0(&self, y : f64, w : f64, v : f64) -> f64 {
            return (- 0.5 * (y - w).powi(2) / (self.variance + v)).exp() / (2.0 * PI * (self.variance + v)).sqrt();
        }

    fn dz0(&self, y : f64, w : f64, v : f64) -> f64 {
        return (y - w) / (v + self.variance) * (- 0.5 * (y - w).powi(2) / (self.variance + v)).exp() / (2.0 * PI * (self.variance + v)).sqrt();
        }
    """

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