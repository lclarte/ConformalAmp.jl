# TODO : Lire le papier de Song mei pour le channel correspondant a la pinball loss 

module PinballChannel
    # corresponds to the loss function 
    # l(y, ŷ) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)

    function loss(y::Real, ŷ::Real, q::Real)
        return q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)
    end

    function loss_der(y::Real, ŷ::Real, q::Real)
        return y > ŷ ? -q : (1 - q)
    end

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; q::Real, rtol::Real = 1e-3, Δ::Real = 1.0)
    """
            Δ is the denominator in the square loss 
        """
        objective(z::Real)   = abs2(z - ω) / (2V) +  loss(y, z, q)
        gradient(_, z::Real) = (z - ω) / V +  loss_der(y, z, q)
        hessian(_, z::Real) = inv(V)

        scalarobj = NLSolvers.ScalarObjective(; f=objective, g=gradient, h=hessian)
        optprob = NLSolvers.OptimizationProblem(scalarobj; inplace=false)
        
        init = ω
        solver = NLSolvers.LineSearch(NLSolvers.Newton())
        options = NLSolvers.OptimizationOptions(; x_reltol=rtol, x_abstol=0.0)
        res = NLSolvers.solve(optprob, init, solver, options)

        prox = res.info.solution
        ∂ωprox = inv(1 + V * logistic_loss_der2(y, prox))  # implicit function theorem

        gₒᵤₜ = (prox - ω) / V
        ∂ωgₒᵤₜ = (∂ωprox - 1) / V

        gₒᵤₜ   = nothing
        ∂ωgₒᵤₜ = nothing
        return gₒᵤₜ, ∂ωgₒᵤₜ 
    end

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; q::Real, rtol::Real = 1e-3, Δ::Real = 1.0)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        gₒᵤₜ, ∂ωgₒᵤₜ = zeros(length(y)), zeros(length(y))
        for i in eachindex(y)
            channel = gₒᵤₜ_and_∂ωgₒᵤₜ(y[i], ω[i], V[i]; q = q, rtol = rtol, Δ = Δ)
            gₒᵤₜ[i], ∂ωgₒᵤₜ[i] = channel[i]
        end
        return gₒᵤₜ, ∂ωgₒᵤₜ
    end

    function ∂ωgₒᵤₜ_and_∂ω∂ωgₒᵤₜ(y::AbstractVector, ω::AbstractVector, V::AbstractVector; rtol::Real, Δ::Real = 1.0)
        """
        Vectorized version of the channel function
        """
        @assert length(y) == length(ω) == length(V)
        ∂ωg   = nothing
        ∂ω∂ωg = nothing
        return ∂ωg, ∂ω∂ωg
    end

    ## 

    function ∂ygₒᵤₜ_and_∂y∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; rtol::Real, Δ::Real = 1.0)
        """
        Needed for full cp
        """
        @assert length(y) == length(ω) == length(V)
        ∂yg   = nothing
        ∂y∂ωg = nothing
        return ∂yg, ∂y∂ωg
    end

    function Z₀_and_∂μZ₀(y::Real, ω::Real, v::Real; rtol::Real, Δ::Real = 1.0)
        """
        Corresponds to the partition function of the Gaussian teacher 
        Here Δ is the variance of the teacher
        """
        Z₀   = nothing
        ∂μZ₀ = nothing
        return Z₀, ∂μZ₀
    end
end