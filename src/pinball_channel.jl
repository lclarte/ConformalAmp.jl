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
        # Since the loss is affine by part, the second derivative of the loss is 0 almost everywhere 
        # and ∂ωprox = 1.0 and ∂ωgₒᵤₜ = 0 ? 
        # ∂ωprox = inv(1 + V * logistic_loss_der2(y, prox)) 
        # ∂ωgₒᵤₜ = (∂ωprox - 1) / V

        gₒᵤₜ = (prox - ω) / V
        ∂ωgₒᵤₜ = 0.0
        return gₒᵤₜ, ∂ωgₒᵤₜ
        """
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
        ∂ωg   = 0.0
        ∂ω∂ωg = 0.0
        return ∂ωg, ∂ω∂ωg
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