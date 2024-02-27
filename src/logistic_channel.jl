module LogisticChannel

    using NLSolvers
    using LinearAlgebra
    using LogExpFunctions: log1pexp

    logistic(x::Real) = 1 / (1 + exp(-x))
    logistic_der(x::Real) = logistic(x) * (1 - logistic(x))

    logistic_loss(y::Real, z::Real) = log1pexp(-y * z)
    logistic_loss_der(y::Real, z::Real) = -y * logistic(-y * z)
    logistic_loss_der2(y::Real, z::Real) = y^2 * logistic_der(-y * z)

    function gₒᵤₜ_and_∂ωgₒᵤₜ(y::Real, ω::Real, V::Real; rtol::Real)
        """
        Channel function for the logistic regression
        """
        objective(z::Real) = abs2(z - ω) / (2V) +  logistic_loss(y, z)
        gradient(_, z::Real) = (z - ω) / V +  logistic_loss_der(y, z)
        hessian(_, z::Real) = inv(V) +  logistic_loss_der2(y, z)

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

    function Z₀_and_∂μZ₀(
        y::Real, μ::Real, v::Real; rtol::Real
    )
        """
        Corresponds to the partition function of the logistic teacher 
        """
        function Z₀_and_∂μZ₀_integrand(u::Real)
            z = u * sqrt(v) + μ
            σ = logistic(y * z)
            σ_der = σ * (1 - σ)
            res = SVector(σ, y[1] * σ_der) * normpdf(u)
            return res
        end
    
        bound = 10.0
        double_integral, err = quadgk(
            Z₀_and_∂μZ₀_integrand_same_labels, -bound, bound; rtol
        )

        Z₀ = double_integral[1]
        ∂μZ₀ = double_integral[2]
        return Z₀, ∂μZ₀
    end

end # module LogisticChannel