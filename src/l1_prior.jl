"""
l1_prior.jl
Channel and prior functions used for GAMP with Lasso
The channel function is shared with ridge regression
"""

function prior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    """
    for l1 penalty
    """
    (; λ) = problem
    ε = 1e-5
        
    function fa(b_, A_) # sigma = 1 / A > 0, r = b / A
        if abs(b_) < λ
            return 0.0
        elseif b_ > λ
            return (b_ - λ) / (A_ + ε)
        else
            return (b_ + λ) / (A_ + ε)
        end
    end

    function fv(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return 1.0 / (A_ + ε)
       end
    end

    return fa.(b, A), fv.(b, A)
end

function ∂bprior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    (; λ) = problem
    ε = 1e-5

    function ∂bfa(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return 1.0 / (A_ + ε)
        end
    end

    return ∂bfa.(b, A), zeros(size(b))
end

function ∂Aprior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    (; λ) = problem
    ε = 1e-5

    function ∂Afa(b_, A_) # sigma = 1 / A > 0, r = b / A
        if abs(b_) < λ
            return 0.0
        elseif b_ > λ
            return - (b_ - λ) / (A_^2. + ε)
        else
            return - (b_ + λ) / (A_^2. + ε)
        end
    end
    
    function ∂Afv(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return - 1.0 / (A_^2. + ε)
       end
    end

    return ∂Afa.(b, A), ∂Afv.(b, A)
end

### Case of Laplace prior (Bayesian setting)

function prior(b::AbstractVector, A::AbstractVector, problem::BayesOptimalLasso)
    """
    for l1 penalty
    """
    function ∂RlogZ(Σ::AbstractVector, R::AbstractVector, λ::Real)
        # convert the Python code above in Julia
        tmp = sqrt.(2.0 * Σ)
        Rm, Rp = (R .- λ * Σ), (R .+ λ * Σ)
        return - λ * (1.0 .+ erf.(Rm ./ tmp) - exp.(2 * R * λ) .* erfc.(Rp ./ tmp)) ./ (1.0 .+ erf.(Rm ./ tmp) .+ exp.(2.0 * λ * R) .* erfc.(Rp ./ tmp))
    end

    function ∂∂RlogZ(Σ::AbstractVector, R::AbstractVector, λ::Real)
        """tmp = np.sqrt(2.0 * Sigma)
        tmp_exp = np.exp(2 * R * lambda_)
        tmp_pi = np.sqrt(2.0 / np.pi)
        Rm, Rp = R - lambda_ * Sigma, R + lambda_ * Sigma
        return 2 * lambda_ * tmp_exp * ( - np.exp(-Rp**2 / tmp**2) * tmp_pi + ( 2 * lambda_ * np.sqrt(Sigma) - tmp_pi * np.exp(-Rm**2 / tmp**2)) * erfc(Rp / tmp) + \
                erf(Rm / tmp) * (2 * lambda_ * np.sqrt(Sigma) * erfc(Rp / tmp) - np.exp(-Rp**2 / tmp**2) * tmp_pi)) / \
                (np.sqrt(Sigma) * (1.0 + erf(Rm / tmp) + tmp_exp * erfc(Rp / tmp) )**2)"""
        tmp = sqrt.(2.0 * Σ)
        tmp_exp = exp.(2 * R * λ)
        tmp_pi = sqrt.(2.0 / π)
        Rm, Rp = R .- λ * Σ, R .+ λ * Σ
        return 2 * λ * tmp_exp .* ( - exp.(- Rp.^2 ./ tmp.^2) .* tmp_pi + ( 2 * λ * sqrt.(Σ) .- tmp_pi .* exp.(- Rm.^2 ./ tmp.^2)) .* erfc.(Rp ./ tmp) .+ 
                erf.(Rm ./ tmp) .* (2 * λ * sqrt.(Σ) .* erfc.(Rp ./ tmp) .- exp.(- Rp.^2 ./ tmp.^2) .* tmp_pi)) ./ 
                (sqrt.(Σ) .* (1.0 .+ erf.(Rm ./ tmp) .+ tmp_exp .* erfc.(Rp ./ tmp) ).^2)
    end

    Σ = 1.0 ./ A
    R = b ./ A
    fa = Σ .* ∂RlogZ(Σ, R, problem.λ) + R
    fv = Σ.^2 .* ∂∂RlogZ(Σ, R, problem.λ) + Σ
    return fa, fv
end