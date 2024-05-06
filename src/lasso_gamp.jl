"""
lasso_gamp.jl
Channel and prior functions used for GAMP with Lasso
The channel function is shared with ridge regression
"""

function prior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    """
    for l1 penalty
    """
    (; λ) = problem

    function fa(b_, A_) # sigma = 1 / A > 0, r = b / A
        if abs(b_) < λ
            return 0.0
        elseif b_ > λ
            return (b_ - λ) / A_
        else
            return (b_ + λ) / A_
        end
    end

    function fv(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return 1.0 / A_
       end
    end

    return fa.(b, A), fv.(b, A)
end

function ∂bprior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    (; λ) = problem

    function ∂bfa(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return 1.0 / A_
        end
    end

    return ∂bfa.(b, A), zeros(size(b))
end

function ∂Aprior(b::AbstractVector, A::AbstractVector, problem::Lasso)
    (; λ) = problem

    function ∂Afa(b_, A_) # sigma = 1 / A > 0, r = b / A
        if abs(b_) < λ
            return 0.0
        elseif b_ > λ
            return -(b_ - λ) / A_^2.
        else
            return -(b_ + λ) / A_^2.
        end
    end
    
    function ∂Afv(b_, A_)
        if abs(b_) < λ
            return 0.0
        else
            return - 1.0 / A_^2.
       end
    end

    return ∂Afa.(b, A), ∂Afv.(b, A)
end