"""
Checks that the leave-one-out estimators correspond obtained with GAMP are correct
by comparing them with "true" ERM estimators.
"""

using ConformalAmp
using LinearAlgebra
using Plots
using Revise
using StableRNGs: AbstractRNG, StableRNG

function compare_gamp_erm(model::String, d::Integer)
    α = 5.0
    n = ceil(Int, α * d)
    λ = 0.1

    if model == "logistic"
        problem = ConformalAmp.Logistic(α = α, λ = λ)
    else
        problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    end

    (; X, w, y) = ConformalAmp.sample_all(StableRNG(0), problem, n)

    (; xhat, vhat, ω) = ConformalAmp.gamp(problem, X, y; rtol=1e-5)
    x̂_cavities = ConformalAmp.get_cavity_means_from_gamp(problem, X, y, xhat, vhat, ω)

    x̂_loo = ConformalAmp.fit_leave_one_out(problem, X, y)
    x̂     = ConformalAmp.fit(problem, X, y)

    index = 1

    # the 1st line should be smaller than the 2nd because 1st line is an approximation error
    # while the 2nd is due to the leave one out
    # OBSERVATION : We need to set a good precitions to gamp in order to have a good approximation

    """
    min, max = 2.0 * minimum(x̂_loo[index, :] -  x̂_cavities[index, :]), 2.0 * maximum(x̂_loo[index, :] -  x̂_cavities[index, :])
    steps = (max - min) / 20
    pl = stephist(x̂_loo[index, :] -  x̂_cavities[index, :]; bins=min:steps:max, label = "Diff. loo / gamp cavities")
    stephist!(pl, x̂_loo[index, :] - x̂;  bins=min:steps:max, label = "Diff. loo / erm")
    display(pl)
    """

    # must be small 
    relative_differences = abs.(x̂_loo[index, :] -  x̂_cavities[index, :]) ./ abs.(x̂_loo[index, :] - x̂)
    pl = stephist(relative_differences; bins=0:0.05:2, label = "Relative diff. loo / gamp cavities")
    display(pl)
end

function compare_at_interpolation(model::String, d::Integer; rng::AbstractRNG = StableRNG(0))
    """
    Check that interpolation, the residuals are 0 and look at the residuals
         1) given by refitting the estimator and 
         2) given by gamp cavities 
    check that they change a lot compared to the approximation of amp
    """
    # small value of alpha to interpolate
    α = 0.5
    n = ceil(Int, α * d)
    if model == "ridge"
        # warning : if we take λ = 0 there are numerical issues for some reason
        # prolly because we don't take the min l2-norm estimator
        λ = 1e-6
    elseif model == "logistic"
        λ = 1e-6
    else 
        error("model not recognized")
    end

    problem = model == "logistic" ? ConformalAmp.Logistic(α = α, λ = λ) : ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, n)
    
    (; xhat, vhat, ω) = ConformalAmp.gamp(problem, X, y; rtol=1e-5)

    # run the leave one out and compare the difference in the predictions 
    ŵ = ConformalAmp.fit(problem, X, y)
    residuals = y .- ConformalAmp.predict(problem, ŵ, X)

    # should be concentrated around 0 
    pl = stephist(residuals; bins=100, label = "Residuals erm")
    display(pl)

    ŵ_loo = ConformalAmp.fit_leave_one_out(problem, X, y)
    ŵ_cavities = ConformalAmp.get_cavity_means_from_gamp(problem, X, y, xhat, vhat, ω, rtol = 1e-5)


    ŷ_loo  = diag(ConformalAmp.predict(problem, ŵ_loo, X))
    ŷ_cavities = diag(ConformalAmp.predict(problem, ŵ_cavities, X))
    residuals_loo = ŷ_loo .- y
    residuals_cavities = ŷ_cavities .- y 

    # normally the residuals should be the same between loo and .fit
    # note that the residuals roughly correspond to the generalisation error 
    
    # pl = scatter(residuals_loo, residuals_cavities, label = "Residuals loo vs cavities")
    # display(pl)
end

# @time compare_gamp_erm("ridge", 1000)
compare_at_interpolation("ridge", 500, rng = StableRNG(0))