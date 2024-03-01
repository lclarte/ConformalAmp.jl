"""
Checks that the leave-one-out estimators correspond obtained with GAMP are correct
by comparing them with "true" ERM estimators.
"""

using ConformalAmp
using LinearAlgebra
using Plots
using ProgressBars
using Revise
using StableRNGs: AbstractRNG, StableRNG

function compare_gamp_erm(model::String, d::Integer)
    """
        Function to compare the leave one out of ERM and GAMP and check that GAMP
        approximates well the leave one out estimators of ERM
    """
    α = 5.0
    λ = 0.1

    if model == "logistic"
        problem = ConformalAmp.Logistic(α = α, λ = λ)
    else
        problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    end

    (; X, w, y) = ConformalAmp.sample_all(StableRNG(0), problem, d)

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
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    
    (; xhat, vhat, ω) = ConformalAmp.gamp(problem, X, y; rtol=1e-5)

    # run the leave one out and compare the difference in the predictions 
    ŵ = ConformalAmp.fit(problem, X, y)
    residuals = y .- ConformalAmp.predict(problem, ŵ, X)

    # should be concentrated around 0 
    pl = stephist(residuals; bins=100, label = "Residuals erm")
    title!("Residuals of ŵ, λ = $λ, α = $α, d = $d")
    savefig("plots/interpolation_residuals_ridge.png")

    ŵ_loo = ConformalAmp.fit_leave_one_out(problem, X, y)
    ŵ_cavities = ConformalAmp.get_cavity_means_from_gamp(problem, X, y, xhat, vhat, ω, rtol = 1e-5)

    # Plot the residuals on the training data
    ŷ_loo              = diag(ConformalAmp.predict(problem, ŵ_loo, X))
    ŷ_cavities         = diag(ConformalAmp.predict(problem, ŵ_cavities, X))
    residuals_loo      = y .- ŷ_loo 
    residuals_cavities = y .- ŷ_cavities
    pl = scatter(residuals_loo, residuals_cavities, label = "Residuals loo vs cavities",
                    xlabel = "Residuals loo", ylabel = "Residuals cavities of amp")
    title!("λ = $λ, α = $α, d = $d")
    savefig("plots/interpolation_residuals_loo_ridge.png")

    """
    # not very intereting because for Full CP we treat the test sample as "training" data
    # but we plot the predictions of the loo estimators for 1 new test sample
    ntest = 1
    Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    # stores the prediction for every leave one out estiamtor on the new test sampel
    ŷ_loo_test = ConformalAmp.predict(problem, ŵ_loo, Xtest)[1, :]
    ŷ_cavities_test = ConformalAmp.predict(problem, ŵ_cavities, Xtest)[1, :]
    pl = scatter(ŷ_loo_test, ŷ_cavities_test, label = "loo vs cavities",
                    xlabel = "loo", ylabel = "cavities", title = "λ = $λ, α = $α, d = $d")
    display(pl)
    """
end

function compare_fcp_interpolation(d::Integer; rng::AbstractRNG = StableRNG(0))
    """
    Compare the residuals for the leave one out when we change the label of one sample (the n-th one)
    """
    δy_list = vcat([0.0], -10.0:2.5:10.0)
    
    λ = 2.0
    α = 0.5
    n = ConformalAmp.get_n(α, d)
    
    residuals_list_erm  = fill(0.0, (length(δy_list), n))
    residuals_list_gamp = fill(0.0, (length(δy_list), n))

    problem = ConformalAmp.Lasso(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    ref_y_n = y[n]

    for i in ProgressBar(eachindex(δy_list))
        δy = δy_list[i]
        y[n] = ref_y_n + δy

        ŵ_erm  = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())
        ŵ_gamp = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5))
        
        residuals_list_erm[i, :]  = y .- diag(ConformalAmp.predict(problem, ŵ_erm,  X))
        residuals_list_gamp[i, :] = y .- diag(ConformalAmp.predict(problem, ŵ_gamp, X))
    end

    # PLOT THE RESIDUALS WHEN CHANGING THE LABEL OF LAST SAMPLE
    pl = scatter(δy_list[2:end],   residuals_list_gamp[2:end, 1], xaxis = "δy", yaxis = "Residuals", label = "residuals with GAMP LOO")
    pl = scatter!(δy_list[2:end], residuals_list_erm[2:end, 1], label = "residuals with ERM LOO")
    title!("Residuals for the 1st sample, λ = $λ, α = $α, d = $d")
    display(pl) 

    # PLOT THE HISTOGRAM (AT δy constant) of the relative difference of residuals 
    # i.e (residual_erm - residual_gamp) / residual_gamp
    # println("Plottingh relative differences for δy = $(δy_list[2])")
    # relative_differences = abs.(residuals_list_erm[2, :] .- residuals_list_gamp[2, :]) ./ residuals_list_erm[2, :]
    # pl = stephist(relative_differences; bins=-2.0:0.2:2.0, label = "Relative diff. loo / gamp cavities")
    # display(pl)
end

# compare_at_interpolation("ridge", 500, rng = StableRNG(0))
compare_fcp_interpolation(300, rng = StableRNG(0))
