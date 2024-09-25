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
using Statistics

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

    (; x̂, v̂, ω) = ConformalAmp.gamp(problem, X, y; rtol=1e-5)
    x̂_cavities = ConformalAmp.get_cavity_means_from_gamp(problem, X, y, x̂, v̂, ω)

    x̂_loo = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())
    x̂     = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())

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

function compare_residuals(model::String, d::Integer; rng::AbstractRNG = StableRNG(0), λ::Real = 1.0)
    """
    Check that interpolation, the residuals are 0 and look at the residuals
         1) given by refitting the estimator and 
         2) given by gamp cavities 
    check that they change a lot compared to the approximation of amp
    """
    # small value of alpha to interpolate
    α = 0.5
    n = ceil(Int, α * d)

    problem = model == "lasso" ? ConformalAmp.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0) : ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    (; x̂, v̂, ω) = ConformalAmp.gamp(problem, X, y; rtol=1e-4)

    # run the leave one out and compare the difference in the predictions 
    ŵ_loo = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())

    gamp_result = ConformalAmp.

    # Plot the residuals on the training data
    ŷ_loo              = diag(ConformalAmp.predict(problem, ŵ_loo, X))
    ŷ_cavities         = diag(ConformalAmp.predict(problem, ŵ_cavities, X))
    residuals_loo      = y .- ŷ_loo 
    residuals_cavities = y .- ŷ_cavities
    pl = scatter(residuals_loo, residuals_cavities, label = "Residuals loo vs cavities",
                    xlabel = "Residuals loo", ylabel = "Residuals cavities of amp")
    title!("λ = $λ, α = $α, d = $d")
    display(pl)
    # savefig("plots/interpolation_residuals_loo_ridge.png")
end

function compare_fcp_last_label_change(d::Integer; rng::AbstractRNG = StableRNG(0), problem_type::String = "ridge")
    """
    Compare the residuals for the leave one out when we change the label of one sample (the n-th one)
    """
    δy_list = vcat([0.0], -10.0:1.0:10.0)
    
    λ = 1.0
    α = 0.5
    n = ConformalAmp.get_n(α, d)
    
    residuals_list_erm  = fill(0.0, (length(δy_list), n))
    residuals_list_gamp = fill(0.0, (length(δy_list), n))

    if problem_type == "ridge"
        problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    elseif problem_type == "lasso"
        problem = ConformalAmp.Lasso(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    else
        error("problem type not recognized")
    end
    
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
    index = n
    pl = scatter(δy_list[2:end],   residuals_list_gamp[2:end, n], xaxis = "δy", yaxis = "Residuals", label = "residuals with GAMP LOO")
    pl = scatter!(δy_list[2:end], residuals_list_erm[2:end, n], label = "residuals with ERM LOO")
    title!("Residuals for the $n-th sample, λ = $λ, α = $α, d = $d")
    display(pl) 

    # PLOT THE HISTOGRAM (AT δy constant) of the relative difference of residuals 
    # i.e (residual_erm - residual_gamp) / residual_gamp
    # println("Plottingh relative differences for δy = $(δy_list[2])")
    # relative_differences = abs.(residuals_list_erm[2, :] .- residuals_list_gamp[2, :]) ./ residuals_list_erm[2, :]
    # pl = stephist(relative_differences; bins=-2.0:0.2:2.0, label = "Relative diff. loo / gamp cavities")
    # display(pl)
end

## Here, we'll plot 1) the density of the changes in the residuals when we change the label of the last sample
# and 2) the relative difference of the residuals given by GAMP and ERM

function plot_residuals_wrt_last_label(problem::ConformalAmp.RegressionProblem, d::Integer; rng::AbstractRNG=StableRNG(0), δy::Real = 0.0)
    n = ConformalAmp.get_n(problem.α, d)
    δy_list = [0.0, δy]
    
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    ref_y_n = y[n]

    predictions_list_erm  = fill(0.0, (length(δy_list), n))
    predictions_list_gamp = fill(0.0, (length(δy_list), n))

    for i in (eachindex(δy_list))
        δy = δy_list[i]
        y[n] = ref_y_n + δy

        ŵ_erm  = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())
        ŵ_gamp = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5))
        
        predictions_list_erm[i, :]  = diag(ConformalAmp.predict(problem, ŵ_erm,  X))
        predictions_list_gamp[i, :] = diag(ConformalAmp.predict(problem, ŵ_gamp, X))
        
        if i > 1
            # do two subplots where the 1st one is the histogream of ERM residuals and the 2nd one is the difference between GAMP and ERM
            relative_difference = abs.(predictions_list_erm[i, :] - predictions_list_erm[1, :]) ./ abs.(predictions_list_erm[1, :])
            relative_difference_erm_gamp = abs.(predictions_list_erm[i, :] - predictions_list_gamp[i, :]) ./ abs.(predictions_list_erm[i, :])
            pl = stephist(relative_difference; bins=0:0.01:1.0, label = "Relat. diff. predictions",
                            xlabel = "Difference", ylabel="Density", normalize=:density)
            stephist!(pl, relative_difference_erm_gamp; bins=0.0:0.01:2.0, label = "Relat. diff. erm / gamp predictions", normalize=:density)
            # set the title
            title!("$(typeof(problem)), δy = $δy, d = $d, λ = $λ")
            display(pl)
        end
    end
end

# alternative : for one or two random samples and the last sample (that we change the label), plot the residuals as a function of δy 

function plot_residuals_wrt_last_label_single_samples(problem::ConformalAmp.RegressionProblem, d::Integer; 
    rng::AbstractRNG=StableRNG(0), δy_max::Real = 5.0, δy_step::Real = 0.25, run_erm::Bool = true)
    n = ConformalAmp.get_n(problem.α, d)
    δy_list = 0.0:δy_step:δy_max
    
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    ref_y_n = y[n]

    predictions_list_erm  = fill(0.0, (length(δy_list), n))
    predictions_list_gamp = fill(0.0, (length(δy_list), n))

    for i in (eachindex(δy_list))
        δy = δy_list[i]
        y[n] = ref_y_n + δy

        ŵ_gamp = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5))
        predictions_list_gamp[i, :] = diag(ConformalAmp.predict(problem, ŵ_gamp, X))
        
        if run_erm
            ŵ_erm  = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())
            predictions_list_erm[i, :]  = diag(ConformalAmp.predict(problem, ŵ_erm , X))
        end
        
    end

    # last index is the one for which we change the label
    indices = [1, 2, 3, 4, 5]
    colors = [:red, :blue, :green, :orange, :purple]

    # for the last sample, the residual is linear in δy because the output for xₙ is constant as it is held out

    pl = plot(xlabel = "δy", ylabel = "ŷ(δy) - ŷ(δy = 0)", title = "$(typeof(problem)), d = $d, λ = $λ")

    for index in indices
        if run_erm
            plot!(δy_list, predictions_list_erm[:, index] .- predictions_list_erm[1, index], linecolor=colors[index], label="")
        end
        plot!(δy_list, predictions_list_gamp[:, index] .- predictions_list_gamp[1, index], linestyle=:dash, linecolor=colors[index], label="")
    end

    display(pl)
end

function plot_label_change_wrt_d(problem::ConformalAmp.RegressionProblem, d_list::Vector{Int}, δy::Real; rng::AbstractRNG=StableRNG(0))
    """
    With δy being fixed, plot the average difference | ŷ - y(δy = 0) | over the train samples as a function of the dimension.
    """

    differences   = []
    ŵ_differences = []
    v̂_differences = []
    V_differences = []
    ω_differences = []

    method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5)
    # method = ConformalAmp.ERM()

    for d in ProgressBar(d_list)

        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        n = ConformalAmp.get_n(problem.α, d)

        ref_y_n = y[n]

        # fit leave one out labels without chaning last label
        ŵ_1      = ConformalAmp.fit(problem, X, y, method)
        if method isa ConformalAmp.GAMP
            gamp_result_1 = ConformalAmp.gamp(problem, X, y)
            v̂_1           = gamp_result_1.v̂
            ω_1           = gamp_result_1.ω
            V_1           = gamp_result_1.V
        end
        ŵ_gamp = ConformalAmp.fit_leave_one_out(problem, X, y, method)
        predictions_gamp_1 = diag(ConformalAmp.predict(problem, ŵ_gamp, X))
        # fit leave one out labels with changing last label
        
        y[n] = ref_y_n + δy
        ŵ_2      = ConformalAmp.fit(problem, X, y, method)
        ŵ_gamp = ConformalAmp.fit_leave_one_out(problem, X, y, method)
        if method isa ConformalAmp.GAMP
            gamp_result_2 = ConformalAmp.gamp(problem, X, y)
            v̂_2            = gamp_result_2.v̂
            ω_2            = gamp_result_2.ω
            V_2            = gamp_result_2.V
            push!(v̂_differences, std(v̂_2 - v̂_1))
            push!(V_differences, std(V_2 - V_1))
            push!(ω_differences, std(ω_2 - ω_1))
        end
        
        push!(ŵ_differences, std(ŵ_2 - ŵ_1))

        predictions_gamp_2 = diag(ConformalAmp.predict(problem, ŵ_gamp, X))
        push!(differences, mean(abs.(predictions_gamp_2 - predictions_gamp_1)))

    end

    # compute the slope of log(differences) w.r.t log(d_list)
    x_, y_ = log.(d_list), log.(differences)
    slope = (mean(x_ .* y_) - mean(x_) * mean(y_)) / (mean(x_ .* x_) - (mean(x_))^2)
    slope_ŵ = (mean(x_ .* log.(ŵ_differences)) - mean(x_) * mean(log.(ŵ_differences))) / (mean(x_ .* x_) - (mean(x_))^2)
    
    pl = plot(d_list, differences, xlabel = "d", ylabel="log difference", title = "$(typeof(problem)), δy = $δy, α = $(problem.α), λ = $(problem.λ)", 
            yaxis=:log, xaxis=:log, label="mean(| ŷᵢ - yᵢ(δy = 0)|); Slope : $(round(slope, digits=2))")
    plot!(pl, d_list, ŵ_differences, label="| ŵ - ŵ(δy=0) | / d); Slope : $(round(slope_ŵ, digits=2))")
    if method isa ConformalAmp.GAMP
        slope_v̂ = (mean(x_ .* log.(v̂_differences)) - mean(x_) * mean(log.(v̂_differences))) / (mean(x_ .* x_) - (mean(x_))^2)
        plot!(pl, d_list, v̂_differences, label="|v̂ - v̂(δy=0) | / d); Slope : $(round(slope_v̂, digits=3))")

        slope_V = (mean(x_ .* log.(V_differences)) - mean(x_) * mean(log.(V_differences))) / (mean(x_ .* x_) - (mean(x_))^2)
        plot!(pl, d_list, V_differences, label="|V - V(δy=0) | / d); Slope : $(round(slope_V, digits=3))")

        slope_ω = (mean(x_ .* log.(ω_differences)) - mean(x_) * mean(log.(ω_differences))) / (mean(x_ .* x_) - (mean(x_))^2)
        plot!(pl, d_list, ω_differences, label="|ω - ω(δy=0) | / d); Slope : $(round(slope_ω, digits=3))")
    end
    # save figure in plots/difference_last_label_wrt_d.png
    # savefig(pl, "plots/difference_last_label_wrt_d.png")
    display(pl)
end

function plot_histogram_weights_label_change_wrt_d(problem::ConformalAmp.RegressionProblem, d_list::Vector{Int}, δy::Real; rng::AbstractRNG=StableRNG(0))
    """
    Plot the histogram of how much the weights change when we change the label of the last sample
    with ERM and GAMP. This is to check that the magnitude of the change is of order 1 / √n
    """
    # method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5)
    method  = ConformalAmp.ERM()

    pl = plot()
    std_differences = []

    for d in ProgressBar(d_list)

        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        n = ConformalAmp.get_n(problem.α, d)

        ref_y_n = y[n]
        # fit leave one out labels without chaning last label
        ŵ_1      = ConformalAmp.fit(problem, X, y, method)
        ŵ_gamp = ConformalAmp.fit_leave_one_out(problem, X, y, method)
        # fit leave one out labels with changing last label
        y[n] = ref_y_n + δy
        ŵ_2      = ConformalAmp.fit(problem, X, y, method)
        ŵ_gamp = ConformalAmp.fit_leave_one_out(problem, X, y, method)

        differences = sqrt(d) .* (ŵ_2 - ŵ_1)

        push!(std_differences, std(differences))

        stephist!(pl, differences, label="d = $d", bins=100, normalize=:pdf)
    end

    for diff in std_differences
        println("d = $d, std of differences : ", diff)
    end

    display(pl)
end

# ========== functions calls 

# compare_at_interpolation("ridge", 500, rng = StableRNG(0))
# compare_fcp_last_label_change(400, rng = StableRNG(2), problem_type = "lasso")

λ = 1.0
d = 100
seed = 10

# plot_residuals_wrt_last_label(ConformalAmp.Ridge(α = 0.5, Δ = 1.0, λ = λ, Δ̂ = 1.0), d,  rng = StableRNG(seed), δy = 5.0)

# plot_residuals_wrt_last_label_single_samples(ConformalAmp.Ridge(α = 0.5, Δ = 1.0, λ = λ, Δ̂ = 1.0), d, rng = StableRNG(seed), δy_step = 1.0, δy_max = 5.0, run_erm = false)

# plot_label_change_wrt_d(ConformalAmp.Lasso(α = 0.5, Δ = 1.0, λ = λ, Δ̂ = 1.0), Vector{Int}(1000:100:5000), 5.0; rng = StableRNG(20))

# plot_histogram_weights_label_change_wrt_d(ConformalAmp.Ridge(α = 0.5, Δ = 1.0, λ = λ, Δ̂ = 1.0), Vector{Int}(500:500:2000), 5.0)

compare_gamp_erm("ridge", 500)