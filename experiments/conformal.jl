"""
Script to compare the intervals that are returned by the LOO _vs_ conformal prediction 
TODO : Comprendre a quel point on peut diminuer la tolerance pour "Taylor-AMP" de maniere
a accelerer les calculs
TODO 2 : Voir si on peut pas simplifier les calculs, notamment pour la penalité L₂ où
certaines derivees sont nulles => checker quels Δ⋅ sont negligeables 
TODO 3 : Implémenter quantile regression
"""

using ConformalAmp
using LinearAlgebra
using Plots
using ProgressBars
using StableRNGs: StableRNG
using Statistics

# UTILITY FUNCTION 

function slope_of_log(x_::AbstractVector, y_::AbstractVector)
    return (mean(log.(x_) .* log.(y_)) - mean(log.(x_)) * mean(log.(y_))) / (mean(log.(x_) .* log.(x_)) - (mean(log.(x_)))^2)
end



function compare_jacknife_fcp_confidence_intervals()
    # define the problem and sample the data
    d   = 500
    α = 0.5
    λ = 1e-6
    coverage = 0.9

    problem = ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)

    rng = StableRNG(0)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    xtest = ConformalAmp.sample_data_any_n(rng, d, 1)[1, :]

    jplus_confidence_set = ConformalAmp.get_confidence_interval(problem, X, y, xtest, ConformalAmp.JacknifePlus(coverage = coverage), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))
    fcp_confidence_set = ConformalAmp.get_confidence_interval(problem, X, y, xtest, ConformalAmp.FullConformal(coverage = coverage, δy_range = 0:0.1:2.0), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))

    println("J+  :",  minimum(jplus_confidence_set), " ", maximum(jplus_confidence_set))
    println("FCP : ", minimum(fcp_confidence_set), " ", maximum(fcp_confidence_set)) 
end

#####

function test_gamp_order_one()
    """
    Compare the "exact" perturbation of x̂ w.r.t δy with the Order 1 approximation for a fixed dimension,
    and as a function of δy
    """
    d = 1000
    α = 0.5
    λ = 1e-4

    problem = ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)
    n = ConformalAmp.get_n(α, d)
    (; X, w, y) = ConformalAmp.sample_all(StableRNG(0), problem, d)

    method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-6)

    result  = ConformalAmp.gamp(problem, X, y; max_iter =  method.max_iter, rtol =  method.rtol)
    Δresult = ConformalAmp.compute_order_one_perturbation_gamp(problem, X, y, result; max_iter = method.max_iter, rtol = method.rtol, δy = 1.0)

    # compute the estimator by a refitting of AMP and compare the results

    differences = []
    differences_wrt_0 = []

    y_ref = y[n]
    δy_range = (10.0).^(-1:0.24:2)

    for δy in ProgressBar(δy_range)
        # refit GAMP
        y[n] = y_ref + δy
        x̂_refit = ConformalAmp.fit(problem, X, y, method)
        x̂_taylor= result.x̂ + δy .* Δresult.x̂
        difference = norm(x̂_refit - x̂_taylor, 2) / d
        diff_wrt_0 = norm(x̂_refit - result.x̂, 2) / d
        push!(differences, difference)
        push!(differences_wrt_0, diff_wrt_0)
    end

    # It's "normal" that the differences are proportional to δy (maybe it should be δy² because we compute the order 1 expansion ?)
    pl = plot(δy_range, differences, xlabel="δy", ylabel="difference", label="refit - taylor", yaxis=:log)
    plot!(δy_range, differences_wrt_0, label="x̂(δy) - x̂(δy = 0)")
    display(pl)
end

function test_gamp_order_one_wrt_d(problem::ConformalAmp.Problem, d_range::AbstractRange; 
                                δy::Real = 5.0, rng_num::Integer = 0, avg_num::Integer = 1, var::String = "x̂")
    """
    Compute the std of the following differences
        * x̂(δy) - x̂(0) computed by refitting 
        * x̂(δy) - x̂(0) computed by Taylor 1 expansion
        * x̂(δy) - x̂'(δy) computed by Taylor 1 expansion vs the refitting
    as a fuinction of the dimension and check that both converge to 0 as 1/ √(d) and that the diff. between the 
    2 quantities is smaller in order than 1 / √d

    Also record the time taken
    """
    rng = StableRNG(rng_num)

    method = ConformalAmp.GAMP(max_iter = 200, rtol = 1e-8)

    # differences_refit and differences_taylor don't have to be close to 0
    # but must be close to one another
    differences_refit = Dict(
        "x̂" => fill(0.0, (avg_num, length(d_range))),
        "v̂" => fill(0.0, (avg_num, length(d_range))),
        "ω" => fill(0.0, (avg_num, length(d_range))),
        "V" => fill(0.0, (avg_num, length(d_range)))
    )

    # also compute the difference between two ERM fit because 
    # for Lasso I think refitting GAMP is imprecise
    differences_refit_erm = Dict(
        "x̂" => fill(0.0, (avg_num, length(d_range))),
    )

    differences_taylor = Dict(
        "x̂" => fill(0.0, (avg_num, length(d_range))),
        "v̂" => fill(0.0, (avg_num, length(d_range))),
        "ω" => fill(0.0, (avg_num, length(d_range))),
        "V" => fill(0.0, (avg_num, length(d_range)))
    )

    differences_refit_taylor = Dict(
        "x̂" => fill(0.0, (avg_num, length(d_range))),
        "v̂" => fill(0.0, (avg_num, length(d_range))),
        "ω" => fill(0.0, (avg_num, length(d_range))),
        "V" => fill(0.0, (avg_num, length(d_range)))
    )

    times_refit  = fill(0.0, (avg_num, length(d_range)))
    times_taylor = fill(0.0, (avg_num, length(d_range)))
    
    for i in 1:avg_num
        for i_d in ProgressBar(eachindex(d_range))
            d = d_range[i_d]
            n = ConformalAmp.get_n(α, d)
            (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
            result  = ConformalAmp.gamp(problem, X, y; max_iter =  method.max_iter, rtol =  method.rtol)
            result_erm = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
            
            debut_taylor = time()
            # ∂ResultGamp × δy
            Δresult = ConformalAmp.compute_order_one_perturbation_gamp(problem, X, y, result; max_iter = method.max_iter, rtol = method.rtol, δy = δy)
            fin_taylor = time()
            
            y[n] = y[n] + δy
            debut_refit = time()
            result_refit = ConformalAmp.gamp(problem, X, y; max_iter = method.max_iter, rtol =  method.rtol)
            fin_refit = time()

            result_erm_refit = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
            differences_refit_erm["x̂"][i, i_d] = std(result_erm - result_erm_refit)

            diff_refit_0 = result_refit - result
            diff_refit_taylor = result_refit - (result + Δresult)
            
            differences_refit["x̂"][i,i_d] = std(diff_refit_0.x̂)
            differences_taylor["x̂"][i,i_d]= std(Δresult.x̂)
            differences_refit_taylor["x̂"][i, i_d] = std(diff_refit_taylor.x̂)
            
            differences_refit["v̂"][i, i_d] = std(diff_refit_0.v̂)
            differences_taylor["v̂"][i, i_d] = std(Δresult.v̂)
            differences_refit_taylor["v̂"][i, i_d] = std(diff_refit_taylor.v̂)

            differences_refit["ω"][i, i_d] = std(diff_refit_0.ω)
            differences_taylor["ω"][i, i_d] = std(Δresult.ω)
            differences_refit_taylor["ω"][i, i_d] = std(diff_refit_taylor.ω)

            differences_refit["V"][i, i_d] = std(diff_refit_0.V)
            differences_taylor["V"][i, i_d] = std(Δresult.ω)
            differences_refit_taylor["V"][i, i_d] = std(diff_refit_taylor.ω)

            times_refit[i, i_d]  = fin_refit - debut_refit
            times_taylor[i, i_d] = fin_taylor - debut_taylor
        end 
    end
    #                     PLOT THE DIFF. BETWEEN ORIGINAL AND THE REFIT (=GROUND TRUTH) AND TAYLOR
    mean_differences_refit = Statistics.mean(differences_refit[var], dims=1)[1, :]
    mean_differences_taylor = Statistics.mean(differences_taylor[var], dims=1)[1, :]
    pl = plot(d_range, mean_differences_refit, xaxis = :log, yaxis = :log, label="Difference refit (GAMP)")
    plot!(d_range, mean_differences_taylor, label="Difference Taylor")

    if var == "x̂"
        mean_differences_erm_refit = Statistics.mean(differences_refit_erm[var], dims=1)[1, :]
        plot!(d_range, mean_differences_erm_refit, label="Diff. refit (ERM)")
    end

    title!(pl, "slope(refit)  = $(slope_of_log(d_range, mean_differences_refit)), 
                slope(taylor) = $(slope_of_log(d_range, mean_differences_taylor))")
    display(pl)

    #                     PLOT THE DIFFERENCE BETWEEN REFIT AND TAYLOR 
    # pl = plot(d_range, differences_refit_taylor["x̂"], xaxis = :log, yaxis = :log, 
    #         label = "$(slope_of_log(d_range, differences_refit_taylor))")
    # display(pl)
    
    #                     PLOT THE COMPUTATION TIME
    # pl = scatter(d_range, times_refit, xaxis = :log, yaxis = :log, label="Full refit")
    # scatter!(pl, d_range, times_taylor, label="Taylor")
    # display(pl)
end

function test_gamp_order_one_loo_wrt_d(problem::ConformalAmp.Problem, d_range::AbstractRange; δy::Real = 1.0, rng_num::Integer = 0)
    """
    Dans cette fonction on va voir si GAMP + Taylor + LOO (Algo 1) permet d'approximer le vrai LOO obtenu par ERM (Algo 2)
    Pour ce faire, en fonction de d, on va faire tourner Algo 1 et Algo 2 pour calculer la matrice d x n des LOO estimators,
    regarder la moyenne de la norme des differences (par exemple) et regarder le rythme a partir duquel elle decay
    """
    rng = StableRNG(rng_num)

    method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-3)
    small_δy = 0.1 # used so that the Taylor approximation is valid

    erm_taylor_difference_norm = []
    erm_gamp_difference_norm   = []
    ΔŴ_erm_norm     = []
    

    for i_d in ProgressBar(eachindex(d_range))
        d = d_range[i_d]
        n = ConformalAmp.get_n(α, d)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        ỹ = copy(y)
        ỹ[n] = y[n] + δy
        ỹ_gamp = copy(y)
        ỹ_gamp[n] = y[n] + small_δy

        # fit ERM twice
        Ŵ_erm   = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())
        Ŵ_erm_2 = ConformalAmp.fit_leave_one_out(problem, X, ỹ, ConformalAmp.ERM())
        ΔŴ_erm = Ŵ_erm_2 - Ŵ_erm 
        push!(ΔŴ_erm_norm, mean([std(ΔŴ_erm[i, :]) for i in 1:1:n]))

        # fit GAMP twice at small_δy and extrapolate for δy
        Ŵ_gamp   =  ConformalAmp.fit_leave_one_out(problem, X, y, method)
        Ŵ_gamp_small_δy =  ConformalAmp.fit_leave_one_out(problem, X, ỹ_gamp, method)
        Ŵ_gamp_δy  = Ŵ_gamp + δy / small_δy * (Ŵ_gamp_small_δy - Ŵ_gamp)
        
        push!(erm_gamp_difference_norm, mean([std(Ŵ_gamp_δy[i, :] - Ŵ_erm_2[i, :]) for i in 1:1:n]))

        result_gamp = ConformalAmp.gamp(problem, X, y; max_iter = method.max_iter, rtol =  method.rtol)
        
        # derivative w.r.t δy
        Δresult_gamp = (δy / small_δy) * ConformalAmp.compute_order_one_perturbation_gamp(problem, X, y, result_gamp; max_iter = method.max_iter, rtol = method.rtol, δy = small_δy)
        Δŵ = Δresult_gamp.x̂
        Δv̂ = Δresult_gamp.v̂
        Δg = Δresult_gamp.g
        v̂  = result_gamp.v̂
        g  = result_gamp.g

        # compute the new cavity means using GAMP
        Ŵ_taylor = Ŵ_gamp + (repeat(Δŵ', n, 1) - X .* (v̂ * Δg' + Δv̂ * g')')

        # push!(difference_norm, mean([norm(ΔŴ_erm[i, :] - ΔŴ_gamp[i, :], 2) for i in 1:1:n]) / d)
        push!(erm_taylor_difference_norm, mean([std(Ŵ_erm_2[i, :] - Ŵ_taylor[i, :]) for i in 1:1:n]) )
        
    end

    slope = slope_of_log(d_range, erm_taylor_difference_norm)
    slope_erm = slope_of_log(d_range, ΔŴ_erm_norm)

    pl = plot(d_range, erm_taylor_difference_norm, xaxis=:log, yaxis=:log, label="Diff. ERM vs Taylor : slope=$(slope)")
    plot!(d_range, erm_gamp_difference_norm, label="Diff. ERM vs GAMP refit : slope=$(slope_erm)")
    plot!(d_range, ΔŴ_erm_norm, label="diff. before and after δy : slope=$(slope_erm)")
    display(pl)

end

# fcp_with_order_one()
# test_gamp_order_one()
   
λ = 1.0
α = 0.5

# TODO : Do profiling to see where we waste the most time in the computation
# test_gamp_order_one_wrt_d( ConformalAmp.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0), 100:100:2000, δy = 0.2, rng_num = 0, avg_num = 5,  var="v̂")

test_gamp_order_one_loo_wrt_d( ConformalAmp.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0), 100:50:750; rng_num = 10, δy = 1.0)