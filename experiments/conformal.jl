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

function test_gamp_order_one_wrt_d(problem::ConformalAmp.Problem, d_range::AbstractRange; δy::Real = 5.0, rng_num::Integer = 0)
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
        "x̂" => [],
        "v̂" => [],
        "ω" => [],
        "V" => []
    )
    differences_taylor = Dict(
        "x̂" => [],
        "v̂" => [],
        "ω" => [],
        "V" => []
    )

    differences_refit_taylor = Dict(
        "x̂" => [],
        "v̂" => [],
        "ω" => [],
        "V" => []
    )

    times_refit  = []
    times_taylor = []
    
    for d in ProgressBar(d_range)
        n = ConformalAmp.get_n(α, d)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        result  = ConformalAmp.gamp(problem, X, y; max_iter =  method.max_iter, rtol =  method.rtol)
        
        debut_taylor = time()
        # ∂ResultGamp × δy
        Δresult = ConformalAmp.compute_order_one_perturbation_gamp(problem, X, y, result; max_iter = method.max_iter, rtol = method.rtol, δy = δy)
        fin_taylor = time()
        
        y[n] = y[n] + δy
        debut_refit = time()
        result_refit = ConformalAmp.gamp(problem, X, y; max_iter = method.max_iter, rtol =  method.rtol)
        fin_refit = time()

        diff_refit_0 = result_refit - result
        diff_refit_taylor = result_refit - (result + δy * Δresult)

        push!(differences_refit["x̂"] ,std(diff_refit_0.x̂))
        push!(differences_taylor["x̂"], std(Δresult.x̂))
        push!(differences_refit_taylor["x̂"], std(diff_refit_taylor.x̂))
        
        push!(differences_refit["v̂"] ,std(diff_refit_0.v̂))
        push!(differences_taylor["v̂"], std(Δresult.v̂))
        push!(differences_refit_taylor["v̂"], std(diff_refit_taylor.v̂))

        push!(differences_refit["ω"] ,std(diff_refit_0.ω))
        push!(differences_taylor["ω"], std(Δresult.ω))
        push!(differences_refit_taylor["ω"], std(diff_refit_taylor.ω))

        push!(differences_refit["V"], std(diff_refit_0.V))
        push!(differences_taylor["V"], std(Δresult.ω))
        push!(differences_refit_taylor["V"], std(diff_refit_taylor.ω))

        push!(times_refit, fin_refit - debut_refit)
        push!(times_taylor, fin_taylor - debut_taylor)
    end 

    #                     PLOT THE DIFF. BETWEEN ORIGINAL AND THE REFIT (=GROUND TRUTH) AND TAYLOR
    var = "ω"
    pl = plot(d_range, differences_refit[var], xaxis = :log, yaxis = :log, label="Difference with refit")
    plot!(d_range, differences_taylor[var], label="Difference with Taylor")
    title!(pl, "slope(refit)  = $(slope_of_log(d_range, differences_refit[var])), 
                slope(taylor) = $(slope_of_log(d_range, differences_taylor[var]))")
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

# fcp_with_order_one()
# test_gamp_order_one()
   
λ = 1e-1
α = 2.0

# TODO : Do profiling to see where we waste the most time in the computation
test_gamp_order_one_wrt_d( ConformalAmp.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0),
                          100:50:2000,
                          δy = 0.5,
                          rng_num = 0)