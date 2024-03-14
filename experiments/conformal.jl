"""
Script to compare the intervals that are returned by the LOO _vs_ conformal prediction 
"""

using ConformalAmp
using LinearAlgebra
using Plots
using ProgressBars
using StableRNGs: StableRNG
using Statistics

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

function test_gamp_order_one_wrt_d(problem::ConformalAmp.Problem, d_range::AbstractRange; δy::Real = 5.0)
    """
    Compute the std of the following differences
        * x̂(δy) - x̂(0) computed by refitting 
        * x̂(δy) - x̂(0) computed by Taylor 1 expansion
        * x̂(δy) - x̂'(δy) computed by Taylor 1 expansion vs the refitting
    as a fuinction of the dimension and check that both converge to 0 as 1/ √(d) and that the diff. between the 
    2 quantities is smaller in order than 1 / √d

    Also record the time taken
    """
    rng = StableRNG(10)

    method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-6)

    differences_refit = []
    differences_taylor= []

    times_refit  = []
    times_taylor = []

    differences_refit_taylor = []

    
    for d in ProgressBar(d_range)
        n = ConformalAmp.get_n(α, d)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        result  = ConformalAmp.gamp(problem, X, y; max_iter =  method.max_iter, rtol =  method.rtol)
        
        debut_taylor = time()
        Δresult = ConformalAmp.compute_order_one_perturbation_gamp(problem, X, y, result; max_iter = method.max_iter, rtol = method.rtol, δy = 1.0)
        fin_taylor = time()
        
        y[n] = y[n] + δy
        debut_refit = time()
        result_refit = ConformalAmp.gamp(problem, X, y; max_iter =  method.max_iter, rtol =  method.rtol)
        fin_refit = time()
        
        push!(differences_refit ,std(result_refit.x̂ - result.x̂))
        push!(differences_taylor,δy * std(Δresult.x̂))
        push!(differences_refit_taylor, std(result_refit.x̂ - result.x̂ - δy * Δresult.x̂))

        push!(times_refit, fin_refit - debut_refit)
        push!(times_taylor, fin_taylor - debut_taylor)
    end 

    # pl = plot(d_range, differences_refit, xaxis = :log, yaxis = :log, label="Difference with refit")
    # plot!(d_range, differences_taylor, label="Difference with Taylor")
    # title!(pl, "log-log slope : $(slope_of_log(d_range, differences_refit))")
    # display(pl)

    # pl = plot(d_range, differences_refit_taylor, xaxis = :log, yaxis = :log, 
    #         label = "$(slope_of_log(d_range, differences_refit_taylor))")
    # display(pl)
    
    pl = plot(d_range, times_refit, xaxis = :log, yaxis = :log, label="Full refit")
    plot!(pl, d_range, times_taylor, label="Taylor")
    display(pl)

end

# fcp_with_order_one()
# test_gamp_order_one()

λ = 1e-4
α = 2.0

test_gamp_order_one_wrt_d( ConformalAmp.Ridge(α = α, λ = λ, Δ = 10.0, Δ̂ = 1.0),
                          100:100:2000,
                          δy = 1.0)