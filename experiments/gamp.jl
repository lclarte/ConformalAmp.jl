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

function difference_erm_gamp(d_range::AbstractRange = 100:100:5000)
    """ 
    Just to plot the rate of convergrence of the difference between ERM and GMAMP
    """
    rng = StableRNG(20)

    λ = 1.0
    α = 3.0
    problem = ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)

    difference_norm = []
    norm_erm        = []
    difference_lf   = []

    
    for d in ProgressBar(d_range)
        x_test = ConformalAmp.sample_data_any_n(rng, d, 1)[1, :]
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        
        Ŵ_erm  = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
        Ŵ_gamp = ConformalAmp.fit(problem, X, y, ConformalAmp.GAMP(max_iter = 200, rtol = 1e-3))
        push!(difference_norm, norm(Ŵ_gamp - Ŵ_erm, 2))
        push!(norm_erm, norm(Ŵ_erm, 2))
        push!(difference_lf, abs((Ŵ_erm - Ŵ_gamp)' * x_test))
    end

    plt = plot(d_range, difference_norm, label="dif.. norm : $(slope_of_log(d_range, difference_norm))", xaxis = :log, yaxis = :log)
    plot!(d_range, norm, label = "norm : $(slope_of_log(d_range, norm_erm))")
    plot!(d_range, difference_lf, label = "diff. lf : $(slope_of_log(d_range, difference_lf))")
    display(plt)
end

difference_erm_gamp()