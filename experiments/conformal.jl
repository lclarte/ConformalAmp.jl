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

function jaccard_index(a1::Real, b1::Real, a2::Real, b2::Real)
    intersection = max(0, min(b1, b2) - max(a1, a2))
    union = max(b1, b2) - min(a1, a2)
    return intersection / union
end

function slope_of_log(x_::AbstractVector, y_::AbstractVector)
    return (mean(log.(x_) .* log.(y_)) - mean(log.(x_)) * mean(log.(y_))) / (mean(log.(x_) .* log.(x_)) - (mean(log.(x_)))^2)
end

##

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

function compare_intervals_fcp_erm_gamptaylor(; ntest::Integer = 10)
    """
    Compare the confidence interval given by ERM() so by refitting everything and GAMPTaylor for a
    single test point at a fixed dimension
    """
    rng = StableRNG(51)
    d = 20
    # problem = ConformalAmp.Ridge(α = 0.5, λ = 1e-1, Δ = 1.0, Δ̂ = 1.0)
    # problem = ConformalAmp.Lasso(α = 0.5, λ = 1e-1, Δ = 1.0, Δ̂ = 1.0)
    problem = ConformalAmp.Pinball(α = 0.5, λ = 1e-1, Δ = 1.0, q = 0.75)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest_array = ConformalAmp.sample_data_any_n(rng, d, ntest)

    algo = ConformalAmp.FullConformal(δy_range = -5.0:0.2:5.0, coverage = 0.9)

    jaccard_list_erm_amptaylor = []
    jaccard_list_erm_amp       = []
    jaccard_list_erm_ermtaylor = []

    for i in ProgressBar(1:ntest)
        xtest = xtest_array[i, :]
        ci_erm = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, ConformalAmp.ERM())
        ci_amptaylor = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo,     
                                ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-3, δy_perturbation = 0.1))
        ci_amp = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo,     
                                ConformalAmp.GAMP(max_iter = 100, rtol = 1e-3))
        ci_ermtaylor = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, ConformalAmp.ERMTaylor())
        push!(jaccard_list_erm_amptaylor, jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_amptaylor), maximum(ci_amptaylor)) )
        push!(jaccard_list_erm_amp, jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_amp), maximum(ci_amp)) )
        push!(jaccard_list_erm_ermtaylor, jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_ermtaylor), maximum(ci_ermtaylor)) )
    end
    return (jaccard_list_erm_amptaylor, jaccard_list_erm_amp, jaccard_list_erm_ermtaylor)
end

#####
   
@time jaccard_list_erm_amptaylor, jaccard_list_erm_amp, jaccard_list_erm_ermtaylor = compare_intervals_fcp_erm_gamptaylor(ntest=100)

plt = stephist(jaccard_list_erm_amptaylor, bins=0.0:0.01:1.01, density=true, label="Amp taylor")
stephist!(jaccard_list_erm_amp, bins=0.0:0.01:1.01, density=true, label="Amp refit")
stephist!(jaccard_list_erm_ermtaylor, bins=0.0:0.01:1.01, density=true, label="Erm refit")
println("Mean jaccard index for ERM vs AMP Taylor : ", mean(jaccard_list_erm_amptaylor))
println("Mean jaccard index for ERM vs AMP refit  : ", mean(jaccard_list_erm_amp))
println("Mean jaccard index for ERM vs ERM Taylor : ", mean(jaccard_list_erm_ermtaylor))
display(plt)