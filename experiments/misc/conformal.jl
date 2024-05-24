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

#### 

function compare_intervals_fcp_erm_gamptaylor(problem::ConformalAmp.Problem; d::Integer = 100, ntest::Integer = 10, seed::Integer = 0)
    """
    Compare the confidence interval given by ERM() so by refitting everything and GAMPTaylor for a
    single test point at a fixed dimension
    """
    rng = StableRNG(seed)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest_array = ConformalAmp.sample_data_any_n(rng, d, ntest)

    algo = ConformalAmp.FullConformal(δy_range = -5.0:0.2:5.0, coverage = 0.9)

    jaccard_list_erm_amptaylor = []
    jaccard_list_erm_amp       = []
    jaccard_list_erm_ermrefit = []

    for i in ProgressBar(1:ntest)
        xtest = xtest_array[i, :]
        ci_erm = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, ConformalAmp.ERM())
        ci_amptaylor = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo,     
                                ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-3))
        ci_amp = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo,     
                                ConformalAmp.GAMP(max_iter = 100, rtol = 1e-3))
        ci_ermrefit = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, ConformalAmp.ERMTaylor())
        push!(jaccard_list_erm_amptaylor, jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_amptaylor), maximum(ci_amptaylor)) )
        push!(jaccard_list_erm_amp, jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_amp), maximum(ci_amp)) )
        push!(jaccard_list_erm_ermrefit, jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_ermrefit), maximum(ci_ermrefit)) )
    end
    return (jaccard_list_erm_amptaylor, jaccard_list_erm_amp, jaccard_list_erm_ermrefit)
end

function compare_length_fcp_erm_gamptaylor(problem::ConformalAmp.Problem; d::Integer = 100, ntest::Integer = 10)
    """
    Compare the sizes of the confidence intervals given by the different algos 
    """
    rng = StableRNG(51)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest_array = ConformalAmp.sample_data_any_n(rng, d, ntest)

    algo = ConformalAmp.FullConformal(δy_range = -2.5:0.02:2.5, coverage = 0.9)

    length_erm = []
    length_amptaylor = []

    for i in ProgressBar(1:ntest)
        xtest = xtest_array[i, :]
        ci_erm = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, ConformalAmp.ERM())
        ci_amptaylor = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo,     
                                ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-3))
        push!(length_erm, maximum(ci_erm) - minimum(ci_erm))
        push!(length_amptaylor, maximum(ci_amptaylor) - minimum(ci_amptaylor))
    end
    return (length_erm, length_amptaylor)
end

## 

function compute_coverage_gamp_taylor(problem::ConformalAmp.Problem; d::Integer = 100, ntest::Integer = 10, rng_seed::Integer = 0)
    """
    We don't need to compute the coverage of ERM since we know that it's correct, the priority is to compute the one of
    GAMP Taylor
    """
    rng = StableRNG(rng_seed)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest   = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest   = ConformalAmp.sample_labels(rng, problem, xtest, w)

    fcp = ConformalAmp.FullConformal(coverage = 0.9, δy_range = -0.0:0.05:5.0)
    method = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
    
    total_covered = 0

    for i in 1:ntest
        ci = ConformalAmp.get_confidence_interval(problem, X, y, xtest[i, :], fcp, method)
        if minimum(ci) <= ytest[i] <= maximum(ci)
            total_covered += 1
        end
    end

    return total_covered / ntest
end

##### EXPERIMENTS

function experiment_jaccard()
    """
    Compute the jaccard index between confidence intervals of ERM and other methods 
    """
    d = 100
    problem = ConformalAmp.Lasso(α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0)
    
    @time jaccard_list_erm_amptaylor, jaccard_list_erm_amp, jaccard_list_erm_ermrefit = compare_intervals_fcp_erm_gamptaylor(problem; d = d, ntest=20)
    
    plt = stephist(jaccard_list_erm_amptaylor, bins=0.0:0.01:1.01, density=true, title="$(problem), d = $d",
        label="Gamp taylor : mean $(round(mean(jaccard_list_erm_amptaylor), sigdigits=2))", )
    stephist!(jaccard_list_erm_amp, bins=0.0:0.01:1.01, density=true, 
        label="Gamp refit : mean = $(round(mean(jaccard_list_erm_amp), sigdigits=2))")
    stephist!(jaccard_list_erm_ermrefit, bins=0.0:0.01:1.01, density=true,
        label="Erm refit : mean = $(round(mean(jaccard_list_erm_ermrefit), sigdigits=2))")

    display(plt)
    # save the plot
    # savefig(plt, "plots/jaccard_$(problem).png")
end

function experiment_length()
    """
    Compare the lengthes of the confidence intervals 
    """
    d = 50
    problem = ConformalAmp.Ridge(α = 2.0, λ = 0.01, Δ = 1.0, Δ̂ = 1.0)
    println("Problem is : $(problem)")
    
    length_erm, length_amptaylor = compare_length_fcp_erm_gamptaylor(problem; d = d, ntest=100)
    mean_relative_difference = mean((length_amptaylor - length_erm) ./ length_erm)

    min_bound = minimum([minimum(length_erm), minimum(length_amptaylor)])
    max_bound = maximum([maximum(length_erm), maximum(length_amptaylor)])
    step      = (max_bound - min_bound) / 50
    
    # plt = stephist(length_erm, bins=min_bound:step:max_bound, density=true, title="$(problem), d = $d",
    #     label="Erm : mean $(round(mean(length_erm), sigdigits=2))")
    # stephist!(length_amptaylor, bins=min_bound:step:max_bound, density=true, label="Amptaylor : mean $(round(mean(length_amptaylor), sigdigits=2))")
    
    plt = scatter(length_erm, length_amptaylor, title="$(problem), d = $d", label="Erm vs Amptaylor", xaxis="ERM length", yaxis="AMP Taylor length")
    # scatter the mean of both 
    scatter!([mean(length_erm)], [mean(length_amptaylor)], label="mean", color=:red)
    # plot the diagonal
    plot!(plt, min_bound:step:max_bound, min_bound:step:max_bound, label="", color=:black)

    println("Average length for ERM is $(round(mean(length_erm), sigdigits=2))")
    println("Average length for AMP taylor is $(round(mean(length_amptaylor), sigdigits=2))")
    println("on average, AMP taylor is $(round(mean_relative_difference, sigdigits=2)) larger than the length of ERM")

    display(plt)
    # save the plot
    # savefig(plt, "plots/length_$(problem).png")
end

function experiment_coverage()
    
    d_list = 50:50:500

    problems = [
        ConformalAmp.Ridge(α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0),
        ConformalAmp.Ridge(α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
        ConformalAmp.Lasso(α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0),
        ConformalAmp.Lasso(α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
    ]

    names = [
        "Ridge(λ = 0.1)",
        "Ridge(λ = 1.0)",
        "Lasso(λ = 0.1)",
        "Lasso(λ = 1.0)"
    ]

    plt = plot(d_list, 0.75 * ones(length(d_list)), label="Target coverage", color=:black)

    for i in eachindex(problems)
        mean_coverages = []
        problem = problems[i]
        for d in ProgressBar(d_list)
            coverages = []
            for seed in 1:200
                c = compute_coverage_gamp_taylor(problem; d = d, ntest = 1, rng_seed = seed+1)
                push!(coverages, c)
            end
        
            push!(mean_coverages, mean(coverages))
        end
        
        plt = scatter!(plt, d_list, mean_coverages, label="$(names[i])", xaxis="d", yaxis="Coverage")
        # draw black line at y = 0.9
        display(plt)
    end
    savefig("plots/coverage_gamp_taylor.pdf")

end

function experiment_time(problem)
    """
    Compare the time of execution of our method and 
    """
    d_range_fcp = (10.0).^(2:0.25:4)
    d_range_erm = (10.0).^(2:0.25:3)

    fcp     = ConformalAmp.FullConformal(coverage = 0.9, δy_range = 0.0:0.5:1.0)
    
    rng = StableRNG(0)

    times_erm = []
    times_amptaylor = []

    for d in ProgressBar(d_range_erm)
        d = Integer(floor(d))
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        xtest = ConformalAmp.sample_data_any_n(rng, d, 1)
        # compute the confidence interval with ERM and AMP Taylor
        time_erm_debut = time()
        ConformalAmp.get_confidence_interval(problem, X, y, xtest[1, :], fcp, ConformalAmp.ERM())
        push!(times_erm, time() - time_erm_debut)
    end

    for d in ProgressBar(d_range_fcp)
        d = Integer(floor(d))
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        xtest = ConformalAmp.sample_data_any_n(rng, d, 1)
        # compute the confidence interval with ERM and AMP Taylor
        time_amptaylor_debut = time()
        ConformalAmp.get_confidence_interval(problem, X, y, xtest[1, :], fcp, ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4))
        push!(times_amptaylor, time() - time_amptaylor_debut)
    end

    return d_range_erm, d_range_fcp, times_erm, times_amptaylor
end

# experiment_coverage()
# experiment_length()
# experiment_jaccard()

"""
problem = ConformalAmp.Lasso(α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)

d_range_erm, d_range_fcp, times_erm, times_amptaylor = experiment_time(
    problem
)

fs = 12
plt = scatter(d_range_erm, times_erm, label="Exact FCP", xaxis="d", yaxis="Time (s)",
yscale = :log10, xscale = :log10, legend=:topleft, color=:black,
xtickfontsize=fs,ytickfontsize=fs, legendfontsize=fs)
scatter!(d_range_fcp, times_amptaylor, label="AMP Taylor")
display(plt)
savefig(plt, "plots/time_erm_vs_amptaylor_$problem.pdf")
"""

# compare_jacknife_fcp_confidence_intervals()