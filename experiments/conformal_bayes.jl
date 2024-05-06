"""
Comparison of the size of the intervals provided by Bayes optimal estimator and those given by conformal prediction. 
e.g. for ridge regression or the LASSO 
TODO : Implémenter le Bayes-optimal pour le lasso, deja fait quelque part 
"""

using ConformalAmp
using Distributions
using Plots
using ProgressBars
using StableRNGs: StableRNG
using Statistics

# Helper functions

function build_problem(problem_str::String, α::Real)
    if problem_str == "ridge"
        problem = ConformalAmp.BayesOptimalRidge(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
    elseif problem_str == "lasso"
        problem = ConformalAmp.BayesOptimalLasso(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
    elseif problem_str == "quantile"
        error("Cannot do quantile regression yet
        ")
    elseif problem_str == "logistic"
        error("Cannot do logistic yet")
    else
        error("Unknown problem")
    end
    return problem
end

# Actual experiments

function test(problem_str::String)
    """
    Test that the overlaps of the Bayes-optimal estimators are all the same 
    """
    α_vals = 0.5:20.0
    d = 500
    rng = StableRNG(0)

    q_overlaps, m_overlaps, v_overlaps = [], [], []

    for α in α_vals
        problem = build_problem(problem_str, α)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        result = ConformalAmp.gamp(problem, X, y; rtol=1e-4)

        push!(m_overlaps, result.x̂' * w / d)
        push!(q_overlaps, result.x̂' * result.x̂ / d)
        push!(v_overlaps, Statistics.mean(result.v̂))
    end

    plt = plot(α_vals, m_overlaps, label="m", xaxis="α", yaxis="overlap", title="$problem_str regression")
    plot!(α_vals, q_overlaps, label="q")
    ρ = problem_str == "lasso" ? 2.0 : 1.0
    plot!(α_vals, ρ .- v_overlaps, label="ρ - v")
    display(plt)
end

function compare_bayes_optimal_conformal_intervals(problem_str::String; λ_erm::Real = 1.0, α::Real = 0.5, seed::Integer = 0, ntest::Integer = 1000)
    """
    For a given dataaset, we do 2 things :
        - Look at the coverage of Bayes-posterior and Full Conformal Prediction (either exact or approximated)
        - Compare the size of the intervals
    """
    rng = StableRNG(seed)
    d = 100
    Δ = 1.0 # don't change this parameter

    algo = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4, δy_perturbation=0.1)
    # algo = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4)
    
    coverage = 0.9
    quantile = Distributions.quantile(Normal(), 0.5 + coverage / 2.0)
    println("CDF between -quantile and quantile is $(cdf(Normal(), quantile) - cdf(Normal(), -quantile))")
    fcp = ConformalAmp.FullConformal(δy_range = 0.1:0.1:6.0, coverage = coverage)
    
    problem = build_problem(problem_str, α)
    erm_problem = problem_str == "ridge" ? ConformalAmp.Ridge(α = α, λ = λ_erm, Δ = 1.0, Δ̂ = 1.0) : ConformalAmp.Lasso(α = α, λ = λ_erm, Δ = 1.0, Δ̂ = 1.0)

    bo_interval_size = []
    cp_interval_size = []
    bo_total, cp_total = 0, 0
   
    for i in ProgressBar(1:ntest)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        Xtest = ConformalAmp.sample_data_any_n(rng, d, 1)
        ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)
    # compute the confidence interval for the Bayes-optimal estimator
        xtest = Xtest[1, :]
        
        # the prediction is a Gaussian, for α coverage, we take the quantile of the Gaussian
        # don't forget to add the aleatoric noise Δ = 1 to the confidence interval
        result = ConformalAmp.gamp(problem, X, y; rtol=1e-4)
        bo_interval = [result.x̂' * xtest - sqrt(Δ + result.v̂' * (xtest.^2)) * quantile, result.x̂' * xtest + sqrt(Δ + result.v̂' * (xtest.^2)) * quantile]
        push!(bo_interval_size, maximum(bo_interval) - minimum(bo_interval))
        if minimum(bo_interval) <= ytest[1] <= maximum(bo_interval)
            bo_total += 1
        end

        cp_interval = ConformalAmp.get_confidence_interval(erm_problem, X, y, xtest, fcp, algo) 
        if cp_interval == []
            push!(cp_interval_size, 0)
        else
            push!(cp_interval_size, maximum(cp_interval) - minimum(cp_interval))
            if minimum(cp_interval) <= ytest[1] <= maximum(cp_interval)
                cp_total += 1
            end
        end
    end

    println("Coverage of Bayes optimal : $(bo_total / ntest)")
    println("Coverage of conformal prediction : $(cp_total / ntest)")

    # stephist(bo_interval_size, title="$problem_str regression - Bayes-optimal intervals", label="Bayes-optimal intervals")
    plt = histogram2d(bo_interval_size, cp_interval_size, xaxis="Bayes optimal", yaxis="Conformal, λ = $(λ_erm)", title="$(100 * coverage)% coverage, α = $α, $problem_str regression",
                      bins=(50, 50), normalize=:pdf, color=:plasma, show_empty_bins=true)
    # plot the line x = y for reference between the min and max of bo_interval_size
    plot!([minimum(bo_interval_size), maximum(bo_interval_size)], [minimum(bo_interval_size), maximum(bo_interval_size)], label="", color="white")
    scatter!([mean(bo_interval_size)], [mean(cp_interval_size)], label="", color="white", ms=10)
    # save the figure
    savefig(plt, "plots/bo_vs_cp_$(problem_str)_λ_$(λ_erm)_α_$(α).png")
    display(plt)
end

compare_bayes_optimal_conformal_intervals("ridge"; λ_erm = 1.0, seed = 0, α = 0.5, ntest = 1000)