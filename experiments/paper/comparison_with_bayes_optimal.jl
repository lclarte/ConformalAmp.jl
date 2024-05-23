"""
Compare the size of the confidence intervals between the Bayes-optimal estimator and conformal prediction (both Taylor AMP and SCP)
"""

using ConformalAmp
using Distributions
using Statistics

function compare_bayes_optimal_conformal_intervals(problem::ConformalAmp.Problem, erm_problem::ConformalAmp.Problem; seed::Integer = 0, ntest::Integer = 1000)
    """
    - problem is the one that will sample the data, must be BayesOptimalRidge or BayesOptimalLasso
    """
    rng = StableRNG(seed)
    d = 250
    Δ = 1.0 # don't change this parameter
    coverage = 0.9

    algo = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
    fcp = ConformalAmp.FullConformal(δy_range = 0.0:0.05:10.0, coverage = coverage)
    
    quantile = Distributions.quantile(Normal(), 0.5 + coverage / 2.0)
    
    bo_interval_size = []
    cp_interval_size = []
    scp_interval_size= []
    
    bo_total, cp_total, scp_total = 0, 0, 0
   
    for i in (1:ntest)
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

        ŵ, q = ConformalAmp.split_conformal(erm_problem, X, y, coverage)
        lower = ŵ' * xtest - q
        upper = ŵ' * xtest + q

        if lower <= ytest[1] <= upper
            scp_total += 1
        end
        push!(scp_interval_size, upper - lower)
    end

    return [
        "bo_coverage" => bo_total / ntest,
        "cp_coverage" => cp_total / ntest,
        "scp_coverage" => scp_total / ntest,
        "bo_mean_size" => mean(bo_interval_size),
        "cp_mean_size" => mean(cp_interval_size),
        "scp_mean_size" => mean(scp_interval_size)    
    ]
end

α = 0.5

problems = [
    ConformalAmp.BayesOptimalRidge(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.BayesOptimalRidge(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.BayesOptimalLasso(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.BayesOptimalLasso(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.BayesOptimalRidge(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.BayesOptimalLasso(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
]

erm_problems = [
    ConformalAmp.Ridge(; α = α, λ = 0.1, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Ridge(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Lasso(; α = α, λ = 0.1, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Lasso(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Lasso(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Ridge(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
]

for i in eachindex(problems)
    println("Bayes is $(problems[i]), ERM is $(erm_problems[i])")
    println(compare_bayes_optimal_conformal_intervals(problems[i], erm_problems[i]; ntest=200))
end