"""
- Experiment to compute the coverage of AMP-Taylor 
"""

using ConformalAmp
using LinearAlgebra
using Plots
using ProgressBars
using StableRNGs: StableRNG
using Statistics

function compute_coverage_gamp_taylor(problem::ConformalAmp.Problem; d::Integer = 100, ntest::Integer = 1, rng_seed::Integer = 0, coverage::Real = 0.9, model::String = "gaussian")
    """
    We don't need to compute the coverage of ERM since we know that it's correct, the priority is to compute the one of
    GAMP Taylor
    """
    rng = StableRNG(rng_seed)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d, model = model)
    xtest   = ConformalAmp.sample_data_any_n(rng, d, ntest, model = model)
    ytest   = ConformalAmp.sample_labels(rng, problem, xtest, w)

    algo = ConformalAmp.FullConformal(coverage = coverage, δy_range = 0.0:0.05:7.5)
    method = ConformalAmp.GAMP(max_iter = 1000, rtol = 1e-5)
    
    total_covered = 0

    for i in 1:ntest
        ci = ConformalAmp.get_confidence_interval(problem, X, y, xtest[i, :], algo, method)
        if minimum(ci) <= ytest[i] <= maximum(ci)
            total_covered += 1
        end
    end

    return total_covered / ntest
end

function experiment_coverage(problem::ConformalAmp.Problem, nseeds::Integer = 10, coverage::Real = 0.9, d_list::AbstractArray = [100, 200]; model::String = "gaussian")
    mean_coverages_gamp = []
    mean_time_gamp = []
    
    for d in d_list
        coverages_gamp = []

        times_gamp = []

        for seed in ProgressBar(1:nseeds)
            timed_result_gamp = @timed compute_coverage_gamp_taylor(problem; d = d, ntest = 1, rng_seed = seed, coverage = coverage, model = model)
            push!(coverages_gamp, timed_result_gamp[1])
            push!(times_gamp, timed_result_gamp[2])
        end
    
        push!(mean_coverages_gamp, mean(coverages_gamp))
        push!(mean_time_gamp, mean(times_gamp))
    end

    return d_list, mean_coverages_gamp, mean_time_gamp
end

problems = [
    # ConformalAmp.Ridge(α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0),
    # ConformalAmp.Ridge(α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    # ConformalAmp.Lasso(α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0),
    # ConformalAmp.Lasso(α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Pinball(Δ = 1.0, q = 0.5, λ = 1.0, α = 0.5)
]
coverage = 0.5

fs = 12

d_list = [100, 200, 500]
plt = plot(d_list, coverage * ones(length(d_list)), label="", color=:black, xaxis="d", yaxis="coverage", legend=:topright,
xtickfontsize=fs,ytickfontsize=fs, legendfontsize=fs)

for problem in problems
    d_list, mean_coverages_gamp, mean_time_gamp = experiment_coverage(problem, 100, coverage, d_list; model = "gaussian")
    scatter!(plt, d_list, mean_coverages_gamp, label="$problem")
    println("Mean coverage for $problem is $(mean_coverages_gamp)")

end
display(plt)

"""
For reference, the experimental results obtained in the paper are : 
problems = Dict(
    "Ridge (λ = 0.1)" => [0.9050000000000001, 0.8970000000000001, 0.89, 0.883, 0.8850000000000001],
    "Ridge (λ = 1.0)" => [0.92, 0.93, 0.91, 0.91, 0.85],
    "Lasso (λ = 0.1)" => [0.8949999999999997, 0.8889999999999999, 0.8629999999999999, 0.8599999999999999, 0.8759999999999998],
    "Lasso (λ = 1.0)" => [0.9069999999999999, 0.8910000000000002, 0.9000000000000001, 0.8980000000000001, 0.9]
)
"""
