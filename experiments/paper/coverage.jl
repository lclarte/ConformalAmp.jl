"""
Experiments to measure the coverage + size of confidence intervals of the different methods : 
- FCP on Lasso + Ridge with GAMP
- Bayes-optimal 
- True FCP
"""

using ConformalAmp
using LinearAlgebra
using Plots
using ProgressBars
using StableRNGs: StableRNG
using Statistics


function compute_coverage_erm(problem::ConformalAmp.Problem; d::Integer = 100, ntest::Integer = 10, rng_seed::Integer = 0, coverage::Real = 0.9)
    """
    We don't need to compute the coverage of ERM since we know that it's correct, the priority is to compute the one of
    GAMP Taylor
    """
    rng = StableRNG(rng_seed)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest   = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest   = ConformalAmp.sample_labels(rng, problem, xtest, w)

    algo = ConformalAmp.FullConformal(coverage = coverage, δy_range = -0.0:0.05:5.0)
    method = ConformalAmp.ERM()
    
    total_covered = 0

    for i in 1:ntest
        ci = ConformalAmp.get_confidence_interval(problem, X, y, xtest[i, :], algo, method)
        if minimum(ci) <= ytest[i] <= maximum(ci)
            total_covered += 1
        end
    end

    return total_covered / ntest
end

function compute_coverage_gamp_taylor(problem::ConformalAmp.Problem; d::Integer = 100, ntest::Integer = 10, rng_seed::Integer = 0, coverage::Real = 0.9)
    """
    We don't need to compute the coverage of ERM since we know that it's correct, the priority is to compute the one of
    GAMP Taylor
    """
    rng = StableRNG(rng_seed)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest   = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest   = ConformalAmp.sample_labels(rng, problem, xtest, w)

    algo = ConformalAmp.FullConformal(coverage = coverage, δy_range = -0.0:0.05:5.0)
    method = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
    
    total_covered = 0

    for i in 1:ntest
        ci = ConformalAmp.get_confidence_interval(problem, X, y, xtest[i, :], algo, method)
        if minimum(ci) <= ytest[i] <= maximum(ci)
            total_covered += 1
        end
    end

    return total_covered / ntest
end

function experiment_coverage(problem::ConformalAmp.Problem, nseeds::Integer = 10, coverage::Real = 0.9)
    # desired coverage

    mean_coverages_gamp = []
    mean_coverages_erm  = []

    mean_time_gamp = []
    mean_time_erm  = []

    d_list = [100, 200, 300, 400, 500]
    
    for d in d_list
        coverages_gamp = []
        coverages_erm  = []

        times_gamp = []
        times_erm  = []

        for seed in ProgressBar(1:nseeds)
            timed_result_gamp = @timed compute_coverage_gamp_taylor(problem; d = d, ntest = 1, rng_seed = seed, coverage = coverage)
            #timed_result_erm = @timed compute_coverage_erm(problem; d = d, ntest = 1, rng_seed = seed, coverage = coverage)
            timed_result_erm = (1.0, 1.0)
            push!(coverages_gamp, timed_result_gamp[1])
            push!(coverages_erm, timed_result_erm[1])

            push!(times_gamp, timed_result_gamp[2])
            push!(times_erm, timed_result_erm[2])
        end
    
        push!(mean_coverages_gamp, mean(coverages_gamp))
        push!(mean_coverages_erm, mean(coverages_erm))

        push!(mean_time_gamp, mean(times_gamp))
        push!(mean_time_erm, mean(times_erm))
    end

    return d_list, mean_coverages_gamp, mean_coverages_erm, mean_time_gamp, mean_time_erm
end

problem = ConformalAmp.Ridge(α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0)
coverage = 0.9

d_list, mean_coverages_gamp, mean_coverages_erm, mean_time_gamp, mean_time_erm = experiment_coverage(problem, 1000, coverage)

println("Mean coverages for GAMP Taylor : $(mean_coverages_gamp)")
println("Mean coverages for ERM  : $(mean_coverages_erm)")

# Ridge, α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0  
# [0.9050000000000001, 0.8970000000000001, 0.89, 0.883, 0.8850000000000001]

# Ridge, α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0  
# [0.92, 0.93, 0.91, 0.91, 0.85]

# Lasso α = 0.5, λ = 0.1, Δ = 1.0, Δ̂ = 1.0
# [0.8949999999999997, 0.8889999999999999, 0.8629999999999999, 0.8599999999999999, 0.8759999999999998]

# Lasso, α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0  
# [0.9069999999999999, 0.8910000000000002, 0.9000000000000001, 0.8980000000000001, 0.9]

### Lines storing the experimental results 
problems = Dict(
    "Ridge (λ = 0.1)" => [0.9050000000000001, 0.8970000000000001, 0.89, 0.883, 0.8850000000000001],
    "Ridge (λ = 1.0)" => [0.92, 0.93, 0.91, 0.91, 0.85],
    "Lasso (λ = 0.1)" => [0.8949999999999997, 0.8889999999999999, 0.8629999999999999, 0.8599999999999999, 0.8759999999999998],
    "Lasso (λ = 1.0)" => [0.9069999999999999, 0.8910000000000002, 0.9000000000000001, 0.8980000000000001, 0.9]
)

fs = 12
plt = plot(d_list, coverage * ones(length(d_list)), label="", color=:black, xaxis="d", yaxis="coverage", legend=:topright,
xtickfontsize=fs,ytickfontsize=fs, legendfontsize=fs)

for i in eachindex(problems) 
    scatter!(plt, d_list, problems[i], label=i)
end

display(plt)
savefig(plt, "experiments/paper/coverage.pdf")