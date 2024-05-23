# test the split conformal prediction function 

using ConformalAmp
using ProgressBars
using StableRNGs: StableRNG
using Statistics


problem = ConformalAmp.Lasso(α = 2.0, Δ = 1.0, λ = 0.01, Δ̂ = 1.0)
d = 50

target_coverage = 0.9
coverages = []

rng = StableRNG(0)

for seed in 1:1:10
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    ŵ, q = ConformalAmp.split_conformal(problem, X, y, target_coverage)

    # sample test data
    ntest = 1000

    xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = ConformalAmp.sample_labels(rng, problem, xtest, w)

    # confidence intervals for the test data

    lower_intervals = xtest * ŵ .- q
    upper_intervals = xtest * ŵ .+ q

    # compute the coverage
    push!(coverages, mean((ytest .>= lower_intervals) .& (ytest .<= upper_intervals)) )
end

mean(coverages)