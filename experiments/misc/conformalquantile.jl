using Plots
using ConformalAmp
using StableRNGs: StableRNG
using Statistics: mean

problem = ConformalAmp.Ridge(α = 2.0, Δ = 1.0, λ = 1.0, Δ̂ = 1.0)
d = 100

rng = StableRNG(0)
(; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

coverages = []
ntest = 1000

for i in 1:100
    Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)


    lower, upper = ConformalAmp.get_intervals_qcp(X, y, Xtest, 0.5, 1.0, rng)

    # compute the coverage
    coverage = mean((ytest .>= lower) .& (ytest .<= upper))

    plt = stephist(lower)
    stephist!(plt, upper)
    stephist!(plt, ytest, label="ytest")

    push!(coverages, coverage)
end

mean(coverages)
