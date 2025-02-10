using ProgressBars
using Statistics: mean
using ConformalAmp
using StableRNGs: StableRNG

# use current time as random seed

coverages = []
mean_lengths = []

println("$problem")

for i in 1:20
    rng = StableRNG(time_ns())

    problem = ConformalAmp.Lasso(α = 0.5, Δ = 1.0, λ = 1.0, Δ̂ = 1.0)
    d = 250
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d; model = "gaussian")

    ntest = 1000
    xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = ConformalAmp.sample_labels(rng, problem, xtest, w)

    κ = 0.1
    # Why we divide κ by 2 : because while the coverage guarantees are for 1.0 - 2 * α (for the right choice of α)
    # in practice we have 1 - α coverage
    algo = ConformalAmp.JackknifePlus(coverage = 1.0 - 2.0 * κ)

    ci = zeros(ntest, 2)

    for i in ProgressBar(1:ntest)
        lower, upper = ConformalAmp.get_confidence_interval(problem, X, y, xtest[i, :], algo, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-3))
        # lower, upper = ConformalAmp.get_confidence_interval(problem, X, y, xtest[i, :], algo, ConformalAmp.ERM())
        ci[i, :] = [lower, upper]
    end

    push!(coverages, mean([ci[i, 1] <= ytest[i] <= ci[i, 2] for i in 1:ntest]))
    push!(mean_lengths, mean([ci[i, 2] - ci[i, 1] for i in 1:ntest]))

end
println("Coverage : ", mean(coverages))
println("Mean length : ", mean(mean_lengths))