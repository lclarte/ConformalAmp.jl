"""
Compare with conformal quantile regression
"""

using ConformalAmp
using ProgressBars
using StableRNGs
rng = StableRNG(10)

using Statistics
# generate the data 

α = 2.0
λ = 1.0
d = 100

problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)

coverages_cqr = []
coverages_amp = []

interval_sizes_cqr = []
interval_sizes_amp = []

ntest = 200
coverage = 0.9


for i in ProgressBar(1:20)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)

    fcp = ConformalAmp.FullConformal(coverage = coverage, δy_range = 0.0:0.05:5.0)

    intervals_amp = []
    sizes_amp     = []

    for x in eachrow(Xtest)
        interval_amp = ConformalAmp.get_confidence_interval(problem, X, y, x, fcp, ConformalAmp.GAMP(rtol = 1e-3, max_iter = 100))
        push!(intervals_amp, interval_amp)
        push!(sizes_amp, maximum(interval_amp) - minimum(interval_amp))
    end
    coverage_amp = mean((ytest .>= minimum.(intervals_amp)) .& (ytest .<= maximum.(intervals_amp)))
    push!(coverages_amp, coverage_amp)
    push!(interval_sizes_amp, mean(sizes_amp))

    lowers, uppers = ConformalAmp.get_intervals_qcp(X, y, Xtest, fcp.coverage, problem.λ, rng)

    # compute the coverage
    coverage_cqr = mean((ytest .>= lowers) .& (ytest .<= uppers))
    push!(coverages_cqr, coverage_cqr)
    push!(interval_sizes_cqr, mean(uppers - lowers))

    print(coverage_amp, " ", coverage_cqr)
end

println( mean(coverages_cqr), " ", mean(coverages_amp) )

println(mean(interval_sizes_cqr), " ", mean(interval_sizes_amp))