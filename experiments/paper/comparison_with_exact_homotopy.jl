# here, we'll use the same data sturcture as in the exact homotopy paper
# https://arxiv.org/pdf/1708.00427
# d = 500, n = 200 and only the 1st five components of the teacher are non zero (either +- 8)
# in that case we don't need to normaléize the student because the label is O(1)

using ConformalAmp
using Plots
using ProgressBars

using StableRNGs: StableRNG
rng = StableRNG(0)

d = 500
n = 200
α = n / d

w = zeros(d)
# sample randomly the first 
w[1:5] = 8 * (2 * rand(5) .- 1)

λ = 0.13 # taken from the R code of the paper
problem = ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)

fcp = ConformalAmp.FullConformal(coverage = 0.9, δy_range = 0.0:0.025:2.5)

coverages = []
mean_lengths = []
ntest = 1

for i in ProgressBar(1:100)
    intervals = []
    lengths = []

    X = ConformalAmp.sample_data_any_n(rng, d, n)
    y = X * w + randn(n)
    Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = Xtest * w + randn(ntest)
    for x in (eachrow(Xtest))
        interval = ConformalAmp.get_confidence_interval(problem, X, y, x, fcp, ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4))
        push!(lengths, maximum(interval) - minimum(interval))
        push!(intervals, interval)
    end
    push!(coverages, mean((ytest .>= minimum.(intervals)) .& (ytest .<= maximum.(intervals))))
    push!(mean_lengths, mean(lengths))
end

println("Coverage: ", mean(coverages))
println("Mean length: ", mean(mean_lengths))