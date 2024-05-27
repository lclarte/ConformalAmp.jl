"""
Generate data to be used by the Python library, and compare with our own prediction intervals 
"""

using ConformalAmp
using Base.Filesystem
using NPZ
using ProgressBars
using StableRNGs: StableRNG
using Statistics

α = 0.5
λ = 1.0
d = 250

problem = ConformalAmp.Lasso(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
coverage = 0.75 # only use this value ! Hardcoded in the Python library
fcp = ConformalAmp.FullConformal(coverage = coverage, δy_range = 0.0:0.05:5.0)

compute_exact_fcp = false

# generate the data

function generate_data(problem, d::Integer, seed::Integer = 0; ntest::Integer = 1000)
    rng = StableRNG(seed)
    
    
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)
    
    # create the folder experiments/python/$prolem if it doesn't exist
    if !isdir("experiments/python/$(problem)_$d")
        mkdir("experiments/python/$(problem)_$d")
    end

    npzwrite("experiments/python/$(problem)_$d/X.npy", X)
    npzwrite("experiments/python/$(problem)_$d/w.npy", w)
    npzwrite("experiments/python/$(problem)_$d/y.npy", y)
    npzwrite("experiments/python/$(problem)_$d/Xtest.npy", Xtest)
    npzwrite("experiments/python/$(problem)_$d/ytest.npy", ytest)

    return X, w, y, Xtest, ytest
end

X, y, w, Xtest, ytest = generate_data(problem, d, ntest = 10)

function load_data(problem, d)
    X = npzread("experiments/python/$(problem)_$d/X.npy")
    w = npzread("experiments/python/$(problem)_$d/w.npy")
    y = npzread("experiments/python/$(problem)_$d/y.npy")
    Xtest = npzread("experiments/python/$(problem)_$d/Xtest.npy")
    ytest = npzread("experiments/python/$(problem)_$d/ytest.npy")

    homotopy_intervals = npzread("experiments/python/$(problem)_$d/homotopy_intervals.npy")
    
    return X, w, y, Xtest, ytest, homotopy_intervals
end

# ===================================LOAD THE DATA ================================================

X, w, y, Xtest, ytest, homotopy_intervals = load_data(problem, d)
ntest = size(Xtest, 1)

# =========================COMPUTE THE INTERVALS EXACTLY ===========================================

if compute_exact_fcp
    exact_fcp_intervals = zeros((ntest, 2))

    for i in ProgressBar(1:ntest)
        xtest = Xtest[i, :]
        result = @timed ConformalAmp.get_confidence_interval(problem, X, y, xtest, fcp, ConformalAmp.ERM())
        exact_fcp_intervals[i, :] = [minimum(result[1]), maximum(result[1])]
    end

    println("Mean interval size for exact FCP: ", mean(exact_fcp_intervals[:, 2] - exact_fcp_intervals[:, 1]))
end

# =========================COMPUTE THE INTERVALS FOR AMP TAYLOR ===========================================

tayloramp_intervals = zeros((ntest, 2))

times = []
for i in 1:ntest
    xtest = Xtest[i, :]
    result = @timed ConformalAmp.get_confidence_interval(problem, X, y, xtest, fcp, ConformalAmp.GAMPTaylor(max_iter = 1000, rtol = 1e-3))
    tayloramp_intervals[i, :] = [minimum(result[1]), maximum(result[1])]
    push!(times, result[2])
end

println("Mean time for Taylor-AMP $(mean(times))")
println("Mean interval size for Taylor-AMP $(mean(tayloramp_intervals[:, 2] - tayloramp_intervals[:, 1]))")

# =========================DISPLAY THE RESULTS FOR HOMOTOPY ===========================================
intervals_sizes = [maximum(x) - minimum(x) for x in eachrow(homotopy_intervals)]
println("Mean interval size for homotopy: ", mean(intervals_sizes))

# =========================COMPARE THE INTERVALS ===========================================

# compute their coverage
function compute_coverage(intervals, ytest)
    ntest = size(ytest, 1)
    coverage = 0.0
    for i in 1:ntest
        if intervals[i, 1] <= ytest[i] <= intervals[i, 2]
            coverage += 1
        end
    end
    return coverage / ntest
end

# println("Exact FCP coverage: ", compute_coverage(exact_fcp_intervals, ytest))
println("Taylor-AMP coverage: ", compute_coverage(tayloramp_intervals, ytest))
println("Homotopy coverage: ", compute_coverage(homotopy_intervals, ytest))