
# import the dataset
using CSV
using Statistics
using DataFrames
using Plots
using ProgressBars
using Random

using ConformalAmp

data = CSV.read("experiments/paper/riboflavin.csv", DataFrame)
X = data[:, 2:end]
println("Shape of the data : ", size(X))
y = data[:, 1]

# normalize the data
X = Matrix(X)
y = Vector(y)

X = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)
X ./= sqrt(size(X, 2))
y = (y .- mean(y)) / std(y)

# run GAMNP on this data

problem = ConformalAmp.Lasso(α = size(X, 1) / size(X, 2), λ = 0.25, Δ = 1.0, Δ̂ = 1.0)

# just to compare the result of the estimators
result = ConformalAmp.fit(problem, X, y, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))
result_erm = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
scatter(result, result_erm, label = "GAMP vs ERM")

# split train test 
n_train = 50

# shuffle the data for the train test split
# allow to choose the seed

time_gamp_list = []
coverage_list = []
mean_length_list = []

coverage = 0.9

# COMMENT THE LINE DEPENDING ON WHICH ALGORITHM YOU WANT TO USE
method = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
# method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5)

seeds = 20
for seed in ProgressBar(1:seeds)
    Random.seed!(seed)
    idx = shuffle(1:size(X, 1))
    X = X[idx, :]
    y = y[idx]

    n_test = size(X, 1) - n_train
    X_train, y_train = X[1:n_train, :], y[1:n_train]
    X_test, y_test = X[n_train+1:end, :], y[n_train+1:end]

    ci_list_gamp = []

    fcp = ConformalAmp.FullConformal(δy_range = 0.0:0.1:5.0, coverage = coverage)

    for x in (eachrow(X_test))
        debut = time()
        ci_gamp = ConformalAmp.get_confidence_interval(problem, X_train, y_train, x, fcp, method)
        fin = time()
        push!(time_gamp_list, fin - debut)
        push!(ci_list_gamp, (minimum(ci_gamp), maximum(ci_gamp)))
    end

    # compute the mean length of the confidence intervals
    mean_length_gamp = mean([ci_list_gamp[i][2] - ci_list_gamp[i][1] for i in 1:n_test])
    # compute the coverage 
    coverage_gamp = mean([ci_list_gamp[i][1] <= y_test[i] <= ci_list_gamp[i][2] for i in 1:n_test])

    coverage_list = [coverage_list; coverage_gamp]
    mean_length_list = [mean_length_list; mean_length_gamp]

end

println("$problem")
println("$method conformal coverage : ", mean(coverage_list), " with std : ", std(coverage_list))
println("$method conformal mean length : ", mean(mean_length_list), " with std : ", std(mean_length_list))
println("Average time for $method : ", mean(time_gamp_list), " with std : ", std(time_gamp_list))