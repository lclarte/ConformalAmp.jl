using DataFrames
using CSV
using RCall
using ConformalAmp
using StableRNGs: StableRNG

rng = StableRNG(0)

R"""
source("experiments/misc/conf_lasso_utils.R")
"""

data = CSV.read("experiments/paper/riboflavin.csv", DataFrame, missingstring=["NA"])

## === loading the data ==== 
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

# split train test 
n_train = 50
n_test = size(X, 1) - n_train

x_train, x_test = X[1:n_train, :], X[n_train+1:end, :]
y_train, y_test = y[1:n_train], y[n_train+1:end]

## ====== 

λ = 0.25

beta0 = ConformalAmp.fit(problem, x_train, y_train, ConformalAmp.ERM())

coverage = 0.9
κ = 1.0 - coverage

start = time()
ci_test = ConformalAmp.get_confidence_interval(problem, x_train, y_train, x_test, ConformalAmp.FullConformal(δy_range = 0.0:0.05:5.0, coverage = coverage), ConformalAmp.ERM())
elapsed = time() - start

# compute the coverage
coverage = mean([ci_test[i, 1] <= y_test[i] <= ci_test[i, 2] for i in 1:n_test])
# compute the mean length
mean_length = mean([ci_test[i, 2] - ci_test[i, 1] for i in 1:n_test])

println("Average time per prediction : ", elapsed / n_test)
println("Coverage : ", coverage)
println("Mean length : ", mean_length)