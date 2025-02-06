using CSV
using DataFrames
using RCall
using ConformalAmp
using StableRNGs: StableRNG
using Statistics

rng = StableRNG(0)

R"""
source("experiments/misc/conf_lasso_utils.R")
"""


## === loading the data ==== 
data = CSV.read("experiments/paper/boston.csv", DataFrame, missingstring=["NA"])
x = data[:, 1:end-1]
y = data[:, end]

# normalize the data
# replace missing values by the mean
x = Matrix(x)
y = Vector(y)

n, d = size(x)
x = coalesce.(x, 0)
x = (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
x ./= sqrt(d)
y = (y .- mean(y)) / std(y)

problem = ConformalAmp.Lasso(α = n / d, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)

n_train = Integer(floor(0.8 * n))
n_test = n - n_train

x_train, x_test = x[1:n_train, :], x[n_train+1:end, :]
y_train, y_test = y[1:n_train], y[n_train+1:end]

## ====== 

λ = 1.0

beta0 = ConformalAmp.fit(problem, x_train, y_train, ConformalAmp.ERM())

coverage = 0.9
κ = 1.0 - coverage

start = time()
ci_test = ConformalAmp.get_confidence_interval(problem, x_train, y_train, x_test, ConformalAmp.FullConformal(δy_range = 0.0:0.05:5.0, coverage = coverage), ConformalAmp.ERM())
elapsed = time() - start

println("Average time per sample : ", elapsed / n_test)

# compute the coverage
coverage = mean([ci_test[i, 1] <= y_test[i] <= ci_test[i, 2] for i in 1:n_test])
# compute the mean length
mean_length = mean([ci_test[i, 2] - ci_test[i, 1] for i in 1:n_test])

println("Coverage : ", coverage)
println("Mean length : ", mean_length)