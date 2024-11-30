using Statistics
using Plots
using DataFrames
using CSV
using ConformalAmp

# load the data
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
ŵ_gamp = ConformalAmp.fit(problem, x, y, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-3))
ŵ_erm = ConformalAmp.fit(problem, x, y, ConformalAmp.ERM())

plt = scatter()
scatter(ŵ_gamp, ŵ_erm, label = "GAMP vs ERM")

n_train = Integer(floor(0.8 * n))
n_test = n - n_train
x_train, x_test = x[1:n_train, :], x[n_train+1:end, :]
y_train, y_test = y[1:n_train], y[n_train+1:end]

coverage = 0.9

ci_gamp_list = []
fcp = ConformalAmp.FullConformal(δy_range = 0.0:0.1:5.0, coverage = coverage)
gamp_time_list = []

# COMMENT THE LINE DEPENDING ON WHICH ALGORITHM YOU WANT TO USE
# method = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4)

println("Using method : $method")

for i in 1:n_test
    # compute the confidencei ntervals 
    debut = time()
    ci_gamp = ConformalAmp.get_confidence_interval(problem, x_train, y_train, x_test[i, :], fcp, method)
    fin = time()
    push!(ci_gamp_list, (minimum(ci_gamp), maximum(ci_gamp)))
    push!(gamp_time_list, fin - debut)
end

# compute the coverage of the GAMP
coverage_gamp = mean([ci_gamp_list[i][1] <= y_test[i] <= ci_gamp_list[i][2] for i in 1:n_test])

# compute the mean length of the confidence intervals
mean_length_gamp = mean([ci_gamp_list[i][2] - ci_gamp_list[i][1] for i in 1:n_test])

println("Coverage of GAMP : ", coverage_gamp)
println("Mean length of GAMP : ", mean_length_gamp)
println("Mean time of GAMP : ", mean(gamp_time_list))