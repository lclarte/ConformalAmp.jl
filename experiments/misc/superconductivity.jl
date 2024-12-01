# import the dataset
using CSV
using Statistics
using DataFrames
using Plots
using ProgressBars

using ConformalAmp

data = CSV.read("experiments/misc/superconductivity.csv", DataFrame)
n_total = 1000
X = data[1:n_total, 1:end-1]
y = data[1:n_total, end]

# normalize the data
X = Matrix(X)
y = Vector(y)

X = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)
X ./= sqrt(size(X, 2))
y = (y .- mean(y)) / std(y)

# run GAMNP on this data

problem = ConformalAmp.Ridge(α = size(X, 1) / size(X, 2), λ = 1.0, Δ = 1.0, Δ̂ = 1.0)

# just to compare the result of the estimators
result = ConformalAmp.fit(problem, X, y, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-3))
result_erm = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
result_vamp = ConformalAmp.fit(problem, X, y, ConformalAmp.VAMP(max_iter = 100, rtol = 1e-5))
scatter(result_vamp, result_erm, label = "VAMP vs ERM")


x̂_1, α_1, γ_1, x̂_2, α_2, γ_2 = ConformalAmp.vamp(problem, X, y; n_iterations = 100, rtol = 1e-10)
Ŵ_vamp = ConformalAmp.iterate_loo_vamp_ridge(X, y, problem.Δ̂, x̂_1, α_1, γ_1, x̂_2, α_2, γ_2)
Ŵ_erm  = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())

plt = scatter()
for i in 1:n_total
    scatter!(plt, Ŵ_vamp[i, :], Ŵ_erm[i, :], label = "")
end
display(plt)

# split train test 
n_train = 500
n_test = 100
X_train, y_train = X[1:n_train, :], y[1:n_train]
X_test = X[n_train+1:n_train+1+n_test, :]
y_test = y[n_train+1:n_train+1+n_test]

ci_list_vamp = []
time_vamp_list = []

coverage = 0.9
fcp =  ConformalAmp.FullConformal(δy_range = 0.0:0.02:4.0, coverage = coverage)
for x in ProgressBar(eachrow(X_test))
    debut = time()
    ci_vamp = ConformalAmp.get_confidence_interval(problem, X_train, y_train, x, fcp, ConformalAmp.VAMP(max_iter = 100, rtol = 1e-3))
    fin = time()
    push!(time_vamp_list, fin - debut)
    try
        push!(ci_list_vamp, (minimum(ci_vamp), maximum(ci_vamp)))
    catch
        push!(ci_list_vamp, (0.0, 0.0))
    end
    # ci_erm = ConformalAmp.get_confidence_interval(problem, X, y, x, ConformalAmp.SplitConformal(coverage = coverage), ConformalAmp.ERM())
    # push!(ci_list_erm, (minimum(ci_erm), maximum(ci_erm)))
end

ci_list_erm = []
for x in ProgressBar(eachrow(X_test))
    ci_erm = ConformalAmp.get_confidence_interval(problem, X_train, y_train, x, ConformalAmp.SplitConformal(coverage = coverage), ConformalAmp.ERM())
    push!(ci_list_erm, (minimum(ci_erm), maximum(ci_erm)))
end

# compute the mean length of the confidence intervals
mean_length_vamp = mean([ci_list_vamp[i][2] - ci_list_vamp[i][1] for i in 1:length(ci_list_vamp)])
mean_length_erm = mean([ci_list_erm[i][2] - ci_list_erm[i][1] for i in 1:length(ci_list_erm)])
# compute the coverage 
coverage_vamp = mean([ci_list_vamp[i][1] <= y_test[i] <= ci_list_vamp[i][2] for i in 1:length(ci_list_vamp)])
coverage_erm = mean([ci_list_erm[i][1] <= y_test[i] <= ci_list_erm[i][2] for i in 1:length(ci_list_erm)])

println("AMP vs Split conformal coverage : ", coverage_vamp, " vs ", coverage_erm)
println("AMP vs Split conformal mean length : ", mean_length_vamp, " vs ", mean_length_erm)