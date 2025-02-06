
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

# split train test 
n_train = 50


# shuffle the data for the train test split
# allow to choose the seed

# COMMENT THE LINE DEPENDING ON WHICH ALGORITHM YOU WANT TO USE
# method = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4)

"""
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
"""
#### 

function run_experiment(X, y, target_coverage::Real, method, seeds::Integer)
    fcp = ConformalAmp.FullConformal(δy_range = 0.0:0.1:5.0, coverage = target_coverage)
    
    coverage_gamp_list = []
    mean_length_gamp_list = []
    gamp_time_list = []

    
    for seed in ProgressBar(1:seeds)
        Random.seed!(seed)
        idx = shuffle(1:size(X, 1))
        X = X[idx, :]
        y = y[idx]
        
        n_test = size(X, 1) - n_train
        X_train, y_train = X[1:n_train, :], y[1:n_train]
        X_test, y_test = X[n_train+1:end, :], y[n_train+1:end]
        
        time_gamp_list = []
        ci_list_gamp = []
    
        fcp = ConformalAmp.FullConformal(δy_range = 0.0:0.1:5.0, coverage = target_coverage)
    
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
    
        push!(coverage_gamp_list,coverage_gamp)
        push!(mean_length_gamp_list, mean_length_gamp)
        push!(gamp_time_list, mean(time_gamp_list))    
    end

    return coverage_gamp_list, mean_length_gamp_list, gamp_time_list
end

target_coverages = 0.55:0.05:0.95

coverage_means = []
coverage_stds  = []

mean_length_means = []
mean_length_stds = []

for target_coverage in target_coverages
    coverage_gamp_list, mean_length_gamp_list, gamp_time_list = run_experiment(X, y, target_coverage, method, 20)
    push!(coverage_means, mean(coverage_gamp_list))
    push!(coverage_stds, std(coverage_gamp_list))
    push!(mean_length_means, mean(mean_length_gamp_list))
    push!(mean_length_stds, std(mean_length_gamp_list))
end

if method isa ConformalAmp.GAMP
    method_name = "GAMP"
else
    method_name = "Taylor-AMP"
end

plot(target_coverages, coverage_means, ribbon = coverage_stds, label = "Coverage of $method_name", xlabel = "Target coverage", ylabel = "Coverage", fmt = :png)
plot!(target_coverages, target_coverages, label = "", linestyle = :dash, color = :black)
savefig("experiments/paper/riboflavin_coverage_$method_name.png")