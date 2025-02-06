using Statistics
using Plots
using DataFrames
using CSV
using ConformalAmp
using ProgressBars
using Random

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

n_train = Integer(floor(0.8 * n))
n_test = n - n_train

target_coverage = 0.9

# COMMENT THE LINE DEPENDING ON WHICH ALGORITHM YOU WANT TO USE
method = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
# method = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5)

println("Using method : $method")

function run_experiment(x, y, target_coverage::Real, method, seeds::Integer)
    fcp = ConformalAmp.FullConformal(δy_range = 0.0:0.05:5.0, coverage = target_coverage)
    coverage_gamp_list = []
    mean_length_gamp_list = []
    gamp_time_list = []
        
    for seed in ProgressBar(1:seeds)
        ci_gamp_list = []
        gamp_time_list = []
        Random.seed!(seed)
        idx = shuffle(1:size(x, 1))
        x = x[idx, :]
        y = y[idx]
    
        x_train, x_test = x[1:n_train, :], x[n_train+1:end, :]
        y_train, y_test = y[1:n_train], y[n_train+1:end]
    
    
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
    
        push!(coverage_gamp_list, coverage_gamp)
        push!(mean_length_gamp_list, mean_length_gamp)
    end

    return coverage_gamp_list, mean_length_gamp_list, gamp_time_list
end

target_coverages = 0.55:0.05:0.95

coverage_means = []
coverage_stds  = []

mean_length_means = []
mean_length_stds = []

for target_coverage in target_coverages
    println("Coverage = $target_coverage")
    coverage_gamp_list, mean_length_gamp_list, gamp_time_list = run_experiment(x, y, target_coverage, method, 100)
    push!(coverage_means, mean(coverage_gamp_list))
    push!(coverage_stds, std(coverage_gamp_list))
    push!(mean_length_means, mean(mean_length_gamp_list))
    push!(mean_length_stds, std(mean_length_gamp_list))
end

# println("Coverage of $method : ", mean(coverage_gamp_list), " ± ", std(coverage_gamp_list))
# println("Mean length of $method : ", mean(mean_length_gamp_list), " ± ", std(mean_length_gamp_list))
# println("Mean time of $method : ", mean(gamp_time_list), " ± ", std(gamp_time_list))

if method isa ConformalAmp.GAMP
    method_name = "GAMP"
else
    method_name = "Taylor-AMP"
end

# do a square plot 
plot(target_coverages, coverage_means, ribbon = coverage_stds, label = "Coverage of $method_name", xlabel = "Target coverage", ylabel = "Coverage", legend = :topleft)
# plot the y = x line 
plot!(target_coverages, target_coverages, label = "y = x", linestyle = :dash, color = :black)
plot!(size = (800, 800))
# save plot 

savefig("experiments/paper/boston_coverage_$method_name.png")