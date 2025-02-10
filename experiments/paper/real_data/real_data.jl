using Statistics
using Plots
using DataFrames
using CSV
using ConformalAmp
using ProgressBars
using Random

include("data.jl")
include("experiment.jl")

dataset_name = "riboflavin"

if dataset_name == "boston"
    x, y = load_boston(); println("Loaded Boston dataset")
    n, d = size(x)
    problem = ConformalAmp.Lasso(α = n / d, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
    n_train = Integer(floor(0.8 * n))
elseif dataset_name == "riboflavin"
    x, y = load_riboflavin(); println("Loaded Riboflavin dataset")
    n, d = size(x)
    problem = ConformalAmp.Lasso(α = n / d, λ = 0.25, Δ = 1.0, Δ̂ = 1.0)
    n_train = 50
elseif dataset_name == "wine"
    x, y = load_wine(); println("Loaded Wine dataset")
    n, d = size(x)
    problem = ConformalAmp.Lasso(α = n / d, λ = 1.0, Δ = 1.0, Δ̂=1.0)
    n_train = Integer(floor(0.9 * n))
end


# load the data

target_coverage = 0.9
# list of FullConformal / JackknifePlus with the Algo used (Taylor-AMP, Exact / ApproximateHomotopy)
experiments = [
    # (ConformalAmp.FullConformal(δy_range = 0.0:0.1:5.0, coverage = target_coverage), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4)),
    (ConformalAmp.FullConformal(δy_range = 0.0:0.05:5.0, coverage = target_coverage), ConformalAmp.ExactHomotopy()),
    (ConformalAmp.JackknifePlus(coverage = target_coverage), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))
]

experiments_names = [
    # "FullConformal with AMP",
    "FullConformal with  Exact Homotopy",
    "Jackknife+"
]

coverage_means = []
coverage_stds  = []

mean_length_means = []
mean_length_stds = []

mean_time_means = []
mean_time_stds = []

println("Coverage = $target_coverage")
seeds = 10

for i in eachindex(experiments)
    fcp, method = experiments[i]
    coverage_gamp_list, mean_length_gamp_list, gamp_time_list = run_experiment(x, y, method, fcp, seeds, n_train)
    push!(coverage_means, mean(coverage_gamp_list))
    push!(coverage_stds, std(coverage_gamp_list))
    push!(mean_length_means, mean(mean_length_gamp_list))
    push!(mean_length_stds, std(mean_length_gamp_list))
    push!(mean_time_means, mean(gamp_time_list))
    push!(mean_time_stds, std(gamp_time_list))
end

for i in eachindex(experiments_names)
    println("For $(experiments_names[i])")
    println("Coverage : $(coverage_means[i]) \\pm $(coverage_stds[i])")
    println("Mean length : $(mean_length_means[i]) \\pm $(mean_length_stds[i])")
    println("Mean time : $(mean_time_means[i]) \\pm $(mean_time_stds[i])")
    println()
    println()
end


### for Boston dataset

# For FullConformal with Taylor-AMP
# Coverage : 0.9050980392156859 ± 0.03219727038205604
# Mean length : 1.5948088235294111 ± 0.04987056473008866
# Mean time : 0.04585503596885532 ± 0.006887180235245329
# 
# 
# For FullConformal with  Exact Homotopy
# Coverage : 0.9046078431372548 ± 0.03252652736501571
# Mean length : 1.565039940419928 ± 0.046907920062581016
# Mean time : 0.10189604759216309 ± NaN
# 
# 
# For Jackknife+
# Coverage : 0.9014705882352936 ± 0.0323925287607827
# Mean length : 1.5715471269360568 ± 0.047082429655938667
# Mean time : 0.0008743417029287301 ± 5.7201755081056386e-5

### for Riboflavin dataset