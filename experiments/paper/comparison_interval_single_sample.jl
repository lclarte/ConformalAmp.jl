"""
For a single sample, with different regularizations λ, we compare the confidence intervals of the real full conformal prediction _vs_ the accelerated one
"""

<<<<<<< HEAD
using ConformalAmp
using ProgressBars
using StableRNGs: StableRNG

rng = StableRNG(10)
d = 100
n = 50
α = n / d

function create_problem(λ)
    return ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0) # we don't use the λ of this one
=======
using NPZ
using ConformalAmp
using ProgressBars
using StableRNGs: StableRNG
using Plots

rng = StableRNG(10)
α = 2.0
d = 100

n = Integer(α * d)

function create_problem(λ)
    return ConformalAmp.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0) 
>>>>>>> clean
end

true_problem = create_problem(1.0)
coverage = 0.9
<<<<<<< HEAD
fcp  = ConformalAmp.FullConformal(δy_range = 0.0:0.1:10.0, coverage = 0.9)
gamp = ConformalAmp.GAMPTaylor(max_iter = 1000, rtol = 1e-5)
=======
fcp  = ConformalAmp.FullConformal(δy_range = 0.0:0.2:5.0, coverage = coverage)
gamp = ConformalAmp.GAMPTaylor(max_iter = 1000, rtol = 1e-3)
>>>>>>> clean

(; X, w, y) = ConformalAmp.sample_all(rng, true_problem, d)

xtest   = ConformalAmp.sample_data_any_n(rng, d, 1)
ytest   = ConformalAmp.sample_labels(rng, true_problem, xtest, w)

xtest = xtest[1, :]
ytest = ytest[1]

<<<<<<< HEAD
λ_range = (10.0).^(-2:0.1:1)
=======
λ_range  = (10.0).^(-2.0:0.25:1.0)
>>>>>>> clean

min_interval = []
max_interval = []

erm_max_interval = []
erm_min_interval = []

for λ in ProgressBar(λ_range)
    # compute the interval with given λ
    problem = create_problem(λ)
    interval = ConformalAmp.get_confidence_interval(problem, X, y, xtest, fcp, gamp)
    erm_interval = ConformalAmp.get_confidence_interval(problem, X, y, xtest, fcp, ConformalAmp.ERM())
<<<<<<< HEAD
    
=======
    # erm_interval = (0.0, 0.0)

>>>>>>> clean
    push!(min_interval, minimum(interval))
    push!(max_interval, maximum(interval))

    push!(erm_min_interval, minimum(erm_interval))
    push!(erm_max_interval, maximum(erm_interval))
end

# plot the intervals in the form of a band
# for each lambda, do a vertical line between min_interval[i] and max_interval[i]

<<<<<<< HEAD

plt = plot(λ_range, min_interval, label="AMP-Taylor", color=:black, xscale=:log10, xaxis="λ", yaxis="y")
plot!(plt, λ_range, max_interval, label="", color=:black)

plot!(plt, λ_range, erm_min_interval, label="Exact", color=:black, linestyle=:dash)
plot!(plt, λ_range, erm_max_interval, label="", color=:black, linestyle=:dash)

# plot a line at y = ytest
plot!(plt, λ_range, ytest * ones(length(λ_range)), label="Target y", color=:black, linestyle=:dashdot)
savefig(plt, "experiments/paper/interval_single_sample_$true_problem.pdf")
=======
fs = 12

# amp_label = "AMP-Taylor"
# erm_label = "Exact"
# target_label = "Target y"

amp_label = ""
erm_label =""
target_label = ""

plt = plot(λ_range, min_interval, marker=:circle, label=amp_label, color=:black, xscale=:log10, xaxis="λ", yaxis="y",
xtickfontsize=fs,ytickfontsize=fs, legendfontsize=fs)
plot!(plt, λ_range, max_interval, marker=:circle, label="", color=:black)

plot!(plt, λ_range, erm_min_interval, marker=:circle, label=erm_label, color=:black, linestyle=:dash)
plot!(plt, λ_range, erm_max_interval, marker=:circle, label="", color=:black, linestyle=:dash)

# plot a line at y = ytest
plot!(plt, λ_range, ytest * ones(length(λ_range)), label=target_label, color=:black, linestyle=:dashdot)
display(plt)
savefig(plt, "experiments/paper/interval_single_sample_$(true_problem)_$(d)_α=$(α).pdf")

# save the results in a npz file 

println("lambda_range = $λ_range")
println("min_interval = $min_interval")
println("max_interval = $max_interval")
println("erm_min_interval = $erm_min_interval")
println("erm_max_interval = $erm_max_interval")
>>>>>>> clean
