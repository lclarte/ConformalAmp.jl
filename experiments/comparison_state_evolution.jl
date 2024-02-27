using BootstrapAsymptotics
using ConformalAmp
using Plots
using ProgressBars
using Revise
using StableRNGs: StableRNG

d = 500
λ = 0.1

rng = StableRNG(0)

α_range = 1.0:0.5:20.0
correlations =  []
correlation_se = []

for α in ProgressBar(α_range)
    n = ceil(Int, α * d)
    problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
    # problem = ConformalAmp.Logistic(α = α, λ = λ)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, n)
    (; xhat, vhat) = ConformalAmp.gamp(problem, X, y; rtol=1e-4)
    # 
    problem_ba = BootstrapAsymptotics.Ridge(α = α, Δ = 1.0, λ = λ)
    # problem_ba = BootstrapAsymptotics.Logistic(α = α, λ = λ)
    res = BootstrapAsymptotics.state_evolution(
        problem_ba,
        BootstrapAsymptotics.FullResampling(),
        BootstrapAsymptotics.FullResampling();
        rtol=1e-4,
        max_iteration=100
    )
    push!(correlations, (w'xhat) / d)
    push!(correlation_se, res.overlaps.m[1])
end

pl = scatter(α_range, correlations)
plot!(pl, α_range, correlation_se)
display(pl)