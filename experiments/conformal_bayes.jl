"""
Comparison of the size of the intervals provided by Bayes optimal estimator and those given by conformal prediction. 
e.g. for ridge regression or the LASSO 
TODO : Implémenter le Bayes-optimal pour le lasso, deja fait quelque part 
"""

using Plots
using ConformalAmp
using StableRNGs: StableRNG
using Statistics

function test(problem_str::String)
    α_vals = 0.5:20.0
    d = 500
    rng = StableRNG(0)

    q_overlaps, m_overlaps, v_overlaps = [], [], []

    for α in α_vals
        if problem_str == "ridge"
            problem = ConformalAmp.BayesOptimalRidge(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
        elseif problem_str == "lasso"
            problem = ConformalAmp.BayesOptimalLasso(; α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
        elseif problem_str == "logistic"
            problem = ConformalAmp.BayesOptimalLogistic(; α = α, λ = 1.0)
        else
            error("Unknown problem")
        end
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        result = ConformalAmp.gamp(problem, X, y; rtol=1e-4)

        push!(m_overlaps, result.x̂' * w / d)
        push!(q_overlaps, result.x̂' * result.x̂ / d)
        push!(v_overlaps, Statistics.mean(result.v̂))
    end

    plt = plot(α_vals, m_overlaps, label="m", xaxis="α", yaxis="overlap", title="$problem_str regression")
    plot!(α_vals, q_overlaps, label="q")
    ρ = problem_str == "lasso" ? 2.0 : 1.0
    plot!(α_vals, ρ .- v_overlaps, label="ρ - v")
    display(plt)
end

test("lasso")