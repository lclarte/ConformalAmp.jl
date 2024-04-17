"""
Comparison of the size of the intervals provided by Bayes optimal estimator and those given by conformal prediction. 
e.g. for ridge regression or the LASSO 
TODO : Implémenter le Bayes-optimal pour le lasso, deja fait quelque part 
"""

using Plots
using ConformalAmp
using StableRNGs: StableRNG

function test_logistic()
    α_vals = 0.5:20.0
    d = 500
    rng = StableRNG(0)

    q_overlaps, m_overlaps = [], []

    for α in α_vals
        problem = ConformalAmp.BayesOptimalLogistic(; α = α)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        result = ConformalAmp.gamp(problem, X, y; rtol=1e-4)

        push!(m_overlaps, result.x̂' * w / d)
        push!(q_overlaps, result.x̂' * result.x̂ / d)

    end

    plt = plot(α_vals, m_overlaps, label="m overlap")
    plot!(α_vals, q_overlaps, label="q overlap")
    display(plt)
end

function test_lasso()
    α_vals = 1.0:1.0:20.0
    d = 200
    rng = StableRNG(0)

    q_overlaps, m_overlaps = [], []

    for α in α_vals
        problem = ConformalAmp.BayesOptimalLasso(; α = α)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        result = ConformalAmp.gamp(problem, X, y; rtol=1e-4)

        push!(m_overlaps, result.x̂' * w / d)
        push!(q_overlaps, result.x̂' * result.x̂ / d)

    end

    plt = plot(α_vals, m_overlaps, label="m overlap")
    plot!(α_vals, q_overlaps, label="q overlap")
    display(plt)
end

test_lasso()