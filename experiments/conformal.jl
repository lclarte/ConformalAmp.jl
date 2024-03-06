"""
Script to compare the intervals that are returned by the LOO _vs_ conformal prediction 
"""

using ConformalAmp
using LinearAlgebra
using Plots
using ProgressBars
using StableRNGs: StableRNG

function compare_jacknife_fcp_confidence_intervals()
    # define the problem and sample the data
    d   = 500
    α = 0.5
    λ = 1e-6
    coverage = 0.9

    problem = ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)

    rng = StableRNG(0)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    xtest = ConformalAmp.sample_data_any_n(rng, d, 1)[1, :]

    jplus_confidence_set = ConformalAmp.get_confidence_interval(problem, X, y, xtest, ConformalAmp.JacknifePlus(coverage = coverage), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))
    fcp_confidence_set = ConformalAmp.get_confidence_interval(problem, X, y, xtest, ConformalAmp.FullConformal(coverage = coverage, δy_range = 0:0.1:2.0), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))

    println("J+  :",  minimum(jplus_confidence_set), " ", maximum(jplus_confidence_set))
    println("FCP : ", minimum(fcp_confidence_set), " ", maximum(fcp_confidence_set)) 
end

# TODO : Continuer ca, regarder si notre approximation d'ordre 1 fonctione bien pour 
# changer l'estimateur x̂ en variant le label de δy > 0

function fcp_with_order_one()
    d = 200
    α = 0.5
    λ = 1.0

    problem = ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)
    n = ConformalAmp.get_n(α, d)
    (; X, w, y) = ConformalAmp.sample_all(StableRNG(0), problem, d)
    # run AMP to compute the estimator
    result = ConformalAmp.gamp(problem, X, y; max_iter = 100, rtol = 1e-5)
    (; x̂) = result
    Δx̂ = ConformalAmp.get_order_one_amp_perturbation(problem, X, y, result; rtol = 1e-5)

    # For a list of δy, compare the change in w obtained with ERM compared with AMP

    δy_range = 0:0.1:2.0
    w_differences_erm_gamp = []
    y₀ = y[n]
    for δy in ProgressBar(δy_range)
        y[n] = y₀ + δy
        # fit using ERM
        ŵerm = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
        push!(w_differences_erm_gamp, norm(ŵerm - (x̂ .+ δy .* Δx̂), 2))
    end
    pl = plot(δy_range, w_differences_erm_gamp, label = "ERM - GAMP", xlabel="δy", ylabel="||ŵ - (x̂ + δy Δx̂)||")
    display(pl)
end

fcp_with_order_one()