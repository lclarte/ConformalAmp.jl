"""
Script to compare the residuals given by exact leave-one-out and Taylor AMP. 
The corresponding figure is in the appendix 
"""

using Plots
using StableRNGs: StableRNG
using Statistics

using ConformalAmp

rng = StableRNG(0)

d_range = [100, 200, 500]

plt = plot()
α = 0.5
problem = ConformalAmp.Ridge(α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)

for d in d_range
    n = Integer(α * d)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    result_gamp = ConformalAmp.gamp(problem, X, y; rtol=1e-4, max_iter = 100)
    Δresult_gamp= ConformalAmp.compute_order_one_perturbation_gamp(problem, X, y, result_gamp; max_iter = 100, rtol = 1e-3)

    Ŵ_0     = ConformalAmp.get_cavity_means_from_gamp(X, result_gamp)
        # we divide by δy_perturbation so that we only have to multiply by δy after
    ΔŴ      = ConformalAmp.get_derivative_cavity_means(X, result_gamp, Δresult_gamp)

    δy = 5.0

    y[end] = y[end] + δy
    w_loo_erm = ConformalAmp.fit_leave_one_out(problem, X, y, ConformalAmp.ERM())
    w_loo_amp = Ŵ_0 + ΔŴ * δy

    # compute the residuals
    residuals_erm = y - diag(X * w_loo_erm')
    residuals_amp = y - diag(X * w_loo_amp')

    scatter!(plt, residuals_erm, residuals_amp, xaxis="ERM", yaxis="GAMP", label="d = $d", alpha=0.5)
    plot!(plt, residuals_erm, residuals_erm, color=:black, label="")
end

display(plt)