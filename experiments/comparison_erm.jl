"""
Checks that the leave-one-out estimators correspond obtained with GAMP are correct
by comparing them with "true" ERM estimators.
"""

using ConformalAmp
using Plots
using Revise
using StableRNGs

α = 5.0
d = 100
n = ceil(Int, α * d)
λ = 0.1

problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
(; X, w, y) = ConformalAmp.sample_all(StableRNG(0), problem, n)

(; xhat, vhat, ω) = ConformalAmp.gamp(problem, X, y; rtol=1e-1)
x̂_cavities = ConformalAmp.get_cavity_means_from_gamp(problem, X, y, xhat, vhat, ω)

x̂_loo = ConformalAmp.fit_leave_one_out(problem, X, y)

for i in 1:1
    pl = scatter(x̂_cavities[i, :], x̂_loo[i, :])
    display(pl)
end